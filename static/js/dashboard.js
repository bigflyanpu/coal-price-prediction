const { createApp, ref, onMounted, nextTick } = Vue;

const app = createApp({
  setup() {
    const loading = ref(true);
    const prediction = ref({});
    const backtestSummary = ref({});
    const dashboardData = ref({});
    
    // Form state
    const csvPath = ref('');
    const isPredicting = ref(false);
    const msg = ref(null); // { type: 'ok' | 'err', text: '' }
    
    // Animation state for numbers
    const flashState = ref({
      d1: false, d2: false, m1: false, y1: false
    });

    let chartInstance = null;

    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
    const fetchJsonWithTimeout = async (url, timeoutMs = 12000) => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
      } finally {
        clearTimeout(timer);
      }
    };

    const fetchDashboard = async () => {
      const endpoints = ['/api/dashboard_full', '/api/dashboard'];
      let lastErr = null;
      try {
        let data = null;
        for (const ep of endpoints) {
          for (let attempt = 1; attempt <= 3; attempt++) {
            try {
              const body = await fetchJsonWithTimeout(ep + '?t=' + new Date().getTime());
              if (ep === '/api/dashboard_full') {
                data = body;
              } else {
                // fallback endpoint response shape
                let fallbackBacktest = body.backtest_summary || {};
                if (!fallbackBacktest || JSON.stringify(fallbackBacktest) === '{}') {
                  try {
                    fallbackBacktest = await fetchJsonWithTimeout('/api/backtest?t=' + new Date().getTime());
                  } catch (_) {
                    // ignore and keep empty fallbackBacktest
                  }
                }
                data = {
                  prediction: body.prediction || {},
                  backtest_summary: fallbackBacktest || {},
                  dashboard_data: body || {},
                };
              }
              break;
            } catch (err) {
              lastErr = err;
              if (attempt < 3) {
                await sleep(700 * attempt);
              }
            }
          }
          if (data) break;
        }

        if (!data) {
          throw lastErr || new Error('无法获取仪表盘数据');
        }

        prediction.value = data.prediction || {};
        backtestSummary.value = data.backtest_summary || {};
        dashboardData.value = data.dashboard_data || {};
        loading.value = false;
        
        // Init chart after DOM updates
        nextTick(() => {
          initChart();
        });
      } catch (err) {
        console.error('Failed to load dashboard data:', err);
        msg.value = { type: 'err', text: '无法加载初始数据，请检查服务状态。' };
        loading.value = false; // 确保发生错误时关闭 loading
      }
    };

    const initChart = () => {
      const chartDom = document.getElementById('mapeChart');
      if (!chartDom) return;
      if (typeof echarts === 'undefined') {
        chartDom.innerHTML = '<div style="padding:12px;color:#ef4444;font-size:12px;">图表库加载失败，请刷新页面重试。</div>';
        return;
      }
      
      chartInstance = echarts.init(chartDom, 'dark');
      
      const style = getComputedStyle(document.body);
      const blue = style.getPropertyValue('--accent-blue').trim() || '#3b82f6';
      const purple = style.getPropertyValue('--accent-purple').trim() || '#8b5cf6';
      const green = style.getPropertyValue('--accent-green').trim() || '#10b981';

      const kpis = dashboardData.value.kpis || {};
      
      const option = {
        backgroundColor: 'transparent',
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'shadow' },
          formatter: '{b} MAPE: {c}%'
        },
        grid: {
          top: '10%',
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          data: ['日度市场', '月度市场', '年度市场'],
          axisTick: { alignWithLabel: true },
          axisLabel: { color: '#9ca3af', fontSize: 13 },
          axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }
        },
        yAxis: {
          type: 'value',
          splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } },
          axisLabel: { color: '#6b7280' }
        },
        series: [
          {
            type: 'bar',
            barWidth: '40%',
            data: [
              { value: kpis.daily_mape || 0, itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: blue }, { offset: 1, color: 'rgba(59, 130, 246, 0.2)' }]) } },
              { value: kpis.monthly_mape || 0, itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: purple }, { offset: 1, color: 'rgba(139, 92, 246, 0.2)' }]) } },
              { value: kpis.yearly_mape || 0, itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: green }, { offset: 1, color: 'rgba(16, 185, 129, 0.2)' }]) } }
            ],
            itemStyle: {
              borderRadius: [6, 6, 0, 0]
            }
          }
        ]
      };
      
      chartInstance.setOption(option);
      
      // Handle resize
      window.addEventListener('resize', () => {
        chartInstance && chartInstance.resize();
      });
    };

    const triggerFlash = () => {
      Object.keys(flashState.value).forEach(k => flashState.value[k] = true);
      setTimeout(() => {
        Object.keys(flashState.value).forEach(k => flashState.value[k] = false);
      }, 400);
    };

    const runPredict = async () => {
      if (isPredicting.value) return;
      
      isPredicting.value = true;
      msg.value = null;
      
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ csv_path: csvPath.value || undefined }),
        });
        
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || '预测失败');
        
        prediction.value.next_day_market_price = data.next_day_market_price;
        prediction.value.next_day_contract_price = data.next_day_contract_price;
        prediction.value.next_month_market_price = data.next_month_market_price;
        prediction.value.next_year_market_price = data.next_year_market_price;
        
        triggerFlash();
        msg.value = { type: 'ok', text: '预测成功，结果已更新。' };
      } catch (err) {
        msg.value = { type: 'err', text: err.message || '请求失败' };
      } finally {
        isPredicting.value = false;
      }
    };

    onMounted(() => {
      fetchDashboard();
    });

    const formatJSON = (obj) => {
      if (!obj) return loading.value ? '加载中...' : '暂无数据';
      const str = JSON.stringify(obj, null, 2);
      if (str === '{}' || str === '[]') return loading.value ? '加载中...' : '暂无数据';
      return str;
    };

    return {
      loading,
      prediction,
      backtestSummary,
      csvPath,
      isPredicting,
      msg,
      flashState,
      runPredict,
      formatJSON
    };
  }
});

app.mount('#app');