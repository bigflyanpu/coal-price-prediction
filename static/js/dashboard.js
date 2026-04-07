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

    const fetchDashboard = async () => {
      try {
        const res = await fetch('/api/dashboard_full');
        const data = await res.json();
        prediction.value = data.prediction;
        backtestSummary.value = data.backtest_summary;
        dashboardData.value = data.dashboard_data;
        loading.value = false;
        
        // Init chart after DOM updates
        nextTick(() => {
          initChart();
        });
      } catch (err) {
        console.error('Failed to load dashboard data:', err);
        msg.value = { type: 'err', text: '无法加载初始数据，请检查服务状态。' };
      }
    };

    const initChart = () => {
      const chartDom = document.getElementById('mapeChart');
      if (!chartDom) return;
      
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

    return {
      loading,
      prediction,
      backtestSummary,
      csvPath,
      isPredicting,
      msg,
      flashState,
      runPredict
    };
  }
});

app.mount('#app');