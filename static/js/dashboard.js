const { createApp, ref, computed, onMounted, nextTick, watch } = Vue;

const app = createApp({
  setup() {
    const loading = ref(true);
    const currentTime = ref('');
    
    // Core Data State
    const prediction = ref({});
    const backtestSummary = ref({});
    const dashboardData = ref({});
    
    // UI State
    const activeTab = ref('daily');
    const policySimulation = ref(0);
    
    // ECharts Instances
    const charts = {};

    // 1. Precision Formatter (3 decimal places)
    const formatPrecision = (val) => {
      if (val === undefined || val === null) return '0.000';
      return Number(val).toFixed(3);
    };

    // 2. Update Time
    setInterval(() => {
      const now = new Date();
      currentTime.value = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
    }, 1000);

    // 3. Top KPIs Computed
    const topKpis = computed(() => {
      const ml = dashboardData.value?.market_layer || {};
      const mp = ml.market_price || [];
      const cp = ml.contract_price || [];
      const lastMp = mp[mp.length - 1] || 0;
      const prevMp = mp[mp.length - 2] || 0;
      const lastCp = cp[cp.length - 1] || 0;
      const prevCp = cp[cp.length - 2] || 0;
      
      // Calculate Spread Warning (Core Feature 1)
      const spread = Math.abs(lastMp - lastCp);
      const isWarning = spread > 15; // Threshold for warning glow

      return [
        { title: '长协基准价', value: lastCp, unit: '元/吨', trend: ((lastCp - prevCp)/prevCp)*100, tag: '稳定锚', tagType: 'success', warning: false },
        { title: '现货市场价', value: lastMp, unit: '元/吨', trend: ((lastMp - prevMp)/prevMp)*100, tag: '高波动', tagType: 'warning', warning: isWarning },
        { title: '港口库存总量 (Mock)', value: 7250.120, unit: '万吨', trend: -1.245, tag: '供需缓冲', tagType: 'info', warning: false },
        { title: '电企日耗量 (Mock)', value: 215.500, unit: '万吨', trend: 2.105, tag: '需求高频', tagType: 'danger', warning: false }
      ];
    });

    // 4. Simulated Yearly Price
    const simulatedYearlyPrice = computed(() => {
      const base = prediction.value.next_year_market_price || 0;
      // Simple simulation: +10% policy strength -> -2% price
      const factor = 1 - (policySimulation.value / 100) * 0.2;
      return base * factor;
    });

    // Fetch Data
    const fetchDashboard = async () => {
      try {
        const res = await fetch('/api/dashboard_full?t=' + new Date().getTime());
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();
        
        prediction.value = data.prediction;
        backtestSummary.value = data.backtest_summary;
        dashboardData.value = data.dashboard_data;
        
        setTimeout(() => {
          loading.value = false;
          nextTick(() => initAllCharts());
        }, 500); // Fake delay for tech feel

      } catch (err) {
        console.error('Failed to load dashboard data:', err);
        loading.value = false;
      }
    };

    // Initialize Charts
    const initAllCharts = () => {
      renderMainDualTrackChart();
      renderGaugeChart();
      renderDailyChart();
      // Other tabs render on change
      renderRadarChart();
      renderSentimentChart();
      
      window.addEventListener('resize', () => {
        Object.values(charts).forEach(c => c && c.resize());
      });
    };

    const handleTabChange = (tabName) => {
      nextTick(() => {
        if (tabName === 'monthly' && !charts.featureImportance) renderFeatureImportanceChart();
        if (tabName === 'yearly' && !charts.yearlySimulation) renderYearlySimulationChart();
      });
    };

    // --- Chart 1: Dual Track & Event Overlay (Core Feature 1 & 2) ---
    const renderMainDualTrackChart = () => {
      const dom = document.getElementById('mainDualTrackChart');
      if (!dom) return;
      charts.main = echarts.init(dom, 'dark');

      const ml = dashboardData.value.market_layer;
      const nlp = dashboardData.value.nlp_layer;
      
      // Calculate markAreas for spread > 15
      const markAreas = [];
      const events = [];
      
      ml.market_price.forEach((mp, i) => {
        const cp = ml.contract_price[i];
        const diff = Math.abs(mp - cp);
        
        // 1. Warning Halo (MarkArea)
        if (diff > 15) {
          markAreas.push([
            { xAxis: ml.timeline[i], itemStyle: { color: 'rgba(239, 68, 68, 0.15)' } },
            { xAxis: ml.timeline[i+1] || ml.timeline[i] }
          ]);
        }

        // 2. Event Overlay (K线打桩)
        // Find spikes in policy strength or extreme sentiment
        if (nlp.policy_strength[i] > 2.8) {
          events.push({
            name: '政策冲击',
            value: [ml.timeline[i], Math.min(...ml.market_price) - 5],
            symbol: 'pin',
            symbolSize: 24,
            itemStyle: { color: '#ef4444' },
            tooltip: {
              formatter: () => `
                <div style="font-family: 'Inter';">
                  <div style="color: #ef4444; font-weight: bold; margin-bottom: 4px;">发改委保供稳价政策出台</div>
                  <div>政策强度指数: ${formatPrecision(nlp.policy_strength[i])}</div>
                  <div>情感偏向: ${formatPrecision(nlp.sentiment_score[i])}</div>
                </div>
              `
            }
          });
        } else if (nlp.sentiment_score[i] < -0.3) {
          events.push({
            name: '极端情绪',
            value: [ml.timeline[i], Math.min(...ml.market_price) - 5],
            symbol: 'diamond',
            symbolSize: 16,
            itemStyle: { color: '#06b6d4' },
            tooltip: {
              formatter: () => `
                <div style="font-family: 'Inter';">
                  <div style="color: #06b6d4; font-weight: bold; margin-bottom: 4px;">市场恐慌情绪蔓延</div>
                  <div>情感指数: ${formatPrecision(nlp.sentiment_score[i])}</div>
                </div>
              `
            }
          });
        }
      });

      const option = {
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
        legend: { data: ['市场价', '长协价', '事件打桩'], top: 0, textStyle: { color: '#9ca3af' } },
        grid: { left: '3%', right: '4%', bottom: '10%', top: '15%', containLabel: true },
        xAxis: { type: 'category', data: ml.timeline, axisLine: { lineStyle: { color: '#374151' } } },
        yAxis: { type: 'value', min: 'dataMin', splitLine: { lineStyle: { color: '#1f2937', type: 'dashed' } } },
        series: [
          {
            name: '市场价',
            type: 'line',
            data: ml.market_price,
            smooth: true,
            symbol: 'none',
            lineStyle: { width: 3, color: '#3b82f6', shadowColor: 'rgba(59, 130, 246, 0.5)', shadowBlur: 10 },
            markArea: { data: markAreas, silent: true } // Dual-track warning halo
          },
          {
            name: '长协价',
            type: 'line',
            data: ml.contract_price,
            step: 'end', // Contract price usually changes in steps
            symbol: 'none',
            lineStyle: { width: 2, color: '#10b981', type: 'dashed' }
          },
          {
            name: '事件打桩',
            type: 'scatter',
            data: events,
            zlevel: 10
          }
        ]
      };
      charts.main.setOption(option);
    };

    // --- Chart 2: Gauges ---
    const renderGaugeChart = () => {
      const dom = document.getElementById('gaugeChart');
      if (!dom) return;
      charts.gauge = echarts.init(dom, 'dark');
      
      const kpis = dashboardData.value.kpis;
      
      const makeGauge = (name, value, center, color) => ({
        name: name,
        type: 'gauge',
        center: center,
        radius: '65%',
        startAngle: 200,
        endAngle: -20,
        min: 0,
        max: 15,
        splitNumber: 3,
        itemStyle: { color: color },
        progress: { show: true, width: 8 },
        pointer: { show: false },
        axisLine: { lineStyle: { width: 8, color: [[1, '#1f2937']] } },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        title: { show: true, offsetCenter: [0, '20%'], fontSize: 12, color: '#9ca3af' },
        detail: { show: true, offsetCenter: [0, '-10%'], fontSize: 18, fontWeight: 'bold', color: '#fff', formatter: '{value}%' },
        data: [{ value: value, name: name }]
      });

      const option = {
        backgroundColor: 'transparent',
        series: [
          makeGauge('日度 MAPE', kpis.daily_mape, ['20%', '50%'], '#3b82f6'),
          makeGauge('月度 MAPE', kpis.monthly_mape, ['50%', '50%'], '#8b5cf6'),
          makeGauge('年度 MAPE', kpis.yearly_mape, ['80%', '50%'], '#10b981')
        ]
      };
      charts.gauge.setOption(option);
    };

    // --- Chart 3: Daily Chart ---
    const renderDailyChart = () => {
      const dom = document.getElementById('dailyChart');
      if (!dom) return;
      charts.daily = echarts.init(dom, 'dark');
      
      const ml = dashboardData.value.market_layer;
      const recentTimeline = ml.timeline.slice(-30);
      const recentPrice = ml.market_price.slice(-30);
      
      // Add prediction point
      recentTimeline.push('次日预测');
      const predData = [...recentPrice];
      predData[predData.length - 1] = prediction.value.next_day_market_price;

      const option = {
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis' },
        grid: { left: '3%', right: '4%', bottom: '5%', top: '10%', containLabel: true },
        xAxis: { type: 'category', data: recentTimeline },
        yAxis: { type: 'value', min: 'dataMin' },
        series: [
          {
            type: 'line',
            data: recentPrice,
            smooth: true,
            lineStyle: { width: 2, color: '#6366f1' },
            areaStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: 'rgba(99, 102, 241, 0.3)' },
                { offset: 1, color: 'rgba(99, 102, 241, 0)' }
              ])
            }
          },
          {
            type: 'scatter',
            data: [[recentTimeline.length - 1, prediction.value.next_day_market_price]],
            symbolSize: 10,
            itemStyle: { color: '#f59e0b', shadowBlur: 10, shadowColor: '#f59e0b' }
          }
        ]
      };
      charts.daily.setOption(option);
    };

    // --- Chart 4: Feature Importance (Monthly) ---
    const renderFeatureImportanceChart = () => {
      const dom = document.getElementById('featureImportanceChart');
      if (!dom) return;
      charts.featureImportance = echarts.init(dom, 'dark');
      
      const features = dashboardData.value.selected_feature_sample.slice(0, 10).reverse();
      // Mock weights for visual
      const weights = features.map((_, i) => (Math.random() * 0.1 + (i+1)*0.05).toFixed(3));

      const option = {
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        grid: { left: '3%', right: '10%', bottom: '5%', top: '5%', containLabel: true },
        xAxis: { type: 'value', splitLine: { show: false } },
        yAxis: { type: 'category', data: features, axisLabel: { color: '#9ca3af', fontSize: 11 } },
        series: [
          {
            type: 'bar',
            data: weights,
            itemStyle: {
              color: new echarts.graphic.LinearGradient(1, 0, 0, 0, [
                { offset: 0, color: '#8b5cf6' },
                { offset: 1, color: '#3b82f6' }
              ]),
              borderRadius: [0, 4, 4, 0]
            },
            label: { show: true, position: 'right', color: '#fff', formatter: '{c}' }
          }
        ]
      };
      charts.featureImportance.setOption(option);
    };

    // --- Chart 5: Yearly Simulation ---
    const renderYearlySimulationChart = () => {
      const dom = document.getElementById('yearlySimulationChart');
      if (!dom) return;
      charts.yearlySimulation = echarts.init(dom, 'dark');
      updateYearlySimulation();
    };

    const updateYearlySimulation = () => {
      if (!charts.yearlySimulation) return;
      
      const basePrice = prediction.value.next_year_market_price;
      const simPrice = simulatedYearlyPrice.value;
      
      const option = {
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis' },
        legend: { data: ['基准预测', '政策推演'], textStyle: { color: '#9ca3af' } },
        grid: { left: '3%', right: '4%', bottom: '5%', top: '15%', containLabel: true },
        xAxis: { type: 'category', data: ['Q1', 'Q2', 'Q3', 'Q4'] },
        yAxis: { type: 'value', min: basePrice * 0.8 },
        series: [
          {
            name: '基准预测',
            type: 'line',
            data: [basePrice*0.98, basePrice*1.02, basePrice*1.05, basePrice],
            lineStyle: { type: 'dashed', color: '#6b7280' },
            itemStyle: { color: '#6b7280' }
          },
          {
            name: '政策推演',
            type: 'line',
            data: [simPrice*0.98, simPrice*1.02, simPrice*1.05, simPrice],
            smooth: true,
            lineStyle: { width: 3, color: '#10b981' },
            areaStyle: { color: 'rgba(16, 185, 129, 0.1)' }
          }
        ]
      };
      charts.yearlySimulation.setOption(option);
    };

    // --- Chart 6: Radar ---
    const renderRadarChart = () => {
      const dom = document.getElementById('radarChart');
      if (!dom) return;
      charts.radar = echarts.init(dom, 'dark');
      
      const option = {
        backgroundColor: 'transparent',
        tooltip: {},
        radar: {
          indicator: [
            { name: '保供稳价强度', max: 5 },
            { name: '环保限产约束', max: 5 },
            { name: '进口关税调节', max: 5 },
            { name: '运力调度倾斜', max: 5 },
            { name: '长协履约监管', max: 5 }
          ],
          splitArea: { areaStyle: { color: ['rgba(255,255,255,0.02)', 'rgba(255,255,255,0.05)'] } },
          axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
          splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }
        },
        series: [{
          type: 'radar',
          data: [
            {
              value: [4.2, 3.0, 2.5, 3.8, 4.5],
              name: '当前政策状态',
              itemStyle: { color: '#06b6d4' },
              areaStyle: { color: 'rgba(6, 182, 212, 0.3)' }
            }
          ]
        }]
      };
      charts.radar.setOption(option);
    };

    // --- Chart 7: Sentiment Heatmap ---
    const renderSentimentChart = () => {
      const dom = document.getElementById('sentimentChart');
      if (!dom) return;
      charts.sentiment = echarts.init(dom, 'dark');
      
      const nlp = dashboardData.value.nlp_layer;
      const scores = nlp.sentiment_score.slice(-60); // Last 60 days
      const dates = dashboardData.value.market_layer.timeline.slice(-60);
      
      const option = {
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis' },
        visualMap: {
          min: -0.5, max: 0.5,
          inRange: { color: ['#ef4444', '#1f2937', '#10b981'] }, // Red (Negative) -> Gray -> Green (Positive)
          show: false
        },
        grid: { left: '3%', right: '4%', bottom: '15%', top: '5%', containLabel: true },
        xAxis: { type: 'category', data: dates, axisLabel: { show: false } },
        yAxis: { type: 'value', splitLine: { show: false } },
        series: [{
          type: 'bar',
          data: scores,
          itemStyle: { borderRadius: 2 }
        }]
      };
      charts.sentiment.setOption(option);
    };

    onMounted(() => {
      fetchDashboard();
    });

    return {
      loading,
      currentTime,
      topKpis,
      activeTab,
      handleTabChange,
      backtestSummary,
      prediction,
      policySimulation,
      simulatedYearlyPrice,
      updateYearlySimulation,
      formatPrecision
    };
  }
});

app.use(ElementPlus);
app.mount('#app');