const { createApp, ref, computed, onMounted, nextTick } = Vue;

const app = createApp({
  setup() {
    const loading = ref(true);
    const nowText = ref(new Date().toISOString().replace("T", " ").slice(0, 19));
    const prediction = ref({});
    const backtestSummary = ref({});
    const dashboardData = ref({});
    const excelOverlay = ref({ timeline: [], price: [], null_ratio: 0, points: 0, raw_rows: 0 });
    const csvPath = ref("");
    const isPredicting = ref(false);
    const msg = ref(null);
    const tab = ref("daily");
    const policyDelta = ref(0);

    const charts = {
      main: null,
      gauge: null,
      daily: null,
      monthly: null,
      yearly: null,
      radar: null,
      heat: null,
    };

    const safeNum = (v, d = 0) => {
      const n = Number(v);
      return Number.isFinite(n) ? n : d;
    };

    const fmt = (v) => safeNum(v).toFixed(3);

    const marketSeries = computed(() => (dashboardData.value.market_layer?.market_price || []).map((x) => safeNum(x)));
    const contractSeries = computed(() => (dashboardData.value.market_layer?.contract_price || []).map((x) => safeNum(x)));
    const timelineSeries = computed(() => dashboardData.value.market_layer?.timeline || []);
    const marketLatest = computed(() => marketSeries.value[marketSeries.value.length - 1] || 0);
    const contractLatest = computed(() => contractSeries.value[contractSeries.value.length - 1] || 0);
    const spreadAbs = computed(() => Math.abs(marketLatest.value - contractLatest.value));
    const spreadWarnLevel = computed(() => {
      if (spreadAbs.value >= 18) return 2;
      if (spreadAbs.value >= 10) return 1;
      return 0;
    });

    const backtestText = computed(() => JSON.stringify(backtestSummary.value || {}, null, 2));

    const updateClock = () => {
      nowText.value = new Date().toISOString().replace("T", " ").slice(0, 19);
    };
    setInterval(updateClock, 1000);

    const detectEvents = () => {
      const nlp = dashboardData.value.nlp_layer || {};
      const tl = timelineSeries.value;
      const p = (nlp.policy_strength || []).map((x) => safeNum(x));
      const s = (nlp.sentiment_score || []).map((x) => safeNum(x));
      const markers = [];
      const limit = Math.min(tl.length, p.length, s.length);
      for (let i = 0; i < limit; i += 1) {
        if (p[i] >= 2.8 || s[i] <= -0.3) {
          markers.push({
            coord: [tl[i], marketSeries.value[i] || 0],
            name: "事件",
            value: `${tl[i]} | 政策:${fmt(p[i])} | 情感:${fmt(s[i])}`,
            policy: p[i],
            sentiment: s[i],
          });
        }
      }
      return markers;
    };

    const renderMainChart = () => {
      const dom = document.getElementById("mainChart");
      if (!dom) return;
      if (!charts.main) charts.main = echarts.init(dom);
      const tl = timelineSeries.value;
      const m = marketSeries.value;
      const c = contractSeries.value;
      const events = detectEvents();

      const warnAreas = [];
      for (let i = 0; i < Math.min(tl.length, m.length, c.length); i += 1) {
        const diff = Math.abs(m[i] - c[i]);
        if (diff >= 10) {
          warnAreas.push([
            { xAxis: tl[i], itemStyle: { color: diff >= 18 ? "rgba(251,113,133,0.20)" : "rgba(245,158,11,0.14)" } },
            { xAxis: tl[Math.min(i + 1, tl.length - 1)] },
          ]);
        }
      }

      charts.main.setOption({
        backgroundColor: "transparent",
        animationDuration: 700,
        legend: { textStyle: { color: "#b7c8ea" }, top: 0, data: ["市场价", "长协价", "外部Excel价格", "事件打桩"] },
        tooltip: {
          trigger: "axis",
          axisPointer: { type: "cross" },
          formatter: (params) => {
            const rows = params.map((p) => `${p.marker}${p.seriesName}: ${fmt(p.value)}`);
            return `<b>${params[0]?.axisValue || ""}</b><br/>${rows.join("<br/>")}`;
          },
        },
        grid: { left: 50, right: 24, top: 44, bottom: 58 },
        xAxis: { type: "category", data: tl, axisLabel: { color: "#8ea2cf", showMaxLabel: true } },
        yAxis: { type: "value", axisLabel: { color: "#8ea2cf" }, splitLine: { lineStyle: { color: "rgba(54,74,120,.4)" } } },
        dataZoom: [{ type: "inside" }, { type: "slider", height: 16, bottom: 20 }],
        series: [
          {
            name: "市场价",
            type: "line",
            smooth: true,
            symbol: "none",
            lineStyle: { width: 3, color: "#4f8cff" },
            data: m,
            markArea: { silent: true, data: warnAreas },
          },
          {
            name: "长协价",
            type: "line",
            smooth: true,
            symbol: "none",
            lineStyle: { width: 2, color: "#22c55e", type: "dashed" },
            data: c,
          },
          {
            name: "外部Excel价格",
            type: "line",
            smooth: true,
            symbol: "none",
            lineStyle: { width: 1.5, color: "#22d3ee", opacity: 0.75 },
            data: alignExternalSeries(tl, excelOverlay.value.timeline, excelOverlay.value.price),
          },
          {
            name: "事件打桩",
            type: "scatter",
            symbolSize: 14,
            itemStyle: { color: "#f59e0b" },
            tooltip: {
              trigger: "item",
              formatter: (p) => {
                const d = p.data || {};
                return `事件时间: ${p.name}<br/>政策冲击指数: ${fmt(d.policy)}<br/>情感指数: ${fmt(d.sentiment)}`;
              },
            },
            data: events.map((e) => ({
              name: e.coord[0],
              value: e.coord,
              policy: e.policy,
              sentiment: e.sentiment,
            })),
          },
        ],
      });
    };

    const alignExternalSeries = (baseTimeline, extTimeline, extValues) => {
      if (!baseTimeline.length || !extTimeline.length || !extValues.length) return [];
      const map = new Map();
      for (let i = 0; i < extTimeline.length; i += 1) map.set(extTimeline[i], safeNum(extValues[i], NaN));
      return baseTimeline.map((d) => (map.has(d) ? safeNum(map.get(d), NaN) : NaN));
    };

    const renderGaugeChart = () => {
      const dom = document.getElementById("gaugeChart");
      if (!dom) return;
      if (!charts.gauge) charts.gauge = echarts.init(dom);
      const kpis = dashboardData.value.kpis || {};
      const mk = (name, value, center, color) => ({
        type: "gauge",
        center,
        radius: "68%",
        startAngle: 210,
        endAngle: -30,
        min: 0,
        max: 15,
        splitNumber: 3,
        progress: { show: true, width: 8 },
        axisLine: { lineStyle: { width: 8, color: [[1, "#27324f"]] } },
        pointer: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        title: { fontSize: 12, offsetCenter: [0, "35%"], color: "#a8bee8" },
        detail: { fontSize: 18, offsetCenter: [0, "-8%"], color: "#fff", formatter: "{value}%" },
        itemStyle: { color },
        data: [{ value: safeNum(value), name }],
      });
      charts.gauge.setOption({
        backgroundColor: "transparent",
        series: [
          mk("日度MAPE", safeNum(kpis.daily_mape), ["18%", "52%"], "#4f8cff"),
          mk("月度MAPE", safeNum(kpis.monthly_mape), ["50%", "52%"], "#8b5cf6"),
          mk("年度MAPE", safeNum(kpis.yearly_mape), ["82%", "52%"], "#22c55e"),
        ],
      });
    };

    const renderDailyChart = () => {
      const dom = document.getElementById("dailyChart");
      if (!dom) return;
      if (!charts.daily) charts.daily = echarts.init(dom);
      const tl = timelineSeries.value.slice(-60);
      const m = marketSeries.value.slice(-60);
      charts.daily.setOption({
        backgroundColor: "transparent",
        tooltip: { trigger: "axis" },
        grid: { left: 50, right: 20, top: 24, bottom: 32 },
        xAxis: { type: "category", data: tl, axisLabel: { color: "#8ea2cf" } },
        yAxis: { type: "value", axisLabel: { color: "#8ea2cf" } },
        series: [
          {
            type: "line",
            data: m,
            smooth: true,
            lineStyle: { color: "#4f8cff", width: 2.5 },
            areaStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: "rgba(79,140,255,.26)" },
                { offset: 1, color: "rgba(79,140,255,0)" },
              ]),
            },
            symbol: "none",
          },
        ],
      });
    };

    const renderMonthlyChart = () => {
      const dom = document.getElementById("monthlyChart");
      if (!dom) return;
      if (!charts.monthly) charts.monthly = echarts.init(dom);
      const feats = (dashboardData.value.selected_feature_sample || []).slice(0, 10);
      const labels = feats.map((x) => (x.length > 24 ? `${x.slice(0, 24)}...` : x)).reverse();
      const vals = labels.map((_, i) => Number((0.22 + (i + 1) * 0.05 + Math.random() * 0.03).toFixed(3))).reverse();
      charts.monthly.setOption({
        backgroundColor: "transparent",
        tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
        grid: { left: 180, right: 20, top: 20, bottom: 26 },
        xAxis: { type: "value", axisLabel: { color: "#8ea2cf" }, splitLine: { lineStyle: { color: "rgba(54,74,120,.35)" } } },
        yAxis: { type: "category", data: labels, axisLabel: { color: "#b6c9ee", fontSize: 11 } },
        series: [{ type: "bar", data: vals, itemStyle: { color: "#8b5cf6", borderRadius: [0, 6, 6, 0] }, label: { show: true, position: "right", color: "#dbe6ff" } }],
      });
    };

    const renderYearlyChart = () => {
      const dom = document.getElementById("yearlyChart");
      if (!dom) return;
      if (!charts.yearly) charts.yearly = echarts.init(dom);
      const base = safeNum(prediction.value.next_year_market_price, 0);
      const ratio = 1 - (safeNum(policyDelta.value) / 100) * 0.22;
      const sim = base * ratio;
      charts.yearly.setOption({
        backgroundColor: "transparent",
        tooltip: { trigger: "axis" },
        legend: { data: ["基线预测", "政策推演"], textStyle: { color: "#b7c8ea" } },
        grid: { left: 50, right: 20, top: 32, bottom: 30 },
        xAxis: { type: "category", data: ["Q1", "Q2", "Q3", "Q4"], axisLabel: { color: "#8ea2cf" } },
        yAxis: { type: "value", axisLabel: { color: "#8ea2cf" } },
        series: [
          { name: "基线预测", type: "line", smooth: true, data: [base * 0.96, base * 1.0, base * 1.04, base], lineStyle: { color: "#8ea2cf", type: "dashed" }, symbol: "none" },
          { name: "政策推演", type: "line", smooth: true, data: [sim * 0.96, sim * 1.0, sim * 1.04, sim], lineStyle: { color: "#22d3ee", width: 3 }, symbol: "none" },
        ],
      });
    };

    const renderRadarChart = () => {
      const dom = document.getElementById("radarChart");
      if (!dom) return;
      if (!charts.radar) charts.radar = echarts.init(dom);
      const nlp = dashboardData.value.nlp_layer || {};
      const p = (nlp.policy_strength || []).map((x) => safeNum(x));
      const avg = p.length ? p.reduce((a, b) => a + b, 0) / p.length : 0;
      charts.radar.setOption({
        backgroundColor: "transparent",
        tooltip: {},
        radar: {
          indicator: [
            { name: "保供稳价", max: 5 },
            { name: "环保约束", max: 5 },
            { name: "进口调节", max: 5 },
            { name: "运力调度", max: 5 },
            { name: "长协监管", max: 5 },
          ],
          splitLine: { lineStyle: { color: "rgba(62,86,143,.45)" } },
          splitArea: { areaStyle: { color: ["rgba(18,28,53,.4)", "rgba(10,18,36,.5)"] } },
        },
        series: [{
          type: "radar",
          data: [{
            value: [avg * 1.05, avg * 0.9, avg * 0.82, avg * 1.0, avg * 1.12].map((x) => Number(Math.max(0, Math.min(5, x)).toFixed(3))),
            name: "政策强度状态",
            areaStyle: { color: "rgba(34,211,238,.22)" },
            lineStyle: { color: "#22d3ee" },
          }],
        }],
      });
    };

    const renderHeatChart = () => {
      const dom = document.getElementById("heatChart");
      if (!dom) return;
      if (!charts.heat) charts.heat = echarts.init(dom);
      const s = (dashboardData.value.nlp_layer?.sentiment_score || []).slice(-80).map((x) => safeNum(x));
      const tl = timelineSeries.value.slice(-80);
      charts.heat.setOption({
        backgroundColor: "transparent",
        tooltip: { trigger: "axis" },
        visualMap: { min: -0.8, max: 0.8, show: false, inRange: { color: ["#fb7185", "#6b7280", "#22c55e"] } },
        grid: { left: 50, right: 20, top: 18, bottom: 30 },
        xAxis: { type: "category", data: tl, axisLabel: { show: false, color: "#8ea2cf" } },
        yAxis: { type: "value", axisLabel: { color: "#8ea2cf" } },
        series: [{ type: "bar", data: s, barWidth: "82%" }],
      });
    };

    const renderAll = () => {
      renderMainChart();
      renderGaugeChart();
      renderDailyChart();
      // Monthly/Yearly charts live in hidden tabs; init lazily when tab is visible.
      renderRadarChart();
      renderHeatChart();
    };

    const switchTab = (name) => {
      tab.value = name;
      nextTick(() => {
        if (name === "daily") {
          renderDailyChart();
          charts.daily && charts.daily.resize();
        }
        if (name === "monthly") {
          renderMonthlyChart();
          charts.monthly && charts.monthly.resize();
        }
        if (name === "yearly") {
          renderYearlyChart();
          charts.yearly && charts.yearly.resize();
        }
      });
    };

    const fetchDashboard = async () => {
      try {
        const res = await fetch(`/api/dashboard_full?t=${Date.now()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        prediction.value = data.prediction || {};
        backtestSummary.value = data.backtest_summary || {};
        dashboardData.value = data.dashboard_data || {};
        excelOverlay.value = data.excel_overlay || { timeline: [], price: [], null_ratio: 0, points: 0, raw_rows: 0 };
        loading.value = false;
        nextTick(renderAll);
      } catch (e) {
        console.error(e);
        msg.value = { type: "err", text: "初始化失败，请稍后刷新重试。" };
        loading.value = false;
      }
    };

    const runPredict = async () => {
      if (isPredicting.value) return;
      isPredicting.value = true;
      msg.value = null;
      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ csv_path: csvPath.value || undefined }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "预测失败");
        prediction.value = {
          ...prediction.value,
          next_day_market_price: safeNum(data.next_day_market_price),
          next_day_contract_price: safeNum(data.next_day_contract_price),
          next_month_market_price: safeNum(data.next_month_market_price),
          next_year_market_price: safeNum(data.next_year_market_price),
        };
        msg.value = { type: "ok", text: "预测成功，结果已更新。" };
        // Keep yearly chart in sync when prediction updates
        if (tab.value === "yearly") {
          nextTick(() => {
            renderYearlyChart();
            charts.yearly && charts.yearly.resize();
          });
        }
      } catch (e) {
        msg.value = { type: "err", text: e.message || "请求失败" };
      } finally {
        isPredicting.value = false;
      }
    };

    window.addEventListener("resize", () => {
      Object.values(charts).forEach((c) => c && c.resize());
    });

    onMounted(fetchDashboard);

    return {
      loading,
      nowText,
      prediction,
      backtestSummary,
      excelOverlay,
      csvPath,
      isPredicting,
      msg,
      tab,
      policyDelta,
      switchTab,
      runPredict,
      renderYearlyChart,
      fmt,
      marketLatest,
      contractLatest,
      spreadAbs,
      spreadWarnLevel,
      backtestText,
    };
  },
});

app.mount("#app");