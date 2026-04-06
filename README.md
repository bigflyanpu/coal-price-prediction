---
title: Coal Price Prediction
emoji: "⛏️"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# 煤炭双轨定价多尺度预测系统（严格对齐PDF）

本版本从原型升级为研究级流程，按 PDF 技术路线实现：

- 数据层：结构化 + 政策文本 + 舆情文本 + 气象数据，统一契约与质检
- NLP层：BERT语义嵌入 + LDA主题，输出12维政策冲击指数
- 特征层：1800+多尺度因子（滞后、滚动、交互、wavelet-like）+ XGBoost筛选至200维
- 模型层：日度 LSTM-Transformer / 月度 LightGBM / 年度 Robust-SVR / 双轨规则映射
- 验证层：2018-2024滚动回测（样本外）
- 系统层：Flask可视化看板 + API + Hugging Face Space部署

文本数据采集链路已升级为“生产化增量模式”：

- 优先抓取可配置政策/舆情 RSS 源（`config/text_sources.json`）
- 自动重试与退避、按时间窗口增量抓取、跨轮去重
- 运行状态与健康报告自动落盘（`reports/text_source_runs.csv`、`reports/text_source_health.json`）
- 内置源级质量评分与告警阈值（`quality_alert`）
- 支持按源覆盖阈值（`quality_alert_overrides`）
- 抓取失败时自动回退：历史本地源数据 -> 模拟兜底数据

## 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 训练（生成模型、回测、报告）

```bash
python train.py
```

可选环境变量：

- `FAST_MODE=1`：快速训练（默认关闭实时文本抓取）
- `LIVE_TEXT_SOURCES=1`：强制开启实时文本抓取
- `LIVE_TEXT_SOURCES=0`：强制关闭实时文本抓取

训练完成后产物：

- 模型：`models/`
- 回测与报告：`reports/rolling_backtest_folds.csv`、`reports/rolling_backtest_summary.json`
- 元数据：`reports/metadata.json`
- 数据质量：`reports/data_quality.csv`、`reports/curated_quality.csv`

## 3. 启动网页

```bash
python app.py
```

访问：`http://127.0.0.1:7860`

网页包含：实时预测、滚动回测摘要、模型数据版本、数据质量监控。

## 4. API

### 4.1 预测

```bash
curl -X POST http://127.0.0.1:7860/api/predict   -H 'Content-Type: application/json'   -d '{}'
```

### 4.2 回测摘要

```bash
curl http://127.0.0.1:7860/api/backtest
```

### 4.3 元数据

```bash
curl http://127.0.0.1:7860/api/metadata
```

### 4.4 数据健康

```bash
curl http://127.0.0.1:7860/api/data-health
```

### 4.5 文本源健康

```bash
curl http://127.0.0.1:7860/api/text-source-health
```

支持过滤参数（便于巡检）：

```bash
curl "http://127.0.0.1:7860/api/text-source-health?kind=policy_text"
curl "http://127.0.0.1:7860/api/text-source-health?status=critical"
curl "http://127.0.0.1:7860/api/text-source-health?kind=sentiment_text&status=warn"
```

## 5. Hugging Face Spaces部署（无需绑卡）

- Space：<https://huggingface.co/spaces/bigflyanpu/coal_price_prediction>
- 应用域名：`https://<你的space名>.hf.space`

如果本机无法直连HF，仓库已提供 GitHub Actions 自动同步流程（`sync-to-hf-space.yml`）。

## 6. 数据契约

契约配置在：`config/data_contract.json`

四类源的必需字段：

- structured：`date market_price contract_price port_inventory rail_transport power_consumption import_volume coal_output industrial_value_added`
- policy_text：`date doc_id title body source url`
- sentiment_text：`date news_id title body media url`
- weather：`date region temperature precipitation wind_speed`
