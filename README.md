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

## 项目总览（给新成员的 60 秒理解）

- **定位**：面向“双轨定价”（市场价+长协价）的多尺度预测系统，覆盖日/月/年三层决策。
- **核心价值**：
  - 日度高频波动预警（市场监控）
  - 月度供需结构分析（经营决策）
  - 年度中枢推演（政策与战略）
- **在线形态**：Flask API + Vue/ECharts 驾驶舱，通过 GitHub Actions 自动同步到 Hugging Face Space。

文本数据采集链路已升级为“生产化增量模式”：

- 优先抓取可配置政策/舆情 RSS 源（`config/text_sources.json`）
- 自动重试与退避、按时间窗口增量抓取、跨轮去重
- 运行状态与健康报告自动落盘（`reports/text_source_runs.csv`、`reports/text_source_health.json`）
- 内置源级质量评分与告警阈值（`quality_alert`）
- 支持按源覆盖阈值（`quality_alert_overrides`）
- 抓取失败时自动回退：历史本地源数据 -> 模拟兜底数据

## 模型与推理逻辑（清晰版）

### 1) 模型分层

- **日度模型**：`LSTM + Transformer`（`LSTMTransformerRegressor`）
  - 输入：日频特征（筛选后）
  - 输出：`next_day_market_price`
- **双轨映射器**：`ContractPriceMapper`
  - 输入：日度市场价预测 + 政策强度
  - 输出：`next_day_contract_price`
- **月度模型**：`LightGBM`
  - 输入：月度聚合特征（含日度预测统计）
  - 输出：`next_month_market_price`
- **年度模型**：`SVR + Ridge` 混合（`YearlyBundle`）
  - 输入：年度聚合特征（含月度预测统计）
  - 输出：`next_year_market_price`

### 2) 在线推理链路（`/api/predict`）

1. 加载模型状态（`models/` 下产物）
2. 对输入数据做特征构建与多尺度聚合
3. 计算日度预测 -> 合同价映射
4. 计算月度预测、年度预测
5. 返回四个核心预测值

> 说明：线上已加入日度预测稳健约束（防止次日价格与现货出现异常跳变）。

### 3) 仓库关键产物

- 模型文件：`models/`
- 基础状态：`models/base_data.joblib`
- 回测结果：`reports/rolling_backtest_summary.json`
- 元信息：`reports/metadata.json`

## 文档

- **[模型层技术说明（基于代码现状）](docs/models_technical.md)**：多尺度模型族、数据依赖、训练与推理差异、落盘产物、与申报书模型族差距及答辩引用建议

## 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 工业级目录树（已重建）

本仓库已落地工业级目录骨架，核心分层如下：

- `data/raw`：原始输入层（Wind/文本/天气/Excel）
- `data/clean`：清洗与对齐后的训练输入层
- `cpp_core`：C++ 高频特征/信号/风控核心
- `python`：Python 训练、评估、服务、CLI
- `configs`：统一配置中心（app/data/model/cpp）
- `schemas`：跨语言数据契约
- `artifacts`：模型、报告、日志、快照产物

目录职责详见：`docs/directory_tree_industrial.md`

## 2. 训练（生成模型、回测、报告）

```bash
python train.py
```

工业化入口（与现有入口并行）：

```bash
python python/cli/run_train.py
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

工业化服务入口（兼容模式）：

```bash
python python/cli/run_service.py
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

## 5.1 GitHub 构建与部署逻辑（当前线上真实流程）

当前不是传统 SSH 发布，而是：

`GitHub(main) -> GitHub Actions(sync-to-hf-space.yml) -> Hugging Face Space(upload_folder)`

### 触发条件

- `push` 到 `main`
- 手动触发 `workflow_dispatch`

### 工作流具体步骤（`.github/workflows/sync-to-hf-space.yml`）

1. `actions/checkout` 拉取仓库
2. `setup-python@v5` 安装 Python 3.11
3. 安装 `huggingface_hub`
4. 读取 GitHub Secret `HF_TOKEN`
5. 执行 `HfApi.upload_folder(...)` 将仓库内容上传到 Space `bigflyanpu/coal_price_prediction`

### 当前忽略上传的路径（非常重要）

工作流中配置了 `ignore_patterns`，以下内容不会从 GitHub 自动同步到 Space：

- `.git/*`
- `.venv/*`
- `__pycache__/*`
- `*.pyc`
- `data/*`
- `models/*`

### 这意味着什么

- **代码会更新**，但 `data/` 与 `models/` 不会被这条工作流覆盖。
- Space 运行依赖它自身容器内已有的模型与数据文件。
- 若线上需要更新模型权重，需要单独设计“模型产物发布策略”（例如单独上传到 HF Dataset/Model Repo，或调整工作流白名单）。

### 线上排障优先级（建议）

1. 先看 GitHub Actions 是否成功
2. 再看 Space 构建日志是否启动成功
3. 若页面“初始化失败”，优先检查 `/api/dashboard_full` 是否返回合法 JSON（无 `NaN/Inf`）

## 6. 数据契约

契约配置在：`config/data_contract.json`

四类源的必需字段：

- structured：`date market_price contract_price port_inventory rail_transport power_consumption import_volume coal_output industrial_value_added`
- policy_text：`date doc_id title body source url`
- sentiment_text：`date news_id title body media url`
- weather：`date region temperature precipitation wind_speed`
