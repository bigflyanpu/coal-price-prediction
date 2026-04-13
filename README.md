---
title: Coal Price Prediction
emoji: "⛏️"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# 煤炭双轨定价多尺度预测系统（已彻底收口到 `core/`）

本仓库已完成结构收口：**所有生产可用代码、配置、模型流程统一在 `core/`**。  
根目录仅保留部署与兼容入口，不再作为主开发路径。

## 1) 唯一项目根

- 主项目树：`core/`
- 主服务入口：`core/app.py`
- 主训练入口：`core/train.py`
- 主依赖文件：`core/requirements.txt`

## 2) 本地运行（唯一命令）

### 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r core/requirements.txt
```

### 训练

```bash
cd core
python train.py
```

### 启动服务

```bash
cd core
gunicorn app:app --bind 0.0.0.0:7860 --timeout 120
```

## 3) 部署链路（已切换到 core）

- Docker 构建：使用 `core/requirements.txt`，并在 `core/` 下训练与启动
- Procfile：`cd core && gunicorn app:app ...`
- Render：build/start 均在 `core/` 下执行
- GitHub Actions -> HF Space：上传时已忽略旧目录，仅保留 `core/` 主树和部署必要文件

## 4) 模型说明（当前线上实现）

- 日度：LSTM + Transformer
- 月度：LightGBM
- 年度：SVR + Ridge 混合
- 双轨映射：ContractPriceMapper（市场价 -> 长协价）
- 评估：MAPE / RMSE / MAE + 滚动回测

## 5) 兼容入口（仅过渡）

- 根目录 `app.py`、`train.py` 仅为兼容代理，会转发到 `core/`。
- 新功能开发与部署请只修改 `core/` 下内容。

## 6) 在线地址

- Hugging Face Space: <https://huggingface.co/spaces/bigflyanpu/coal_price_prediction>

