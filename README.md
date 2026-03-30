---
title: Coal Price Prediction
emoji: "⛏️"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# 煤炭价格多尺度预测系统（按 PDF 技术路线实现）

本项目实现了一个可运行的端到端原型：

- 日度模型：`LSTM + Transformer`（短期价格波动）
- 月度模型：`LightGBM`（融合日度预测均值）
- 年度模型：`RobustScaler + SVR`（长期趋势）
- 双轨映射：将市场价预测映射到长协价预测
- 系统层：`Flask` 网页 + API + 可部署配置（Render/Railway/Fly/自建云）

## 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 训练模型

```bash
python train.py
```

- 若 `data/coal_prices.csv` 不存在，会自动生成示例数据并训练。
- 训练产物保存在 `models/`。

## 3. 本地启动网页

```bash
python app.py
```

浏览器打开：`http://127.0.0.1:7860`

## 4. API 调用

```bash
curl -X POST http://127.0.0.1:7860/api/predict   -H 'Content-Type: application/json'   -d '{}'
```

或者使用你自己的 CSV：

```bash
curl -X POST http://127.0.0.1:7860/api/predict   -H 'Content-Type: application/json'   -d '{"csv_path":"/absolute/path/to/your.csv"}'
```

## 5. 部署到 Hugging Face Spaces（无需绑卡，推荐）

本项目已支持 **Docker Space**，可直接部署为长期稳定公网地址：

- 形如：`https://<你的space名>.hf.space`
- 例如：`https://bigflyanpu-coal-price-prediction.hf.space`

### 5.1 创建 Space

1. 打开 [Hugging Face Spaces](https://huggingface.co/new-space)
2. Owner 选你的账号
3. Space name 建议填：`coal-price-prediction`
4. SDK 选择：`Docker`
5. Visibility 选 `Public`（大家都能访问）
6. 点击 `Create Space`

### 5.2 上传代码（网页操作）

在新 Space 页面点 `Files` -> `Add file` -> `Upload files`，把项目文件上传（保持目录结构）：

- `Dockerfile`
- `app.py`
- `requirements.txt`
- `src/`
- `templates/`
- `train.py`
- `README.md`

上传后会自动触发构建和部署，首次构建通常几分钟到十几分钟。

### 5.3 上传代码（Git 命令，可选）

如果你想用命令行推送：

```bash
git clone https://huggingface.co/spaces/<你的用户名>/<你的space名>
cd <你的space名>
# 复制项目文件到当前目录后：
git add .
git commit -m "Deploy coal price prediction app"
git push
```

### 5.4 部署完成后访问

访问你的固定地址：

- `https://<你的space名>.hf.space`

---

## 6. Render 方案（需要绑卡）

项目已包含 `render.yaml`，可直接一键部署并获得稳定地址（`*.onrender.com`）。

### 5.1 一键部署（推荐）

1. 把项目上传到 GitHub 仓库。
2. 登录 [Render](https://render.com)，点击 `New +` -> `Blueprint`。
3. 连接你的 GitHub 仓库并选择当前项目。
4. Render 会自动读取 `render.yaml` 并创建服务。
5. 首次部署完成后，你会得到固定域名：
   - `https://<your-service-name>.onrender.com`

> `onrender.com` 子域名是长期稳定的，不会像临时隧道链接那样频繁变化。

### 5.2 绑定你自己的自定义域名（可选）

如果你有自己的域名（例如 `coal.yourdomain.com`）：

1. 在 Render 服务页进入 `Settings` -> `Custom Domains` -> `Add Custom Domain`。
2. 填入你的域名，例如 `coal.yourdomain.com`。
3. 到域名服务商（阿里云/腾讯云/Cloudflare）添加 DNS 记录：
   - 类型：`CNAME`
   - 主机记录：`coal`
   - 记录值：Render 给出的目标地址
4. 等待 DNS 生效后，Render 会自动签发 HTTPS 证书。

### 5.3 每次更新自动上线

- 推送代码到 GitHub 主分支后，Render 会自动重新部署。
- 地址保持不变，访问链接长期可用。

## 7. 数据格式要求（最小字段）

CSV 至少包含：

- `date`（日期）
- `market_price`
- `contract_price`
- `policy_index`
- `sentiment_score`
- `temperature`
- `precipitation`
- `port_inventory`
- `rail_transport`
- `power_consumption`
- `import_volume`

