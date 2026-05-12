# Core 运行根目录

`core/` 是当前训练与服务的主运行目录。

## 训练

```bash
cd core
python train.py
```

## 服务

```bash
cd core
gunicorn app:app --bind 0.0.0.0:7860 --timeout 120
```

## Runtime directories

- `models/`: 训练产物（运行时生成）
- `reports/`: 指标、回测、可观测性与论文资产（运行时生成）
- `data/`: 原始/中间/整理数据层（运行时生成或维护）

## 说明

- 生产服务通常使用 `gunicorn app:app`，此路径不会读取 `app.py` 中 `__main__` 分支的 `APP_ENV` 配置。
- 技术全景与流程说明请参考 `core/docs/煤价预测系统技术总文档.md`。
