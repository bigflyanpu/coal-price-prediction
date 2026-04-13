# 工业级目录树与职责说明

本文档定义本项目的工业级目录组织方式，目标是将数据、模型、C++核心、部署流程解耦并标准化。

## 1. 顶层目录职责

- `configs/`：唯一配置入口，按 app/data/model/cpp 分域管理。
- `schemas/`：跨语言数据协议，保证 Python 与 C++ 的输入输出一致。
- `data/raw/`：原始数据落盘层，不允许在此层做特征加工。
- `data/clean/`：清洗对齐后数据层，作为训练与回测标准输入。
- `data/mart/`：面向前端/API 的轻量宽表层。
- `cpp_core/`：高性能特征、信号、风控计算核心。
- `python/`：数据处理、训练、评估、服务、CLI 全流程 Python 实现。
- `artifacts/`：模型、报告、日志、快照等不可逆产物。
- `tests/`：单元、集成、契约测试。
- `scripts/`：环境初始化、迁移、发布脚本。

## 2. 跨语言数据流建议（执行标准）

1. Python ingestion 将外部源写入 `data/raw/*`。
2. Python preprocessing 输出 `data/clean/*`，并做 schema 校验。
3. Python 通过 pybind 调用 C++ 核心，生成高频特征/风控信号到 `data/clean/features`。
4. Python models 训练并产出到 `artifacts/models`。
5. Python evaluation 回测并输出 `artifacts/reports`。
6. Python serving 读取 `artifacts/models` + `data/mart` 对外提供 API。

## 3. 兼容策略

- 保留当前根目录入口（`app.py`、`train.py`）用于线上稳定运行。
- 新增工业化入口（`python/cli/run_train.py`、`python/serving/app.py`）用于后续迁移。
- 迁移完成后可将根目录入口降级为兼容代理。

