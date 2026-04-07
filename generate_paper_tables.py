from __future__ import annotations

from src.reporting import build_paper_experiment_tables


if __name__ == "__main__":
    meta = build_paper_experiment_tables("reports")
    print("论文实验表格已生成")
    print(meta)
