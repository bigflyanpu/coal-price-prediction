from __future__ import annotations

from src.data_audit import build_data_gap_audit


if __name__ == "__main__":
    meta = build_data_gap_audit(start="2018-01-01", end="2024-12-31", report_dir="reports", raw_dir="data/raw")
    print("数据缺口审计已生成")
    print(meta)
