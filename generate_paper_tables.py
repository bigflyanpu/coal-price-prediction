from __future__ import annotations

from src.reporting import build_paper_assets


if __name__ == "__main__":
    meta = build_paper_assets("reports")
    print("论文表格与图表资产已生成")
    print(meta)
