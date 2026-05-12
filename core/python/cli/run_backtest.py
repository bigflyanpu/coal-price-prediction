from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import pandas as pd

from src.backtest import rolling_backtest


if __name__ == "__main__":
    curated_path = Path(os.getenv("BACKTEST_DATA_PATH", "data/curated/daily.csv"))
    if not curated_path.exists():
        raise FileNotFoundError(f"回测输入不存在: {curated_path}")
    df = pd.read_csv(curated_path, parse_dates=["date"])
    start_year = int(os.getenv("BACKTEST_START_YEAR", "2021"))
    end_year = int(os.getenv("BACKTEST_END_YEAR", "2024"))
    result = rolling_backtest(df, start_test_year=start_year, end_test_year=end_year)
    print("[cli] backtest summary")
    print(result.backtest_summary)
