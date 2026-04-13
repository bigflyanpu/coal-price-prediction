from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import train_all


if __name__ == "__main__":
    result = train_all(fast_mode=True, refresh_cache=False, verbose=True)
    print("[cli] backtest summary")
    print(result.backtest_summary)
