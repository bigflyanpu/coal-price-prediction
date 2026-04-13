from __future__ import annotations

import os
import sys
from pathlib import Path

BASE = Path(__file__).parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.pipeline import train_all


if __name__ == "__main__":
    fast_mode = os.getenv("FAST_MODE", "0") == "1"
    refresh_cache = os.getenv("REFRESH_CACHE", "0") == "1"
    result = train_all(fast_mode=fast_mode, refresh_cache=refresh_cache, verbose=True)
    print("训练完成（严格对齐版）")
    print("在线留出集指标:", result.online_metrics)
    print("滚动回测摘要:", result.backtest_summary)
    print("元数据:", result.metadata)
