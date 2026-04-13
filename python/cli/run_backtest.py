from __future__ import annotations

from src.pipeline import train_all


if __name__ == "__main__":
    result = train_all(fast_mode=True, refresh_cache=False, verbose=True)
    print("[cli] backtest summary")
    print(result.backtest_summary)
