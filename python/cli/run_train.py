from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import train_all


def main() -> None:
    fast_mode = os.getenv("FAST_MODE", "0") == "1"
    refresh_cache = os.getenv("REFRESH_CACHE", "0") == "1"
    result = train_all(fast_mode=fast_mode, refresh_cache=refresh_cache, verbose=True)
    print("[cli] train finished")
    print(result.metadata)


if __name__ == "__main__":
    main()
