"""Legacy compatibility training entrypoint.

Canonical training entry is `core/train.py`.
"""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "core" / "train.py"), run_name="__main__")
