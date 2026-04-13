"""Legacy compatibility training entrypoint.

Canonical training entry is `core/train.py`.
"""

import runpy
import os
from pathlib import Path


if __name__ == "__main__":
    core_dir = Path(__file__).resolve().parent / "core"
    os.chdir(core_dir)
    runpy.run_path(str(core_dir / "train.py"), run_name="__main__")
