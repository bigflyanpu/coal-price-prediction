from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(f"[migrate] repository root: {root}")
    print("[migrate] directory tree already bootstrapped. Add file migration rules as needed.")


if __name__ == "__main__":
    main()
