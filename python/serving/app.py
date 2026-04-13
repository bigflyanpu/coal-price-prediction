"""Industrial serving entry (compat mode).

Current implementation reuses root `app.py` to avoid breaking production,
while providing a stable import path for future migration.
"""

from __future__ import annotations

from app import app  # noqa: F401
