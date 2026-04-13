from __future__ import annotations

from python.serving.app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
