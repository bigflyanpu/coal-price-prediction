from __future__ import annotations

import os
from typing import Any

import numpy as np

_CPP_ENABLED = os.getenv("COAL_CPP_CORE", "0") == "1"
_CPP_MODULE = None
_CPP_ERROR = ""
if _CPP_ENABLED:
    try:
        import coal_cpp_core as _CPP_MODULE  # type: ignore
    except Exception as exc:  # pragma: no cover
        _CPP_MODULE = None
        _CPP_ERROR = str(exc)


def cpp_status() -> dict[str, Any]:
    if _CPP_MODULE is None:
        return {"enabled": False, "error": _CPP_ERROR}
    try:
        version = int(_CPP_MODULE.io_bridge_version())
    except Exception:
        version = -1
    return {"enabled": True, "version": version}


def clamp_daily_prediction(pred: float, last_price: float, max_abs_jump: float) -> float:
    if _CPP_MODULE is not None:
        try:
            return float(_CPP_MODULE.clamp_daily_prediction(pred, last_price, max_abs_jump))
        except Exception:
            pass
    return float(np.clip(pred, last_price - max_abs_jump, last_price + max_abs_jump))


def spread_signal_level(market_price: float, contract_price: float, warn_thr: float = 20.0, critical_thr: float = 40.0) -> int:
    if _CPP_MODULE is not None:
        try:
            signal = _CPP_MODULE.spread_signal(market_price, contract_price, warn_thr, critical_thr)
            return int(signal)
        except Exception:
            pass
    spread = abs(float(market_price) - float(contract_price))
    if spread >= critical_thr:
        return 2
    if spread >= warn_thr:
        return 1
    return 0

