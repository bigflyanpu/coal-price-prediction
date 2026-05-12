from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIGS_DIR = BASE_DIR / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


@dataclass
class DailyTrainRuntimeConfig:
    epochs: int = 18
    learning_rate: float = 8e-4
    batch_size: int = 64


@dataclass
class ModelRouteConfig:
    daily_variant: str = "lstm_transformer"
    monthly_variant: str = "lightgbm"


def load_daily_train_config() -> DailyTrainRuntimeConfig:
    raw = _load_yaml(CONFIGS_DIR / "model" / "daily_lstm_transformer.yaml")
    params = raw.get("params", {}) if isinstance(raw, dict) else {}
    return DailyTrainRuntimeConfig(
        epochs=int(params.get("epochs", 18)),
        learning_rate=float(params.get("learning_rate", 8e-4)),
        batch_size=int(params.get("batch_size", 64)),
    )


def load_monthly_candidate_defaults() -> dict[str, Any]:
    raw = _load_yaml(CONFIGS_DIR / "model" / "monthly_lightgbm.yaml")
    params = raw.get("params", {}) if isinstance(raw, dict) else {}
    return {
        "n_estimators": int(params.get("n_estimators", 450)),
        "learning_rate": float(params.get("learning_rate", 0.03)),
        "max_depth": int(params.get("max_depth", 5)),
        "num_leaves": int(params.get("num_leaves", 48)),
    }


def load_yearly_default_params() -> dict[str, Any]:
    raw = _load_yaml(CONFIGS_DIR / "model" / "yearly_svr.yaml")
    params = raw.get("params", {}) if isinstance(raw, dict) else {}
    svr = params.get("svr", {}) if isinstance(params, dict) else {}
    return {
        "C": float(svr.get("C", 12.0)),
        "epsilon": float(svr.get("epsilon", 0.05)),
        "kernel": str(svr.get("kernel", "rbf")),
        "ridge_alpha": float(params.get("ridge_alpha", 2.0)),
        "blend_weight_svr": float(params.get("blend_weight_svr", 0.65)),
    }


def load_app_runtime_config(env_name: str) -> dict[str, Any]:
    env = (env_name or "dev").strip().lower()
    cfg = _load_yaml(CONFIGS_DIR / "app" / f"{env}.yaml")
    return {
        "host": str(cfg.get("host", "0.0.0.0")),
        "port": int(cfg.get("port", 7860)),
        "log_level": str(cfg.get("log_level", "info")),
        "env": str(cfg.get("env", env)),
    }


def load_model_route_config() -> ModelRouteConfig:
    raw = _load_yaml(CONFIGS_DIR / "model" / "ensemble.yaml")
    routes = raw.get("model_routes", {}) if isinstance(raw, dict) else {}
    return ModelRouteConfig(
        daily_variant=str(routes.get("daily_variant", "lstm_transformer")).strip().lower(),
        monthly_variant=str(routes.get("monthly_variant", "lightgbm")).strip().lower(),
    )

