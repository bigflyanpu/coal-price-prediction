from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    lags: tuple[int, ...] = (1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60)
    windows: tuple[int, ...] = (3, 5, 7, 10, 14, 21, 30, 45, 60)
    max_features_keep: int = 200


def _wavelet_energy(series: pd.Series, level: int = 3) -> pd.Series:
    """Compute wavelet-like energy decomposition with pywt fallback."""
    try:
        import pywt

        vals = series.ffill().bfill().to_numpy()
        coeffs = pywt.wavedec(vals, "db2", level=level)
        recon = np.zeros_like(vals, dtype=float)
        for c in coeffs[1:]:
            recon[: len(c)] += np.abs(c)
        return pd.Series(np.interp(np.arange(len(vals)), np.linspace(0, len(vals) - 1, len(recon)), recon), index=series.index)
    except Exception:
        smoothed = series.rolling(14, min_periods=1).mean()
        return (series - smoothed).abs()


def build_feature_library(df: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    out = df.sort_values("date").copy()

    base_numeric = [
        c
        for c in out.columns
        if c != "date" and pd.api.types.is_numeric_dtype(out[c])
    ]

    # Temporal basis
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_year"] = out["date"].dt.dayofyear
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    # Lag / rolling / trend / volatility
    for col in base_numeric:
        for lag in cfg.lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
            out[f"{col}_diff_{lag}"] = out[col] - out[col].shift(lag)
            out[f"{col}_pct_{lag}"] = out[col].pct_change(lag)

        for w in cfg.windows:
            roll = out[col].rolling(w, min_periods=2)
            out[f"{col}_roll_mean_{w}"] = roll.mean()
            out[f"{col}_roll_std_{w}"] = roll.std()
            out[f"{col}_roll_min_{w}"] = roll.min()
            out[f"{col}_roll_max_{w}"] = roll.max()
            out[f"{col}_z_{w}"] = (out[col] - out[f"{col}_roll_mean_{w}"]) / (out[f"{col}_roll_std_{w}"] + 1e-6)

        out[f"{col}_ewm_7"] = out[col].ewm(span=7, adjust=False).mean()
        out[f"{col}_ewm_30"] = out[col].ewm(span=30, adjust=False).mean()
        out[f"{col}_wavelet_energy"] = _wavelet_energy(out[col])

    # Cross-source interactions
    interactions = [
        ("policy_strength", "sentiment_score"),
        ("policy_strength", "market_price"),
        ("power_consumption", "temperature"),
        ("port_inventory", "rail_transport"),
        ("import_volume", "coal_output"),
        ("sentiment_heat", "policy_uncertainty"),
    ]
    for a, b in interactions:
        if a in out.columns and b in out.columns:
            out[f"{a}_x_{b}"] = out[a] * out[b]
            out[f"{a}_div_{b}"] = out[a] / (out[b].abs() + 1e-6)
            out[f"{a}_plus_{b}"] = out[a] + out[b]

    out = out.replace([np.inf, -np.inf], np.nan)
    burn_in = max(cfg.lags) if cfg.lags else 0
    if burn_in > 0 and len(out) > burn_in:
        out = out.iloc[burn_in:].copy()
    out = out.ffill().bfill().fillna(0.0)
    return out.reset_index(drop=True)


def select_core_features_xgboost(
    feature_df: pd.DataFrame,
    target_col: str,
    keep_top_k: int = 200,
) -> Tuple[pd.DataFrame, List[str]]:
    candidates = [c for c in feature_df.columns if c not in {"date", target_col, "contract_price"}]
    x = feature_df[candidates]
    y = feature_df[target_col]
    if x.empty or len(x) == 0:
        raise ValueError("feature_df 为空，无法进行特征筛选。请检查数据预处理是否产生全空样本。")

    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=2,
        )
        model.fit(x, y)
        imp = pd.Series(model.feature_importances_, index=candidates).sort_values(ascending=False)
    except Exception:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=2)
        model.fit(x, y)
        imp = pd.Series(model.feature_importances_, index=candidates).sort_values(ascending=False)

    selected = list(imp.head(min(keep_top_k, len(imp))).index)
    return feature_df[["date", target_col, "contract_price"] + selected].copy(), selected


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {c: "mean" for c in df.columns if c != "date"}
    if "precipitation" in agg_dict:
        agg_dict["precipitation"] = "sum"
    monthly = df.set_index("date").resample("MS").agg(agg_dict).reset_index()
    monthly["month_id"] = monthly["date"].dt.month
    monthly["quarter"] = monthly["date"].dt.quarter
    return monthly


def aggregate_yearly(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {c: "mean" for c in df.columns if c != "date"}
    if "precipitation" in agg_dict:
        agg_dict["precipitation"] = "sum"
    yearly = df.set_index("date").resample("YS").agg(agg_dict).reset_index()
    yearly["year"] = yearly["date"].dt.year
    return yearly


def enrich_yearly_features(yearly_df: pd.DataFrame) -> pd.DataFrame:
    out = yearly_df.sort_values("date").copy()

    base_cols = [
        "coal_output",
        "import_volume",
        "industrial_value_added",
        "policy_strength",
        "sentiment_heat",
        "sentiment_score",
        "power_consumption",
        "port_inventory",
        "rail_transport",
    ]
    for col in base_cols:
        if col in out.columns:
            out[f"{col}_yoy"] = out[col].pct_change().replace([np.inf, -np.inf], np.nan)
            out[f"{col}_trend"] = out[col].diff()
            out[f"{col}_ma2"] = out[col].rolling(2, min_periods=1).mean()
            out[f"{col}_vol3"] = out[col].rolling(3, min_periods=2).std()

    if "monthly_pred_std" in out.columns and "monthly_pred_mean" in out.columns:
        out["scenario_vol_ratio"] = out["monthly_pred_std"] / (out["monthly_pred_mean"].abs() + 1e-6)
        out["monthly_pred_cv"] = out["monthly_pred_std"] / (out["monthly_pred_mean"].abs() + 1e-6)
    if "monthly_pred_max" in out.columns and "monthly_pred_min" in out.columns:
        out["monthly_pred_range"] = out["monthly_pred_max"] - out["monthly_pred_min"]

    if "policy_strength" in out.columns and "sentiment_score" in out.columns:
        out["policy_sentiment_gap"] = out["policy_strength"] - out["sentiment_score"]
    if "coal_output" in out.columns and "import_volume" in out.columns and "power_consumption" in out.columns:
        out["supply_demand_ratio"] = (out["coal_output"] + out["import_volume"]) / (out["power_consumption"].abs() + 1e-6)
    if "port_inventory" in out.columns and "rail_transport" in out.columns:
        out["inventory_transport_ratio"] = out["port_inventory"] / (out["rail_transport"].abs() + 1e-6)

    if "year" in out.columns:
        out["year_index"] = out["year"] - int(out["year"].min())

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return out
