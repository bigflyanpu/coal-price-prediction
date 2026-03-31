from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .features import aggregate_monthly, aggregate_yearly, build_feature_library, select_core_features_xgboost
from .models import (
    ContractPriceMapper,
    DailyTrainerConfig,
    evaluate_metrics,
    predict_daily_model,
    train_daily_model,
    predict_yearly_bundle,
    train_best_monthly_model,
    train_best_yearly_model,
)


@dataclass
class RollingBacktestResult:
    fold_metrics: pd.DataFrame
    summary_metrics: Dict[str, Dict[str, float]]


def rolling_backtest(df: pd.DataFrame, start_test_year: int = 2021, end_test_year: int = 2024) -> RollingBacktestResult:
    rows: List[dict] = []

    for test_year in range(start_test_year, end_test_year + 1):
        train_df = df[df["date"].dt.year < test_year].copy()
        test_df = df[df["date"].dt.year == test_year].copy()
        if len(train_df) < 365 or len(test_df) < 30:
            continue

        train_feat = build_feature_library(train_df)
        train_sel, selected = select_core_features_xgboost(train_feat, target_col="market_price", keep_top_k=200)

        test_feat = build_feature_library(pd.concat([train_df.tail(120), test_df], ignore_index=True))
        test_feat = test_feat[test_feat["date"].dt.year == test_year].copy()

        x_cols = [c for c in selected if c in test_feat.columns]
        x_train = train_sel[x_cols]
        y_train = train_sel["market_price"]

        x_test = test_feat[x_cols]
        y_test = test_feat["market_price"]

        daily_bundle = train_daily_model(x_train.to_numpy(), y_train.to_numpy(), DailyTrainerConfig())
        daily_pred = predict_daily_model(daily_bundle, x_test.to_numpy())
        daily_metrics = evaluate_metrics(y_test.to_numpy(), daily_pred)

        # Fit mapping on train period to avoid test leakage.
        train_contract = train_sel["contract_price"].to_numpy()
        daily_pred_train = predict_daily_model(daily_bundle, x_train.to_numpy())
        mapper = ContractPriceMapper().fit(daily_pred_train, train_contract)
        contract_pred = mapper.predict(daily_pred, test_feat.get("policy_strength", pd.Series(np.zeros(len(test_feat)))).to_numpy())
        contract_metrics = evaluate_metrics(test_feat["contract_price"].to_numpy(), contract_pred)

        daily_pred_train_frame = pd.DataFrame({"date": train_sel["date"].to_numpy(), "daily_pred": daily_pred_train})
        daily_pred_test_frame = pd.DataFrame({"date": test_feat["date"].to_numpy(), "daily_pred": daily_pred})
        daily_pred_all = pd.concat([daily_pred_train_frame, daily_pred_test_frame], ignore_index=True)

        month_all = aggregate_monthly(pd.concat([train_df, test_df], ignore_index=True))
        pred_month = (
            daily_pred_all.set_index("date")["daily_pred"]
            .resample("MS")
            .agg(["mean", "std", "min", "max"])
            .rename(
                columns={
                    "mean": "daily_pred_mean",
                    "std": "daily_pred_std",
                    "min": "daily_pred_min",
                    "max": "daily_pred_max",
                }
            )
            .reset_index()
        )
        month_all = month_all.merge(pred_month, on="date", how="left").sort_values("date").ffill().bfill()
        month_train = month_all[month_all["date"].dt.year < test_year].copy()
        month_test = month_all[month_all["date"].dt.year == test_year].copy()
        month_drop = [c for c in ["date", "market_price", "contract_price"] if c in month_train.columns]
        x_m_train = month_train.drop(columns=month_drop)
        y_m_train = month_train["market_price"]
        x_m_test = month_test.drop(columns=month_drop)
        y_m_test = month_test["market_price"]
        if len(x_m_train) >= 12 and len(x_m_test) >= 1:
            monthly_model, _, _ = train_best_monthly_model(x_m_train, y_m_train)
            month_pred = monthly_model.predict(x_m_test)
            monthly_metrics = evaluate_metrics(y_m_test.to_numpy(), month_pred)
        else:
            monthly_metrics = {"rmse": np.nan, "mape": np.nan, "mae": np.nan}

        month_all_pred = monthly_model.predict(month_all.drop(columns=month_drop)) if len(x_m_train) >= 12 else np.zeros(len(month_all))
        yearly_all = aggregate_yearly(pd.concat([train_df, test_df], ignore_index=True))
        month_pred_tmp = month_all[["date"]].copy()
        month_pred_tmp["monthly_pred"] = month_all_pred
        pred_year = (
            month_pred_tmp.set_index("date")["monthly_pred"]
            .resample("YS")
            .agg(["mean", "std", "min", "max"])
            .rename(
                columns={
                    "mean": "monthly_pred_mean",
                    "std": "monthly_pred_std",
                    "min": "monthly_pred_min",
                    "max": "monthly_pred_max",
                }
            )
            .reset_index()
        )
        yearly_all = yearly_all.merge(pred_year, on="date", how="left")
        for col in ["coal_output", "import_volume", "industrial_value_added", "policy_strength", "sentiment_heat"]:
            if col in yearly_all.columns:
                yearly_all[f"{col}_yoy"] = yearly_all[col].pct_change().replace([np.inf, -np.inf], np.nan)
                yearly_all[f"{col}_trend"] = yearly_all[col].diff()
        if "monthly_pred_std" in yearly_all.columns and "monthly_pred_mean" in yearly_all.columns:
            yearly_all["scenario_vol_ratio"] = yearly_all["monthly_pred_std"] / (yearly_all["monthly_pred_mean"].abs() + 1e-6)
        yearly_all = yearly_all.sort_values("date").ffill().bfill()
        year_train = yearly_all[yearly_all["date"].dt.year < test_year].copy()
        year_test = yearly_all[yearly_all["date"].dt.year == test_year].copy()
        year_drop = [c for c in ["date", "market_price", "contract_price"] if c in year_train.columns]
        x_y_train = year_train.drop(columns=year_drop)
        y_y_train = year_train["market_price"]
        x_y_test = year_test.drop(columns=year_drop)
        y_y_test = year_test["market_price"]
        if len(x_y_train) >= 3 and len(x_y_test) >= 1:
            yearly_bundle, _, _ = train_best_yearly_model(x_y_train, y_y_train)
            year_pred = predict_yearly_bundle(yearly_bundle, x_y_test)
            yearly_metrics = evaluate_metrics(y_y_test.to_numpy(), year_pred)
        else:
            yearly_metrics = {"rmse": np.nan, "mape": np.nan, "mae": np.nan}

        rows.extend(
            [
                {"test_year": test_year, "scale": "daily_market", **daily_metrics},
                {"test_year": test_year, "scale": "daily_contract", **contract_metrics},
                {"test_year": test_year, "scale": "monthly_market", **monthly_metrics},
                {"test_year": test_year, "scale": "yearly_market", **yearly_metrics},
            ]
        )

    fold_metrics = pd.DataFrame(rows)
    summary = {}
    if not fold_metrics.empty:
        for scale, grp in fold_metrics.groupby("scale"):
            summary[scale] = {
                "rmse": float(np.nanmean(grp["rmse"])),
                "mape": float(np.nanmean(grp["mape"])),
                "mae": float(np.nanmean(grp["mae"])),
            }
    return RollingBacktestResult(fold_metrics=fold_metrics, summary_metrics=summary)
