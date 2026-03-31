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
    train_monthly_model,
    train_yearly_model,
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

        daily_model = train_daily_model(x_train.to_numpy(), y_train.to_numpy(), DailyTrainerConfig())
        daily_pred = predict_daily_model(daily_model, x_test.to_numpy())
        daily_metrics = evaluate_metrics(y_test.to_numpy(), daily_pred)

        mapper = ContractPriceMapper().fit(daily_pred, test_feat["contract_price"].to_numpy())
        contract_pred = mapper.predict(daily_pred, test_feat.get("policy_strength", pd.Series(np.zeros(len(test_feat)))).to_numpy())
        contract_metrics = evaluate_metrics(test_feat["contract_price"].to_numpy(), contract_pred)

        month_train = aggregate_monthly(train_df)
        month_test = aggregate_monthly(test_df)
        month_drop = [c for c in ["date", "market_price", "contract_price"] if c in month_train.columns]
        x_m_train = month_train.drop(columns=month_drop)
        y_m_train = month_train["market_price"]
        x_m_test = month_test.drop(columns=month_drop)
        y_m_test = month_test["market_price"]
        if len(x_m_train) >= 12 and len(x_m_test) >= 1:
            monthly_model = train_monthly_model(x_m_train, y_m_train)
            month_pred = monthly_model.predict(x_m_test)
            monthly_metrics = evaluate_metrics(y_m_test.to_numpy(), month_pred)
        else:
            monthly_metrics = {"rmse": np.nan, "mape": np.nan, "mae": np.nan}

        year_train = aggregate_yearly(train_df)
        year_test = aggregate_yearly(test_df)
        year_drop = [c for c in ["date", "market_price", "contract_price"] if c in year_train.columns]
        x_y_train = year_train.drop(columns=year_drop)
        y_y_train = year_train["market_price"]
        x_y_test = year_test.drop(columns=year_drop)
        y_y_test = year_test["market_price"]
        if len(x_y_train) >= 3 and len(x_y_test) >= 1:
            yearly_bundle = train_yearly_model(x_y_train, y_y_train)
            year_pred = yearly_bundle.model.predict(yearly_bundle.scaler.transform(x_y_test))
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
