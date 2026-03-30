from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch

from .data import load_or_create_data
from .features import aggregate_monthly, aggregate_yearly, build_daily_features
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
class TrainOutput:
    daily_metrics: Dict[str, float]
    monthly_metrics: Dict[str, float]
    yearly_metrics: Dict[str, float]


def _split_xy(df: pd.DataFrame, target_col: str, drop_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=drop_cols)
    y = df[target_col]
    return x, y


def train_all(data_path: str | Path = "data/coal_prices.csv", model_dir: str | Path = "models") -> TrainOutput:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_create_data(data_path)
    daily = build_daily_features(df)

    daily_drop = ["date", "market_price", "contract_price"]
    x_daily, y_daily = _split_xy(daily, "market_price", daily_drop)

    split_idx = int(len(daily) * 0.8)
    x_daily_train, x_daily_test = x_daily.iloc[:split_idx], x_daily.iloc[split_idx:]
    y_daily_train, y_daily_test = y_daily.iloc[:split_idx], y_daily.iloc[split_idx:]
    contract_daily_test = daily["contract_price"].iloc[split_idx:].to_numpy()

    daily_model = train_daily_model(x_daily_train.to_numpy(), y_daily_train.to_numpy(), DailyTrainerConfig())
    daily_pred_test = predict_daily_model(daily_model, x_daily_test.to_numpy())
    daily_metrics = evaluate_metrics(y_daily_test.to_numpy(), daily_pred_test)

    # Monthly model uses monthly aggregation + daily prediction monthly mean.
    monthly = aggregate_monthly(df)
    monthly_features = monthly.copy()
    daily_pred_series = pd.Series(
        predict_daily_model(daily_model, x_daily.to_numpy()),
        index=daily["date"],
        name="daily_pred",
    )
    monthly_pred_mean = daily_pred_series.resample("MS").mean().reset_index(drop=True)
    monthly_features["daily_pred_mean"] = monthly_pred_mean

    monthly_drop = ["date", "market_price", "contract_price"]
    x_month, y_month = _split_xy(monthly_features, "market_price", monthly_drop)
    split_m = max(4, int(len(x_month) * 0.8))
    x_month_train, x_month_test = x_month.iloc[:split_m], x_month.iloc[split_m:]
    y_month_train, y_month_test = y_month.iloc[:split_m], y_month.iloc[split_m:]

    monthly_model = train_monthly_model(x_month_train, y_month_train)
    month_pred_test = monthly_model.predict(x_month_test)
    monthly_metrics = evaluate_metrics(y_month_test.to_numpy(), month_pred_test)

    yearly = aggregate_yearly(df)
    yearly_features = yearly.copy()
    month_pred_full = monthly_model.predict(x_month)
    yearly_features["monthly_pred_mean"] = (
        pd.Series(month_pred_full, index=monthly["date"]).resample("YS").mean().reset_index(drop=True)
    )

    yearly_drop = ["date", "market_price", "contract_price"]
    x_year, y_year = _split_xy(yearly_features, "market_price", yearly_drop)
    split_y = max(2, int(len(x_year) * 0.8))
    x_year_train, x_year_test = x_year.iloc[:split_y], x_year.iloc[split_y:]
    y_year_train, y_year_test = y_year.iloc[:split_y], y_year.iloc[split_y:]

    yearly_bundle = train_yearly_model(x_year_train, y_year_train)
    year_pred_test = yearly_bundle.model.predict(yearly_bundle.scaler.transform(x_year_test))
    yearly_metrics = evaluate_metrics(y_year_test.to_numpy(), year_pred_test)

    mapper = ContractPriceMapper().fit(daily_pred_test, contract_daily_test)

    torch.save(daily_model.state_dict(), model_dir / "daily_model.pt")
    joblib.dump({"columns": list(x_daily.columns)}, model_dir / "daily_meta.joblib")
    joblib.dump(monthly_model, model_dir / "monthly_model.joblib")
    joblib.dump({"columns": list(x_month.columns)}, model_dir / "monthly_meta.joblib")
    joblib.dump(yearly_bundle, model_dir / "yearly_bundle.joblib")
    joblib.dump({"columns": list(x_year.columns)}, model_dir / "yearly_meta.joblib")
    joblib.dump(mapper, model_dir / "contract_mapper.joblib")
    joblib.dump(df, model_dir / "base_data.joblib")

    return TrainOutput(
        daily_metrics=daily_metrics,
        monthly_metrics=monthly_metrics,
        yearly_metrics=yearly_metrics,
    )
