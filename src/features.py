from __future__ import annotations

import pandas as pd


def build_daily_features(df: pd.DataFrame, target_col: str = "market_price") -> pd.DataFrame:
    out = df.sort_values("date").copy()

    for lag in [1, 3, 7, 14, 30]:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)

    for window in [7, 14, 30]:
        out[f"{target_col}_roll_mean_{window}"] = out[target_col].rolling(window, min_periods=1).mean()
        out[f"{target_col}_roll_std_{window}"] = out[target_col].rolling(window, min_periods=2).std()

    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_year"] = out["date"].dt.dayofyear

    return out.dropna().reset_index(drop=True)


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.set_index("date")
        .resample("MS")
        .agg(
            {
                "market_price": "mean",
                "contract_price": "mean",
                "policy_index": "mean",
                "sentiment_score": "mean",
                "temperature": "mean",
                "precipitation": "sum",
                "port_inventory": "mean",
                "rail_transport": "mean",
                "power_consumption": "mean",
                "import_volume": "mean",
            }
        )
        .reset_index()
    )
    monthly["month_id"] = monthly["date"].dt.month
    monthly["quarter"] = monthly["date"].dt.quarter
    return monthly


def aggregate_yearly(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.set_index("date")
        .resample("YS")
        .agg(
            {
                "market_price": "mean",
                "contract_price": "mean",
                "policy_index": "mean",
                "sentiment_score": "mean",
                "temperature": "mean",
                "precipitation": "sum",
                "port_inventory": "mean",
                "rail_transport": "mean",
                "power_consumption": "mean",
                "import_volume": "mean",
            }
        )
        .reset_index()
    )
    yearly["year"] = yearly["date"].dt.year
    return yearly
