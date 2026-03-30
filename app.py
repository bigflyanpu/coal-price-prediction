from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request

from src.features import aggregate_monthly, aggregate_yearly, build_daily_features
from src.models import LSTMTransformerRegressor

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"

app = Flask(__name__)


def load_models():
    daily_meta = joblib.load(MODEL_DIR / "daily_meta.joblib")
    monthly_meta = joblib.load(MODEL_DIR / "monthly_meta.joblib")
    yearly_meta = joblib.load(MODEL_DIR / "yearly_meta.joblib")

    daily = LSTMTransformerRegressor(input_size=len(daily_meta["columns"]))
    daily.load_state_dict(torch.load(MODEL_DIR / "daily_model.pt", map_location="cpu"))
    daily.eval()

    monthly = joblib.load(MODEL_DIR / "monthly_model.joblib")
    yearly_bundle = joblib.load(MODEL_DIR / "yearly_bundle.joblib")
    mapper = joblib.load(MODEL_DIR / "contract_mapper.joblib")
    base_data = joblib.load(MODEL_DIR / "base_data.joblib")

    return {
        "daily": daily,
        "daily_cols": daily_meta["columns"],
        "monthly": monthly,
        "monthly_cols": monthly_meta["columns"],
        "yearly_bundle": yearly_bundle,
        "yearly_cols": yearly_meta["columns"],
        "mapper": mapper,
        "base_data": base_data,
    }


STATE = None


def ensure_state():
    global STATE
    if STATE is None:
        STATE = load_models()
    return STATE


def predict_next(df: pd.DataFrame) -> dict:
    state = ensure_state()

    daily_features = build_daily_features(df)
    x_daily_last = daily_features[state["daily_cols"]].iloc[[-1]].to_numpy()

    with torch.no_grad():
        day_pred = (
            state["daily"](torch.tensor(x_daily_last[:, None, :], dtype=torch.float32))
            .cpu()
            .numpy()
            .reshape(-1)[0]
        )

    month_df = aggregate_monthly(df)
    month_df["daily_pred_mean"] = np.nan
    month_df.iloc[-1, month_df.columns.get_loc("daily_pred_mean")] = day_pred
    month_row = month_df[state["monthly_cols"]].ffill().iloc[[-1]]
    month_pred = float(state["monthly"].predict(month_row)[0])

    year_df = aggregate_yearly(df)
    year_df["monthly_pred_mean"] = np.nan
    year_df.iloc[-1, year_df.columns.get_loc("monthly_pred_mean")] = month_pred
    year_row = year_df[state["yearly_cols"]].ffill().iloc[[-1]]
    year_pred = float(
        state["yearly_bundle"].model.predict(state["yearly_bundle"].scaler.transform(year_row))[0]
    )

    contract_pred = float(state["mapper"].predict(np.array([day_pred]))[0])

    return {
        "next_day_market_price": round(float(day_pred), 2),
        "next_day_contract_price": round(contract_pred, 2),
        "next_month_market_price": round(month_pred, 2),
        "next_year_market_price": round(year_pred, 2),
    }


@app.route("/")
def index():
    state = ensure_state()
    pred = predict_next(state["base_data"])
    return render_template("index.html", prediction=pred)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    state = ensure_state()
    payload = request.get_json(silent=True) or {}
    path = payload.get("csv_path")

    if path:
        p = Path(path)
        if not p.exists():
            return jsonify({"error": "csv_path 不存在"}), 400
        df = pd.read_csv(p, parse_dates=["date"])
    else:
        df = state["base_data"]

    result = predict_next(df)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
