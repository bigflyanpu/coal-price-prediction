from __future__ import annotations

import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request

from src.features import aggregate_monthly, aggregate_yearly, build_feature_library
from src.models import LSTMTransformerRegressor

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"

app = Flask(__name__)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_models():
    daily_meta = joblib.load(MODEL_DIR / "daily_meta.joblib")
    monthly_meta = joblib.load(MODEL_DIR / "monthly_meta.joblib")
    yearly_meta = joblib.load(MODEL_DIR / "yearly_meta.joblib")

    daily = LSTMTransformerRegressor(input_size=len(daily_meta["columns"]))
    daily.load_state_dict(torch.load(MODEL_DIR / "daily_model.pt", map_location="cpu"))
    daily.eval()

    return {
        "daily": daily,
        "daily_cols": daily_meta["columns"],
        "monthly": joblib.load(MODEL_DIR / "monthly_model.joblib"),
        "monthly_cols": monthly_meta["columns"],
        "yearly_bundle": joblib.load(MODEL_DIR / "yearly_bundle.joblib"),
        "yearly_cols": yearly_meta["columns"],
        "mapper": joblib.load(MODEL_DIR / "contract_mapper.joblib"),
        "base_data": joblib.load(MODEL_DIR / "base_data.joblib"),
    }


STATE = None


def ensure_state():
    global STATE
    if STATE is None:
        STATE = load_models()
    return STATE


def _daily_last_vector(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    feat = build_feature_library(df)
    aligned = feat.reindex(columns=["date", "market_price", "contract_price", *feature_cols], fill_value=0)
    return aligned[feature_cols].iloc[[-1]].to_numpy()


def predict_next(df: pd.DataFrame) -> dict:
    state = ensure_state()
    daily_last_x = _daily_last_vector(df, state["daily_cols"])

    with torch.no_grad():
        day_pred = (
            state["daily"](torch.tensor(daily_last_x[:, None, :], dtype=torch.float32))
            .cpu()
            .numpy()
            .reshape(-1)[0]
        )

    policy_strength = float(df.get("policy_strength", pd.Series([0])).iloc[-1]) if len(df) else 0.0
    contract_pred = float(state["mapper"].predict(np.array([day_pred]), np.array([policy_strength]))[0])

    month_df = aggregate_monthly(df)
    month_row = month_df.reindex(columns=state["monthly_cols"], fill_value=0).iloc[[-1]]
    month_pred = float(state["monthly"].predict(month_row)[0])

    year_df = aggregate_yearly(df)
    year_row = year_df.reindex(columns=state["yearly_cols"], fill_value=0).iloc[[-1]]
    year_pred = float(
        state["yearly_bundle"].model.predict(state["yearly_bundle"].scaler.transform(year_row))[0]
    )

    return {
        "next_day_market_price": round(float(day_pred), 2),
        "next_day_contract_price": round(contract_pred, 2),
        "next_month_market_price": round(month_pred, 2),
        "next_year_market_price": round(year_pred, 2),
    }


@app.route("/")
def index():
    state = ensure_state()
    prediction = predict_next(state["base_data"])
    backtest_summary = _load_json(REPORT_DIR / "rolling_backtest_summary.json", {})
    metadata = _load_json(REPORT_DIR / "metadata.json", {})
    data_quality = []
    q_path = REPORT_DIR / "data_quality.csv"
    if q_path.exists():
        data_quality = pd.read_csv(q_path).to_dict(orient="records")

    return render_template(
        "index.html",
        prediction=prediction,
        backtest_summary=backtest_summary,
        metadata=metadata,
        data_quality=data_quality,
    )


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

    return jsonify(predict_next(df))


@app.route("/api/backtest")
def backtest_api():
    return jsonify(_load_json(REPORT_DIR / "rolling_backtest_summary.json", {}))


@app.route("/api/metadata")
def metadata_api():
    return jsonify(_load_json(REPORT_DIR / "metadata.json", {}))


@app.route("/api/data-health")
def data_health_api():
    path = REPORT_DIR / "data_quality.csv"
    if not path.exists():
        return jsonify([])
    return jsonify(pd.read_csv(path).to_dict(orient="records"))


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
