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
from src.models import DailyBundle, LSTMTransformerRegressor, predict_daily_model, predict_yearly_bundle

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"

app = Flask(__name__)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_text_source_health() -> dict:
    return _load_json(REPORT_DIR / "text_source_health.json", {})


def _filter_text_source_health(health: dict, kind: str | None, status: str | None) -> dict:
    kind_filter = (kind or "").strip().lower()
    status_filter = (status or "").strip().lower()
    valid_kind = {"policy_text", "sentiment_text"}
    valid_status = {"good", "warn", "critical"}

    if kind_filter and kind_filter not in valid_kind:
        kind_filter = ""
    if status_filter and status_filter not in valid_status:
        status_filter = ""

    out = {}
    for k, item in health.items():
        if kind_filter and k != kind_filter:
            continue
        entry = dict(item)
        details = list(item.get("sources_detail", []))
        if status_filter:
            details = [d for d in details if str(d.get("quality_status", "")).lower() == status_filter]
        entry["sources_detail"] = details
        out[k] = entry
    return out


def load_models():
    daily_meta = joblib.load(MODEL_DIR / "daily_meta.joblib")
    monthly_meta = joblib.load(MODEL_DIR / "monthly_meta.joblib")
    yearly_meta = joblib.load(MODEL_DIR / "yearly_meta.joblib")

    daily = LSTMTransformerRegressor(input_size=len(daily_meta["columns"]))
    daily.load_state_dict(torch.load(MODEL_DIR / "daily_model.pt", map_location="cpu"))
    daily.eval()

    daily_bundle = DailyBundle(
        model=daily,
        x_scaler=daily_meta["x_scaler"],
        y_scaler=daily_meta["y_scaler"],
    )

    return {
        "daily_bundle": daily_bundle,
        "daily_cols": daily_meta["columns"],
        "monthly": joblib.load(MODEL_DIR / "monthly_model.joblib"),
        "monthly_cols": monthly_meta["columns"],
        "yearly_bundle": joblib.load(MODEL_DIR / "yearly_bundle.joblib"),
        "yearly_cols": yearly_meta["columns"],
        "mapper": joblib.load(MODEL_DIR / "contract_mapper.joblib"),
        "base_data": joblib.load(MODEL_DIR / "base_data.joblib"),
    }


STATE = None
STATE_SIGNATURE = None


def _model_signature() -> tuple[float, ...]:
    files = [
        MODEL_DIR / "daily_model.pt",
        MODEL_DIR / "daily_meta.joblib",
        MODEL_DIR / "monthly_model.joblib",
        MODEL_DIR / "monthly_meta.joblib",
        MODEL_DIR / "yearly_bundle.joblib",
        MODEL_DIR / "yearly_meta.joblib",
        MODEL_DIR / "contract_mapper.joblib",
        MODEL_DIR / "base_data.joblib",
    ]
    signature = []
    for p in files:
        if not p.exists():
            signature.append(0.0)
        else:
            signature.append(p.stat().st_mtime)
    return tuple(signature)


def ensure_state():
    global STATE, STATE_SIGNATURE
    sig = _model_signature()
    if STATE is None or STATE_SIGNATURE != sig:
        STATE = load_models()
        STATE_SIGNATURE = sig
    return STATE


def _daily_last_vector(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    feat = build_feature_library(df)
    aligned = feat.reindex(columns=["date", "market_price", "contract_price", *feature_cols], fill_value=0)
    return aligned[feature_cols].iloc[[-1]].to_numpy()


def predict_next(df: pd.DataFrame) -> dict:
    state = ensure_state()
    daily_last_x = _daily_last_vector(df, state["daily_cols"])
    day_pred = float(predict_daily_model(state["daily_bundle"], daily_last_x).reshape(-1)[0])

    policy_strength = float(df.get("policy_strength", pd.Series([0])).iloc[-1]) if len(df) else 0.0
    contract_pred = float(state["mapper"].predict(np.array([day_pred]), np.array([policy_strength]))[0])

    month_df = aggregate_monthly(df)
    month_row = month_df.reindex(columns=state["monthly_cols"], fill_value=0).iloc[[-1]]
    month_pred = float(state["monthly"].predict(month_row)[0])

    year_df = aggregate_yearly(df)
    year_row = year_df.reindex(columns=state["yearly_cols"], fill_value=0).iloc[[-1]]
    year_pred = float(predict_yearly_bundle(state["yearly_bundle"], year_row)[0])

    return {
        "next_day_market_price": round(float(day_pred), 2),
        "next_day_contract_price": round(contract_pred, 2),
        "next_month_market_price": round(month_pred, 2),
        "next_year_market_price": round(year_pred, 2),
    }


def build_dashboard_data(
    base_data: pd.DataFrame,
    prediction: dict,
    backtest_summary: dict,
    metadata: dict,
    data_quality: list[dict],
    text_source_health: dict,
) -> dict:
    recent = base_data.sort_values("date").tail(90).copy()
    recent["date"] = pd.to_datetime(recent["date"])
    timeline = [d.strftime("%Y-%m-%d") for d in recent["date"]]

    policy_series = recent.get("policy_strength", pd.Series(np.zeros(len(recent)))).round(4).tolist()
    sentiment_series = recent.get("sentiment_score", pd.Series(np.zeros(len(recent)))).round(4).tolist()
    market_series = recent.get("market_price", pd.Series(np.zeros(len(recent)))).round(2).tolist()
    contract_series = recent.get("contract_price", pd.Series(np.zeros(len(recent)))).round(2).tolist()

    mape_daily = backtest_summary.get("daily_market", {}).get("mape")
    mape_monthly = backtest_summary.get("monthly_market", {}).get("mape")
    mape_yearly = backtest_summary.get("yearly_market", {}).get("mape")

    source_counts = {}
    for source_name in ["structured", "policy_text", "sentiment_text", "weather"]:
        p = BASE / "data" / "raw" / f"{source_name}.csv"
        if p.exists():
            try:
                source_counts[source_name] = int(len(pd.read_csv(p)))
            except Exception:
                source_counts[source_name] = 0
        else:
            source_counts[source_name] = 0

    folds = []
    fold_path = REPORT_DIR / "rolling_backtest_folds.csv"
    if fold_path.exists():
        fold_df = pd.read_csv(fold_path)
        folds = fold_df.to_dict(orient="records")

    return {
        "prediction": prediction,
        "kpis": {
            "daily_mape": None if mape_daily is None else round(float(mape_daily) * 100, 2),
            "monthly_mape": None if mape_monthly is None else round(float(mape_monthly) * 100, 2),
            "yearly_mape": None if mape_yearly is None else round(float(mape_yearly) * 100, 2),
            "features_total": metadata.get("features_total", 0),
            "features_selected": metadata.get("features_selected", 0),
        },
        "data_layer": {
            "source_counts": source_counts,
            "quality_ok_count": sum(1 for r in data_quality if r.get("ok")),
            "quality_total_count": len(data_quality),
        },
        "nlp_layer": {
            "timeline": timeline,
            "policy_strength": policy_series,
            "sentiment_score": sentiment_series,
            "text_source_health": text_source_health,
        },
        "market_layer": {
            "timeline": timeline,
            "market_price": market_series,
            "contract_price": contract_series,
        },
        "backtest_folds": folds,
        "model_versions": metadata.get("model_versions", {}),
        "selected_feature_sample": metadata.get("selected_feature_sample", []),
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
    text_source_health = _load_text_source_health()
    dashboard_data = build_dashboard_data(
        base_data=state["base_data"],
        prediction=prediction,
        backtest_summary=backtest_summary,
        metadata=metadata,
        data_quality=data_quality,
        text_source_health=text_source_health,
    )

    return render_template(
        "index.html",
        prediction=prediction,
        backtest_summary=backtest_summary,
        metadata=metadata,
        data_quality=data_quality,
        dashboard_data=dashboard_data,
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


@app.route("/api/text-source-health")
def text_source_health_api():
    kind = request.args.get("kind")
    status = request.args.get("status")
    data = _filter_text_source_health(_load_text_source_health(), kind=kind, status=status)
    return jsonify(data)


@app.route("/api/dashboard")
def dashboard_api():
    state = ensure_state()
    prediction = predict_next(state["base_data"])
    backtest_summary = _load_json(REPORT_DIR / "rolling_backtest_summary.json", {})
    metadata = _load_json(REPORT_DIR / "metadata.json", {})
    q_path = REPORT_DIR / "data_quality.csv"
    data_quality = pd.read_csv(q_path).to_dict(orient="records") if q_path.exists() else []
    text_source_health = _load_text_source_health()
    return jsonify(
        build_dashboard_data(
            base_data=state["base_data"],
            prediction=prediction,
            backtest_summary=backtest_summary,
            metadata=metadata,
            data_quality=data_quality,
            text_source_health=text_source_health,
        )
    )


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
