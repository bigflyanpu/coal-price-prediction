from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

from .runtime_config import load_monthly_candidate_defaults, load_yearly_default_params


def stabilize_daily_predictions(
    raw_preds: np.ndarray,
    history_market: np.ndarray,
    future_market: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply the same volatility-based daily guard used by online inference.
    The guard is applied sequentially so offline evaluation follows serve behavior.
    """
    preds = np.asarray(raw_preds, dtype=float).reshape(-1)
    history = list(np.asarray(history_market, dtype=float).reshape(-1))
    future = np.asarray(future_market, dtype=float).reshape(-1) if future_market is not None else None
    out: list[float] = []
    for i, pred in enumerate(preds):
        market = pd.Series(history, dtype=float).dropna()
        if market.empty:
            clipped = float(pred)
        else:
            last_price = float(market.iloc[-1])
            lookback = market.tail(30)
            diff_std = float(lookback.diff().std(skipna=True) or 0.0)
            pct_std = float(lookback.pct_change().std(skipna=True) or 0.0)
            move_cap_abs = max(10.0, diff_std * 2.5, abs(last_price) * max(0.015, pct_std * 2.5))
            lower = last_price - move_cap_abs
            upper = last_price + move_cap_abs
            clipped = float(np.clip(pred, lower, upper))
        out.append(clipped)
        if future is not None and i < len(future):
            history.append(float(future[i]))
        else:
            history.append(clipped)
    return np.asarray(out, dtype=float)


class LSTMTransformerRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 96, n_heads: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=2)
        enc = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        x = self.transformer(x)
        return self.regressor(x[:, -1, :])


class GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 96):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=2)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gru(x)
        return self.regressor(x[:, -1, :])


@dataclass
class DailyTrainerConfig:
    epochs: int = 18
    lr: float = 8e-4
    batch_size: int = 64


@dataclass
class DailyBundle:
    model: nn.Module
    x_scaler: RobustScaler
    y_scaler: RobustScaler
    variant: str = "lstm_transformer"


def _to_sequence(x: np.ndarray) -> np.ndarray:
    return x[:, None, :]


def train_daily_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    cfg: DailyTrainerConfig = DailyTrainerConfig(),
    *,
    variant: str = "lstm_transformer",
) -> DailyBundle:
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    train_x_scaled = x_scaler.fit_transform(train_x)
    train_y_scaled = y_scaler.fit_transform(train_y.reshape(-1, 1)).reshape(-1)

    model_variant = (variant or "lstm_transformer").strip().lower()
    if model_variant == "gru":
        model = GRURegressor(input_size=train_x.shape[1])
    else:
        model = LSTMTransformerRegressor(input_size=train_x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.HuberLoss()

    x_t = torch.tensor(_to_sequence(train_x_scaled), dtype=torch.float32)
    y_t = torch.tensor(train_y_scaled.reshape(-1, 1), dtype=torch.float32)

    model.train()
    n = len(x_t)
    for _ in range(cfg.epochs):
        idx = torch.randperm(n)
        for i in range(0, n, cfg.batch_size):
            b = idx[i : i + cfg.batch_size]
            pred = model(x_t[b])
            loss = loss_fn(pred, y_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return DailyBundle(model=model, x_scaler=x_scaler, y_scaler=y_scaler, variant=model_variant)


def predict_daily_model(bundle: DailyBundle, x: np.ndarray) -> np.ndarray:
    x_scaled = bundle.x_scaler.transform(x)
    bundle.model.eval()
    with torch.no_grad():
        pred_scaled = (
            bundle.model(torch.tensor(_to_sequence(x_scaled), dtype=torch.float32))
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )
    pred = bundle.y_scaler.inverse_transform(pred_scaled).reshape(-1)
    return pred


def train_monthly_model(train_x: pd.DataFrame, train_y: pd.Series) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=48,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1,
    )
    model.fit(train_x, train_y)
    return model


@dataclass
class MonthlyModelWrapper:
    variant: str
    model: Any
    feature_cols: list[str]

    def predict(self, x: pd.DataFrame, *, dates: pd.Series | None = None) -> np.ndarray:
        if self.variant == "prophet":
            if dates is None:
                raise ValueError("Prophet 预测需要 dates 参数")
            pred_df = pd.DataFrame({"ds": pd.to_datetime(dates, errors="coerce")}).dropna()
            if pred_df.empty:
                raise ValueError("Prophet dates 为空")
            yhat = self.model.predict(pred_df)["yhat"].to_numpy()
            return yhat
        aligned = x.reindex(columns=self.feature_cols, fill_value=0.0)
        return self.model.predict(aligned)


def train_best_monthly_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    *,
    train_dates: pd.Series | None = None,
    variant: str = "lightgbm",
) -> Tuple[MonthlyModelWrapper, dict, float]:
    model_variant = (variant or "lightgbm").strip().lower()
    if model_variant == "prophet":
        if train_dates is None:
            model_variant = "lightgbm"
        else:
            try:
                from prophet import Prophet  # type: ignore

                ds = pd.to_datetime(train_dates, errors="coerce")
                base = pd.DataFrame({"ds": ds, "y": pd.to_numeric(train_y, errors="coerce")}).dropna()
                if len(base) < 12:
                    raise ValueError("prophet样本不足")
                split = max(12, int(len(base) * 0.8))
                if split >= len(base):
                    split = max(1, len(base) - 1)
                tr, val = base.iloc[:split], base.iloc[split:]
                model_val = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
                model_val.fit(tr)
                if len(val) > 0:
                    pred = model_val.predict(val[["ds"]])["yhat"].to_numpy()
                    score = float(mean_absolute_percentage_error(val["y"].to_numpy(), pred))
                else:
                    pred = model_val.predict(tr[["ds"]])["yhat"].to_numpy()
                    score = float(mean_absolute_percentage_error(tr["y"].to_numpy(), pred))
                model_full = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
                model_full.fit(base)
                wrapper = MonthlyModelWrapper(variant="prophet", model=model_full, feature_cols=list(train_x.columns))
                return wrapper, {"variant": "prophet"}, score
            except Exception:
                model_variant = "lightgbm"

    split = max(12, int(len(train_x) * 0.8))
    if split >= len(train_x):
        split = max(1, len(train_x) - 1)
    x_tr, x_val = train_x.iloc[:split], train_x.iloc[split:]
    y_tr, y_val = train_y.iloc[:split], train_y.iloc[split:]

    cfg_defaults = load_monthly_candidate_defaults()
    candidates = [
        {"n_estimators": 300, "learning_rate": 0.04, "max_depth": 4, "num_leaves": 31},
        cfg_defaults,
        {"n_estimators": 600, "learning_rate": 0.02, "max_depth": 6, "num_leaves": 63},
    ]

    best_params = candidates[0]
    best_score = float("inf")
    for params in candidates:
        model = LGBMRegressor(
            random_state=42,
            subsample=0.85,
            colsample_bytree=0.85,
            verbose=-1,
            **params,
        )
        model.fit(x_tr, y_tr)
        if len(x_val) > 0:
            pred = model.predict(x_val)
            score = float(mean_absolute_percentage_error(y_val, pred))
        else:
            pred = model.predict(x_tr)
            score = float(mean_absolute_percentage_error(y_tr, pred))
        if score < best_score:
            best_score = score
            best_params = params

    best_model = LGBMRegressor(
        random_state=42,
        subsample=0.85,
        colsample_bytree=0.85,
        verbose=-1,
        **best_params,
    )
    best_model.fit(train_x, train_y)
    wrapper = MonthlyModelWrapper(variant="lightgbm", model=best_model, feature_cols=list(train_x.columns))
    return wrapper, {**best_params, "variant": "lightgbm"}, best_score


@dataclass
class YearlyBundle:
    scaler: RobustScaler
    model: SVR
    ridge: Ridge
    blend_weight_svr: float


def _build_yearly_cv_folds(n_samples: int) -> list[dict]:
    min_train = max(3, int(np.ceil(n_samples * 0.55)))
    if min_train >= n_samples:
        min_train = max(2, n_samples - 1)
    max_folds = min(5, max(1, n_samples - min_train))

    folds: list[dict] = []
    for i in range(max_folds):
        train_end = min_train + i
        if train_end >= n_samples:
            break
        val_end = min(n_samples, train_end + 2)
        val_len = max(1, val_end - train_end)
        # Downweight one-point folds to reduce tiny-split variance.
        fold_weight = 0.7 if val_len == 1 else 1.0
        folds.append(
            {
                "train_end": int(train_end),
                "val_end": int(val_end),
                "val_len": int(val_len),
                "fold_weight": float(fold_weight),
            }
        )

    if not folds:
        folds = [{"train_end": max(1, n_samples - 1), "val_end": n_samples, "val_len": 1, "fold_weight": 0.7}]
    return folds


def train_yearly_model(train_x: pd.DataFrame, train_y: pd.Series) -> YearlyBundle:
    defaults = load_yearly_default_params()
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(train_x)
    model = SVR(C=defaults["C"], epsilon=defaults["epsilon"], kernel=defaults["kernel"], gamma="scale")
    model.fit(x_scaled, train_y)
    ridge = Ridge(alpha=float(defaults["ridge_alpha"]), random_state=42)
    ridge.fit(train_x, train_y)
    return YearlyBundle(
        scaler=scaler,
        model=model,
        ridge=ridge,
        blend_weight_svr=float(defaults["blend_weight_svr"]),
    )


def predict_yearly_bundle(bundle: YearlyBundle, x: pd.DataFrame) -> np.ndarray:
    x = x.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    svr_pred = bundle.model.predict(bundle.scaler.transform(x))
    ridge_pred = bundle.ridge.predict(x)
    return bundle.blend_weight_svr * svr_pred + (1.0 - bundle.blend_weight_svr) * ridge_pred


def train_best_yearly_model(train_x: pd.DataFrame, train_y: pd.Series) -> Tuple[YearlyBundle, dict, float]:
    train_x = (
        train_x.replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
    )
    train_y = pd.Series(train_y).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    n = len(train_x)
    if n < 3:
        bundle = train_yearly_model(train_x, train_y)
        return bundle, {"fallback": True}, float("inf")

    candidates = [
        {"C": 4.0, "epsilon": 0.12, "kernel": "linear", "gamma": "scale", "ridge_alpha": 0.8},
        {"C": 6.0, "epsilon": 0.10, "kernel": "rbf", "gamma": "scale", "ridge_alpha": 1.0},
        {"C": 8.0, "epsilon": 0.08, "kernel": "rbf", "gamma": "auto", "ridge_alpha": 1.5},
        {"C": 10.0, "epsilon": 0.08, "kernel": "rbf", "gamma": "scale", "ridge_alpha": 2.0},
        {"C": 14.0, "epsilon": 0.06, "kernel": "rbf", "gamma": "auto", "ridge_alpha": 3.0},
        {"C": 18.0, "epsilon": 0.05, "kernel": "rbf", "gamma": "scale", "ridge_alpha": 4.0},
        {"C": 24.0, "epsilon": 0.03, "kernel": "rbf", "gamma": "auto", "ridge_alpha": 6.0},
    ]
    blend_weights = [0.35, 0.45, 0.55, 0.65, 0.75]
    cv_folds = _build_yearly_cv_folds(n)
    lambda_rmse = 0.20
    lambda_mae = 0.15

    best_params = {
        **candidates[0],
        "blend_weight_svr": blend_weights[0],
    }
    best_score = float("inf")
    experiment_rows: list[dict] = []

    for params in candidates:
        fold_scores = []
        for fold in cv_folds:
            train_end = int(fold["train_end"])
            val_end = int(fold["val_end"])
            fold_weight = float(fold["fold_weight"])
            x_tr, x_val = train_x.iloc[:train_end], train_x.iloc[train_end:val_end]
            y_tr, y_val = train_y.iloc[:train_end], train_y.iloc[train_end:val_end]
            if len(x_tr) < 2 or len(x_val) < 1:
                continue

            scaler = RobustScaler()
            x_tr_scaled = scaler.fit_transform(x_tr)
            svr_params = {k: v for k, v in params.items() if k not in {"ridge_alpha"}}
            svr = SVR(**svr_params)
            svr.fit(x_tr_scaled, y_tr)

            ridge = Ridge(alpha=float(params["ridge_alpha"]), random_state=42)
            ridge.fit(x_tr, y_tr)

            svr_pred = svr.predict(scaler.transform(x_val))
            ridge_pred = ridge.predict(x_val)

            for w in blend_weights:
                pred = w * svr_pred + (1.0 - w) * ridge_pred
                mape = float(mean_absolute_percentage_error(y_val, pred))
                rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
                mae = float(np.mean(np.abs(np.asarray(y_val) - np.asarray(pred))))
                baseline = float(np.mean(np.abs(np.asarray(y_val)))) + 1e-6
                rmse_norm = rmse / baseline
                mae_norm = mae / baseline
                score = float(mape + lambda_rmse * rmse_norm + lambda_mae * mae_norm)
                fold_scores.append((w, score, mape, rmse, mae, fold_weight))

        if not fold_scores:
            continue

        # aggregate by blend weight to select robust setting.
        weight_to_scores: dict[float, list[tuple[float, float, float, float, float]]] = {}
        for w, sc, mape, rmse, mae, f_w in fold_scores:
            weight_to_scores.setdefault(float(w), []).append((float(sc), float(mape), float(rmse), float(mae), float(f_w)))

        for w, score_pack in weight_to_scores.items():
            score_arr = np.array([row[0] for row in score_pack], dtype=float)
            mape_arr = np.array([row[1] for row in score_pack], dtype=float)
            rmse_arr = np.array([row[2] for row in score_pack], dtype=float)
            mae_arr = np.array([row[3] for row in score_pack], dtype=float)
            fw_arr = np.array([row[4] for row in score_pack], dtype=float)
            mean_score = float(np.average(score_arr, weights=fw_arr))
            mean_mape = float(np.average(mape_arr, weights=fw_arr))
            mean_rmse = float(np.average(rmse_arr, weights=fw_arr))
            mean_mae = float(np.average(mae_arr, weights=fw_arr))
            experiment_rows.append(
                {
                    "C": params["C"],
                    "epsilon": params["epsilon"],
                    "kernel": params["kernel"],
                    "gamma": params["gamma"],
                    "ridge_alpha": params["ridge_alpha"],
                    "blend_weight_svr": w,
                    "cv_score": mean_score,
                    "cv_mape": mean_mape,
                    "cv_mape_raw": mean_mape,
                    "cv_rmse": mean_rmse,
                    "cv_mae": mean_mae,
                    "n_folds": len(score_pack),
                }
            )
            if mean_score < best_score or (abs(mean_score - best_score) < 1e-12 and mean_mape < best_params.get("cv_mape_raw", float("inf"))):
                best_score = mean_score
                best_params = {
                    **params,
                    "blend_weight_svr": w,
                    "cv_mape_raw": mean_mape,
                    "cv_rmse": mean_rmse,
                    "cv_mae": mean_mae,
                }

    final_scaler = RobustScaler()
    x_full_scaled = final_scaler.fit_transform(train_x)
    svr_params = {k: v for k, v in best_params.items() if k not in {"blend_weight_svr", "ridge_alpha", "cv_mape_raw", "cv_rmse", "cv_mae"}}
    final_model = SVR(**svr_params)
    final_model.fit(x_full_scaled, train_y)
    final_ridge = Ridge(alpha=float(best_params.get("ridge_alpha", 2.0)), random_state=42)
    final_ridge.fit(train_x, train_y)

    exp_df = pd.DataFrame(experiment_rows)
    if not exp_df.empty:
        exp_df = exp_df.sort_values("cv_mape")

    return (
        YearlyBundle(
            scaler=final_scaler,
            model=final_model,
            ridge=final_ridge,
            blend_weight_svr=float(best_params["blend_weight_svr"]),
        ),
        {
            "best": best_params,
            "search_space": {
                "candidates": candidates,
                "blend_weights": blend_weights,
                "cv_folds": cv_folds,
                "lambda_rmse": lambda_rmse,
                "lambda_mae": lambda_mae,
            },
            "cv_top": exp_df.head(8).to_dict(orient="records") if not exp_df.empty else [],
        },
        float(best_score),
    )


@dataclass
class DualTrackRuleConfig:
    low_ratio: float = 0.85
    high_ratio: float = 1.05


class ContractPriceMapper:
    """Map market predictions to contract prices with corridor constraints."""

    def __init__(self, rule_cfg: DualTrackRuleConfig = DualTrackRuleConfig()) -> None:
        self.reg = LinearRegression()
        self.rule_cfg = rule_cfg

    def fit(self, market_pred: np.ndarray, contract_true: np.ndarray) -> "ContractPriceMapper":
        self.reg.fit(market_pred.reshape(-1, 1), contract_true)
        return self

    def predict(self, market_pred: np.ndarray, policy_strength: np.ndarray | None = None) -> np.ndarray:
        raw = self.reg.predict(market_pred.reshape(-1, 1))
        floor = market_pred * self.rule_cfg.low_ratio
        cap = market_pred * self.rule_cfg.high_ratio

        if policy_strength is not None:
            adj = np.clip(policy_strength, 0, 3)
            floor = floor + 2.0 * adj
            cap = cap + 2.0 * adj
        return np.clip(raw, floor, cap)


@dataclass
class SentimentForecastBundle:
    model: Ridge
    lags: tuple[int, ...]
    x_scaler: RobustScaler


def _build_lag_matrix(series: pd.Series, lags: tuple[int, ...]) -> tuple[pd.DataFrame, pd.Series]:
    tmp = pd.DataFrame({"y": pd.to_numeric(series, errors="coerce")})
    for lag in lags:
        tmp[f"lag_{lag}"] = tmp["y"].shift(lag)
    tmp = tmp.dropna().reset_index(drop=True)
    x = tmp[[f"lag_{lag}" for lag in lags]]
    y = tmp["y"]
    return x, y


def train_sentiment_forecast_model(
    sentiment_series: pd.Series,
    *,
    lags: tuple[int, ...] = (1, 2, 3, 5, 7),
) -> tuple[SentimentForecastBundle, dict[str, float]]:
    x, y = _build_lag_matrix(sentiment_series, lags)
    if len(x) < 20:
        raise ValueError("舆情序列样本不足，无法训练独立预测模型")
    x = x.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    y = y.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    x = x.clip(lower=-5.0, upper=5.0)
    y = y.clip(lower=-5.0, upper=5.0)
    split = max(10, int(len(x) * 0.8))
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    x_scaler = RobustScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_eval = x_test if len(x_test) > 0 else x_train
    x_eval_scaled = x_scaler.transform(x_eval)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(x_train_scaled, y_train)
    pred = model.predict(x_eval_scaled)
    target = y_test.to_numpy() if len(y_test) > 0 else y_train.to_numpy()
    metrics = evaluate_sentiment_metrics(target, pred)
    return SentimentForecastBundle(model=model, lags=lags, x_scaler=x_scaler), metrics


def predict_sentiment_next(bundle: SentimentForecastBundle, sentiment_series: pd.Series) -> float:
    seq = pd.to_numeric(sentiment_series, errors="coerce").dropna().to_numpy()
    if len(seq) < max(bundle.lags):
        raise ValueError("舆情序列长度不足，无法预测下一时点")
    row = np.array([[seq[-lag] for lag in bundle.lags]], dtype=float)
    row = np.clip(row, -5.0, 5.0)
    row_scaled = bundle.x_scaler.transform(row)
    return float(bundle.model.predict(row_scaled)[0])


def evaluate_sentiment_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-6
    smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))
    return {"rmse": rmse, "mae": mae, "smape": smape}


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    safe_true = np.where(np.abs(y_true) < 1e-6, np.sign(y_true) * 1e-6 + (y_true == 0) * 1e-6, y_true)
    mape = float(mean_absolute_percentage_error(safe_true, y_pred))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"rmse": rmse, "mape": mape, "mae": mae}
