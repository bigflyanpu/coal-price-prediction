from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR


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


@dataclass
class DailyTrainerConfig:
    epochs: int = 18
    lr: float = 8e-4
    batch_size: int = 64


@dataclass
class DailyBundle:
    model: LSTMTransformerRegressor
    x_scaler: RobustScaler
    y_scaler: RobustScaler


def _to_sequence(x: np.ndarray) -> np.ndarray:
    return x[:, None, :]


def train_daily_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    cfg: DailyTrainerConfig = DailyTrainerConfig(),
) -> DailyBundle:
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    train_x_scaled = x_scaler.fit_transform(train_x)
    train_y_scaled = y_scaler.fit_transform(train_y.reshape(-1, 1)).reshape(-1)

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
    return DailyBundle(model=model, x_scaler=x_scaler, y_scaler=y_scaler)


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


def train_best_monthly_model(train_x: pd.DataFrame, train_y: pd.Series) -> Tuple[LGBMRegressor, dict, float]:
    split = max(12, int(len(train_x) * 0.8))
    if split >= len(train_x):
        split = max(1, len(train_x) - 1)
    x_tr, x_val = train_x.iloc[:split], train_x.iloc[split:]
    y_tr, y_val = train_y.iloc[:split], train_y.iloc[split:]

    candidates = [
        {"n_estimators": 300, "learning_rate": 0.04, "max_depth": 4, "num_leaves": 31},
        {"n_estimators": 450, "learning_rate": 0.03, "max_depth": 5, "num_leaves": 48},
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
    return best_model, best_params, best_score


@dataclass
class YearlyBundle:
    scaler: RobustScaler
    model: SVR
    ridge: Ridge
    blend_weight_svr: float


def train_yearly_model(train_x: pd.DataFrame, train_y: pd.Series) -> YearlyBundle:
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(train_x)
    model = SVR(C=12.0, epsilon=0.05, kernel="rbf", gamma="scale")
    model.fit(x_scaled, train_y)
    ridge = Ridge(alpha=2.0, random_state=42)
    ridge.fit(train_x, train_y)
    return YearlyBundle(scaler=scaler, model=model, ridge=ridge, blend_weight_svr=0.65)


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
        {"C": 6.0, "epsilon": 0.10, "kernel": "rbf", "ridge_alpha": 1.0},
        {"C": 10.0, "epsilon": 0.08, "kernel": "rbf", "ridge_alpha": 2.0},
        {"C": 14.0, "epsilon": 0.06, "kernel": "rbf", "ridge_alpha": 3.0},
        {"C": 20.0, "epsilon": 0.05, "kernel": "rbf", "ridge_alpha": 4.0},
    ]
    blend_weights = [0.45, 0.55, 0.65, 0.75]

    val_split_points: List[int] = []
    start = max(3, int(n * 0.6))
    for s in range(start, n):
        if s < n:
            val_split_points.append(s)
    if not val_split_points:
        val_split_points = [max(1, n - 1)]

    best_params = {
        **candidates[0],
        "blend_weight_svr": blend_weights[0],
    }
    best_score = float("inf")
    experiment_rows: list[dict] = []

    for params in candidates:
        fold_scores = []
        for split in val_split_points:
            x_tr, x_val = train_x.iloc[:split], train_x.iloc[split:]
            y_tr, y_val = train_y.iloc[:split], train_y.iloc[split:]
            if len(x_tr) < 2 or len(x_val) < 1:
                continue

            scaler = RobustScaler()
            x_tr_scaled = scaler.fit_transform(x_tr)
            svr_params = {k: v for k, v in params.items() if k not in {"ridge_alpha"}}
            svr = SVR(gamma="scale", **svr_params)
            svr.fit(x_tr_scaled, y_tr)

            ridge = Ridge(alpha=float(params["ridge_alpha"]), random_state=42)
            ridge.fit(x_tr, y_tr)

            svr_pred = svr.predict(scaler.transform(x_val))
            ridge_pred = ridge.predict(x_val)

            for w in blend_weights:
                pred = w * svr_pred + (1.0 - w) * ridge_pred
                score = float(mean_absolute_percentage_error(y_val, pred))
                fold_scores.append((w, score))

        if not fold_scores:
            continue

        # aggregate by blend weight to select robust setting.
        weight_to_scores: dict[float, list[float]] = {}
        for w, sc in fold_scores:
            weight_to_scores.setdefault(float(w), []).append(float(sc))

        for w, scores in weight_to_scores.items():
            mean_score = float(np.mean(scores))
            experiment_rows.append(
                {
                    "C": params["C"],
                    "epsilon": params["epsilon"],
                    "kernel": params["kernel"],
                    "ridge_alpha": params["ridge_alpha"],
                    "blend_weight_svr": w,
                    "cv_mape": mean_score,
                    "n_folds": len(scores),
                }
            )
            if mean_score < best_score:
                best_score = mean_score
                best_params = {
                    **params,
                    "blend_weight_svr": w,
                }

    final_scaler = RobustScaler()
    x_full_scaled = final_scaler.fit_transform(train_x)
    svr_params = {k: v for k, v in best_params.items() if k not in {"blend_weight_svr", "ridge_alpha"}}
    final_model = SVR(gamma="scale", **svr_params)
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
                "val_splits": val_split_points,
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


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"rmse": rmse, "mape": mape, "mae": mae}
