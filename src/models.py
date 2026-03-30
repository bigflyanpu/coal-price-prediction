from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR


class LSTMTransformerRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, n_heads: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.regressor(x)


@dataclass
class DailyTrainerConfig:
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 32


def _to_sequence(x: np.ndarray) -> np.ndarray:
    return x[:, None, :]


def train_daily_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    cfg: DailyTrainerConfig = DailyTrainerConfig(),
) -> LSTMTransformerRegressor:
    model = LSTMTransformerRegressor(input_size=train_x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    x_t = torch.tensor(_to_sequence(train_x), dtype=torch.float32)
    y_t = torch.tensor(train_y.reshape(-1, 1), dtype=torch.float32)

    model.train()
    n = len(x_t)
    for _ in range(cfg.epochs):
        idx = torch.randperm(n)
        for i in range(0, n, cfg.batch_size):
            b = idx[i : i + cfg.batch_size]
            xb = x_t[b]
            yb = y_t[b]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def predict_daily_model(model: LSTMTransformerRegressor, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(_to_sequence(x), dtype=torch.float32)).cpu().numpy().reshape(-1)
    return pred


def train_monthly_model(train_x: pd.DataFrame, train_y: pd.Series) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    model.fit(train_x, train_y)
    return model


@dataclass
class YearlyBundle:
    scaler: RobustScaler
    model: SVR


def train_yearly_model(train_x: pd.DataFrame, train_y: pd.Series) -> YearlyBundle:
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(train_x)
    model = SVR(C=10.0, epsilon=0.08, kernel="rbf")
    model.fit(x_scaled, train_y)
    return YearlyBundle(scaler=scaler, model=model)


class ContractPriceMapper:
    def __init__(self) -> None:
        self.reg = LinearRegression()

    def fit(self, market_pred: np.ndarray, contract_true: np.ndarray) -> "ContractPriceMapper":
        self.reg.fit(market_pred.reshape(-1, 1), contract_true)
        return self

    def predict(self, market_pred: np.ndarray) -> np.ndarray:
        return self.reg.predict(market_pred.reshape(-1, 1))


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    return {"rmse": rmse, "mape": mape}
