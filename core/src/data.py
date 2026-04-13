from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DataConfig:
    seed: int = 42
    start: str = "2018-01-01"
    end: str = "2024-12-31"


def generate_demo_data(path: str | Path, cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Generate a synthetic daily coal market dataset for demo/training.

    The generated columns mimic the PDF proposal fields:
    market_price, policy/sentiment/weather and several fundamentals.
    """
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start, cfg.end, freq="D")
    n = len(dates)

    trend = np.linspace(520, 860, n)
    yearly = 25 * np.sin(np.arange(n) * 2 * np.pi / 365.25)
    weekly = 6 * np.sin(np.arange(n) * 2 * np.pi / 7)

    policy_index = np.clip(rng.normal(0.0, 1.0, n).cumsum() / 50.0, -3, 3)
    sentiment_score = np.clip(rng.normal(0.0, 0.6, n), -2, 2)
    temperature = 12 + 16 * np.sin((np.arange(n) + 50) * 2 * np.pi / 365.25) + rng.normal(0, 2, n)
    precipitation = np.abs(rng.normal(20, 8, n))

    port_inventory = 2200 + 120 * np.sin(np.arange(n) * 2 * np.pi / 60) + rng.normal(0, 30, n)
    rail_transport = 900 + 40 * np.sin(np.arange(n) * 2 * np.pi / 45) + rng.normal(0, 20, n)
    power_consumption = 520 + 55 * np.sin((np.arange(n) + 140) * 2 * np.pi / 365.25) + rng.normal(0, 14, n)
    import_volume = 300 + 15 * np.sin(np.arange(n) * 2 * np.pi / 90) + rng.normal(0, 9, n)

    noise = rng.normal(0, 4.0, n)
    market_price = (
        trend
        + yearly
        + weekly
        - 0.02 * (port_inventory - 2200)
        + 0.04 * (power_consumption - 520)
        + 7.5 * policy_index
        + 3.0 * sentiment_score
        + noise
    )

    # Contract price is smoother than market price (dual-track mapping target).
    contract_price = 0.72 * market_price + 0.28 * pd.Series(market_price).rolling(30, min_periods=1).mean().to_numpy() + 8

    df = pd.DataFrame(
        {
            "date": dates,
            "market_price": market_price,
            "contract_price": contract_price,
            "policy_index": policy_index,
            "sentiment_score": sentiment_score,
            "temperature": temperature,
            "precipitation": precipitation,
            "port_inventory": port_inventory,
            "rail_transport": rail_transport,
            "power_consumption": power_consumption,
            "import_volume": import_volume,
        }
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_or_create_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return generate_demo_data(path)
