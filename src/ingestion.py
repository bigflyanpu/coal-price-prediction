from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data_contract import DataContract, ValidationResult


@dataclass
class IngestionConfig:
    seed: int = 42
    start: str = "2018-01-01"
    end: str = "2024-12-31"
    source_dir: str | Path = "data/sources"
    raw_dir: str | Path = "data/raw"
    curated_dir: str | Path = "data/curated"


class MultiSourceIngestor:
    def __init__(self, cfg: IngestionConfig = IngestionConfig()) -> None:
        self.cfg = cfg
        self.contract = DataContract()
        self.rng = np.random.default_rng(cfg.seed)

    def _date_index(self) -> pd.DatetimeIndex:
        return pd.date_range(self.cfg.start, self.cfg.end, freq="D")

    def _load_or_generate(self, name: str, generator) -> pd.DataFrame:
        path = Path(self.cfg.source_dir) / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = generator()
        return df

    def ingest_structured(self) -> pd.DataFrame:
        def _gen() -> pd.DataFrame:
            dates = self._date_index()
            n = len(dates)
            trend = np.linspace(520, 860, n)
            yearly = 30 * np.sin(np.arange(n) * 2 * np.pi / 365.25)
            policy_impact = np.clip(self.rng.normal(0.0, 1.0, n).cumsum() / 40.0, -4, 4)
            power = 520 + 45 * np.sin((np.arange(n) + 120) * 2 * np.pi / 365.25) + self.rng.normal(0, 14, n)
            port = 2200 + 130 * np.sin(np.arange(n) * 2 * np.pi / 60) + self.rng.normal(0, 35, n)
            rail = 900 + 35 * np.sin(np.arange(n) * 2 * np.pi / 45) + self.rng.normal(0, 18, n)
            imports = 280 + 18 * np.sin(np.arange(n) * 2 * np.pi / 90) + self.rng.normal(0, 7, n)
            coal_output = 1100 + 30 * np.sin(np.arange(n) * 2 * np.pi / 28) + self.rng.normal(0, 10, n)
            iv_added = 5.8 + 0.4 * np.sin(np.arange(n) * 2 * np.pi / 180) + self.rng.normal(0, 0.08, n)
            market_price = trend + yearly + 0.05 * (power - 520) - 0.02 * (port - 2200) + 5.6 * policy_impact + self.rng.normal(0, 4, n)
            contract_price = 0.76 * market_price + 0.24 * pd.Series(market_price).rolling(30, min_periods=1).mean().to_numpy() + 6
            return pd.DataFrame(
                {
                    "date": dates,
                    "market_price": market_price,
                    "contract_price": contract_price,
                    "port_inventory": port,
                    "rail_transport": rail,
                    "power_consumption": power,
                    "import_volume": imports,
                    "coal_output": coal_output,
                    "industrial_value_added": iv_added,
                }
            )

        df = self._load_or_generate("structured", _gen)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def ingest_policy_text(self) -> pd.DataFrame:
        def _gen() -> pd.DataFrame:
            dates = self._date_index()
            kws = ["保供", "限产", "价格调控", "运输保障", "长协", "进口政策", "安全检查", "环保督察"]
            records = []
            for i, dt in enumerate(dates):
                num_docs = int(self.rng.integers(0, 3))
                for j in range(num_docs):
                    kw = kws[int(self.rng.integers(0, len(kws)))]
                    records.append(
                        {
                            "date": dt,
                            "doc_id": f"POL-{i:05d}-{j}",
                            "title": f"关于煤炭市场{kw}的通知",
                            "body": f"为保障能源稳定供应，针对{kw}发布阶段性政策，要求重点企业落实责任。",
                            "source": "gov",
                            "url": f"https://example.gov/policy/{i}-{j}",
                            "department": "国家部委",
                            "doc_type": "通知",
                        }
                    )
            return pd.DataFrame(records)

        df = self._load_or_generate("policy_text", _gen)
        if df.empty:
            df = _gen()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def ingest_sentiment_text(self) -> pd.DataFrame:
        def _gen() -> pd.DataFrame:
            dates = self._date_index()
            medias = [f"media_{i:02d}" for i in range(1, 51)]
            tones = ["上涨", "回落", "紧张", "宽松", "震荡"]
            records = []
            for i, dt in enumerate(dates):
                num_news = int(self.rng.integers(2, 8))
                for j in range(num_news):
                    tone = tones[int(self.rng.integers(0, len(tones)))]
                    media = medias[int(self.rng.integers(0, len(medias)))]
                    records.append(
                        {
                            "date": dt,
                            "news_id": f"NEWS-{i:05d}-{j}",
                            "title": f"煤炭市场{tone}信号",
                            "body": f"{media} 报道煤炭价格{tone}，电厂采购节奏与港口库存出现变化。",
                            "media": media,
                            "url": f"https://example.news/{i}-{j}",
                            "author": "editor",
                            "topic": "coal",
                        }
                    )
            return pd.DataFrame(records)

        df = self._load_or_generate("sentiment_text", _gen)
        if df.empty:
            df = _gen()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def ingest_weather(self) -> pd.DataFrame:
        def _gen() -> pd.DataFrame:
            dates = self._date_index()
            regions = ["shanxi", "inner_mongolia", "shaanxi", "qinhuangdao", "jiangsu"]
            rows = []
            for dt in dates:
                for region in regions:
                    temp = 12 + 16 * np.sin((dt.dayofyear + 35) * 2 * np.pi / 365.25) + self.rng.normal(0, 2)
                    rows.append(
                        {
                            "date": dt,
                            "region": region,
                            "temperature": temp,
                            "precipitation": abs(self.rng.normal(18, 7)),
                            "wind_speed": abs(self.rng.normal(2.8, 0.9)),
                            "humidity": np.clip(self.rng.normal(55, 12), 20, 95),
                            "pressure": self.rng.normal(1008, 8),
                        }
                    )
            return pd.DataFrame(rows)

        df = self._load_or_generate("weather", _gen)
        if df.empty:
            df = _gen()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["date", "region"]).reset_index(drop=True)

    def _save_raw(self, name: str, df: pd.DataFrame) -> None:
        out = Path(self.cfg.raw_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / f"{name}.csv", index=False)

    def _save_curated(self, name: str, df: pd.DataFrame) -> None:
        out = Path(self.cfg.curated_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / f"{name}.csv", index=False)

    def build_curated_daily(
        self,
        structured: pd.DataFrame,
        policy_daily: pd.DataFrame,
        sentiment_daily: pd.DataFrame,
        weather_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        daily = structured.merge(policy_daily, on="date", how="left")
        daily = daily.merge(sentiment_daily, on="date", how="left")
        daily = daily.merge(weather_daily, on="date", how="left")
        daily = daily.sort_values("date").ffill().bfill()
        return daily.reset_index(drop=True)

    def run(self) -> tuple[pd.DataFrame, dict[str, ValidationResult]]:
        structured = self.ingest_structured()
        policy = self.ingest_policy_text()
        sentiment = self.ingest_sentiment_text()
        weather = self.ingest_weather()

        self._save_raw("structured", structured)
        self._save_raw("policy_text", policy)
        self._save_raw("sentiment_text", sentiment)
        self._save_raw("weather", weather)

        weather_daily = (
            weather.groupby("date")[["temperature", "precipitation", "wind_speed"]]
            .mean()
            .reset_index()
        )

        # policy_daily and sentiment_daily will be replaced by NLP index outputs in later stage.
        policy_daily = policy.groupby("date").size().rename("policy_doc_count").reset_index()
        policy_daily["policy_strength"] = np.log1p(policy_daily["policy_doc_count"]) 
        policy_daily["policy_uncertainty"] = policy_daily["policy_doc_count"].rolling(7, min_periods=1).std().fillna(0).to_numpy()
        policy_daily = policy_daily.drop(columns=["policy_doc_count"]) 

        sent_daily = sentiment.groupby("date").size().rename("sentiment_heat").reset_index()
        sent_daily["sentiment_score"] = np.tanh((sent_daily["sentiment_heat"] - sent_daily["sentiment_heat"].mean()) / (sent_daily["sentiment_heat"].std() + 1e-6))

        curated = self.build_curated_daily(structured, policy_daily, sent_daily, weather_daily)
        self._save_curated("daily", curated)

        results = {
            "structured": self.contract.validate_source("structured", structured),
            "policy_text": self.contract.validate_source("policy_text", policy),
            "sentiment_text": self.contract.validate_source("sentiment_text", sentiment),
            "weather": self.contract.validate_source("weather", weather),
            "curated_daily": self.contract.validate_curated_daily(curated),
        }
        report_path = Path("reports")
        report_path.mkdir(parents=True, exist_ok=True)
        self.contract.dump_quality_report(list(results.values()), report_path / "data_quality.csv")
        return curated, results
