from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Dict, Any
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import torch

from .backtest import rolling_backtest
from .data_contract import DataContract
from .features import (
    FeatureConfig,
    aggregate_monthly,
    aggregate_yearly,
    build_feature_library,
    enrich_yearly_features,
    select_core_features_xgboost,
)
from .ingestion import IngestionConfig, MultiSourceIngestor
from .models import (
    ContractPriceMapper,
    DailyTrainerConfig,
    evaluate_metrics,
    predict_daily_model,
    train_daily_model,
    predict_yearly_bundle,
    train_best_monthly_model,
    train_best_yearly_model,
)
from .nlp_index import NLPConfig, PolicySentimentIndexer
from .reporting import build_paper_experiment_tables


@dataclass
class TrainOutput:
    online_metrics: Dict[str, Dict[str, float]]
    backtest_summary: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any]


@dataclass
class TrainConfig:
    fast_mode: bool = False
    refresh_cache: bool = False
    verbose: bool = True


class CoalResearchPipeline:
    def __init__(
        self,
        data_path: str | Path = "data/curated/daily.csv",
        model_dir: str | Path = "models",
        train_cfg: TrainConfig = TrainConfig(),
    ) -> None:
        self.data_path = Path(data_path)
        self.model_dir = Path(model_dir)
        self.train_cfg = train_cfg
        self.model_dir.mkdir(parents=True, exist_ok=True)
        Path("reports").mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path("reports/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, msg: str) -> None:
        if self.train_cfg.verbose:
            print(f"[pipeline] {msg}")

    def _build_research_dataset(self) -> pd.DataFrame:
        if self.data_path.exists() and not self.train_cfg.refresh_cache:
            self._log(f"reuse curated dataset from {self.data_path}")
            return pd.read_csv(self.data_path, parse_dates=["date"])

        self._log("stage A: multi-source ingestion")
        live_text_enabled = os.getenv("LIVE_TEXT_SOURCES")
        if live_text_enabled is None:
            use_live_text = not self.train_cfg.fast_mode
        else:
            use_live_text = live_text_enabled == "1"
        ingestor = MultiSourceIngestor(IngestionConfig(use_live_text_sources=use_live_text))
        _, _ = ingestor.run()

        structured = pd.read_csv("data/raw/structured.csv", parse_dates=["date"])
        policy_raw = pd.read_csv("data/raw/policy_text.csv", parse_dates=["date"])
        sentiment_raw = pd.read_csv("data/raw/sentiment_text.csv", parse_dates=["date"])
        weather_raw = pd.read_csv("data/raw/weather.csv", parse_dates=["date"])

        policy_cache = self.cache_dir / "policy_daily.csv"
        sent_cache = self.cache_dir / "sentiment_daily.csv"

        if policy_cache.exists() and sent_cache.exists() and not self.train_cfg.refresh_cache:
            self._log("stage B: reuse cached NLP indices")
            policy_daily = pd.read_csv(policy_cache, parse_dates=["date"])
            sent_daily = pd.read_csv(sent_cache, parse_dates=["date"])
        else:
            self._log("stage B: build policy/sentiment indices")
            nlp = PolicySentimentIndexer(
                NLPConfig(
                    policy_dims=12,
                    lda_topics=6,
                    use_bert=not self.train_cfg.fast_mode,
                    bert_local_files_only=self.train_cfg.fast_mode,
                )
            )
            policy_daily, sent_daily = nlp.build_indices(policy_raw, sentiment_raw)
            policy_daily.to_csv(policy_cache, index=False)
            sent_daily.to_csv(sent_cache, index=False)

        weather_daily = (
            weather_raw.groupby("date")[["temperature", "precipitation", "wind_speed"]]
            .mean()
            .reset_index()
        )

        daily = structured.merge(policy_daily, on="date", how="left")
        daily = daily.merge(sent_daily, on="date", how="left")
        daily = daily.merge(weather_daily, on="date", how="left")

        index_cols = [f"policy_index_{i+1}" for i in range(12)]
        for c in index_cols + ["policy_strength", "policy_uncertainty", "sentiment_score", "sentiment_heat", "sentiment_volatility"]:
            if c not in daily.columns:
                daily[c] = 0.0

        daily = daily.sort_values("date").ffill().bfill().reset_index(drop=True)
        daily.to_csv(self.data_path, index=False)

        contract = DataContract()
        contract_result = contract.validate_curated_daily(daily)
        contract.dump_quality_report([contract_result], "reports/curated_quality.csv")
        return daily

    def _split_train_test(self, df: pd.DataFrame, ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(len(df) * ratio)
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def _build_monthly_enhanced(self, daily_df: pd.DataFrame, daily_pred_frame: pd.DataFrame) -> pd.DataFrame:
        monthly = aggregate_monthly(daily_df)
        pred_month = (
            daily_pred_frame.set_index("date")["daily_pred"]
            .resample("MS")
            .agg(["mean", "std", "min", "max"])
            .rename(
                columns={
                    "mean": "daily_pred_mean",
                    "std": "daily_pred_std",
                    "min": "daily_pred_min",
                    "max": "daily_pred_max",
                }
            )
            .reset_index()
        )
        monthly = monthly.merge(pred_month, on="date", how="left")
        monthly = monthly.sort_values("date").ffill().bfill()
        return monthly

    def _build_yearly_enhanced(self, daily_df: pd.DataFrame, monthly_df: pd.DataFrame, monthly_pred: np.ndarray) -> pd.DataFrame:
        yearly = aggregate_yearly(daily_df)
        monthly_tmp = monthly_df[["date"]].copy()
        monthly_tmp["monthly_pred"] = monthly_pred
        pred_year = (
            monthly_tmp.set_index("date")["monthly_pred"]
            .resample("YS")
            .agg(["mean", "std", "min", "max"])
            .rename(
                columns={
                    "mean": "monthly_pred_mean",
                    "std": "monthly_pred_std",
                    "min": "monthly_pred_min",
                    "max": "monthly_pred_max",
                }
            )
            .reset_index()
        )
        yearly = yearly.merge(pred_year, on="date", how="left")
        yearly = enrich_yearly_features(yearly)
        yearly = yearly.sort_values("date").ffill().bfill()
        return yearly

    def train(self) -> TrainOutput:
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        self._log(f"train start (fast_mode={self.train_cfg.fast_mode}, refresh_cache={self.train_cfg.refresh_cache})")
        daily_df = self._build_research_dataset()

        feat_cache = self.cache_dir / "feature_library.joblib"
        sel_cache = self.cache_dir / "selected_feature_df.joblib"
        cols_cache = self.cache_dir / "selected_feature_cols.json"
        if feat_cache.exists() and sel_cache.exists() and cols_cache.exists() and not self.train_cfg.refresh_cache:
            self._log("stage C: reuse cached feature library and selected features")
            feature_df = joblib.load(feat_cache)
            selected_df = joblib.load(sel_cache)
            selected_features = json.loads(cols_cache.read_text(encoding="utf-8"))
        else:
            self._log("stage C: build feature library and XGBoost selection")
            feature_df = build_feature_library(daily_df, FeatureConfig())
            selected_df, selected_features = select_core_features_xgboost(
                feature_df,
                target_col="market_price",
                keep_top_k=200,
            )
            joblib.dump(feature_df, feat_cache)
            joblib.dump(selected_df, sel_cache)
            cols_cache.write_text(json.dumps(selected_features, ensure_ascii=False), encoding="utf-8")

        self._log("stage D: train daily/monthly/yearly models")
        train_df, test_df = self._split_train_test(selected_df)
        x_cols = selected_features
        x_train = train_df[x_cols]
        y_train = train_df["market_price"]
        x_test = test_df[x_cols]
        y_test = test_df["market_price"]

        daily_bundle = train_daily_model(x_train.to_numpy(), y_train.to_numpy(), DailyTrainerConfig())
        daily_pred_test = predict_daily_model(daily_bundle, x_test.to_numpy())
        daily_metrics = evaluate_metrics(y_test.to_numpy(), daily_pred_test)

        policy_for_rule = test_df.get("policy_strength", pd.Series(np.zeros(len(test_df))))
        daily_pred_train = predict_daily_model(daily_bundle, x_train.to_numpy())
        mapper = ContractPriceMapper().fit(daily_pred_train, train_df["contract_price"].to_numpy())
        contract_pred = mapper.predict(daily_pred_test, policy_for_rule.to_numpy())
        contract_metrics = evaluate_metrics(test_df["contract_price"].to_numpy(), contract_pred)

        full_daily_for_agg = daily_df.copy()
        daily_pred_full = predict_daily_model(daily_bundle, selected_df[x_cols].to_numpy())
        daily_pred_frame = pd.DataFrame({"date": selected_df["date"].to_numpy(), "daily_pred": daily_pred_full})
        monthly = self._build_monthly_enhanced(full_daily_for_agg, daily_pred_frame)
        month_drop = ["date", "market_price", "contract_price"]
        x_month = monthly.drop(columns=month_drop)
        y_month = monthly["market_price"]
        split_m = max(12, int(len(monthly) * 0.8))
        x_m_train, x_m_test = x_month.iloc[:split_m], x_month.iloc[split_m:]
        y_m_train, y_m_test = y_month.iloc[:split_m], y_month.iloc[split_m:]
        monthly_model, monthly_params, monthly_val_mape = train_best_monthly_model(x_m_train, y_m_train)
        month_pred = monthly_model.predict(x_m_test)
        monthly_metrics = evaluate_metrics(y_m_test.to_numpy(), month_pred)

        month_pred_full = monthly_model.predict(x_month)
        yearly = self._build_yearly_enhanced(full_daily_for_agg, monthly, month_pred_full)
        year_drop = ["date", "market_price", "contract_price"]
        x_year = yearly.drop(columns=year_drop)
        y_year = yearly["market_price"]
        split_y = max(3, int(len(yearly) * 0.8))
        x_y_train, x_y_test = x_year.iloc[:split_y], x_year.iloc[split_y:]
        y_y_train, y_y_test = y_year.iloc[:split_y], y_year.iloc[split_y:]
        yearly_bundle, yearly_params, yearly_val_mape = train_best_yearly_model(x_y_train, y_y_train)
        year_pred = predict_yearly_bundle(yearly_bundle, x_y_test)
        yearly_metrics = evaluate_metrics(y_y_test.to_numpy(), year_pred)
        Path("reports/yearly_model_experiments.json").write_text(
            json.dumps(
                {
                    "best_params": yearly_params.get("best", yearly_params),
                    "val_mape": yearly_val_mape,
                    "cv_top": yearly_params.get("cv_top", []),
                    "search_space": yearly_params.get("search_space", {}),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self._log("stage E: rolling backtest")
        if self.train_cfg.fast_mode:
            backtest = rolling_backtest(daily_df, start_test_year=2024, end_test_year=2024)
        else:
            backtest = rolling_backtest(daily_df, start_test_year=2021, end_test_year=2024)
        backtest.fold_metrics.to_csv("reports/rolling_backtest_folds.csv", index=False)
        Path("reports/rolling_backtest_summary.json").write_text(
            json.dumps(backtest.summary_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        torch.save(daily_bundle.model.state_dict(), self.model_dir / "daily_model.pt")
        joblib.dump(
            {
                "columns": x_cols,
                "x_scaler": daily_bundle.x_scaler,
                "y_scaler": daily_bundle.y_scaler,
            },
            self.model_dir / "daily_meta.joblib",
        )
        joblib.dump(monthly_model, self.model_dir / "monthly_model.joblib")
        joblib.dump({"columns": list(x_month.columns)}, self.model_dir / "monthly_meta.joblib")
        joblib.dump(yearly_bundle, self.model_dir / "yearly_bundle.joblib")
        joblib.dump({"columns": list(x_year.columns)}, self.model_dir / "yearly_meta.joblib")
        joblib.dump(mapper, self.model_dir / "contract_mapper.joblib")
        joblib.dump(daily_df, self.model_dir / "base_data.joblib")

        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_window": {
                "start": str(daily_df["date"].min().date()),
                "end": str(daily_df["date"].max().date()),
            },
            "features_total": int(len([c for c in feature_df.columns if c not in {"date", "market_price", "contract_price"}])),
            "features_selected": int(len(x_cols)),
            "selected_feature_sample": x_cols[:20],
            "model_versions": {
                "daily": "LSTMTransformer_v2",
                "monthly": "LightGBM_v3_tuned",
                "yearly": "RobustSVR_v3_tuned",
                "dual_track": "RuleMapper_v2",
                "policy_index": "BERT_LDA_12d",
            },
            "tuning": {
                "monthly_params": monthly_params,
                "monthly_val_mape": monthly_val_mape,
                "yearly_params": yearly_params,
                "yearly_val_mape": yearly_val_mape,
            },
        }
        Path("reports/metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        online_metrics = {
            "daily_market": daily_metrics,
            "daily_contract": contract_metrics,
            "monthly_market": monthly_metrics,
            "yearly_market": yearly_metrics,
        }
        Path("reports/online_holdout_metrics.json").write_text(
            json.dumps(online_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        build_paper_experiment_tables("reports")
        self._log("stage F: artifacts and reports saved")

        return TrainOutput(online_metrics=online_metrics, backtest_summary=backtest.summary_metrics, metadata=metadata)


def train_all(
    data_path: str | Path = "data/curated/daily.csv",
    model_dir: str | Path = "models",
    *,
    fast_mode: bool = False,
    refresh_cache: bool = False,
    verbose: bool = True,
) -> TrainOutput:
    cfg = TrainConfig(fast_mode=fast_mode, refresh_cache=refresh_cache, verbose=verbose)
    return CoalResearchPipeline(data_path=data_path, model_dir=model_dir, train_cfg=cfg).train()
