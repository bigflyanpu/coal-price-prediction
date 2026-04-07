from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_csv(path: Path, parse_date: bool = False) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if parse_date:
            return pd.read_csv(path, parse_dates=["date"])
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _detect_missing_date_ranges(dates: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, str]]:
    if dates.empty:
        return [{"start": str(start.date()), "end": str(end.date())}]
    full = pd.date_range(start=start, end=end, freq="D")
    existing = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize().unique()
    missing = pd.DatetimeIndex(full.difference(pd.DatetimeIndex(existing)))
    if missing.empty:
        return []

    blocks: list[dict[str, str]] = []
    block_start = missing[0]
    prev = missing[0]
    for d in missing[1:]:
        if (d - prev).days > 1:
            blocks.append({"start": str(block_start.date()), "end": str(prev.date())})
            block_start = d
        prev = d
    blocks.append({"start": str(block_start.date()), "end": str(prev.date())})
    return blocks


def build_data_gap_audit(
    start: str,
    end: str,
    *,
    report_dir: str | Path = "reports",
    raw_dir: str | Path = "data/raw",
    contract_path: str | Path = "config/data_contract.json",
    text_source_config_path: str | Path = "config/text_sources.json",
) -> dict[str, Any]:
    report_dir = Path(report_dir)
    raw_dir = Path(raw_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    contract = _load_json(Path(contract_path), {})
    text_cfg = _load_json(Path(text_source_config_path), {})
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    structured = _read_csv(raw_dir / "structured.csv", parse_date=True)
    policy = _read_csv(raw_dir / "policy_text.csv", parse_date=True)
    sentiment = _read_csv(raw_dir / "sentiment_text.csv", parse_date=True)
    weather = _read_csv(raw_dir / "weather.csv", parse_date=True)

    # 1) 数据源台账
    registry_rows: list[dict[str, Any]] = []
    source_map = {
        "structured": structured,
        "policy_text": policy,
        "sentiment_text": sentiment,
        "weather": weather,
    }
    for name, df in source_map.items():
        req = contract.get("sources", {}).get(name, {}).get("required", [])
        date_min = str(pd.to_datetime(df["date"], errors="coerce").min().date()) if (not df.empty and "date" in df.columns) else None
        date_max = str(pd.to_datetime(df["date"], errors="coerce").max().date()) if (not df.empty and "date" in df.columns) else None
        registry_rows.append(
            {
                "source_name": name,
                "source_file": str(raw_dir / f"{name}.csv"),
                "exists": (raw_dir / f"{name}.csv").exists(),
                "records": int(len(df)),
                "date_min": date_min,
                "date_max": date_max,
                "required_columns": ",".join(req),
            }
        )

    # 补充文本源配置台账
    for kind, key in [("policy_text", "policy_feeds"), ("sentiment_text", "sentiment_feeds")]:
        for feed in text_cfg.get(key, []):
            registry_rows.append(
                {
                    "source_name": f"{kind}:{feed.get('name', 'unknown')}",
                    "source_file": feed.get("url", ""),
                    "exists": True,
                    "records": None,
                    "date_min": None,
                    "date_max": None,
                    "required_columns": "url",
                }
            )

    registry_df = pd.DataFrame(registry_rows)
    registry_df.to_csv(report_dir / "data_source_registry.csv", index=False)

    # 2) 覆盖率日报
    idx = pd.date_range(start=start_dt, end=end_dt, freq="D")
    daily = pd.DataFrame({"date": idx})

    def _count_daily(df: pd.DataFrame, col: str, out_col: str):
        if df.empty or "date" not in df.columns:
            daily[out_col] = 0
            return
        temp = df.copy()
        temp["date"] = pd.to_datetime(temp["date"], errors="coerce").dt.normalize()
        cnt = temp.groupby("date").size().rename(col).reset_index()
        merged = daily.merge(cnt, on="date", how="left")
        daily[out_col] = merged[col].fillna(0).astype(int)

    _count_daily(structured, "structured_cnt", "structured_cnt")
    _count_daily(policy, "policy_doc_cnt", "policy_doc_cnt")
    _count_daily(sentiment, "sentiment_doc_cnt", "sentiment_doc_cnt")
    if weather.empty or "date" not in weather.columns:
        daily["weather_region_cnt"] = 0
    else:
        w = weather.copy()
        w["date"] = pd.to_datetime(w["date"], errors="coerce").dt.normalize()
        if "region" in w.columns:
            c = w.groupby("date")["region"].nunique().rename("weather_region_cnt").reset_index()
        else:
            c = w.groupby("date").size().rename("weather_region_cnt").reset_index()
        daily = daily.merge(c, on="date", how="left")
        daily["weather_region_cnt"] = daily["weather_region_cnt"].fillna(0).astype(int)

    daily["structured_ok"] = (daily["structured_cnt"] > 0).astype(int)
    daily["policy_ok"] = (daily["policy_doc_cnt"] > 0).astype(int)
    daily["sentiment_ok"] = (daily["sentiment_doc_cnt"] > 0).astype(int)
    daily["weather_ok"] = (daily["weather_region_cnt"] > 0).astype(int)
    daily["coverage_score"] = daily[["structured_ok", "policy_ok", "sentiment_ok", "weather_ok"]].mean(axis=1).round(4)
    daily.to_csv(report_dir / "data_coverage_daily.csv", index=False)

    # 3) 缺口清单
    expected_weather_regions = int(daily["weather_region_cnt"].max()) if not daily.empty else 0
    gap = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": str(start_dt.date()), "end": str(end_dt.date())},
        "missing_date_ranges": {
            "structured": _detect_missing_date_ranges(structured.get("date", pd.Series(dtype="datetime64[ns]")), start_dt, end_dt),
            "policy_text": _detect_missing_date_ranges(policy.get("date", pd.Series(dtype="datetime64[ns]")), start_dt, end_dt),
            "sentiment_text": _detect_missing_date_ranges(sentiment.get("date", pd.Series(dtype="datetime64[ns]")), start_dt, end_dt),
            "weather": _detect_missing_date_ranges(weather.get("date", pd.Series(dtype="datetime64[ns]")), start_dt, end_dt),
        },
        "zero_count_days": {
            "policy_text": int((daily["policy_doc_cnt"] == 0).sum()),
            "sentiment_text": int((daily["sentiment_doc_cnt"] == 0).sum()),
            "weather": int((daily["weather_region_cnt"] == 0).sum()),
        },
        "weather_region_expectation": {
            "expected_regions_per_day": expected_weather_regions,
            "days_below_expectation": int((daily["weather_region_cnt"] < expected_weather_regions).sum()) if expected_weather_regions > 0 else None,
        },
        "null_rate_required_columns": {},
    }

    for source_name, df in source_map.items():
        req = contract.get("sources", {}).get(source_name, {}).get("required", [])
        col_null = {}
        for col in req:
            if col in df.columns and not df.empty:
                col_null[col] = float(df[col].isna().mean())
            else:
                col_null[col] = 1.0
        gap["null_rate_required_columns"][source_name] = col_null

    (report_dir / "data_gap_checklist.json").write_text(json.dumps(gap, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(report_dir / "data_source_registry.csv"),
        "coverage_daily_path": str(report_dir / "data_coverage_daily.csv"),
        "gap_checklist_path": str(report_dir / "data_gap_checklist.json"),
        "coverage_days": int(len(daily)),
    }
    (report_dir / "data_gap_audit_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
