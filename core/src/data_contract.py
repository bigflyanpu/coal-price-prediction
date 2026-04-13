from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, List

import pandas as pd


@dataclass
class ValidationResult:
    name: str
    missing_columns: List[str]
    null_rate: Dict[str, float]
    duplicate_rows: int

    @property
    def ok(self) -> bool:
        return not self.missing_columns and self.duplicate_rows == 0


class DataContract:
    def __init__(self, config_path: str | Path = "config/data_contract.json") -> None:
        self.config_path = Path(config_path)
        self.contract = json.loads(self.config_path.read_text(encoding="utf-8"))

    def required_columns(self, source_name: str) -> list[str]:
        return list(self.contract["sources"][source_name]["required"])

    def validate_source(self, source_name: str, df: pd.DataFrame) -> ValidationResult:
        req = self.required_columns(source_name)
        missing = [c for c in req if c not in df.columns]
        null_rate = {}
        for col in req:
            if col in df.columns:
                null_rate[col] = float(df[col].isna().mean())
        duplicates = int(df.duplicated().sum())
        return ValidationResult(name=source_name, missing_columns=missing, null_rate=null_rate, duplicate_rows=duplicates)

    def validate_curated_daily(self, df: pd.DataFrame) -> ValidationResult:
        req = list(self.contract["curated_daily_required"])
        missing = [c for c in req if c not in df.columns]
        null_rate = {}
        for col in req:
            if col in df.columns:
                null_rate[col] = float(df[col].isna().mean())
        duplicates = int(df.duplicated(subset=["date"]).sum()) if "date" in df.columns else int(df.duplicated().sum())
        return ValidationResult(name="curated_daily", missing_columns=missing, null_rate=null_rate, duplicate_rows=duplicates)

    def dump_quality_report(self, results: list[ValidationResult], path: str | Path) -> None:
        rows = []
        for r in results:
            rows.append(
                {
                    "source": r.name,
                    "ok": r.ok,
                    "missing_columns": ",".join(r.missing_columns),
                    "duplicate_rows": r.duplicate_rows,
                    "avg_null_rate": round(sum(r.null_rate.values()) / len(r.null_rate), 6) if r.null_rate else 0.0,
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)
