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


def _to_percent(v: Any) -> float | None:
    try:
        return float(v) * 100.0
    except Exception:
        return None


def build_paper_experiment_tables(report_dir: str | Path = "reports") -> dict[str, Any]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    online = _load_json(report_dir / "online_holdout_metrics.json", {})
    backtest = _load_json(report_dir / "rolling_backtest_summary.json", {})
    yearly_exp = _load_json(report_dir / "yearly_model_experiments.json", {})

    rows: list[dict[str, Any]] = []
    scales = ["daily_market", "daily_contract", "monthly_market", "yearly_market"]
    for scale in scales:
        rows.append(
            {
                "section": "main_metrics",
                "metric": scale,
                "online_mape_pct": _to_percent(online.get(scale, {}).get("mape")),
                "backtest_mape_pct": _to_percent(backtest.get(scale, {}).get("mape")),
                "online_rmse": online.get(scale, {}).get("rmse"),
                "backtest_rmse": backtest.get(scale, {}).get("rmse"),
                "online_mae": online.get(scale, {}).get("mae"),
                "backtest_mae": backtest.get(scale, {}).get("mae"),
            }
        )

    cv_top = yearly_exp.get("cv_top", [])
    for i, item in enumerate(cv_top[:10], start=1):
        rows.append(
            {
                "section": "yearly_cv_top",
                "metric": f"rank_{i}",
                "online_mape_pct": _to_percent(item.get("cv_mape")),
                "backtest_mape_pct": None,
                "online_rmse": None,
                "backtest_rmse": None,
                "online_mae": None,
                "backtest_mae": None,
                "C": item.get("C"),
                "epsilon": item.get("epsilon"),
                "ridge_alpha": item.get("ridge_alpha"),
                "blend_weight_svr": item.get("blend_weight_svr"),
                "n_folds": item.get("n_folds"),
            }
        )

    table_df = pd.DataFrame(rows)
    csv_path = report_dir / "paper_experiment_tables.csv"
    table_df.to_csv(csv_path, index=False)

    main_df = table_df[table_df["section"] == "main_metrics"].copy()
    cv_df = table_df[table_df["section"] == "yearly_cv_top"].copy()

    md_lines = [
        "# 论文实验表格草稿",
        "",
        f"- 生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        f"- 数据来源：`{report_dir / 'online_holdout_metrics.json'}`、`{report_dir / 'rolling_backtest_summary.json'}`、`{report_dir / 'yearly_model_experiments.json'}`",
        "",
        "## 表1：主结果（在线留出 vs 滚动回测）",
        "",
    ]
    if main_df.empty:
        md_lines.append("无可用数据。")
    else:
        main_fmt = main_df[
            [
                "metric",
                "online_mape_pct",
                "backtest_mape_pct",
                "online_rmse",
                "backtest_rmse",
                "online_mae",
                "backtest_mae",
            ]
        ].copy()
        try:
            md_lines.append(main_fmt.to_markdown(index=False))
        except Exception:
            md_lines.append(main_fmt.to_csv(index=False))

    md_lines.extend(["", "## 表2：年度模型时序CV候选（Top-10）", ""])
    if cv_df.empty:
        md_lines.append("无可用数据。")
    else:
        cv_fmt = cv_df[
            [
                "metric",
                "online_mape_pct",
                "C",
                "epsilon",
                "ridge_alpha",
                "blend_weight_svr",
                "n_folds",
            ]
        ].copy()
        try:
            md_lines.append(cv_fmt.to_markdown(index=False))
        except Exception:
            md_lines.append(cv_fmt.to_csv(index=False))

    md_path = report_dir / "paper_experiment_tables.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "rows_total": int(len(table_df)),
        "main_metric_rows": int(len(main_df)),
        "yearly_cv_rows": int(len(cv_df)),
    }
    (report_dir / "paper_experiment_tables_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta
