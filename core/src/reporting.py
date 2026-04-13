from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
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


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
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


def build_paper_figures(report_dir: str | Path = "reports") -> dict[str, Any]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    plt = _try_import_matplotlib()

    figure_entries: list[dict[str, Any]] = []
    generated_at = datetime.now(timezone.utc).isoformat()

    table_path = report_dir / "paper_experiment_tables.csv"
    if not table_path.exists():
        build_paper_experiment_tables(report_dir)
    table_df = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()

    if plt is None:
        meta = {
            "generated_at": generated_at,
            "figures": [
                {
                    "name": "all",
                    "status": "skipped",
                    "reason": "matplotlib_not_available",
                }
            ],
        }
        (report_dir / "paper_figures_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return meta

    # Figure 1: main scale MAPE comparison.
    main_df = table_df[table_df["section"] == "main_metrics"].copy() if not table_df.empty else pd.DataFrame()
    fig1_path = report_dir / "paper_fig_main_mape.png"
    if not main_df.empty:
        main_df = main_df.sort_values("metric").reset_index(drop=True)
        labels = main_df["metric"].tolist()
        online_vals = pd.to_numeric(main_df["online_mape_pct"], errors="coerce").fillna(0).to_numpy()
        backtest_vals = pd.to_numeric(main_df["backtest_mape_pct"], errors="coerce").fillna(0).to_numpy()
        x = np.arange(len(labels))
        width = 0.36

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(x - width / 2, online_vals, width=width, label="Online Holdout")
        ax.bar(x + width / 2, backtest_vals, width=width, label="Rolling Backtest")
        ax.set_title("Main Scale MAPE Comparison")
        ax.set_ylabel("MAPE (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig1_path, dpi=180)
        plt.close(fig)
        figure_entries.append({"name": "main_mape_compare", "path": str(fig1_path), "status": "ok"})
    else:
        figure_entries.append({"name": "main_mape_compare", "status": "skipped", "reason": "no_main_metrics"})

    # Figure 2: yearly CV ranking.
    cv_df = table_df[table_df["section"] == "yearly_cv_top"].copy() if not table_df.empty else pd.DataFrame()
    fig2_path = report_dir / "paper_fig_yearly_cv_top.png"
    if not cv_df.empty:
        cv_df["cv_mape_pct"] = pd.to_numeric(cv_df["online_mape_pct"], errors="coerce")
        cv_df = cv_df.sort_values("cv_mape_pct").head(10).copy()
        labels = [
            f"C={r.C},e={r.epsilon},a={r.ridge_alpha},w={r.blend_weight_svr}"
            for r in cv_df.itertuples(index=False)
        ]
        vals = cv_df["cv_mape_pct"].fillna(0).to_numpy()

        fig, ax = plt.subplots(figsize=(10, 5.2))
        y = np.arange(len(labels))
        ax.barh(y, vals)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("CV MAPE (%)")
        ax.set_title("Yearly Model Time-Series CV Candidates (Top-10)")
        fig.tight_layout()
        fig.savefig(fig2_path, dpi=180)
        plt.close(fig)
        figure_entries.append({"name": "yearly_cv_top10", "path": str(fig2_path), "status": "ok"})
    else:
        figure_entries.append({"name": "yearly_cv_top10", "status": "skipped", "reason": "no_yearly_cv_metrics"})

    # Figure 3: rolling backtest yearly trend.
    folds_path = report_dir / "rolling_backtest_folds.csv"
    fig3_path = report_dir / "paper_fig_backtest_trend.png"
    if folds_path.exists():
        folds_df = pd.read_csv(folds_path)
        if not folds_df.empty and {"test_year", "scale", "mape"}.issubset(folds_df.columns):
            fig, ax = plt.subplots(figsize=(9.5, 4.8))
            for scale, grp in folds_df.groupby("scale"):
                s = grp.sort_values("test_year")
                ax.plot(s["test_year"], s["mape"] * 100.0, marker="o", label=scale)
            ax.set_title("Rolling Backtest MAPE Trend by Scale")
            ax.set_xlabel("Test Year")
            ax.set_ylabel("MAPE (%)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig3_path, dpi=180)
            plt.close(fig)
            figure_entries.append({"name": "backtest_mape_trend", "path": str(fig3_path), "status": "ok"})
        else:
            figure_entries.append({"name": "backtest_mape_trend", "status": "skipped", "reason": "invalid_fold_schema"})
    else:
        figure_entries.append({"name": "backtest_mape_trend", "status": "skipped", "reason": "missing_fold_file"})

    meta = {
        "generated_at": generated_at,
        "figures": figure_entries,
    }
    (report_dir / "paper_figures_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def build_paper_assets(report_dir: str | Path = "reports") -> dict[str, Any]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    tables_meta = build_paper_experiment_tables(report_dir)
    figures_meta = build_paper_figures(report_dir)
    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": tables_meta,
        "figures": figures_meta,
    }
    (report_dir / "paper_assets_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return index
