from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _pick_text(elem: ET.Element, names: list[str]) -> str:
    for n in names:
        hit = elem.find(n)
        if hit is not None and hit.text:
            return hit.text.strip()
    return ""


def _parse_dt(text: str) -> datetime | None:
    raw = _safe_text(text)
    if not raw:
        return None

    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc)
    except Exception:
        pass

    try:
        ts = pd.to_datetime(raw, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def _to_day(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)


@dataclass
class SourceRun:
    run_at: str
    kind: str
    source_name: str
    source_url: str
    status: str
    fetched: int
    accepted: int
    error: str


class TextSourceCollector:
    def __init__(self, config_path: str | Path = "config/text_sources.json") -> None:
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        request_cfg = self.config.get("request", {})
        incr_cfg = self.config.get("incremental", {})

        self.timeout_sec = int(request_cfg.get("timeout_sec", 10))
        self.retries = int(request_cfg.get("retries", 2))
        self.backoff_sec = float(request_cfg.get("backoff_sec", 1.5))
        self.user_agent = str(request_cfg.get("user_agent", "coal-price-prediction/1.0"))
        self.lookback_days = int(incr_cfg.get("lookback_days", 14))
        quality_cfg = self.config.get("quality_alert", {})
        self.min_success_rate = float(quality_cfg.get("min_success_rate", 0.7))
        self.min_coverage_ratio = float(quality_cfg.get("min_coverage_ratio", 0.25))
        self.min_accept_ratio = float(quality_cfg.get("min_accept_ratio", 0.05))
        self.quality_alert_overrides = self.config.get("quality_alert_overrides", {})

        self.runtime_dir = Path(incr_cfg.get("runtime_dir", "data/runtime"))
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.runtime_dir / "text_source_state.json"
        self.run_log_path = Path("reports") / "text_source_runs.csv"
        self.health_path = Path("reports") / "text_source_health.json"
        Path("reports").mkdir(parents=True, exist_ok=True)

    def _source_thresholds(self, kind: str, source_name: str, feed_cfg: dict[str, Any]) -> dict[str, float]:
        thresholds = {
            "min_success_rate": self.min_success_rate,
            "min_accept_ratio": self.min_accept_ratio,
        }

        kind_overrides = self.quality_alert_overrides.get(kind, {})
        source_overrides = kind_overrides.get(source_name, {})
        for k in ["min_success_rate", "min_accept_ratio"]:
            if k in source_overrides:
                thresholds[k] = float(source_overrides[k])

        feed_quality = feed_cfg.get("quality_alert", {})
        for k in ["min_success_rate", "min_accept_ratio"]:
            if k in feed_quality:
                thresholds[k] = float(feed_quality[k])

        return thresholds

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _request_text(self, url: str) -> str:
        last_error = ""
        for i in range(self.retries):
            try:
                req = Request(url, headers={"User-Agent": self.user_agent})
                with urlopen(req, timeout=self.timeout_sec) as resp:
                    body = resp.read()
                return body.decode("utf-8", errors="ignore")
            except Exception as exc:
                last_error = str(exc)
                if i < self.retries - 1:
                    time.sleep(self.backoff_sec * (i + 1))
        raise RuntimeError(last_error or "unknown error")

    def _parse_rss_items(self, xml_text: str) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        root = ET.fromstring(xml_text)

        # RSS 2.0 / Atom fallback parser.
        for item in root.findall(".//item"):
            items.append(
                {
                    "title": _pick_text(item, ["title"]),
                    "link": _pick_text(item, ["link"]),
                    "summary": _pick_text(item, ["description"]),
                    "author": _pick_text(item, ["author"]),
                    "published": _pick_text(item, ["pubDate"]),
                }
            )

        if not items:
            ns = {"a": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//a:entry", ns):
                link_elem = entry.find("a:link", ns)
                href = ""
                if link_elem is not None:
                    href = _safe_text(link_elem.attrib.get("href"))
                items.append(
                    {
                        "title": _pick_text(entry, ["{http://www.w3.org/2005/Atom}title"]),
                        "link": href,
                        "summary": _pick_text(entry, ["{http://www.w3.org/2005/Atom}summary", "{http://www.w3.org/2005/Atom}content"]),
                        "author": _pick_text(entry, ["{http://www.w3.org/2005/Atom}author"]),
                        "published": _pick_text(entry, ["{http://www.w3.org/2005/Atom}updated", "{http://www.w3.org/2005/Atom}published"]),
                    }
                )
        return items

    def _build_policy_rows(
        self,
        items: list[dict[str, str]],
        source_name: str,
        source_url: str,
        feed_cfg: dict[str, Any],
        start_dt: datetime,
        end_dt: datetime,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for it in items:
            published = _parse_dt(it.get("published", ""))
            if published is None:
                continue
            day = _to_day(published)
            if day < start_dt or day > end_dt:
                continue
            title = _safe_text(it.get("title"))
            body = _safe_text(it.get("summary"))
            link = _safe_text(it.get("link"))
            if not title and not body:
                continue
            seed = "|".join([source_name, title, link, day.isoformat()])
            out.append(
                {
                    "date": day.date().isoformat(),
                    "doc_id": _sha1(seed),
                    "title": title[:300],
                    "body": body[:2000],
                    "source": source_name,
                    "url": link,
                    "department": _safe_text(feed_cfg.get("department")) or "未知部门",
                    "doc_type": _safe_text(feed_cfg.get("doc_type")) or "政策信息",
                    "_source_url": source_url,
                }
            )
        return out

    def _build_sent_rows(
        self,
        items: list[dict[str, str]],
        source_name: str,
        source_url: str,
        feed_cfg: dict[str, Any],
        start_dt: datetime,
        end_dt: datetime,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for it in items:
            published = _parse_dt(it.get("published", ""))
            if published is None:
                continue
            day = _to_day(published)
            if day < start_dt or day > end_dt:
                continue
            title = _safe_text(it.get("title"))
            body = _safe_text(it.get("summary"))
            link = _safe_text(it.get("link"))
            author = _safe_text(it.get("author"))
            host = urlparse(link).netloc or source_name
            if not title and not body:
                continue
            seed = "|".join([source_name, title, link, day.isoformat()])
            out.append(
                {
                    "date": day.date().isoformat(),
                    "news_id": _sha1(seed),
                    "title": title[:300],
                    "body": body[:2000],
                    "media": host,
                    "url": link,
                    "author": author[:100] or "unknown",
                    "topic": _safe_text(feed_cfg.get("topic")) or "coal",
                    "_source_url": source_url,
                }
            )
        return out

    def _append_run_logs(self, runs: list[SourceRun]) -> None:
        if not runs:
            return
        df = pd.DataFrame([r.__dict__ for r in runs])
        if self.run_log_path.exists():
            old = pd.read_csv(self.run_log_path)
            df = pd.concat([old, df], axis=0, ignore_index=True)
        df.to_csv(self.run_log_path, index=False)

    def _write_health(
        self,
        kind: str,
        df: pd.DataFrame,
        runs: list[SourceRun],
        start: str,
        end: str,
        feeds: list[dict[str, Any]],
    ) -> None:
        success = sum(1 for r in runs if r.status == "success")
        failed = sum(1 for r in runs if r.status != "success")
        dates = pd.to_datetime(df["date"], errors="coerce") if not df.empty else pd.Series([], dtype="datetime64[ns]")
        coverage_start = None if dates.empty else str(dates.min().date())
        coverage_end = None if dates.empty else str(dates.max().date())

        expected_days = max(1, len(pd.date_range(start, end, freq="D")))
        actual_days = 0 if dates.empty else int(dates.dt.date.nunique())
        coverage_ratio = float(actual_days / expected_days)
        source_count = len(runs)
        success_rate = float(success / source_count) if source_count else 0.0
        fetched_total = int(sum(max(r.fetched, 0) for r in runs))
        accepted_total = int(sum(max(r.accepted, 0) for r in runs))
        end_dt = pd.to_datetime(end, utc=True, errors="coerce")
        is_recent_window = False if pd.isna(end_dt) else bool((datetime.now(timezone.utc) - end_dt.to_pydatetime()) <= timedelta(days=30))
        if is_recent_window:
            accept_ratio = float(accepted_total / fetched_total) if fetched_total > 0 else 0.0
        else:
            # Historical replay windows may naturally have zero newly accepted items.
            accept_ratio = 1.0

        score = 100.0 * (0.45 * success_rate + 0.35 * min(1.0, coverage_ratio) + 0.20 * min(1.0, accept_ratio))
        if score >= 80:
            score_status = "good"
        elif score >= 60:
            score_status = "warn"
        else:
            score_status = "critical"

        alerts: list[str] = []
        if success_rate < self.min_success_rate:
            alerts.append(
                f"source_success_rate_low({success_rate:.2f} < {self.min_success_rate:.2f})"
            )
        if coverage_ratio < self.min_coverage_ratio:
            alerts.append(
                f"coverage_ratio_low({coverage_ratio:.2f} < {self.min_coverage_ratio:.2f})"
            )
        if is_recent_window and accept_ratio < self.min_accept_ratio:
            alerts.append(
                f"accept_ratio_low({accept_ratio:.2f} < {self.min_accept_ratio:.2f})"
            )

        feed_map: dict[str, dict[str, Any]] = {}
        for feed in feeds:
            source_name = _safe_text(feed.get("name")) or "unknown_source"
            feed_map[source_name] = feed

        sources_detail: list[dict[str, Any]] = []
        for r in runs:
            thresholds = self._source_thresholds(kind, r.source_name, feed_map.get(r.source_name, {}))
            src_success_rate = 1.0 if r.status == "success" else 0.0
            if is_recent_window:
                src_accept_ratio = float(r.accepted / r.fetched) if r.fetched > 0 else 0.0
            else:
                src_accept_ratio = 1.0
            src_score = 100.0 * (0.6 * src_success_rate + 0.4 * min(1.0, src_accept_ratio))
            if src_score >= 80:
                src_status = "good"
            elif src_score >= 60:
                src_status = "warn"
            else:
                src_status = "critical"

            src_alerts: list[str] = []
            if src_success_rate < float(thresholds["min_success_rate"]):
                src_alerts.append(
                    f"source_success_rate_low({src_success_rate:.2f} < {thresholds['min_success_rate']:.2f})"
                )
            if is_recent_window and src_accept_ratio < float(thresholds["min_accept_ratio"]):
                src_alerts.append(
                    f"accept_ratio_low({src_accept_ratio:.2f} < {thresholds['min_accept_ratio']:.2f})"
                )

            sources_detail.append(
                {
                    "source_name": r.source_name,
                    "source_url": r.source_url,
                    "status": r.status,
                    "fetched": int(r.fetched),
                    "accepted": int(r.accepted),
                    "error": r.error,
                    "quality_score": round(src_score, 2),
                    "quality_status": src_status,
                    "quality_alerts": src_alerts,
                    "thresholds": {
                        "min_success_rate": float(thresholds["min_success_rate"]),
                        "min_accept_ratio": float(thresholds["min_accept_ratio"]),
                    },
                }
            )

        health = {}
        if self.health_path.exists():
            try:
                health = json.loads(self.health_path.read_text(encoding="utf-8"))
            except Exception:
                health = {}
        health[kind] = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "records_total": int(len(df)),
            "sources_total": int(len(runs)),
            "sources_success": int(success),
            "sources_failed": int(failed),
            "coverage_start": coverage_start,
            "coverage_end": coverage_end,
            "coverage_ratio": round(coverage_ratio, 4),
            "quality_score": round(score, 2),
            "quality_status": score_status,
            "quality_alerts": alerts,
            "quality_metrics": {
                "success_rate": round(success_rate, 4),
                "accept_ratio": round(accept_ratio, 4),
                "is_recent_window": is_recent_window,
            },
            "sources_detail": sources_detail,
        }
        self.health_path.write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8")

    def collect(
        self,
        kind: str,
        start: str,
        end: str,
        existing_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if kind not in {"policy_text", "sentiment_text"}:
            raise ValueError(f"unsupported kind: {kind}")

        feeds_key = "policy_feeds" if kind == "policy_text" else "sentiment_feeds"
        feeds = list(self.config.get(feeds_key, []))
        if not feeds:
            return pd.DataFrame()

        now_utc = datetime.now(timezone.utc)
        start_dt = _to_day(pd.to_datetime(start, utc=True).to_pydatetime())
        end_dt = _to_day(pd.to_datetime(end, utc=True).to_pydatetime())
        state = self._load_state()

        rows: list[dict[str, Any]] = []
        runs: list[SourceRun] = []
        for feed in feeds:
            source_name = _safe_text(feed.get("name")) or "unknown_source"
            source_url = _safe_text(feed.get("url"))
            if not source_url:
                runs.append(
                    SourceRun(
                        run_at=now_utc.isoformat(),
                        kind=kind,
                        source_name=source_name,
                        source_url="",
                        status="failed",
                        fetched=0,
                        accepted=0,
                        error="empty url",
                    )
                )
                continue

            source_state = state.get(kind, {}).get(source_name, {})
            last_success = _parse_dt(str(source_state.get("last_success_at", "")))
            if last_success is None:
                fetch_start = start_dt
            else:
                fetch_start = max(start_dt, _to_day(last_success) - timedelta(days=self.lookback_days))

            try:
                raw_xml = self._request_text(source_url)
                items = self._parse_rss_items(raw_xml)
                if kind == "policy_text":
                    accepted_rows = self._build_policy_rows(items, source_name, source_url, feed, fetch_start, end_dt)
                else:
                    accepted_rows = self._build_sent_rows(items, source_name, source_url, feed, fetch_start, end_dt)
                rows.extend(accepted_rows)

                state.setdefault(kind, {})
                state[kind][source_name] = {
                    "last_success_at": now_utc.isoformat(),
                    "last_count": len(accepted_rows),
                    "url": source_url,
                }
                runs.append(
                    SourceRun(
                        run_at=now_utc.isoformat(),
                        kind=kind,
                        source_name=source_name,
                        source_url=source_url,
                        status="success",
                        fetched=len(items),
                        accepted=len(accepted_rows),
                        error="",
                    )
                )
            except Exception as exc:
                runs.append(
                    SourceRun(
                        run_at=now_utc.isoformat(),
                        kind=kind,
                        source_name=source_name,
                        source_url=source_url,
                        status="failed",
                        fetched=0,
                        accepted=0,
                        error=str(exc)[:400],
                    )
                )

        self._save_state(state)
        self._append_run_logs(runs)

        live = pd.DataFrame(rows)
        if existing_df is not None and not existing_df.empty:
            existing = existing_df.copy()
            existing["date"] = pd.to_datetime(existing["date"]).dt.date.astype(str)
            live = pd.concat([existing, live], axis=0, ignore_index=True)

        if live.empty:
            self._write_health(kind, live, runs, start, end, feeds)
            return live

        id_col = "doc_id" if kind == "policy_text" else "news_id"
        keep_cols = (
            ["date", "doc_id", "title", "body", "source", "url", "department", "doc_type"]
            if kind == "policy_text"
            else ["date", "news_id", "title", "body", "media", "url", "author", "topic"]
        )
        live = live.drop_duplicates(subset=[id_col], keep="last")
        live["date"] = pd.to_datetime(live["date"], errors="coerce")
        live = live[(live["date"] >= pd.to_datetime(start)) & (live["date"] <= pd.to_datetime(end))]
        live = live.sort_values("date")
        live["date"] = live["date"].dt.date.astype(str)
        live = live.reindex(columns=keep_cols, fill_value="")
        self._write_health(kind, live, runs, start, end, feeds)
        return live.reset_index(drop=True)
