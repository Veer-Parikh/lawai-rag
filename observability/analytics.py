"""Aggregate the privacy-safe eval and feedback logs into admin analytics."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

TIERS = ("HIGH", "MEDIUM", "LOW")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; 0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return float(ordered[rank - 1])


def _timestamp(entry: dict[str, Any]) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(entry["timestamp"])
    except (KeyError, TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def summarize(
    requests: Iterable[dict[str, Any]],
    feedback: Iterable[dict[str, Any]] = (),
    *,
    days: int = 14,
    today: date | None = None,
) -> dict[str, Any]:
    today = today or datetime.now(timezone.utc).date()
    start = today - timedelta(days=days - 1)

    window: list[tuple[date, dict[str, Any]]] = []
    for entry in requests:
        stamp = _timestamp(entry)
        if stamp and start <= stamp.date() <= today:
            window.append((stamp.date(), entry))
    window.sort(key=lambda item: item[1]["timestamp"])

    latencies = [float(e.get("latency_ms", 0)) for _, e in window]
    answered = [e for _, e in window if e.get("had_llm_call")]
    tiers = Counter(str(e.get("confidence_tier", "LOW")).upper() for _, e in window)

    per_day: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for day, entry in window:
        per_day[day].append(entry)
    daily = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        entries = per_day.get(day, [])
        day_latency = [float(e.get("latency_ms", 0)) for e in entries]
        day_tiers = Counter(str(e.get("confidence_tier", "LOW")).upper() for e in entries)
        daily.append({
            "date": day.isoformat(),
            "total": len(entries),
            **{tier.lower(): day_tiers.get(tier, 0) for tier in TIERS},
            "p50_latency_ms": percentile(day_latency, 50),
            "p95_latency_ms": percentile(day_latency, 95),
        })

    feedback_window = []
    for item in feedback:
        stamp = _timestamp(item)
        if stamp and start <= stamp.date() <= today:
            feedback_window.append(item)

    total = len(window)
    return {
        "window_days": days,
        "window_start": start.isoformat(),
        "window_end": today.isoformat(),
        "totals": {
            "requests": total,
            "answered": len(answered),
            "abstained": total - len(answered),
            "abstention_rate": round((total - len(answered)) / total, 4) if total else 0.0,
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
            "tokens_used": sum(int(e.get("gemini_tokens_used", 0)) for e in answered),
            "citation_warnings": sum(int(e.get("citation_warnings", 0)) for _, e in window),
            "rewritten_queries": sum(1 for _, e in window if e.get("query_rewritten")),
            "sessions": len({e.get("session_id_hash") for _, e in window if e.get("session_id_hash")}),
            "feedback": len(feedback_window),
        },
        "confidence": {tier: tiers.get(tier, 0) for tier in TIERS},
        "domains": dict(Counter(
            domain
            for _, e in window
            for domain in str(e.get("domain_detected", "unknown")).split(",")
            if domain
        ).most_common()),
        "router_stages": dict(Counter(str(e.get("domain_router_stage", "unknown")) for _, e in window).most_common()),
        "daily": daily,
        "recent": [
            {
                "timestamp": e["timestamp"],
                "query_ref": str(e.get("query_hash", ""))[:10],
                "domain": e.get("domain_detected"),
                "confidence_tier": e.get("confidence_tier"),
                "top_chunk_score": round(float(e.get("top_chunk_score", 0.0)), 4),
                "chunks_used": e.get("chunks_used", 0),
                "citation_warnings": e.get("citation_warnings", 0),
                "latency_ms": e.get("latency_ms", 0),
                "answered": bool(e.get("had_llm_call")),
            }
            for _, e in reversed(window[-25:])
        ],
        "feedback": {
            "by_kind": dict(Counter(str(item.get("kind", "other")) for item in feedback_window).most_common()),
            "recent": [
                {
                    "timestamp": item.get("timestamp"),
                    "kind": item.get("kind"),
                    "query_ref": str(item.get("query_hash", ""))[:10],
                    "comment": item.get("comment"),
                }
                for item in sorted(feedback_window, key=lambda i: i.get("timestamp", ""), reverse=True)[:10]
            ],
        },
    }
