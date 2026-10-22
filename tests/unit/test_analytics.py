from __future__ import annotations

from datetime import date

from observability.analytics import percentile, summarize


def _entry(day: int, tier: str, latency: int, *, llm: bool, tokens: int = 0) -> dict:
    return {
        "query_hash": f"hash{day}{tier}{latency}",
        "session_id_hash": "",
        "domain_detected": "criminal",
        "domain_router_stage": "override",
        "confidence_tier": tier,
        "top_chunk_score": 0.7,
        "chunks_used": 3 if llm else 0,
        "citation_warnings": 1 if llm else 0,
        "query_rewritten": False,
        "latency_ms": latency,
        "gemini_tokens_used": tokens,
        "had_llm_call": llm,
        "timestamp": f"2026-10-{day:02d}T09:00:00+00:00",
    }


def test_percentile_nearest_rank():
    assert percentile([], 95) == 0.0
    assert percentile([10, 20, 30, 40], 50) == 20.0
    assert percentile([10, 20, 30, 40], 95) == 40.0


def test_summary_counts_abstentions_and_tiers_inside_window():
    requests = [
        _entry(20, "HIGH", 1000, llm=True, tokens=500),
        _entry(21, "LOW", 400, llm=False),
        _entry(21, "MEDIUM", 2000, llm=True, tokens=300),
        _entry(1, "HIGH", 9999, llm=True, tokens=999),  # outside a 7-day window
    ]

    summary = summarize(requests, days=7, today=date(2026, 10, 21))

    assert summary["totals"]["requests"] == 3
    assert summary["totals"]["answered"] == 2
    assert summary["totals"]["abstention_rate"] == round(1 / 3, 4)
    assert summary["totals"]["tokens_used"] == 800
    assert summary["confidence"] == {"HIGH": 1, "MEDIUM": 1, "LOW": 1}
    assert len(summary["daily"]) == 7
    assert summary["daily"][-1] == {
        "date": "2026-10-21", "total": 2, "high": 0, "medium": 1, "low": 1,
        "p50_latency_ms": 400.0, "p95_latency_ms": 2000.0,
    }
    assert summary["recent"][0]["query_ref"] == "hash21MEDI"


def test_feedback_is_grouped_by_kind():
    feedback = [
        {"kind": "wrong_source", "query_hash": "abc", "timestamp": "2026-10-21T08:00:00+00:00"},
        {"kind": "wrong_source", "query_hash": "def", "timestamp": "2026-10-21T09:00:00+00:00"},
        {"kind": "helpful", "query_hash": "ghi", "timestamp": "2026-10-20T09:00:00+00:00"},
    ]

    summary = summarize([], feedback, days=7, today=date(2026, 10, 21))

    assert summary["feedback"]["by_kind"] == {"wrong_source": 2, "helpful": 1}
    assert summary["feedback"]["recent"][0]["query_ref"] == "def"
