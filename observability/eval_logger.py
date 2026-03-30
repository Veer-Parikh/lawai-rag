"""
observability/eval_logger.py — Append-only JSONL eval log for every /chat request.

Logs privacy-safe metrics (never raw queries) to observability/eval.jsonl.
Used for offline retrieval quality analysis and Gemini cost tracking.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

import config
from utils.logger import get_logger

logger = get_logger(__name__)
_log_lock = threading.Lock()


def log_request(
    *,
    query_hash: str,
    session_id_hash: str,
    domain_detected: str,
    domain_router_stage: str,
    confidence_tier: str,
    top_chunk_score: float,
    chunks_used: int,
    reranker_used: bool,
    cross_refs_appended: int,
    citation_warnings: int,
    query_rewritten: bool,
    latency_ms: int,
    gemini_tokens_used: int,
    had_llm_call: bool,
) -> None:
    """
    Append a single eval log entry to eval.jsonl.
    Thread-safe. Non-blocking: errors are swallowed to never impact request flow.
    """
    entry = {
        "query_hash": query_hash,
        "session_id_hash": session_id_hash,
        "domain_detected": domain_detected,
        "domain_router_stage": domain_router_stage,
        "confidence_tier": confidence_tier,
        "top_chunk_score": top_chunk_score,
        "chunks_used": chunks_used,
        "reranker_used": reranker_used,
        "cross_refs_appended": cross_refs_appended,
        "citation_warnings": citation_warnings,
        "query_rewritten": query_rewritten,
        "latency_ms": latency_ms,
        "gemini_tokens_used": gemini_tokens_used,
        "had_llm_call": had_llm_call,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        log_path = config.EVAL_LOG_PATH
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        with _log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning(f"Eval logging failed (non-critical): {exc}")
