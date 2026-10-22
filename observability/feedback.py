"""Append-only research feedback log; stores query hashes, never raw queries."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

import config
from utils.logger import get_logger

logger = get_logger(__name__)
_lock = threading.Lock()

FEEDBACK_KINDS = ("helpful", "wrong_source", "stale_status", "missing_authority", "poor_explanation")


def record_feedback(*, kind: str, query_hash: str, role: str, comment: str | None = None) -> dict:
    if kind not in FEEDBACK_KINDS:
        raise ValueError(f"unsupported feedback kind: {kind}")
    entry = {
        "kind": kind,
        "query_hash": query_hash,
        "role": role,
        "comment": (comment or "").strip()[:500] or None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = config.FEEDBACK_LOG_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _lock:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    return entry
