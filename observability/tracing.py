"""Dependency-free structured trace records for local development.

The interface is intentionally compatible with a future OpenTelemetry adapter.
"""
from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterator


@dataclass
class TraceRecord:
    trace_id: str
    span_id: str
    name: str
    started_at: str
    attributes: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, default=str)


class TraceRecorder:
    def __init__(self) -> None:
        self._records: list[TraceRecord] = []
        self._lock = threading.Lock()

    @contextmanager
    def span(
        self,
        name: str,
        *,
        trace_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[TraceRecord]:
        record = TraceRecord(
            trace_id=trace_id or uuid.uuid4().hex,
            span_id=uuid.uuid4().hex,
            name=name,
            started_at=datetime.now(timezone.utc).isoformat(),
            attributes=attributes or {},
        )
        started = perf_counter()
        try:
            yield record
        except Exception as exc:
            record.error = type(exc).__name__
            raise
        finally:
            record.duration_ms = round((perf_counter() - started) * 1000, 3)
            with self._lock:
                self._records.append(record)

    def snapshot(self) -> list[TraceRecord]:
        with self._lock:
            return list(self._records)
