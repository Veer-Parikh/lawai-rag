"""Safe source refresh primitives; parsing and publication remain explicit steps."""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from ingestion.source_registry import SourceSpec


@dataclass(frozen=True)
class RetrievedSource:
    source: SourceSpec
    content: bytes
    content_hash: str
    retrieved_at: str


class SourceSnapshotStore:
    """Durable content-hash registry used to make refreshes idempotent."""

    def __init__(self, path: str | Path = "data/source_refresh.sqlite3") -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_connection = (
            sqlite3.connect(":memory:") if str(self.path) == ":memory:" else None
        )
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS source_snapshots (
                    source_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            self._close(connection)

    def _connect(self) -> sqlite3.Connection:
        return self._memory_connection or sqlite3.connect(str(self.path))

    def _close(self, connection: sqlite3.Connection) -> None:
        if self._memory_connection is None:
            connection.close()

    def record_if_changed(self, retrieved: RetrievedSource) -> bool:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT content_hash FROM source_snapshots WHERE source_id = ?",
                (retrieved.source.source_id,),
            ).fetchone()
            if row is not None and row[0] == retrieved.content_hash:
                return False
            connection.execute(
                """
                INSERT INTO source_snapshots (source_id, content_hash, retrieved_at)
                VALUES (?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    retrieved_at = excluded.retrieved_at
                """,
                (
                    retrieved.source.source_id,
                    retrieved.content_hash,
                    retrieved.retrieved_at,
                ),
            )
            connection.commit()
            return True
        finally:
            self._close(connection)

    def latest_hash(self, source_id: str) -> str | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT content_hash FROM source_snapshots WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            return row[0] if row else None
        finally:
            self._close(connection)


def fetch_source(source: SourceSpec, *, timeout_seconds: int = 20) -> RetrievedSource:
    request = Request(
        source.url,
        headers={"User-Agent": "BharatLawSourceRefresh/1.0"},
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        content = response.read()
    return RetrievedSource(
        source=source,
        content=content,
        content_hash=hashlib.sha256(content).hexdigest(),
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
