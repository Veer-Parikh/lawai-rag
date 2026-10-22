"""Durable record of which corpus files were ingested, when, and with what result."""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class LedgerEntry:
    path: str
    doc_type: str
    status: str  # "ingested" | "dry_run" | "failed"
    chunks_created: int
    chunks_uploaded: int
    error: str | None
    job_id: str | None
    sha256: str | None
    ingested_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class IngestionLedger:
    """Latest result per file path, plus an append-only history of attempts."""

    def __init__(self, path: str | Path = "data/ingestion_ledger.sqlite3") -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_connection = (
            sqlite3.connect(":memory:", check_same_thread=False)
            if str(self.path) == ":memory:"
            else None
        )
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ingestion_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunks_created INTEGER NOT NULL DEFAULT 0,
                    chunks_uploaded INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    job_id TEXT,
                    sha256 TEXT,
                    ingested_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attempts_path ON ingestion_attempts(path);
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

    @staticmethod
    def normalize(path: str | Path) -> str:
        return Path(path).as_posix()

    def record(
        self,
        *,
        path: str | Path,
        doc_type: str,
        chunks_created: int,
        chunks_uploaded: int,
        error: str | None,
        dry_run: bool,
        job_id: str | None = None,
        sha256: str | None = None,
        now: datetime | None = None,
    ) -> LedgerEntry:
        status = "failed" if error else ("dry_run" if dry_run else "ingested")
        entry = LedgerEntry(
            path=self.normalize(path),
            doc_type=doc_type,
            status=status,
            chunks_created=chunks_created,
            chunks_uploaded=chunks_uploaded,
            error=error,
            job_id=job_id,
            sha256=sha256,
            ingested_at=(now or datetime.now(timezone.utc)).isoformat(),
        )
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO ingestion_attempts
                    (path, doc_type, status, chunks_created, chunks_uploaded,
                     error, job_id, sha256, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.path, entry.doc_type, entry.status, entry.chunks_created,
                    entry.chunks_uploaded, entry.error, entry.job_id, entry.sha256,
                    entry.ingested_at,
                ),
            )
            connection.commit()
        finally:
            self._close(connection)
        return entry

    def latest_by_path(self) -> dict[str, LedgerEntry]:
        """Most recent attempt per path (dry runs never mask a real ingestion)."""
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT path, doc_type, status, chunks_created, chunks_uploaded,
                       error, job_id, sha256, ingested_at
                FROM ingestion_attempts
                ORDER BY id ASC
                """
            ).fetchall()
        finally:
            self._close(connection)
        latest: dict[str, LedgerEntry] = {}
        for row in rows:
            entry = LedgerEntry(*row)
            previous = latest.get(entry.path)
            if entry.status == "dry_run" and previous and previous.status == "ingested":
                continue
            latest[entry.path] = entry
        return latest

    def history(self, limit: int = 50) -> list[LedgerEntry]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT path, doc_type, status, chunks_created, chunks_uploaded,
                       error, job_id, sha256, ingested_at
                FROM ingestion_attempts
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            self._close(connection)
        return [LedgerEntry(*row) for row in rows]
