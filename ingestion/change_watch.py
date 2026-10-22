"""Watch authoritative sources for changes and keep a change log.

A check fetches each watched source, fingerprints its content, and records an
outcome: ``new`` (first successful fetch), ``changed``, ``unchanged`` or
``failed``. Corpus uploads are logged as ``added``. The log feeds the
researcher change feed and the admin sources page.

Only the four authorities that can change the law's text or effect are watched
(Gazette, India Code, MHA, Supreme Court); explanatory portals are not.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ingestion.refresh import RetrievedSource, SourceSnapshotStore, fetch_source
from ingestion.source_registry import SourceRegistry, SourceSpec

WATCHED_AUTHORITIES = {"gazette", "india_code", "mha", "supreme_court"}
ALLOWED_HOSTS = {
    "www.indiacode.nic.in", "indiacode.nic.in", "egazette.gov.in", "www.mha.gov.in", "mha.gov.in",
    "www.sci.gov.in", "sci.gov.in", "main.sci.gov.in", "judgments.ecourts.gov.in",
}
NOTEWORTHY = ("new", "changed", "added")


@dataclass(frozen=True)
class ChangeEvent:
    id: int
    source_id: str
    title: str
    url: str | None
    authority: str | None
    outcome: str
    checked_at: str
    content_hash: str | None
    previous_hash: str | None
    error: str | None
    detail: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def fingerprint(content: bytes) -> str:
    """Hash of the meaningful text, so markup churn and session tokens don't read as changes."""
    text = content.decode("utf-8", errors="ignore")
    if "<" in text[:2000]:
        text = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", text)
        text = re.sub(r"(?is)<input[^>]+type=[\"']?hidden[^>]*>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"\(S\([a-z0-9]+\)\)", "", text)  # ASP.NET cookieless session IDs
    text = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(text.encode("utf-8") if text else content).hexdigest()


def watched_sources(manifest_path: str | Path) -> tuple[list[SourceSpec], list[dict]]:
    """Sources to watch from the manifest, plus the entries deliberately not watched."""
    path = Path(manifest_path)
    if not path.exists():
        return [], []
    registry = SourceRegistry(allowed_hosts=ALLOWED_HOSTS)
    watched, skipped = [], []
    for row in json.loads(path.read_text(encoding="utf-8")):
        if row.get("authority") not in WATCHED_AUTHORITIES:
            skipped.append({**row, "reason": "Explanatory or secondary source"})
            continue
        try:
            watched.append(registry.register(SourceSpec(
                source_id=row["source_id"], title=row["title"], url=row["url"],
                publisher=row.get("publisher", ""), authority=row["authority"],
                document_type=row.get("document_type", ""),
            )))
        except ValueError as exc:
            skipped.append({**row, "reason": str(exc)})
    return watched, skipped


class ChangeLog:
    def __init__(self, path: str | Path = "data/change_log.sqlite3") -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if str(self.path) == ":memory:" else None
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS change_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT,
                    authority TEXT,
                    outcome TEXT NOT NULL,
                    checked_at TEXT NOT NULL,
                    content_hash TEXT,
                    previous_hash TEXT,
                    error TEXT,
                    detail TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_change_events_source ON change_events(source_id);
                """
            )
            connection.commit()
        finally:
            self._close(connection)

    def _connect(self) -> sqlite3.Connection:
        return self._memory or sqlite3.connect(str(self.path))

    def _close(self, connection: sqlite3.Connection) -> None:
        if self._memory is None:
            connection.close()

    def record(self, *, source_id: str, title: str, url: str | None, authority: str | None, outcome: str,
               content_hash: str | None = None, previous_hash: str | None = None,
               error: str | None = None, detail: str | None = None, now: datetime | None = None) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO change_events
                    (source_id, title, url, authority, outcome, checked_at, content_hash, previous_hash, error, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (source_id, title, url, authority, outcome,
                 (now or datetime.now(timezone.utc)).isoformat(), content_hash, previous_hash, error, detail),
            )
            connection.commit()
        finally:
            self._close(connection)

    def _events(self, where: str = "", params: tuple = (), limit: int = 100) -> list[ChangeEvent]:
        connection = self._connect()
        try:
            rows = connection.execute(
                f"""
                SELECT id, source_id, title, url, authority, outcome, checked_at,
                       content_hash, previous_hash, error, detail
                FROM change_events {where} ORDER BY id DESC LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
        finally:
            self._close(connection)
        return [ChangeEvent(*row) for row in rows]

    def feed(self, limit: int = 50) -> list[ChangeEvent]:
        placeholders = ",".join("?" for _ in NOTEWORTHY)
        return self._events(f"WHERE outcome IN ({placeholders})", NOTEWORTHY, limit)

    def latest_by_source(self) -> dict[str, dict]:
        """Last check and last noteworthy change per source."""
        summary: dict[str, dict] = {}
        for event in reversed(self._events(limit=5000)):
            entry = summary.setdefault(event.source_id, {"last_check": None, "last_change": None})
            if event.outcome != "added":
                entry["last_check"] = event.to_dict()
            if event.outcome in NOTEWORTHY:
                entry["last_change"] = event.to_dict()
        return summary


def check_sources(
    sources: list[SourceSpec],
    *,
    snapshots: SourceSnapshotStore,
    log: ChangeLog,
    fetch: Callable[[SourceSpec], RetrievedSource] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    fetch = fetch or fetch_source
    counts = {"new": 0, "changed": 0, "unchanged": 0, "failed": 0}
    for position, source in enumerate(sources):
        if on_progress:
            on_progress(position, len(sources), source.title)
        try:
            retrieved = fetch(source)
            digest = fingerprint(retrieved.content)
            previous = snapshots.latest_hash(source.source_id)
            snapshots.record_if_changed(RetrievedSource(source, retrieved.content, digest, retrieved.retrieved_at))
            outcome = "new" if previous is None else ("unchanged" if previous == digest else "changed")
            log.record(source_id=source.source_id, title=source.title, url=source.url, authority=source.authority,
                       outcome=outcome, content_hash=digest, previous_hash=previous,
                       detail=f"{len(retrieved.content):,} bytes fetched")
        except Exception as exc:
            outcome = "failed"
            log.record(source_id=source.source_id, title=source.title, url=source.url,
                       authority=source.authority, outcome=outcome, error=str(exc)[:300])
        counts[outcome] += 1
    if on_progress:
        on_progress(len(sources), len(sources), "")
    return counts
