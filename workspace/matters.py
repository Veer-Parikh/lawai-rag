"""Matters: named folders of saved research answers, owned by one signed-in subject."""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_ANSWER_BYTES = 200_000


@dataclass(frozen=True)
class Entry:
    id: str
    matter_id: str
    created_at: str
    question: str
    as_of: str | None
    answer: dict[str, Any]
    note: str


@dataclass(frozen=True)
class Matter:
    id: str
    owner: str
    title: str
    reference: str
    notes: str
    created_at: str
    updated_at: str
    entry_count: int = 0
    entries: list[Entry] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("owner")
        return payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MatterStore:
    def __init__(self, path: str | Path = "data/workspace.sqlite3") -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if str(self.path) == ":memory:" else None
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS matters (
                    id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    title TEXT NOT NULL,
                    reference TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_matters_owner ON matters(owner);
                CREATE TABLE IF NOT EXISTS matter_entries (
                    id TEXT PRIMARY KEY,
                    matter_id TEXT NOT NULL REFERENCES matters(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    question TEXT NOT NULL,
                    as_of TEXT,
                    answer_json TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_entries_matter ON matter_entries(matter_id);
                """
            )
            connection.commit()
        finally:
            self._close(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = self._memory or sqlite3.connect(str(self.path))
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _close(self, connection: sqlite3.Connection) -> None:
        if self._memory is None:
            connection.close()

    def _owned(self, connection: sqlite3.Connection, owner: str, matter_id: str) -> tuple:
        row = connection.execute(
            "SELECT id, owner, title, reference, notes, created_at, updated_at FROM matters WHERE id = ?",
            (matter_id,),
        ).fetchone()
        if row is None or row[1] != owner:
            raise KeyError(f"matter not found: {matter_id}")
        return row

    def create(self, owner: str, *, title: str, reference: str = "", notes: str = "") -> Matter:
        title = title.strip()
        if not title:
            raise ValueError("give the matter a title")
        matter_id, stamp = str(uuid.uuid4()), _now()
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO matters (id, owner, title, reference, notes, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (matter_id, owner, title, reference.strip(), notes, stamp, stamp),
            )
            connection.commit()
        finally:
            self._close(connection)
        return self.get(owner, matter_id)

    def list(self, owner: str) -> list[Matter]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT m.id, m.owner, m.title, m.reference, m.notes, m.created_at, m.updated_at, COUNT(e.id)
                FROM matters m LEFT JOIN matter_entries e ON e.matter_id = m.id
                WHERE m.owner = ? GROUP BY m.id ORDER BY m.updated_at DESC
                """,
                (owner,),
            ).fetchall()
        finally:
            self._close(connection)
        return [Matter(*row[:7], entry_count=row[7]) for row in rows]

    def get(self, owner: str, matter_id: str) -> Matter:
        connection = self._connect()
        try:
            row = self._owned(connection, owner, matter_id)
            entries = connection.execute(
                """
                SELECT id, matter_id, created_at, question, as_of, answer_json, note
                FROM matter_entries WHERE matter_id = ? ORDER BY created_at
                """,
                (matter_id,),
            ).fetchall()
        finally:
            self._close(connection)
        parsed = [Entry(*entry[:5], answer=json.loads(entry[5]), note=entry[6]) for entry in entries]
        return Matter(*row, entry_count=len(parsed), entries=parsed)

    def update(self, owner: str, matter_id: str, **changes: str | None) -> Matter:
        allowed = {key: value for key, value in changes.items() if key in ("title", "reference", "notes") and value is not None}
        if "title" in allowed and not allowed["title"].strip():
            raise ValueError("give the matter a title")
        connection = self._connect()
        try:
            self._owned(connection, owner, matter_id)
            for key, value in allowed.items():
                connection.execute(f"UPDATE matters SET {key} = ? WHERE id = ?", (value.strip() if key != "notes" else value, matter_id))
            connection.execute("UPDATE matters SET updated_at = ? WHERE id = ?", (_now(), matter_id))
            connection.commit()
        finally:
            self._close(connection)
        return self.get(owner, matter_id)

    def delete(self, owner: str, matter_id: str) -> None:
        connection = self._connect()
        try:
            self._owned(connection, owner, matter_id)
            connection.execute("DELETE FROM matter_entries WHERE matter_id = ?", (matter_id,))
            connection.execute("DELETE FROM matters WHERE id = ?", (matter_id,))
            connection.commit()
        finally:
            self._close(connection)

    def add_entry(self, owner: str, matter_id: str, *, question: str, answer: dict[str, Any],
                  as_of: str | None = None, note: str = "") -> Entry:
        payload = json.dumps(answer, ensure_ascii=False)
        if len(payload.encode("utf-8")) > MAX_ANSWER_BYTES:
            raise ValueError("this answer is too large to save")
        entry_id, stamp = str(uuid.uuid4()), _now()
        connection = self._connect()
        try:
            self._owned(connection, owner, matter_id)
            connection.execute(
                "INSERT INTO matter_entries (id, matter_id, created_at, question, as_of, answer_json, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (entry_id, matter_id, stamp, question, as_of, payload, note),
            )
            connection.execute("UPDATE matters SET updated_at = ? WHERE id = ?", (stamp, matter_id))
            connection.commit()
        finally:
            self._close(connection)
        return next(entry for entry in self.get(owner, matter_id).entries if entry.id == entry_id)

    def update_entry(self, owner: str, matter_id: str, entry_id: str, *, note: str) -> Entry:
        connection = self._connect()
        try:
            self._owned(connection, owner, matter_id)
            cursor = connection.execute(
                "UPDATE matter_entries SET note = ? WHERE id = ? AND matter_id = ?", (note, entry_id, matter_id)
            )
            if cursor.rowcount == 0:
                raise KeyError(f"entry not found: {entry_id}")
            connection.commit()
        finally:
            self._close(connection)
        return next(entry for entry in self.get(owner, matter_id).entries if entry.id == entry_id)

    def delete_entry(self, owner: str, matter_id: str, entry_id: str) -> None:
        connection = self._connect()
        try:
            self._owned(connection, owner, matter_id)
            cursor = connection.execute("DELETE FROM matter_entries WHERE id = ? AND matter_id = ?", (entry_id, matter_id))
            if cursor.rowcount == 0:
                raise KeyError(f"entry not found: {entry_id}")
            connection.commit()
        finally:
            self._close(connection)
