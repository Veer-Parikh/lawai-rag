"""Section correspondences between the old and new criminal laws.

Rows are seeded from ``legal/data/crosswalk.json`` and stored in SQLite so an
administrator can mark each one verified against the official tables.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

SEED_PATH = Path(__file__).parent / "data" / "crosswalk.json"
RELATIONS = ("equivalent", "modified", "omitted", "new")


@dataclass(frozen=True)
class CrosswalkRow:
    id: str
    from_code: str | None
    from_section: str | None
    from_heading: str | None
    to_code: str | None
    to_section: str | None
    to_heading: str | None
    relation: str
    note: str
    verified: bool
    verified_by: str | None
    verified_at: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def base_section(section: str | None) -> str:
    """'103(1)' -> '103'; '65B' -> '65B'."""
    return re.sub(r"\(.*$", "", (section or "").strip()).upper()


def row_id(row: dict) -> str:
    left = f"{row.get('from_code') or 'none'}-{row.get('from_section') or ''}"
    right = f"{row.get('to_code') or 'none'}-{row.get('to_section') or ''}"
    return re.sub(r"[^a-z0-9]+", "-", f"{left}-to-{right}".lower()).strip("-")


_QUERY = re.compile(r"^(?:(?P<code>ipc|crpc|iea|bns|bnss|bsa)\b)?[\s.,]*(?:s(?:ec(?:tion)?)?\.?\s*)?(?P<section>\d{1,3}[a-z]{0,3})$", re.I)


class CrosswalkStore:
    def __init__(self, path: str | Path = "data/crosswalk.sqlite3", *, seed_path: Path = SEED_PATH) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if str(self.path) == ":memory:" else None
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS crosswalk (
                    id TEXT PRIMARY KEY,
                    from_code TEXT, from_section TEXT, from_heading TEXT,
                    to_code TEXT, to_section TEXT, to_heading TEXT,
                    relation TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    verified INTEGER NOT NULL DEFAULT 0,
                    verified_by TEXT, verified_at TEXT
                )
                """
            )
            connection.commit()
        finally:
            self._close(connection)
        self.seed(seed_path)

    def _connect(self) -> sqlite3.Connection:
        return self._memory or sqlite3.connect(str(self.path))

    def _close(self, connection: sqlite3.Connection) -> None:
        if self._memory is None:
            connection.close()

    def seed(self, seed_path: Path) -> int:
        """Insert seed rows that are not stored yet; never overwrites reviewed rows."""
        if not seed_path.exists():
            return 0
        payload = json.loads(seed_path.read_text(encoding="utf-8"))
        connection = self._connect()
        try:
            before = connection.total_changes
            for row in payload["rows"]:
                if row["relation"] not in RELATIONS:
                    raise ValueError(f"unknown relation {row['relation']!r}")
                connection.execute(
                    """
                    INSERT OR IGNORE INTO crosswalk
                        (id, from_code, from_section, from_heading, to_code, to_section, to_heading, relation, note)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row_id(row), row.get("from_code"), row.get("from_section"), row.get("from_heading"),
                        row.get("to_code"), row.get("to_section"), row.get("to_heading"),
                        row["relation"], row.get("note") or "",
                    ),
                )
            connection.commit()
            return connection.total_changes - before
        finally:
            self._close(connection)

    def _rows(self, where: str = "", params: tuple = ()) -> list[CrosswalkRow]:
        connection = self._connect()
        try:
            rows = connection.execute(
                f"""
                SELECT id, from_code, from_section, from_heading, to_code, to_section, to_heading,
                       relation, note, verified, verified_by, verified_at
                FROM crosswalk {where}
                ORDER BY COALESCE(from_code, to_code), CAST(COALESCE(from_section, to_section) AS INTEGER),
                         COALESCE(from_section, to_section)
                """,
                params,
            ).fetchall()
        finally:
            self._close(connection)
        return [CrosswalkRow(*row[:9], bool(row[9]), row[10], row[11]) for row in rows]

    def all(self) -> list[CrosswalkRow]:
        return self._rows()

    def get(self, identifier: str) -> CrosswalkRow:
        rows = self._rows("WHERE id = ?", (identifier,))
        if not rows:
            raise KeyError(f"crosswalk row not found: {identifier}")
        return rows[0]

    def search(self, query: str = "", *, code: str | None = None, relation: str | None = None) -> list[CrosswalkRow]:
        rows = self.all()
        if code:
            rows = [r for r in rows if code.lower() in {(r.from_code or "").lower(), (r.to_code or "").lower()}]
        if relation:
            rows = [r for r in rows if r.relation == relation]
        query = query.strip()
        if not query:
            return rows
        exact = _QUERY.match(query)
        if exact:
            wanted_code = (exact.group("code") or "").lower()
            wanted = exact.group("section").upper()

            def hit(r: CrosswalkRow, side: str) -> bool:
                row_code = (getattr(r, f"{side}_code") or "").lower()
                return base_section(getattr(r, f"{side}_section")) == wanted and (not wanted_code or row_code == wanted_code)

            return [r for r in rows if hit(r, "from") or hit(r, "to")]
        needle = query.lower()
        return [
            r for r in rows
            if needle in " ".join(filter(None, [r.from_heading, r.to_heading, r.note])).lower()
        ]

    def for_section(self, code: str, section: str) -> list[CrosswalkRow]:
        wanted = base_section(section)
        return [
            r for r in self.all()
            if (r.from_code or "").lower() == code.lower() and base_section(r.from_section) == wanted
            or (r.to_code or "").lower() == code.lower() and base_section(r.to_section) == wanted
        ]

    def set_verified(self, identifier: str, *, verified: bool, by: str, note: str | None = None) -> CrosswalkRow:
        self.get(identifier)
        connection = self._connect()
        try:
            connection.execute(
                """
                UPDATE crosswalk
                SET verified = ?, verified_by = ?, verified_at = ?, note = COALESCE(?, note)
                WHERE id = ?
                """,
                (
                    int(verified),
                    by if verified else None,
                    datetime.now(timezone.utc).isoformat() if verified else None,
                    note,
                    identifier,
                ),
            )
            connection.commit()
        finally:
            self._close(connection)
        return self.get(identifier)
