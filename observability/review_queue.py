"""Review queue: researcher problem reports, triaged by an administrator, promoted into eval cases.

The question and answer text are stored only when the researcher chose to
share them. Without them an item can still be triaged, but not promoted.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("open", "confirmed", "dismissed", "promoted")
PROBLEM_KINDS = ("wrong_source", "stale_status", "missing_authority", "poor_explanation")
EVAL_CATEGORY = {
    "wrong_source": "source-priority",
    "stale_status": "temporal",
    "missing_authority": "citation",
    "poor_explanation": "explanation",
}


@dataclass(frozen=True)
class ReviewItem:
    id: int
    created_at: str
    kind: str
    query_hash: str
    question: str | None
    answer: str | None
    citations: list[str]
    sources: list[dict]
    confidence_tier: str | None
    comment: str | None
    status: str
    reviewer_note: str | None
    expected_sections: list[str]
    must_abstain: bool
    resolved_by: str | None
    resolved_at: str | None
    eval_case_id: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class ReviewQueue:
    def __init__(self, path: str | Path = "data/review_queue.sqlite3") -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if str(self.path) == ":memory:" else None
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS review_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    question TEXT,
                    answer TEXT,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    confidence_tier TEXT,
                    comment TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    reviewer_note TEXT,
                    expected_sections_json TEXT NOT NULL DEFAULT '[]',
                    must_abstain INTEGER NOT NULL DEFAULT 0,
                    resolved_by TEXT,
                    resolved_at TEXT,
                    eval_case_id TEXT
                )
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

    _COLUMNS = (
        "id, created_at, kind, query_hash, question, answer, citations_json, sources_json, confidence_tier, "
        "comment, status, reviewer_note, expected_sections_json, must_abstain, resolved_by, resolved_at, eval_case_id"
    )

    @staticmethod
    def _item(row: tuple) -> ReviewItem:
        return ReviewItem(
            id=row[0], created_at=row[1], kind=row[2], query_hash=row[3], question=row[4], answer=row[5],
            citations=json.loads(row[6]), sources=json.loads(row[7]), confidence_tier=row[8], comment=row[9],
            status=row[10], reviewer_note=row[11], expected_sections=json.loads(row[12]),
            must_abstain=bool(row[13]), resolved_by=row[14], resolved_at=row[15], eval_case_id=row[16],
        )

    def add(self, *, kind: str, query_hash: str, question: str | None = None, answer: str | None = None,
            citations: list[str] | None = None, sources: list[dict] | None = None,
            confidence_tier: str | None = None, comment: str | None = None) -> ReviewItem:
        if kind not in PROBLEM_KINDS:
            raise ValueError(f"only problem reports enter the review queue, not {kind!r}")
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                INSERT INTO review_items
                    (created_at, kind, query_hash, question, answer, citations_json, sources_json, confidence_tier, comment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (datetime.now(timezone.utc).isoformat(), kind, query_hash, question, answer,
                 json.dumps(citations or []), json.dumps(sources or []), confidence_tier, comment),
            )
            connection.commit()
            new_id = cursor.lastrowid
        finally:
            self._close(connection)
        return self.get(new_id)

    def get(self, item_id: int) -> ReviewItem:
        connection = self._connect()
        try:
            row = connection.execute(f"SELECT {self._COLUMNS} FROM review_items WHERE id = ?", (item_id,)).fetchone()
        finally:
            self._close(connection)
        if row is None:
            raise KeyError(f"review item not found: {item_id}")
        return self._item(row)

    def list(self, status: str | None = None, limit: int = 100) -> list[ReviewItem]:
        connection = self._connect()
        try:
            rows = connection.execute(
                f"SELECT {self._COLUMNS} FROM review_items {'WHERE status = ?' if status else ''} ORDER BY id DESC LIMIT ?",
                (status, limit) if status else (limit,),
            ).fetchall()
        finally:
            self._close(connection)
        return [self._item(row) for row in rows]

    def counts(self) -> dict[str, int]:
        connection = self._connect()
        try:
            rows = dict(connection.execute("SELECT status, COUNT(*) FROM review_items GROUP BY status").fetchall())
        finally:
            self._close(connection)
        return {status: rows.get(status, 0) for status in STATUSES}

    def resolve(self, item_id: int, *, status: str, by: str, note: str | None = None,
                expected_sections: list[str] | None = None, must_abstain: bool | None = None) -> ReviewItem:
        if status not in ("open", "confirmed", "dismissed"):
            raise ValueError("status must be open, confirmed or dismissed")
        current = self.get(item_id)
        if current.status == "promoted":
            raise ValueError("this item is already an eval case")
        connection = self._connect()
        try:
            connection.execute(
                """
                UPDATE review_items
                SET status = ?, reviewer_note = COALESCE(?, reviewer_note),
                    expected_sections_json = COALESCE(?, expected_sections_json),
                    must_abstain = COALESCE(?, must_abstain),
                    resolved_by = ?, resolved_at = ?
                WHERE id = ?
                """,
                (
                    status, note,
                    json.dumps(expected_sections) if expected_sections is not None else None,
                    int(must_abstain) if must_abstain is not None else None,
                    None if status == "open" else by,
                    None if status == "open" else datetime.now(timezone.utc).isoformat(),
                    item_id,
                ),
            )
            connection.commit()
        finally:
            self._close(connection)
        return self.get(item_id)

    def promote(self, item_id: int, *, dataset_path: str | Path, by: str) -> dict:
        """Append the item to an eval dataset as a reviewed case."""
        item = self.get(item_id)
        if item.status != "confirmed":
            raise ValueError("confirm the problem before turning it into an eval case")
        if not item.question:
            raise ValueError("the researcher did not share the question, so it cannot become an eval case")
        case = {
            "id": f"review-{item.id}",
            "category": EVAL_CATEGORY[item.kind],
            "question": item.question,
            "jurisdiction": "IN",
            "expected_source_ids": [],
            "expected_anchors": item.expected_sections,
            "must_abstain": item.must_abstain,
            "provenance": {
                "source": "review-queue",
                "reported_as": item.kind,
                "reviewed_by": by,
                "review_note": item.reviewer_note,
            },
        }
        path = Path(dataset_path)
        cases = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        cases = [existing for existing in cases if existing.get("id") != case["id"]] + [case]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        connection = self._connect()
        try:
            connection.execute(
                "UPDATE review_items SET status = 'promoted', eval_case_id = ?, resolved_by = ?, resolved_at = ? WHERE id = ?",
                (case["id"], by, datetime.now(timezone.utc).isoformat(), item_id),
            )
            connection.commit()
        finally:
            self._close(connection)
        return case
