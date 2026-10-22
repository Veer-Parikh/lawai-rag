"""Durable SQLite storage for versioned legal documents."""
from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

from legal.documents import (
    DocumentStatus,
    LegalDocument,
    ProvisionVersion,
)


class DocumentStore:
    def __init__(self, path: str | Path = "data/catalog.sqlite3") -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_connection = (
            sqlite3.connect(":memory:") if str(self.path) == ":memory:" else None
        )
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return self._memory_connection or sqlite3.connect(str(self.path))

    def _close(self, connection: sqlite3.Connection) -> None:
        if self._memory_connection is None:
            connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS legal_documents (
                    document_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()
        finally:
            self._close(connection)

    def upsert(self, document: LegalDocument) -> None:
        payload = json.dumps(document.to_dict(), sort_keys=True)
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO legal_documents (document_id, payload_json)
                VALUES (?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (document.document_id, payload),
            )
            connection.commit()
        finally:
            self._close(connection)

    def get(self, document_id: str) -> LegalDocument:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT payload_json FROM legal_documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        finally:
            self._close(connection)
        if row is None:
            raise KeyError(f"document version not found: {document_id}")
        return self._from_dict(json.loads(row[0]))

    def list(self) -> list[LegalDocument]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT payload_json FROM legal_documents ORDER BY document_id"
            ).fetchall()
        finally:
            self._close(connection)
        return [self._from_dict(json.loads(row[0])) for row in rows]

    @staticmethod
    def _from_dict(payload: dict) -> LegalDocument:
        provisions = [
            ProvisionVersion(
                provision_id=item["provision_id"],
                section_number=item["section_number"],
                heading=item["heading"],
                text=item["text"],
                effective_from=date.fromisoformat(item["effective_from"]),
                effective_until=(
                    date.fromisoformat(item["effective_until"])
                    if item["effective_until"]
                    else None
                ),
                status=DocumentStatus(item["status"]),
                page_anchor=item.get("page_anchor"),
            )
            for item in payload["provisions"]
        ]
        date_fields = {
            key: date.fromisoformat(payload[key]) if payload.get(key) else None
            for key in (
                "publication_date",
                "assent_date",
                "retrieved_at",
                "effective_date",
            )
        }
        return LegalDocument(
            document_id=payload["document_id"],
            title=payload["title"],
            act_code=payload["act_code"],
            source_url=payload["source_url"],
            publisher=payload["publisher"],
            status=DocumentStatus(payload["status"]),
            provisions=provisions,
            document_hash=payload.get("document_hash"),
            language=payload.get("language", "en"),
            source_authority=payload.get("source_authority", "india_code"),
            **date_fields,
        )
