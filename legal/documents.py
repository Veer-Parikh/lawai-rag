"""Canonical legal document and provision contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class DocumentStatus(str, Enum):
    ENACTED = "enacted"
    NOTIFIED = "notified"
    IN_FORCE = "in_force"
    SUPERSEDED = "superseded"
    REPEALED = "repealed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProvisionVersion:
    provision_id: str
    section_number: str
    heading: str
    text: str
    effective_from: date
    effective_until: date | None = None
    status: DocumentStatus = DocumentStatus.IN_FORCE
    page_anchor: str | None = None

    def applies_on(self, on_date: date) -> bool:
        if self.status is not DocumentStatus.IN_FORCE:
            return False
        if on_date < self.effective_from:
            return False
        return self.effective_until is None or on_date < self.effective_until


@dataclass(frozen=True)
class AmendmentRelation:
    relation_id: str
    from_document_id: str
    to_document_id: str
    relation_type: str
    instrument_id: str
    affected_sections: tuple[str, ...] = ()


@dataclass
class LegalDocument:
    document_id: str
    title: str
    act_code: str
    source_url: str
    publisher: str
    status: DocumentStatus
    provisions: list[ProvisionVersion] = field(default_factory=list)
    publication_date: date | None = None
    assent_date: date | None = None
    retrieved_at: date | None = None
    effective_date: date | None = None
    document_hash: str | None = None
    language: str = "en"
    source_authority: str = "india_code"

    def provision_at(self, section_number: str, on_date: date) -> ProvisionVersion:
        matches = [
            provision
            for provision in self.provisions
            if provision.section_number == section_number and provision.applies_on(on_date)
        ]
        if not matches:
            raise KeyError(
                f"No in-force provision {self.act_code} section {section_number} "
                f"on {on_date.isoformat()}"
            )
        return max(matches, key=lambda provision: provision.effective_from)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("publication_date", "assent_date", "retrieved_at", "effective_date"):
            if payload[key] is not None:
                payload[key] = payload[key].isoformat()
        for provision in payload["provisions"]:
            provision["effective_from"] = provision["effective_from"].isoformat()
            if provision["effective_until"] is not None:
                provision["effective_until"] = provision["effective_until"].isoformat()
            provision["status"] = provision["status"].value
        payload["status"] = self.status.value
        return payload
