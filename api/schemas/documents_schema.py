from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class ProvisionChange(BaseModel):
    section_number: str
    change_type: str
    before_text: str | None = None
    after_text: str | None = None


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    act_code: str
    publisher: str
    source_url: str
    status: str
    publication_date: date | None = None
    effective_date: date | None = None
    source_authority: str


class DocumentChangeResponse(BaseModel):
    from_document_id: str
    to_document_id: str
    changed_sections: list[str]
    added_sections: list[str]
    removed_sections: list[str]
