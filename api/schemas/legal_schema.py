"""
api/schemas/legal_schema.py — Pydantic models for the crosswalk, provisions and timelines.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CrosswalkRowModel(BaseModel):
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


class CrosswalkResponse(BaseModel):
    note: str
    total: int
    verified: int
    rows: list[CrosswalkRowModel]


class CrosswalkVerifyRequest(BaseModel):
    verified: bool
    note: str | None = Field(None, max_length=500)


class ProvisionResponse(BaseModel):
    code: str
    section: str
    act_title: str
    heading: str | None
    text: str
    page_anchor: str | None
    status: str
    status_label: str
    source_url: str
    extracted_from: str


class TimelineEvent(BaseModel):
    date: str
    kind: str
    title: str
    detail: str
    code: str | None
    section: str | None
    instrument: str | None
    url: str | None


class TimelineResponse(BaseModel):
    code: str
    section: str
    act_title: str
    heading: str | None
    status: str
    status_label: str
    source_url: str
    text: str | None
    page_anchor: str | None
    events: list[TimelineEvent]
    related: list[CrosswalkRowModel]


class ActSummary(BaseModel):
    code: str
    title: str
    act_number: str
    status: str
    in_force_from: str
    repealed_on: str | None
    repealed_by: str | None
    successor: str | None
    predecessor: str | None
    commencement: str | None
    not_in_force: list[str]
    sections_in_catalog: int
    catalog_built_at: str | None


class CatalogBuildRequest(BaseModel):
    codes: list[str] | None = None
