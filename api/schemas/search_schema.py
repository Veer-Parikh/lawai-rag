"""
api/schemas/search_schema.py — Pydantic models for GET /search endpoint.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    text_preview: str
    doc_type: str
    act_name: str | None = None
    case_name: str | None = None
    citation: str | None = None
    section: str | None = None
    source_file: str
    namespace: str | None = None


class SearchResponse(BaseModel):
    query: str
    domain: str
    results: list[SearchResult]
    total_results: int
    latency_ms: int
