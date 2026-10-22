"""
api/schemas/chat_schema.py — Pydantic models for /chat endpoint.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="User's legal question")
    domain: str | None = Field(None, description="Optional domain override")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks for LLM context")
    session_id: str | None = Field(None, description="Session ID for multi-turn query rewriting")
    as_of: date | None = Field(None, description="Historical legal status date")


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str | None = None
    document_hash: str | None = None
    source_file: str
    source_url: str | None = None
    publisher: str | None = None
    source_authority: str | None = None
    retrieved_at: str | None = None
    language: str | None = None
    extraction_status: str | None = None
    conflict_state: str | None = None
    doc_type: str
    act_name: str | None = None
    case_name: str | None = None
    citation: str | None = None
    section: str | None = None
    publication_date: str | None = None
    effective_date: str | None = None
    effective_until: str | None = None
    legal_status: str | None = None
    page_anchor: str | None = None
    score: float


class CitationReference(BaseModel):
    text: str
    verified: bool
    source_chunk_ids: list[str]
    source_urls: list[str]
    page_anchors: list[str]


class ChatResponse(BaseModel):
    answer: str
    relevant_sections: list[str]
    legal_explanation: str
    disclaimer: str
    citation_warnings: list[str]
    citations: list[CitationReference]
    confidence_tier: str
    sources: list[SourceChunk]
    domain_detected: str
    domain_router_stage: str
    latency_ms: int
    context_chunks_used: int
