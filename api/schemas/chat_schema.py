"""
api/schemas/chat_schema.py — Pydantic models for /chat endpoint.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="User's legal question")
    domain: str | None = Field(None, description="Optional domain override")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks for LLM context")
    session_id: str | None = Field(None, description="Session ID for multi-turn query rewriting")


class SourceChunk(BaseModel):
    chunk_id: str
    source_file: str
    doc_type: str
    act_name: str | None = None
    case_name: str | None = None
    citation: str | None = None
    section: str | None = None
    score: float


class ChatResponse(BaseModel):
    answer: str
    relevant_sections: list[str]
    legal_explanation: str
    disclaimer: str
    citation_warnings: list[str]
    confidence_tier: str
    sources: list[SourceChunk]
    domain_detected: str
    domain_router_stage: str
    latency_ms: int
    context_chunks_used: int
