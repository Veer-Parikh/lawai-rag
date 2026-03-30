"""
api/schemas/admin_schema.py — Pydantic models for /admin endpoints.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    target_path: str = Field(..., description="Path to ingest, relative to project root")
    doc_type: Literal["statute", "judgment", "mapping", "all"] = "all"
    dry_run: bool = Field(False, description="Parse without uploading to Pinecone")


class IngestStatus(BaseModel):
    job_id: str
    status: Literal["started", "running", "completed", "failed"]
    files_found: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    total_chunks_created: int = 0
    total_chunks_uploaded: int = 0
    errors: list[str] = []
