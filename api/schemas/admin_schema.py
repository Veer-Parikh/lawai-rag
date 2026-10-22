"""
api/schemas/admin_schema.py — Pydantic models for /admin endpoints.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    target_path: str = Field(..., description="File or folder to ingest, relative to the data directory")
    doc_type: Literal["statute", "judgment", "mapping", "all"] = "all"
    dry_run: bool = Field(False, description="Parse without uploading to Pinecone")
    skip_ingested: bool = Field(False, description="Skip files already ingested successfully")


class IngestStatus(BaseModel):
    job_id: str
    status: Literal["started", "running", "completed", "failed"]
    target_path: str = ""
    dry_run: bool = False
    requested_by: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    progress_done: int = 0
    progress_total: int = 0
    current_file: str = ""
    files_found: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    total_chunks_created: int = 0
    total_chunks_uploaded: int = 0
    errors: list[str] = []


class UploadResponse(BaseModel):
    path: str
    size_bytes: int
    sha256: str | None = None


class AnalyticsResponse(BaseModel):
    window_days: int
    window_start: str
    window_end: str
    totals: dict[str, Any]
    confidence: dict[str, int]
    domains: dict[str, int]
    router_stages: dict[str, int]
    daily: list[dict[str, Any]]
    recent: list[dict[str, Any]]
    feedback: dict[str, Any]
    http: dict[str, Any]


class CorpusResponse(BaseModel):
    summary: dict[str, Any]
    files: list[dict[str, Any]]
    unavailable: list[dict[str, Any]]
    destinations: list[dict[str, Any]]
    history: list[dict[str, Any]]
