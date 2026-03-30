"""
api/routers/admin.py — POST /admin/ingest and GET /admin/ingest/status.

Protected by ADMIN_API_KEY header.
Ingestion runs in a background thread to avoid blocking the HTTP response.
"""
from __future__ import annotations

import threading
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Header

import config
from api.schemas.admin_schema import IngestRequest, IngestStatus
from utils.logger import get_logger

router = APIRouter(prefix="/admin")
logger = get_logger(__name__)

# In-memory job tracker
_jobs: dict[str, IngestStatus] = {}


def _require_admin_key(x_admin_key: str = Header(..., alias="X-Admin-Key")) -> None:
    if not config.ADMIN_API_KEY or x_admin_key != config.ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")


def _run_ingestion_background(job_id: str, request: IngestRequest) -> None:
    from ingestion.run_ingestion import run
    _jobs[job_id].status = "running"
    try:
        stats = run(
            path=request.target_path,
            doc_type=request.doc_type,
            dry_run=request.dry_run,
        )
        _jobs[job_id].status = "completed"
        _jobs[job_id].files_found = stats.get("files_found", 0)
        _jobs[job_id].files_processed = stats.get("files_processed", 0)
        _jobs[job_id].files_skipped = stats.get("files_skipped", 0)
        _jobs[job_id].total_chunks_created = stats.get("total_chunks_created", 0)
        _jobs[job_id].total_chunks_uploaded = stats.get("total_chunks_uploaded", 0)
        _jobs[job_id].errors = stats.get("errors", [])
    except Exception as exc:
        logger.error(f"Ingestion job {job_id} failed: {exc}")
        _jobs[job_id].status = "failed"
        _jobs[job_id].errors = [str(exc)]


@router.post("/ingest", response_model=IngestStatus, dependencies=[Depends(_require_admin_key)])
async def trigger_ingest(request: IngestRequest) -> IngestStatus:
    job_id = str(uuid.uuid4())
    job = IngestStatus(job_id=job_id, status="started")
    _jobs[job_id] = job

    thread = threading.Thread(
        target=_run_ingestion_background,
        args=(job_id, request),
        daemon=True,
    )
    thread.start()
    logger.info(f"Ingestion job {job_id} started for path: {request.target_path}")
    return job


@router.get("/ingest/status/{job_id}", response_model=IngestStatus, dependencies=[Depends(_require_admin_key)])
async def ingest_status(job_id: str) -> IngestStatus:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _jobs[job_id]


