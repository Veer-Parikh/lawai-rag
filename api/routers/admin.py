"""
api/routers/admin.py — Administrator endpoints: ingestion, corpus inventory, analytics.

Protected by an admin bearer token (POST /auth/admin) or the legacy
X-Admin-Key header. Ingestion runs in a background thread to avoid blocking
the HTTP response.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

import config
from api.auth import Principal, require_admin
from api.schemas.admin_schema import (
    AnalyticsResponse,
    CorpusResponse,
    IngestRequest,
    IngestStatus,
    UploadResponse,
)
from ingestion.change_watch import ChangeLog
from ingestion.corpus import UPLOAD_DESTINATIONS, resolve_inside, scan_corpus, store_upload
from ingestion.ledger import IngestionLedger
from observability.analytics import read_jsonl, summarize
from utils.logger import get_logger

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])
logger = get_logger(__name__)

# In-memory job tracker (completed job results also land in the ingestion ledger)
_jobs: dict[str, IngestStatus] = {}
_jobs_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ledger() -> IngestionLedger:
    return IngestionLedger(config.INGESTION_LEDGER_PATH)


def _run_ingestion_background(job_id: str, request: IngestRequest, target: Path) -> None:
    from ingestion.run_ingestion import run

    job = _jobs[job_id]
    job.status = "running"

    def on_progress(done: int, total: int, current: str) -> None:
        job.progress_done = done
        job.progress_total = total
        job.current_file = current

    try:
        stats = run(
            path=str(target),
            doc_type=request.doc_type,
            dry_run=request.dry_run,
            skip_ingested=request.skip_ingested,
            job_id=job_id,
            on_progress=on_progress,
        )
        job.status = "completed"
        job.files_found = stats.get("files_found", 0)
        job.files_processed = stats.get("files_processed", 0)
        job.files_skipped = stats.get("files_skipped", 0)
        job.total_chunks_created = stats.get("total_chunks_created", 0)
        job.total_chunks_uploaded = stats.get("total_chunks_uploaded", 0)
        job.errors = stats.get("errors", [])
    except Exception as exc:
        logger.error(f"Ingestion job {job_id} failed: {exc}")
        job.status = "failed"
        job.errors = [str(exc)]
    finally:
        job.finished_at = _now()
        job.current_file = ""


@router.post("/ingest", response_model=IngestStatus)
async def trigger_ingest(
    request: IngestRequest,
    principal: Principal = Depends(require_admin),
) -> IngestStatus:
    try:
        target = resolve_inside(config.DATA_DIR, request.target_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Nothing to ingest at {request.target_path}")

    with _jobs_lock:
        if any(job.status in ("started", "running") for job in _jobs.values()):
            raise HTTPException(status_code=409, detail="An ingestion job is already running")
        job_id = str(uuid.uuid4())
        job = IngestStatus(
            job_id=job_id,
            status="started",
            target_path=request.target_path,
            dry_run=request.dry_run,
            requested_by=principal.name,
            started_at=_now(),
        )
        _jobs[job_id] = job

    thread = threading.Thread(
        target=_run_ingestion_background,
        args=(job_id, request, target),
        daemon=True,
    )
    thread.start()
    logger.info(f"Ingestion job {job_id} started for path: {request.target_path}")
    return job


@router.get("/ingest/status/{job_id}", response_model=IngestStatus)
async def ingest_status(job_id: str) -> IngestStatus:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _jobs[job_id]


@router.get("/ingest/jobs", response_model=list[IngestStatus])
async def ingest_jobs() -> list[IngestStatus]:
    return sorted(_jobs.values(), key=lambda job: job.started_at or "", reverse=True)


@router.get("/corpus", response_model=CorpusResponse)
async def corpus() -> CorpusResponse:
    ledger = _ledger()
    inventory = scan_corpus(config.DATA_DIR, ledger)
    return CorpusResponse(
        **inventory,
        history=[entry.to_dict() for entry in ledger.history(limit=40)],
    )


@router.post("/corpus/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    destination: str = Form(...),
    title: str | None = Form(None),
    source_url: str | None = Form(None),
    publisher: str | None = Form(None),
    principal: Principal = Depends(require_admin),
) -> UploadResponse:
    content = await file.read()
    try:
        stored = store_upload(
            config.DATA_DIR,
            destination=destination,
            filename=file.filename or "document.pdf",
            content=content,
            title=title,
            source_url=source_url or None,
            publisher=publisher,
            uploaded_by=principal.name,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    relative = stored.resolve().relative_to(Path(config.DATA_DIR).resolve()).as_posix()
    logger.info(f"Corpus upload stored at {relative} by {principal.name}")
    ChangeLog(config.CHANGE_LOG_PATH).record(
        source_id=f"upload:{relative}",
        title=(title or "").strip() or stored.stem.replace("_", " "),
        url=source_url or None,
        authority=UPLOAD_DESTINATIONS[destination]["authority"],
        outcome="added",
        detail=f"Added to {UPLOAD_DESTINATIONS[destination]['label']} by {principal.name}",
    )
    return UploadResponse(path=relative, size_bytes=len(content))


@router.get("/analytics", response_model=AnalyticsResponse)
async def analytics(days: int = Query(14, ge=1, le=90)) -> AnalyticsResponse:
    from api.routers.metrics import metrics

    summary = summarize(
        read_jsonl(config.EVAL_LOG_PATH),
        read_jsonl(config.FEEDBACK_LOG_PATH),
        days=days,
    )
    return AnalyticsResponse(**summary, http=metrics.snapshot())


@router.get("/index")
async def index_stats() -> dict:
    """Live vector counts per Pinecone namespace (best-effort)."""
    try:
        from ingestion.pinecone_uploader import _get_client

        stats = _get_client().Index(config.PINECONE_INDEX_NAME).describe_index_stats()
        namespaces = {
            name: getattr(info, "vector_count", None) or info.get("vector_count", 0)
            for name, info in (stats.get("namespaces") or {}).items()
        }
        return {
            "available": True,
            "index": config.PINECONE_INDEX_NAME,
            "dimension": stats.get("dimension"),
            "total_vectors": stats.get("total_vector_count", sum(namespaces.values())),
            "namespaces": namespaces,
        }
    except Exception as exc:
        logger.warning(f"Index stats unavailable: {exc}")
        return {"available": False, "index": config.PINECONE_INDEX_NAME, "error": str(exc)}
