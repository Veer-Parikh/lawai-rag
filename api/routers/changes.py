"""
api/routers/changes.py — Change feed for researchers, source watch for administrators.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

import config
from api.auth import Principal, require_admin
from api.jobs import jobs
from ingestion.change_watch import ChangeLog, check_sources, watched_sources
from ingestion.refresh import SourceSnapshotStore
from utils.logger import get_logger

router = APIRouter()
admin_router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])
logger = get_logger(__name__)


class CheckRequest(BaseModel):
    source_ids: list[str] | None = None


def change_log() -> ChangeLog:
    return ChangeLog(config.CHANGE_LOG_PATH)


def _manifest_path() -> Path:
    return Path(config.DATA_DIR) / "source_manifest.json"


def run_check(source_ids: list[str] | None = None, progress=None) -> dict[str, int]:
    sources, _ = watched_sources(_manifest_path())
    if source_ids:
        sources = [source for source in sources if source.source_id in source_ids]
    return check_sources(
        sources,
        snapshots=SourceSnapshotStore(config.SOURCE_SNAPSHOT_PATH),
        log=change_log(),
        on_progress=progress,
    )


@router.get("/changes")
async def change_feed(limit: int = Query(50, ge=1, le=200)) -> dict:
    """New, changed and newly added official documents, newest first."""
    return {"events": [event.to_dict() for event in change_log().feed(limit)]}


@admin_router.get("/sources")
async def list_sources() -> dict:
    sources, skipped = watched_sources(_manifest_path())
    latest = change_log().latest_by_source()
    job = jobs.latest("source_check")
    return {
        "sources": [
            {
                "source_id": source.source_id,
                "title": source.title,
                "url": source.url,
                "authority": source.authority,
                "document_type": source.document_type,
                "last_check": latest.get(source.source_id, {}).get("last_check"),
                "last_change": latest.get(source.source_id, {}).get("last_change"),
            }
            for source in sources
        ],
        "not_watched": [
            {"source_id": row["source_id"], "title": row["title"], "authority": row.get("authority"), "reason": row["reason"]}
            for row in skipped
        ],
        "interval_hours": config.CHANGE_WATCH_INTERVAL_HOURS,
        "job": job.to_dict() if job else None,
    }


@admin_router.post("/sources/check")
async def start_source_check(body: CheckRequest, principal: Principal = Depends(require_admin)) -> dict:
    try:
        job = jobs.start("source_check", lambda progress: run_check(body.source_ids, progress), requested_by=principal.name)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return job.to_dict()


def start_periodic_watch() -> threading.Thread | None:
    """Check sources every CHANGE_WATCH_INTERVAL_HOURS (0 disables)."""
    hours = config.CHANGE_WATCH_INTERVAL_HOURS
    if hours <= 0:
        return None

    def loop() -> None:
        while True:
            time.sleep(hours * 3600)
            try:
                jobs.start("source_check", lambda progress: run_check(None, progress), requested_by="Scheduled check")
            except RuntimeError:
                logger.info("Scheduled source check skipped: a check is already running")

    thread = threading.Thread(target=loop, daemon=True, name="source-watch")
    thread.start()
    logger.info(f"Source watch scheduled every {hours} hours")
    return thread
