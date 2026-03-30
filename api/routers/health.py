"""
api/routers/health.py — GET /health: liveness and readiness check.
"""
from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import config
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health")
async def health() -> JSONResponse:
    t_start = time.time()
    checks: dict[str, str] = {}

    # Check Pinecone connectivity
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        indexes = [idx.name for idx in pc.list_indexes()]
        index_ready = config.PINECONE_INDEX_NAME in indexes
        checks["pinecone"] = "ok" if index_ready else f"index '{config.PINECONE_INDEX_NAME}' not found"
    except Exception as exc:
        checks["pinecone"] = f"error: {exc}"

    # Check Gemini connectivity
    try:
        import google.generativeai as genai
        genai.configure(api_key=config.GEMINI_API_KEY)
        # List models (lightweight check)
        genai.list_models()
        checks["gemini"] = "ok"
    except Exception as exc:
        checks["gemini"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503

    return JSONResponse(
        content={
            "status": "healthy" if all_ok else "degraded",
            "checks": checks,
            "latency_ms": int((time.time() - t_start) * 1000),
        },
        status_code=status_code,
    )
