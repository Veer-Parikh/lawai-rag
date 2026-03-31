"""
api/main.py — FastAPI application factory.

Mounts all routers, registers middleware, and exposes startup/shutdown hooks.

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.middleware import register_middleware
from api.routers import admin, chat, health, search
from utils.logger import get_logger

logger = get_logger(__name__)


import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up heavy singletons in background so port binds immediately."""
    def warmup_models():
        logger.info("Bharat Law API starting up — warming up models...")
        try:
            from ingestion.embedder import embed_query
            embed_query("warm up")
            logger.info("Embedding model loaded ✓")
        except Exception as exc:
            logger.warning(f"Embedding model warm-up failed: {exc}")

        try:
            from retrieval.reranker import _get_reranker
            _get_reranker()
            logger.info("Reranker model loaded ✓")
        except Exception as exc:
            logger.warning(f"Reranker warm-up failed: {exc}")

    # Run the heavy downloading/loading in a background thread 
    # so uvicorn can instantly bind the port and start accepting traffic
    asyncio.get_running_loop().run_in_executor(None, warmup_models)

    logger.info("Bharat Law API ready.")
    yield
    logger.info("Bharat Law API shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Bharat Law",
        description="Production-grade RAG-based legal chatbot for Indian law.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    register_middleware(app)

    # Mount routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(chat.router, tags=["Chat"])
    app.include_router(search.router, tags=["Search"])
    app.include_router(admin.router, tags=["Admin"])

    return app


app = create_app()
