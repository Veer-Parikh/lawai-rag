"""
api/routers/search.py — GET /search: raw retrieval endpoint (no LLM generation).

Returns ranked chunks from Pinecone for debugging retrieval quality
or building a "browse legal documents" frontend feature.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Query

from api.schemas.search_schema import SearchResponse, SearchResult
from ingestion.embedder import embed_query
from retrieval.domain_router import classify_query
from retrieval.pinecone_retriever import retrieve
from retrieval.reranker import rerank
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=2, description="Search query"),
    domain: str | None = Query(None, description="Domain override"),
    top_k: int = Query(10, ge=1, le=50),
    rerank_results: bool = Query(True, description="Apply cross-encoder reranking"),
) -> SearchResponse:
    t_start = time.time()

    # Domain routing
    if domain:
        from retrieval.domain_router import DomainResult
        domain_result = DomainResult(domains=[domain], stage="override")
    else:
        domain_result = classify_query(q)

    # Embed
    query_vector = embed_query(q)

    # Retrieve
    raw_chunks = retrieve(
        query_vector=query_vector,
        namespaces=domain_result.namespaces,
        top_k=top_k * 2,  # over-fetch for reranking
    )

    # Optional reranking
    if rerank_results and raw_chunks:
        results = rerank(q, raw_chunks, top_n=top_k)
    else:
        results = raw_chunks[:top_k]

    latency_ms = int((time.time() - t_start) * 1000)

    search_results = [
        SearchResult(
            chunk_id=c.get("chunk_id", ""),
            score=float(c.get("score", 0.0)),
            text_preview=(c.get("text", ""))[:300],
            doc_type=c.get("doc_type", ""),
            act_name=c.get("act_name"),
            case_name=c.get("case_name"),
            citation=c.get("citation"),
            section=c.get("section_number"),
            source_file=c.get("source_file", ""),
            namespace=c.get("namespace"),
        )
        for c in results
    ]

    return SearchResponse(
        query=q,
        domain=",".join(domain_result.domains),
        results=search_results,
        total_results=len(search_results),
        latency_ms=latency_ms,
    )
