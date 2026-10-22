"""
api/routers/chat.py — POST /chat: core RAG Q&A endpoint.

Full pipeline:
  query rewriting → domain routing → embedding → retrieval → reranking →
  confidence assessment → context building → Gemini generation → eval logging
"""
from __future__ import annotations

import hashlib
import time

from fastapi import APIRouter

from api.schemas.chat_schema import (
    ChatRequest,
    ChatResponse,
    CitationReference,
    SourceChunk,
)
from ingestion.embedder import embed_query
from llm.gemini_client import generate_answer
from llm.prompt_builder import build_prompt, get_system_prompt
from observability.eval_logger import log_request
from retrieval.context_builder import build_context
from retrieval.domain_router import classify_query
from retrieval.pinecone_retriever import retrieve
from retrieval.query_rewriter import add_to_session, rewrite_query
from retrieval.reranker import rerank
from retrieval.service import retrieve_hybrid_candidates
from retrieval.retrieval_quality_assessor import (
    ConfidenceTier,
    assess,
    build_low_confidence_response,
)
from utils.exceptions import GeminiError
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    t_start = time.time()

    query_hash = hashlib.sha256(request.query.encode()).hexdigest()
    session_id_hash = (
        hashlib.sha256(request.session_id.encode()).hexdigest()
        if request.session_id
        else ""
    )

    # ── Step 1: Query rewriting ────────────────────────────────────────────────
    query, was_rewritten = rewrite_query(request.query, request.session_id)

    # ── Step 2: Domain routing ─────────────────────────────────────────────────
    if request.domain:
        from retrieval.domain_router import DomainResult
        domain_result = DomainResult(domains=[request.domain], stage="override")
    else:
        domain_result = classify_query(query)

    # ── Step 3: Embed query ────────────────────────────────────────────────────
    query_vector = embed_query(query)

    # ── Step 4: Retrieve ───────────────────────────────────────────────────────
    raw_chunks = retrieve(
        query_vector=query_vector,
        namespaces=domain_result.namespaces,
    )
    raw_chunks = retrieve_hybrid_candidates(
        query,
        raw_chunks,
        top_n=max(request.top_k * 2, 10),
        as_of=request.as_of,
    )

    # ── Step 5: Rerank ─────────────────────────────────────────────────────────
    # Cap chunks sent to cross-encoder to heavily optimize local CPU latency
    reranked = rerank(query, raw_chunks[:10], top_n=request.top_k)

    # ── Step 6: Confidence assessment ─────────────────────────────────────────
    assessment = assess(reranked)

    # Do not let model memory fill an evidence gap in a legal workflow.
    if not reranked or assessment.tier is ConfidenceTier.LOW:
        latency_ms = int((time.time() - t_start) * 1000)
        log_request(
            query_hash=query_hash,
            session_id_hash=session_id_hash,
            domain_detected=",".join(domain_result.domains),
            domain_router_stage=domain_result.stage,
            confidence_tier=ConfidenceTier.LOW.value,
            top_chunk_score=assessment.top_score,
            chunks_used=0,
            reranker_used=True,
            cross_refs_appended=0,
            citation_warnings=0,
            query_rewritten=was_rewritten,
            latency_ms=latency_ms,
            gemini_tokens_used=0,
            had_llm_call=False,
        )
        fallback = build_low_confidence_response()
        return ChatResponse(
            domain_detected=",".join(domain_result.domains),
            domain_router_stage=domain_result.stage,
            latency_ms=latency_ms,
            context_chunks_used=0,
            **fallback,
        )

    # ── Step 6b: Diversity re-fetch for narrow context ────────────────────────
    if assessment.needs_diversity and raw_chunks:
        current_source = reranked[0].get("source_file", "")
        diversity_chunks = [c for c in raw_chunks if c.get("source_file") != current_source]
        reranked.extend(diversity_chunks[:3])
        reranked = rerank(query, reranked, top_n=request.top_k)

    # ── Step 7: Build context ──────────────────────────────────────────────────
    context_string, all_chunks = build_context(reranked, fetch_cross_refs=True)
    cross_refs_appended = len(all_chunks) - len(reranked)

    # ── Step 8: Prompt construction ────────────────────────────────────────────
    user_prompt = build_prompt(context_string, query, assessment.tier)
    system_prompt = get_system_prompt()

    # ── Step 9: LLM generation ─────────────────────────────────────────────────
    try:
        llm_result = generate_answer(system_prompt, user_prompt, all_chunks)
    except GeminiError as exc:
        logger.error(f"Gemini generation failed: {exc}")
        raise

    # ── Step 10: Build response ────────────────────────────────────────────────
    sources = [
        SourceChunk(
            chunk_id=c.get("chunk_id", ""),
            document_id=c.get("document_id"),
            document_hash=c.get("document_hash"),
            source_file=c.get("source_file", ""),
            source_url=c.get("source_url"),
            publisher=c.get("publisher"),
            source_authority=c.get("source_authority"),
            retrieved_at=c.get("retrieved_at"),
            language=c.get("language"),
            extraction_status=c.get("extraction_status"),
            conflict_state=c.get("conflict_state"),
            doc_type=c.get("doc_type", ""),
            act_name=c.get("act_name"),
            case_name=c.get("case_name"),
            citation=c.get("citation"),
            section=c.get("section_number"),
            publication_date=c.get("publication_date"),
            effective_date=c.get("effective_date"),
            effective_until=c.get("effective_until"),
            legal_status=c.get("legal_status"),
            page_anchor=c.get("page_anchor"),
            score=float(c.get("score", 0.0)),
        )
        for c in all_chunks
    ]
    citations = [
        CitationReference(**citation)
        for citation in llm_result["citations"]
    ]

    latency_ms = int((time.time() - t_start) * 1000)

    # ── Step 11: Store query in session ───────────────────────────────────────
    if request.session_id:
        add_to_session(request.session_id, request.query)

    # ── Step 12: Eval logging ──────────────────────────────────────────────────
    log_request(
        query_hash=query_hash,
        session_id_hash=session_id_hash,
        domain_detected=",".join(domain_result.domains),
        domain_router_stage=domain_result.stage,
        confidence_tier=assessment.tier.value,
        top_chunk_score=assessment.top_score,
        chunks_used=len(all_chunks),
        reranker_used=True,
        cross_refs_appended=cross_refs_appended,
        citation_warnings=len(llm_result["citation_warnings"]),
        query_rewritten=was_rewritten,
        latency_ms=latency_ms,
        gemini_tokens_used=llm_result["tokens_used"],
        had_llm_call=True,
    )

    return ChatResponse(
        answer=llm_result["answer"],
        relevant_sections=llm_result["relevant_sections"],
        legal_explanation=llm_result["legal_explanation"],
        disclaimer=llm_result["disclaimer"],
        citation_warnings=llm_result["citation_warnings"],
        citations=citations,
        confidence_tier=assessment.tier.value,
        sources=sources,
        domain_detected=",".join(domain_result.domains),
        domain_router_stage=domain_result.stage,
        latency_ms=latency_ms,
        context_chunks_used=len(all_chunks),
    )