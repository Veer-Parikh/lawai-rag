"""
api/routers/feedback.py — POST /feedback: research-quality feedback on an answer.

The raw query is hashed exactly as /chat hashes it. Problem reports also enter
the review queue; the question and answer text are kept only when the
researcher ticks "share my question with reviewers".
"""
from __future__ import annotations

import hashlib
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

import config
from api.auth import Principal, require_principal
from observability.feedback import record_feedback
from observability.review_queue import PROBLEM_KINDS, ReviewQueue

router = APIRouter()


class SourceRef(BaseModel):
    title: str = Field(..., max_length=300)
    section: str | None = Field(None, max_length=40)
    status: str | None = Field(None, max_length=40)


class FeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    kind: Literal["helpful", "wrong_source", "stale_status", "missing_authority", "poor_explanation"]
    comment: str | None = Field(None, max_length=500)
    share_question: bool = False
    answer: str | None = Field(None, max_length=12000)
    citations: list[str] = Field(default_factory=list, max_length=30)
    sources: list[SourceRef] = Field(default_factory=list, max_length=30)
    confidence_tier: str | None = Field(None, max_length=10)


class FeedbackResponse(BaseModel):
    recorded: bool
    kind: str
    queued_for_review: bool


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    body: FeedbackRequest,
    principal: Principal = Depends(require_principal),
) -> FeedbackResponse:
    query_hash = hashlib.sha256(body.query.encode()).hexdigest()
    record_feedback(kind=body.kind, query_hash=query_hash, role=principal.role, comment=body.comment)
    queued = body.kind in PROBLEM_KINDS
    if queued:
        ReviewQueue(config.REVIEW_QUEUE_PATH).add(
            kind=body.kind,
            query_hash=query_hash,
            question=body.query if body.share_question else None,
            answer=body.answer if body.share_question else None,
            citations=body.citations,
            sources=[source.model_dump() for source in body.sources],
            confidence_tier=body.confidence_tier,
            comment=(body.comment or "").strip() or None,
        )
    return FeedbackResponse(recorded=True, kind=body.kind, queued_for_review=queued)
