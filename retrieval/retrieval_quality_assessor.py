"""
retrieval/retrieval_quality_assessor.py — 3-tier confidence scoring gate before LLM.

Evaluates the top chunk scores AFTER reranking and decides:
  HIGH   (≥0.80): call LLM normally
  MEDIUM (0.65–0.79): call LLM with conservative prompt injection
  LOW    (<0.65): block LLM call entirely, return structured fallback
  NARROW: all top chunks from same source file → diversity re-fetch trigger
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceTier(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class AssessmentResult:
    tier: ConfidenceTier
    top_score: float
    is_narrow: bool          # True if all chunks from same source file
    needs_diversity: bool    # True if narrow context was detected


def assess(chunks: list[dict[str, Any]]) -> AssessmentResult:
    """
    Assess the quality of a retrieved chunk set.

    Args:
        chunks: Reranked list of chunk dicts, expected to have 'score' and 'source_file'.

    Returns:
        AssessmentResult describing confidence tier and narrow-context flag.
    """
    if not chunks:
        return AssessmentResult(
            tier=ConfidenceTier.LOW,
            top_score=0.0,
            is_narrow=False,
            needs_diversity=False,
        )

    top_score = float(chunks[0].get("score", 0.0))

    # Determine tier
    if top_score >= 0.72:
        tier = ConfidenceTier.HIGH
    elif top_score >= 0.50:
        tier = ConfidenceTier.MEDIUM
    else:
        tier = ConfidenceTier.LOW

    # Check for narrow context: all chunks from same source file
    source_files = {c.get("source_file", "") for c in chunks}
    is_narrow = len(source_files) == 1 and len(chunks) > 1

    logger.info(
        f"Retrieval quality: tier={tier.value}, top_score={top_score:.3f}, "
        f"narrow_context={is_narrow}, sources={len(source_files)}"
    )

    return AssessmentResult(
        tier=tier,
        top_score=top_score,
        is_narrow=is_narrow,
        needs_diversity=is_narrow,
    )


def build_low_confidence_response() -> dict:
    """
    Structured fallback response for LOW confidence retrievals.
    Never calls the LLM — returns immediately to save API cost.
    """
    return {
        "answer": (
            "I was unable to find sufficiently relevant legal provisions or case law "
            "in the database to answer your question accurately. "
            "Please try rephrasing your question with more specific legal terms, "
            "relevant Act names, or section numbers."
        ),
        "relevant_sections": [],
        "legal_explanation": "",
        "disclaimer": (
            "This response is for informational purposes only and does not constitute "
            "legal advice. Consult a qualified lawyer for your specific situation."
        ),
        "citation_warnings": [],
        "confidence_tier": ConfidenceTier.LOW.value,
        "sources": [],
    }
