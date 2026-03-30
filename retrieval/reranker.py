"""
retrieval/reranker.py — Cross-encoder reranking using BAAI/bge-reranker-base.

Takes top-k chunks from vector retrieval and reranks them
for semantic relevance to the actual query text.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import math

import config
from utils.logger import get_logger


def _sigmoid(x: float) -> float:
    """Normalize raw logit to 0.0–1.0 probability."""
    return 1.0 / (1.0 + math.exp(-x))

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder
    logger.info(f"Loading reranker model: {config.RERANKER_MODEL_NAME}")
    model = CrossEncoder(config.RERANKER_MODEL_NAME)
    return model


def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int = config.RETRIEVAL_TOP_K_FINAL,
) -> list[dict[str, Any]]:
    """
    Rerank retrieved chunks using BAAI/bge-reranker-base cross-encoder.

    Args:
        query: The user query string.
        chunks: List of chunk dicts (must contain 'text' key).
        top_n: Number of top chunks to return after reranking.

    Returns:
        Top-n chunks sorted by reranker score (descending), with 'score' updated.
    """
    if not chunks:
        return []

    if len(chunks) == 1:
        return chunks

    model = _get_reranker()

    # Build (query, passage) pairs
    pairs = [(query, chunk.get("text", "")) for chunk in chunks]

    try:
        raw_scores = model.predict(pairs)
        # Normalize raw logits → [0.0, 1.0] so they align with
        # the 0.65 / 0.80 thresholds in retrieval_quality_assessor.
        for chunk, score in zip(chunks, raw_scores):
            chunk["score"] = _sigmoid(float(score))

        reranked = sorted(chunks, key=lambda x: x["score"], reverse=True)
        result = reranked[:top_n]
        logger.info(
            f"Reranked {len(chunks)} → top-{len(result)} "
            f"(top score: {result[0]['score']:.3f})"
        )
        return result
    except Exception as exc:
        logger.warning(f"Reranking failed: {exc} — returning original order")
        return chunks[:top_n]