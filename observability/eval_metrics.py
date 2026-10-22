"""Small dependency-free retrieval evaluation metrics."""
from __future__ import annotations


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    *,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_batches: list[list[str]],
    relevant_batches: list[set[str]],
) -> float:
    if len(retrieved_batches) != len(relevant_batches):
        raise ValueError("retrieved and relevant batches must have equal length")
    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_batches, relevant_batches):
        rank = next(
            (index for index, item in enumerate(retrieved, start=1) if item in relevant),
            None,
        )
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
