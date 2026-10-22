"""Provider-neutral hybrid retrieval utilities."""
from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any


class BM25Retriever:
    """Small in-process BM25 index for exact legal terms and section numbers."""

    def __init__(self, documents: list[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._tokens = [self._tokenize(doc.get("text", "")) for doc in documents]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0
        self._document_frequency: defaultdict[str, int] = defaultdict(int)
        for tokens in self._tokens:
            for token in set(tokens):
                self._document_frequency[token] += 1

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str, *, top_n: int = 10) -> list[dict[str, Any]]:
        if top_n <= 0:
            return []
        query_terms = self._tokenize(query)
        document_count = len(self.documents)
        scored = []
        for index, document in enumerate(self.documents):
            term_counts = defaultdict(int)
            for token in self._tokens[index]:
                term_counts[token] += 1
            score = 0.0
            for term in query_terms:
                frequency = term_counts[term]
                if not frequency:
                    continue
                inverse_frequency = math.log(
                    1 + (document_count - self._document_frequency[term] + 0.5)
                    / (self._document_frequency[term] + 0.5)
                )
                length_factor = 1 - self.b + self.b * (
                    self._lengths[index] / self._average_length
                    if self._average_length
                    else 0
                )
                score += inverse_frequency * (
                    frequency * (self.k1 + 1) / (frequency + self.k1 * length_factor)
                )
            if score:
                result = dict(document)
                result["score"] = score
                result["source"] = "bm25"
                scored.append(result)
        scored.sort(key=lambda item: (-item["score"], item.get("chunk_id", "")))
        return scored[:top_n]


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
    *,
    k: int = 60,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Fuse ranked candidate lists while preserving candidate metadata."""
    if k <= 0:
        raise ValueError("k must be positive")
    if top_n <= 0:
        return []

    fused: dict[str, dict[str, Any]] = {}
    scores: defaultdict[str, float] = defaultdict(float)
    stages: defaultdict[str, list[str]] = defaultdict(list)

    for ranking in rankings:
        seen_in_stage: set[str] = set()
        for rank, candidate in enumerate(ranking, start=1):
            chunk_id = candidate.get("chunk_id")
            if not chunk_id or chunk_id in seen_in_stage:
                continue
            seen_in_stage.add(chunk_id)
            scores[chunk_id] += 1.0 / (k + rank)
            stage = candidate.get("source") or candidate.get("retrieval_stage")
            if stage and stage not in stages[chunk_id]:
                stages[chunk_id].append(stage)
            fused.setdefault(chunk_id, dict(candidate))

    result = []
    for chunk_id, candidate in fused.items():
        candidate["fusion_score"] = scores[chunk_id]
        candidate["retrieval_stages"] = stages[chunk_id]
        result.append(candidate)
    result.sort(key=lambda item: (-item["fusion_score"], item["chunk_id"]))
    return result[:top_n]
