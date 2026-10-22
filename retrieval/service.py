"""Provider-neutral legal retrieval orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from retrieval.hybrid import BM25Retriever, reciprocal_rank_fusion


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    as_of: date | None = None
    jurisdiction: str = "IN"
    top_n: int = 10

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("query text is required")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")


class HybridRetrievalService:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        *,
        dense_search: Callable[[str], list[dict[str, Any]]] | None = None,
    ) -> None:
        self._documents = {doc["chunk_id"]: doc for doc in documents}
        self._bm25 = BM25Retriever(documents)
        self._dense_search = dense_search or (lambda _query: [])

    def search(self, query: RetrievalQuery) -> list[dict[str, Any]]:
        allowed = {
            chunk_id: document
            for chunk_id, document in self._documents.items()
            if self._is_allowed(document, query)
        }
        lexical = [
            result
            for result in self._bm25.search(query.text, top_n=max(query.top_n * 4, 20))
            if result["chunk_id"] in allowed
        ]
        dense = [
            result
            for result in self._dense_search(query.text)
            if result.get("chunk_id") in allowed
        ]
        fused = reciprocal_rank_fusion([lexical, dense], top_n=query.top_n)
        return [{**allowed[item["chunk_id"]], **item} for item in fused]

    @staticmethod
    def _is_allowed(document: dict[str, Any], query: RetrievalQuery) -> bool:
        if document.get("jurisdiction", "IN") != query.jurisdiction:
            return False
        if document.get("legal_status") in {"superseded", "repealed", "unknown"}:
            return False
        if query.as_of is None:
            return True
        effective_date = document.get("effective_date")
        if not effective_date:
            return False
        if date.fromisoformat(effective_date) > query.as_of:
            return False
        effective_until = document.get("effective_until")
        return not effective_until or query.as_of < date.fromisoformat(effective_until)


def retrieve_hybrid_candidates(
    text: str,
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 10,
    as_of: date | None = None,
    jurisdiction: str = "IN",
) -> list[dict[str, Any]]:
    """Fuse provider-ranked candidates with BM25 and apply legal filters."""
    service = HybridRetrievalService(
        candidates,
        dense_search=lambda _query: candidates,
    )
    fused = service.search(
        RetrievalQuery(
            text=text,
            as_of=as_of,
            jurisdiction=jurisdiction,
            top_n=top_n,
        )
    )
    provider_scores = {
        candidate["chunk_id"]: candidate.get("score", 0.0)
        for candidate in candidates
    }
    return [
        {
            **item,
            "rrf_score": item.get("score", 0.0),
            "score": provider_scores.get(item["chunk_id"], item.get("score", 0.0)),
        }
        for item in fused
    ]


def filter_legal_candidates(
    candidates: list[dict[str, Any]],
    *,
    as_of: date | None = None,
    jurisdiction: str = "IN",
) -> list[dict[str, Any]]:
    """Apply legal status and temporal filters to provider results."""
    return [
        candidate
        for candidate in candidates
        if HybridRetrievalService._is_allowed(candidate, RetrievalQuery(
            text="provider-filter",
            as_of=as_of,
            jurisdiction=jurisdiction,
        ))
    ]
