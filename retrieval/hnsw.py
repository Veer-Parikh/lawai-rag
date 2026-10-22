"""Optional HNSW dense-retrieval adapter.

The adapter keeps the index provider out of request handlers. Install
``hnswlib`` in deployments that use a local HNSW index.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class HnswRetriever:
    def __init__(self, index_path: str | Path, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        try:
            import hnswlib
        except ImportError as exc:
            raise RuntimeError(
                "HNSW retrieval requires the optional hnswlib dependency"
            ) from exc
        self._index = hnswlib.Index(space="cosine", dim=dimension)
        self._index.load_index(str(index_path))

    def search(self, vector: list[float], *, top_n: int = 10) -> list[dict[str, Any]]:
        if top_n <= 0:
            return []
        labels, distances = self._index.knn_query([vector], k=top_n)
        return [
            {
                "chunk_id": str(label),
                "score": 1.0 - float(distance),
                "source": "hnsw",
            }
            for label, distance in zip(labels[0], distances[0])
        ]
