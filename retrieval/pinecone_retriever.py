"""
retrieval/pinecone_retriever.py — Execute filtered top-k vector searches across namespaces.

Queries multiple namespaces in parallel, merges results, and deduplicates by chunk_id.
"""
from __future__ import annotations

import concurrent.futures
from typing import Any

from pinecone import Pinecone

import config
from utils.logger import get_logger
from utils.exceptions import PineconeError

logger = get_logger(__name__)

_pc_client: Pinecone | None = None


def _get_index():
    global _pc_client
    if _pc_client is None:
        _pc_client = Pinecone(api_key=config.PINECONE_API_KEY)
    return _pc_client.Index(config.PINECONE_INDEX_NAME)


def _query_namespace(
    index,
    namespace: str,
    vector: list[float],
    top_k: int,
    metadata_filter: dict | None,
) -> list[dict[str, Any]]:
    """Query a single namespace and return match dicts."""
    try:
        results = index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            filter=metadata_filter or None,
        )
        matches = []
        for m in results.matches:
            entry = {"chunk_id": m.id, "score": m.score, "namespace": namespace}
            if m.metadata:
                entry.update(m.metadata)
            matches.append(entry)
        return matches
    except Exception as exc:
        logger.warning(f"Pinecone query failed for namespace '{namespace}': {exc}")
        return []


def retrieve(
    query_vector: list[float],
    namespaces: list[str],
    top_k: int = config.RETRIEVAL_TOP_K_FETCH,
    metadata_filter: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Query all target namespaces in parallel, merge, deduplicate, and sort by score.

    Args:
        query_vector: 768-dim query embedding.
        namespaces: List of Pinecone namespace strings to query.
        top_k: Fetch this many results per namespace before merging.
        metadata_filter: Additional Pinecone metadata filter dict.

    Returns:
        Deduplicated list of result dicts sorted by score descending.
    """
    # Build combined filter for judgments
    def _build_filter(ns: str) -> dict | None:
        filters = []
        if metadata_filter:
            filters.append(metadata_filter)
        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    index = _get_index()

    # Parallel queries
    all_matches: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(namespaces)) as executor:
        futures = {
            executor.submit(
                _query_namespace, index, ns, query_vector, top_k, _build_filter(ns)
            ): ns
            for ns in namespaces
        }
        for future in concurrent.futures.as_completed(futures):
            all_matches.extend(future.result())

    # Deduplicate by chunk_id, keeping highest score
    seen: dict[str, dict] = {}
    for match in all_matches:
        cid = match["chunk_id"]
        if cid not in seen or match["score"] > seen[cid]["score"]:
            seen[cid] = match

    sorted_matches = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    logger.info(
        f"Retrieved {len(sorted_matches)} unique chunks from {len(namespaces)} namespace(s)"
    )
    return sorted_matches
