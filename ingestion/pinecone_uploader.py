"""
ingestion/pinecone_uploader.py — Batch upsert vectors and metadata to Pinecone Serverless.
"""
from __future__ import annotations

import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec

import config
from utils.logger import get_logger
from utils.exceptions import PineconeError

logger = get_logger(__name__)

_pc_client: Pinecone | None = None


def _get_client() -> Pinecone:
    global _pc_client
    if _pc_client is None:
        _pc_client = Pinecone(api_key=config.PINECONE_API_KEY)
    return _pc_client


def ensure_index_exists() -> None:
    """Create the Pinecone Serverless index if it does not already exist."""
    pc = _get_client()
    existing = [idx.name for idx in pc.list_indexes()]
    if config.PINECONE_INDEX_NAME not in existing:
        logger.info(f"Creating Pinecone index '{config.PINECONE_INDEX_NAME}'")
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=config.PINECONE_CLOUD,
                region=config.PINECONE_REGION,
            ),
        )
        # Wait for index to be ready
        for _ in range(20):
            desc = pc.describe_index(config.PINECONE_INDEX_NAME)
            if desc.status.get("ready", False):
                break
            time.sleep(3)
        logger.info("Pinecone index ready")
    else:
        logger.info(f"Pinecone index '{config.PINECONE_INDEX_NAME}' already exists")


def upsert_chunks(
    chunks_with_metadata: list[dict[str, Any]],
    namespace: str,
    vectors: list[list[float]],
) -> int:
    """
    Upsert a list of chunks to Pinecone.

    Args:
        chunks_with_metadata: List of metadata dicts (each containing 'chunk_id' and 'text').
        namespace: Target Pinecone namespace.
        vectors: Corresponding embedding vectors (same length as chunks_with_metadata).

    Returns:
        Number of vectors upserted.
    """
    if len(chunks_with_metadata) != len(vectors):
        raise PineconeError(
            f"Mismatch: {len(chunks_with_metadata)} chunks vs {len(vectors)} vectors"
        )

    pc = _get_client()
    index = pc.Index(config.PINECONE_INDEX_NAME)
    batch_size = config.PINECONE_UPSERT_BATCH_SIZE
    total_upserted = 0

    for i in range(0, len(chunks_with_metadata), batch_size):
        meta_batch = chunks_with_metadata[i : i + batch_size]
        vec_batch = vectors[i : i + batch_size]

        # Build Pinecone upsert records
        upsert_records = []
        for meta, vec in zip(meta_batch, vec_batch):
            # Pinecone metadata: only store filterable scalar fields
            # Exclude large text fields; store text separately as 'text_preview'
            pinecone_meta = {}
            for k, v in meta.items():
                if k == "chunk_id" or v is None:
                    continue
                # Hard limit string metadata to 32,000 chars to avoid Pinecone 40KB limit
                if isinstance(v, str) and len(v) > 32000:
                    pinecone_meta[k] = v[:32000] + "... [TRUNCATED]"
                else:
                    pinecone_meta[k] = v

            upsert_records.append(
                {
                    "id": meta["chunk_id"],
                    "values": vec,
                    "metadata": pinecone_meta,
                }
            )

        try:
            index.upsert(vectors=upsert_records, namespace=namespace)
            total_upserted += len(upsert_records)
            logger.info(
                f"Upserted batch {i // batch_size + 1} "
                f"({len(upsert_records)} vectors) to namespace '{namespace}'"
            )
        except Exception as exc:
            logger.error(f"Pinecone upsert failed at batch {i // batch_size + 1}: {exc}")
            raise PineconeError(str(exc)) from exc

    return total_upserted
