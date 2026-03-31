"""
ingestion/embedder.py — BGE-base-en embedding wrapper.

Loads BAAI/bge-base-en once at module import (lazy singleton).
Embeds text in configurable batches to avoid OOM on large corpora.
"""
from __future__ import annotations

from functools import lru_cache

# Heavy import moved inside _get_model to prevent blocking startup
# from sentence_transformers import SentenceTransformer

import config
from utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings using BGE-base-en.

    Processes in batches of EMBEDDING_BATCH_SIZE to avoid OOM.
    BGE models perform best with the query prefix for asymmetric retrieval,
    but for document embeddings we do NOT add a prefix.

    Returns:
        List of 768-dimensional float vectors.
    """
    model = _get_model()
    all_embeddings: list[list[float]] = []
    batch_size = config.EMBEDDING_BATCH_SIZE

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,   # Required for cosine similarity
            show_progress_bar=False,
        )
        all_embeddings.extend(embeddings.tolist())
        logger.info(f"Embedded batch {i // batch_size + 1} ({len(batch)} texts)")

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single user query with the BGE query instruction prefix.
    BGE asymmetric: query side uses a prefix for better retrieval.
    """
    model = _get_model()
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    embedding = model.encode([prefixed], normalize_embeddings=True)
    return embedding[0].tolist()
