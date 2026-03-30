"""
retrieval/context_builder.py — Assemble retrieved chunks into the structured XML context string.

Also handles cross-reference auto-append:
if a statute chunk has cross_references metadata, fetch the referenced chunks from Pinecone
and append them with a [CROSS-REFERENCE] label.
"""
from __future__ import annotations

from typing import Any

import config
from utils.logger import get_logger

logger = get_logger(__name__)


def _fetch_cross_ref_chunk(ref: str, namespace: str) -> dict | None:
    """
    Attempt to fetch a specific cross-referenced section from Pinecone.
    e.g. ref="BNS Section 103" → query namespace for that section.
    """
    import re
    from pinecone import Pinecone

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)

    # Extract section number from ref string e.g. "BNS Section 103" → "103"
    sec_match = re.search(r"Section\s+(\d+[\w\.\-]*)", ref, re.IGNORECASE)
    if not sec_match:
        return None

    section_number = sec_match.group(1)

    try:
        result = index.query(
            vector=[0.0] * config.EMBEDDING_DIMENSION,  # dummy vector
            top_k=1,
            namespace=namespace,
            include_metadata=True,
            filter={"section_number": {"$eq": section_number}},
        )
        if result.matches:
            m = result.matches[0]
            entry = {"chunk_id": m.id, "score": 0.0, "is_cross_ref": True}
            if m.metadata:
                entry.update(m.metadata)
            return entry
    except Exception as exc:
        logger.warning(f"Cross-ref fetch failed for '{ref}': {exc}")

    return None


def _format_source_tag(chunk: dict) -> str:
    """Build the [SOURCE: ...] tag for a chunk."""
    parts = []
    if chunk.get("act_name") and chunk.get("section_number"):
        parts.append(f"{chunk['act_name']}, Section {chunk['section_number']}")
    elif chunk.get("act_name"):
        parts.append(chunk["act_name"])

    if chunk.get("case_name"):
        parts.append(chunk["case_name"])
    if chunk.get("citation"):
        parts.append(chunk["citation"])

    doc_type = chunk.get("doc_type", "")
    domain = chunk.get("domain", "")
    if doc_type:
        parts.append(f"{domain} | {doc_type}")

    return " | ".join(parts) if parts else "Unknown Source"


def build_context(
    chunks: list[dict[str, Any]],
    fetch_cross_refs: bool = True,
) -> tuple[str, list[dict]]:
    """
    Build the XML-tagged legal context string for the prompt.

    Args:
        chunks: Reranked chunk list from retrieval.
        fetch_cross_refs: Whether to auto-append cross-reference chunks.

    Returns:
        Tuple of (context_string, all_chunks_used_including_cross_refs)
    """
    all_chunks: list[dict] = list(chunks)
    cross_ref_appended = 0

    # Auto-append cross-reference chunks (capped)
    if fetch_cross_refs:
        existing_chunk_ids = {c["chunk_id"] for c in chunks}
        for chunk in chunks:
            refs = chunk.get("cross_references", [])
            if not refs:
                continue
            for ref in refs:
                if cross_ref_appended >= config.RETRIEVAL_MAX_CROSS_REFS:
                    break
                # Determine which namespace to look in
                if "BNS" in ref or "BNSS" in ref:
                    ns = config.NAMESPACE_STATUTES_CRIMINAL
                elif "IPC" in ref:
                    ns = config.NAMESPACE_STATUTES_CRIMINAL
                else:
                    ns = config.NAMESPACE_STATUTES_CRIMINAL

                xref_chunk = _fetch_cross_ref_chunk(ref, ns)
                if xref_chunk and xref_chunk["chunk_id"] not in existing_chunk_ids:
                    all_chunks.append(xref_chunk)
                    existing_chunk_ids.add(xref_chunk["chunk_id"])
                    cross_ref_appended += 1

    if cross_ref_appended:
        logger.info(f"Appended {cross_ref_appended} cross-reference chunk(s)")

    # Build XML-tagged context string
    lines = ["[LEGAL CONTEXT]"]
    for i, chunk in enumerate(all_chunks):
        source_tag = _format_source_tag(chunk)
        is_xref = chunk.get("is_cross_ref", False)
        label = f"Chunk {i + 1}" + (" [CROSS-REFERENCE]" if is_xref else "")
        text = chunk.get("text", "")

        lines.append(f"<{label}>")
        lines.append(f"[SOURCE: {source_tag}]")
        lines.append(text)
        lines.append(f"</{label}>")

    lines.append("[/LEGAL CONTEXT]")
    context_string = "\n".join(lines)

    return context_string, all_chunks
