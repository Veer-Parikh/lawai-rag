"""
ingestion/metadata_builder.py — Construct the full Pinecone metadata dict per chunk.

Takes raw chunk data from statute_chunker or judgment_chunker,
plus file-level context, and emits the final metadata dict.
"""
from __future__ import annotations

import re
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

# Known Acts and their associated domains
_ACT_DOMAIN_MAP: dict[str, str] = {
    "Indian Penal Code": "criminal",
    "Bharatiya Nyaya Sanhita": "criminal",
    "Bharatiya Nagarik Suraksha Sanhita": "criminal",
    "Indian Evidence Act": "criminal",
    "Bharatiya Sakshya Adhiniyam": "criminal",
    "Code of Criminal Procedure": "criminal",
    "Contract Act": "civil",
    "Specific Relief Act": "civil",
    "Transfer of Property Act": "civil",
    "Civil Procedure Code": "civil",
    "Consumer Protection Act": "economic",
    "Information Technology Act": "economic",
    "Hindu Marriage Act": "family",
    "Hindu Succession Act": "family",
    "Special Marriage Act": "family",
    "Protection of Women from Domestic Violence Act": "family",
    "Constitution of India": "constitutional",
}


def _infer_domain_from_path(source_path: str) -> str:
    """Infer domain from directory structure e.g. data/statutes/criminal/..."""
    parts = Path(source_path).parts
    for part in parts:
        low = part.lower()
        if low in ("criminal", "civil", "economic", "family", "constitutional"):
            return low
    return "misc"


def _infer_domain_from_act(act_name: str) -> str:
    for known_act, domain in _ACT_DOMAIN_MAP.items():
        if known_act.lower() in act_name.lower():
            return domain
    return "misc"


def _infer_year_from_path(source_path: str) -> int | None:
    """Extract year from path like data/supreme_court_judgments/2022/..."""
    match = re.search(r"20\d{2}", source_path)
    return int(match.group(0)) if match else None


def build_statute_metadata(
    chunk: dict,
    source_path: str,
    total_chunks: int,
    act_name: str,
    chapter: str = "",
    amendment_year: int | None = None,
    cross_ref_index: dict[str, list[str]] | None = None,
) -> dict:
    """Build metadata for a statute chunk."""
    domain = _infer_domain_from_act(act_name) or _infer_domain_from_path(source_path)
    namespace = f"statutes-{domain}"

    # Look up cross-references for this section
    cross_refs: list[str] = []
    if cross_ref_index and chunk.get("section_number"):
        act_ref_ipc = f"IPC Section {chunk['section_number']}"
        act_ref_bns = f"BNS Section {chunk['section_number']}"
        cross_refs = cross_ref_index.get(act_ref_ipc, []) + cross_ref_index.get(act_ref_bns, [])
        # Deduplicate
        cross_refs = list(dict.fromkeys(cross_refs))

    return {
        "chunk_id": chunk["chunk_id"],
        "source_file": source_path,
        "doc_type": "statute",
        "domain": domain,
        "namespace": namespace,
        "chunk_index": chunk["chunk_index"],
        "total_chunks": total_chunks,
        # Statute-specific
        "act_name": act_name,
        "section_number": chunk.get("section_number", ""),
        "section_heading": chunk.get("section_heading", ""),
        "chapter": chapter,
        "amendment_year": amendment_year,
        "cross_references": cross_refs,
        # Judgment-specific (null for statutes)
        "case_name": None,
        "citation": None,
        "year": None,
        "bench": None,
        "subject_tags": [],
        "is_headnote": False,
    }


def build_judgment_metadata(
    chunk: dict,
    source_path: str,
    total_chunks: int,
    case_name: str,
    citation: str,
    year: int | None = None,
    bench: str = "",
    subject_tags: list[str] | None = None,
) -> dict:
    """Build metadata for a judgment chunk."""
    inferred_year = year or _infer_year_from_path(source_path)

    return {
        "chunk_id": chunk["chunk_id"],
        "source_file": source_path,
        "doc_type": "judgment",
        "domain": "general",
        "namespace": "judgments-all",
        "chunk_index": chunk["chunk_index"],
        "total_chunks": total_chunks,
        # Statute-specific (null for judgments)
        "act_name": None,
        "section_number": None,
        "section_heading": None,
        "chapter": None,
        "amendment_year": None,
        "cross_references": [],
        # Judgment-specific
        "case_name": case_name,
        "citation": citation,
        "year": inferred_year,
        "bench": bench,
        "subject_tags": subject_tags or [],
        "is_headnote": chunk.get("is_headnote", False),
    }


def build_mapping_metadata(
    chunk_id: str,
    source_path: str,
    chunk_index: int,
    total_chunks: int,
    ipc_ref: str,
    bns_ref: str,
    description: str,
) -> dict:
    """Build metadata for a mapping chunk."""
    return {
        "chunk_id": chunk_id,
        "source_file": source_path,
        "doc_type": "mapping",
        "domain": "criminal",
        "namespace": "mappings",
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "act_name": "IPC-BNS Mapping",
        "section_number": f"{ipc_ref} | {bns_ref}",
        "section_heading": f"{ipc_ref} ↔ {bns_ref}",
        "chapter": "",
        "amendment_year": None,
        "cross_references": [ipc_ref, bns_ref],
        "case_name": None,
        "citation": None,
        "year": None,
        "bench": None,
        "subject_tags": [],
        "is_headnote": False,
        "mapping_description": description,
    }
