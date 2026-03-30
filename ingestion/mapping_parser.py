"""
ingestion/mapping_parser.py — Parse IPC↔BNS cross-reference mapping documents.

These docs map old IPC sections to new BNS/BNSS equivalents.
Emits structured pairs used to populate cross_references metadata.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from utils.text_cleaner import clean_text
from utils.logger import get_logger

logger = get_logger(__name__)


class MappingPair(TypedDict):
    ipc_ref: str    # e.g. "IPC Section 302"
    bns_ref: str    # e.g. "BNS Section 103"
    description: str


# Match lines like: "302 | 103 | Murder" or "Section 302 → Section 103"
_TABLE_ROW = re.compile(
    r"(?:Section\s*)?(\d+[\w\.\-]*)\s*[|→\-–]\s*(?:Section\s*)?(\d+[\w\.\-]*)\s*[|→\-–]?\s*(.*)"
)

# Match inline mapping: "IPC Section 302 corresponds to BNS Section 103"
_INLINE_MAP = re.compile(
    r"IPC\s+[Ss]ection\s+(\d+[\w\.\-]*)\s+(?:corresponds to|is replaced by|maps to|equivalent)\s+BNS\s+[Ss]ection\s+(\d+[\w\.\-]*)",
    re.IGNORECASE,
)


def parse_mapping_doc(pages: list[dict]) -> list[MappingPair]:
    """
    Parse a mapping PDF and extract IPC↔BNS pairs.

    Args:
        pages: List of PageData dicts from pdf_loader.

    Returns:
        List of MappingPair dicts.
    """
    full_text = "\n".join(clean_text(p["raw_text"]) for p in pages)
    pairs: list[MappingPair] = []
    seen: set[tuple[str, str]] = set()

    # Try table-format rows first
    for line in full_text.split("\n"):
        m = _TABLE_ROW.match(line.strip())
        if m:
            ipc_sec, bns_sec, desc = m.group(1), m.group(2), m.group(3).strip()
            key = (ipc_sec, bns_sec)
            if key not in seen:
                seen.add(key)
                pairs.append(
                    MappingPair(
                        ipc_ref=f"IPC Section {ipc_sec}",
                        bns_ref=f"BNS Section {bns_sec}",
                        description=desc,
                    )
                )

    # Try inline sentence format
    for m in _INLINE_MAP.finditer(full_text):
        ipc_sec, bns_sec = m.group(1), m.group(2)
        key = (ipc_sec, bns_sec)
        if key not in seen:
            seen.add(key)
            pairs.append(
                MappingPair(
                    ipc_ref=f"IPC Section {ipc_sec}",
                    bns_ref=f"BNS Section {bns_sec}",
                    description="",
                )
            )

    logger.info(f"Mapping parser: extracted {len(pairs)} IPC↔BNS pairs")
    return pairs


def build_cross_ref_index(all_pairs: list[MappingPair]) -> dict[str, list[str]]:
    """
    Build a lookup dict: section ref → list of cross-references.
    e.g. "IPC Section 302" → ["BNS Section 103"]
         "BNS Section 103" → ["IPC Section 302"]
    Used by metadata_builder.py.
    """
    index: dict[str, list[str]] = {}
    for pair in all_pairs:
        index.setdefault(pair["ipc_ref"], []).append(pair["bns_ref"])
        index.setdefault(pair["bns_ref"], []).append(pair["ipc_ref"])
    return index
