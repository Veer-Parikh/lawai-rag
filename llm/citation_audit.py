"""Extract and verify inline legal citations against retrieved evidence."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_CASE_CITATION = re.compile(
    r"\b(?:AIR\s+\d{4}\s+SC\s+\d+|\(\d{4}\)\s+\d+\s+SCC\s+\d+|JT\s+\d{4}\s+\(\d+\)\s+SC\s+\d+)\b",
    re.IGNORECASE,
)
_SECTION_CITATION = re.compile(
    r"\b(?:Section|Article|Clause)\s+\d+[\w.\-]*(?:,\s*(?:the\s+)?[A-Z][A-Za-z0-9() ,.&'-]+)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CitationReference:
    text: str
    verified: bool
    source_chunk_ids: list[str]
    source_urls: list[str]
    page_anchors: list[str]


@dataclass(frozen=True)
class CitationAudit:
    references: list[CitationReference]
    warnings: list[str]


def _matches(reference: str, chunk: dict[str, Any]) -> bool:
    normalized = " ".join(reference.lower().split())
    candidates = [
        chunk.get("citation"),
        chunk.get("section_heading"),
        (
            f"Section {chunk['section_number']}, {chunk['act_name']}"
            if chunk.get("section_number") and chunk.get("act_name")
            else None
        ),
        (
            f"Article {chunk['section_number']}, {chunk['act_name']}"
            if chunk.get("section_number") and chunk.get("act_name")
            else None
        ),
    ]
    return any(
        candidate and normalized in " ".join(str(candidate).lower().split())
        for candidate in candidates
    )


def audit_answer_citations(
    answer: str,
    chunks: list[dict[str, Any]],
) -> CitationAudit:
    references: list[CitationReference] = []
    warnings: list[str] = []
    matches = list(_CASE_CITATION.finditer(answer)) + list(_SECTION_CITATION.finditer(answer))
    for match in sorted(matches, key=lambda item: item.start()):
        text = match.group(0).strip().rstrip(".,;:")
        matching_chunks = [chunk for chunk in chunks if _matches(text, chunk)]
        reference = CitationReference(
            text=text,
            verified=bool(matching_chunks),
            source_chunk_ids=[chunk["chunk_id"] for chunk in matching_chunks],
            source_urls=[
                chunk["source_url"]
                for chunk in matching_chunks
                if chunk.get("source_url")
            ],
            page_anchors=[
                chunk["page_anchor"]
                for chunk in matching_chunks
                if chunk.get("page_anchor")
            ],
        )
        references.append(reference)
        if not reference.verified:
            warnings.append(f"Unverified inline citation: {text}")
    return CitationAudit(references=references, warnings=warnings)
