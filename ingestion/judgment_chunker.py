"""
ingestion/judgment_chunker.py — Paragraph-aware chunking for SC judgment PDFs.

Strategy:
  1. Detect structural markers: BENCH, JUDGMENT, HEADNOTE, ORDER, HELD.
  2. Extract headnotes as standalone high-signal chunks.
  3. Split body by paragraph (double newline), merge into max-token windows
     with JUDGMENT_OVERLAP_TOKENS overlap between successive chunks.
"""
from __future__ import annotations

import re
import uuid
from typing import TypedDict

from config import JUDGMENT_MAX_TOKENS, JUDGMENT_OVERLAP_TOKENS
from utils.text_cleaner import clean_text
from utils.logger import get_logger

logger = get_logger(__name__)

# Structural section markers (case-insensitive)
_MARKER_PATTERN = re.compile(
    r"(?im)^(BENCH|JUDGMENT|HEADNOTE[S]?|ORDER|HELD|FACTS|ISSUE[S]?|RATIO|DECISION)\s*:?\s*$"
)

# Indian SC citation patterns
_CITATION_PATTERN = re.compile(
    r"(?:AIR\s+\d{4}\s+SC\s+\d+|\(\d{4}\)\s+\d+\s+SCC\s+\d+|JT\s+\d{4}\s+\(\d+\)\s+SC\s+\d+)"
)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class RawJudgmentChunk(TypedDict):
    chunk_id: str
    text: str
    chunk_index: int
    is_headnote: bool
    section_label: str   # e.g. "HEADNOTE", "JUDGMENT", "HELD"


def _extract_citation(full_text: str) -> str:
    """Try to find the first SC citation string in the document."""
    match = _CITATION_PATTERN.search(full_text)
    return match.group(0) if match else ""


def _extract_case_name(full_text: str) -> str:
    """
    Heuristic: the case name usually appears in the first 5 lines.
    Pattern: two proper-noun phrases separated by 'v.' or 'vs.'
    """
    first_lines = "\n".join(full_text.split("\n")[:10])
    match = re.search(r"([A-Z][^\n]+?)\s+v(?:s)?\.?\s+([A-Z][^\n]+)", first_lines)
    if match:
        return f"{match.group(1).strip()} v. {match.group(2).strip()}"
    return ""


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text on blank lines; discard very short paragraphs."""
    paras = re.split(r"\n\s*\n", text)
    
    # If the PDF lacks double newlines, fallback to single newlines
    if len(paras) < 5 and len(text) > 2000:
        paras = text.split("\n")
        
    result = []
    for p in paras:
        p = p.strip()
        if len(p) > 30:
            result.append(p)
    return result


def _merge_paragraphs_with_overlap(
    paragraphs: list[str],
    max_tokens: int,
    overlap_tokens: int,
    section_label: str,
) -> list[RawJudgmentChunk]:
    """
    Greedy merge: accumulate paragraphs until hitting max_tokens,
    then start new chunk with last `overlap_tokens` worth of text.
    """
    chunks: list[RawJudgmentChunk] = []
    current_paras: list[str] = []
    current_tokens = 0
    index = 0

    def flush(paras: list[str]) -> None:
        nonlocal index
        if not paras:
            return
        text = "\n\n".join(paras)
        chunks.append(
            RawJudgmentChunk(
                chunk_id=str(uuid.uuid4()),
                text=text,
                chunk_index=index,
                is_headnote=False,
                section_label=section_label,
            )
        )
        index += 1

    for para in paragraphs:
        para_tokens = _approx_tokens(para)
        
        # If a single paragraph is massive, hard split it by words
        if para_tokens > max_tokens:
            if current_paras:
                flush(current_paras)
            
            words = para.split()
            chunk_words = []
            token_estimate = 0
            
            expanded_words = []
            for word in words:
                if len(word) > 2000:
                    expanded_words.extend([word[i:i+2000] for i in range(0, len(word), 2000)])
                else:
                    expanded_words.append(word)

            for word in expanded_words:
                word_tokens = max(1, len(word) // 4)
                if token_estimate + word_tokens > max_tokens and chunk_words:
                    flush([" ".join(chunk_words)])
                    chunk_words = []
                    token_estimate = 0
                chunk_words.append(word)
                token_estimate += word_tokens
                
            if chunk_words:
                current_paras = [" ".join(chunk_words)]
                current_tokens = token_estimate
            else:
                current_paras = []
                current_tokens = 0
            continue

        if current_tokens + para_tokens > max_tokens and current_paras:
            flush(current_paras)
            # Overlap: carry the last paragraph(s) that fit in overlap_tokens
            overlap_paras: list[str] = []
            overlap_count = 0
            for p in reversed(current_paras):
                if overlap_count + _approx_tokens(p) <= overlap_tokens:
                    overlap_paras.insert(0, p)
                    overlap_count += _approx_tokens(p)
                else:
                    break
            current_paras = overlap_paras
            current_tokens = overlap_count

        current_paras.append(para)
        current_tokens += para_tokens

    flush(current_paras)
    return chunks


def chunk_judgment(
    pages: list[dict],
    case_name: str = "",
    citation: str = "",
) -> tuple[list[RawJudgmentChunk], str, str]:
    """
    Chunk a Supreme Court judgment document.

    Args:
        pages: List of PageData dicts from pdf_loader.load_pdf().
        case_name: Override case name if known; otherwise extracted heuristically.
        citation: Override citation if known.

    Returns:
        Tuple of (list of RawJudgmentChunk, resolved_case_name, resolved_citation)
    """
    full_text = "\n".join(clean_text(p["raw_text"]) for p in pages)

    # Resolve case name and citation
    resolved_case = case_name or _extract_case_name(full_text)
    resolved_citation = citation or _extract_citation(full_text)

    # Find structural markers
    markers = list(_MARKER_PATTERN.finditer(full_text))
    all_chunks: list[RawJudgmentChunk] = []

    if not markers:
        # No markers: treat entire text as JUDGMENT section
        paragraphs = _split_into_paragraphs(full_text)
        all_chunks = _merge_paragraphs_with_overlap(
            paragraphs, JUDGMENT_MAX_TOKENS, JUDGMENT_OVERLAP_TOKENS, "JUDGMENT"
        )
        # Re-index
        for i, c in enumerate(all_chunks):
            c["chunk_index"] = i
        logger.info(
            f"Judgment '{resolved_case}': no markers, produced {len(all_chunks)} chunks"
        )
        return all_chunks, resolved_case, resolved_citation

    # Build section blocks
    sections: list[tuple[str, str]] = []
    for i, match in enumerate(markers):
        label = match.group(1).strip().upper()
        start = match.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(full_text)
        body = full_text[start:end].strip()
        sections.append((label, body))

    global_index = 0
    for label, body in sections:
        if not body:
            continue

        if label.startswith("HEADNOTE"):
            # Headnote → single standalone chunk (high signal)
            all_chunks.append(
                RawJudgmentChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=f"HEADNOTE:\n{body}",
                    chunk_index=global_index,
                    is_headnote=True,
                    section_label="HEADNOTE",
                )
            )
            global_index += 1
        elif label in ("BENCH",):
            # Very short — single chunk
            all_chunks.append(
                RawJudgmentChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=f"{label}:\n{body}",
                    chunk_index=global_index,
                    is_headnote=False,
                    section_label=label,
                )
            )
            global_index += 1
        else:
            paragraphs = _split_into_paragraphs(body)
            section_chunks = _merge_paragraphs_with_overlap(
                paragraphs, JUDGMENT_MAX_TOKENS, JUDGMENT_OVERLAP_TOKENS, label
            )
            for c in section_chunks:
                c["chunk_index"] = global_index
                global_index += 1
            all_chunks.extend(section_chunks)

    logger.info(
        f"Judgment '{resolved_case}': {len(sections)} sections → {len(all_chunks)} chunks"
    )
    return all_chunks, resolved_case, resolved_citation
