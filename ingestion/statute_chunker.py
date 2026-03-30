"""
ingestion/statute_chunker.py — Section-aware chunking for Indian statute PDFs.

Strategy:
  1. Join all page texts into a single document string.
  2. Split on section/article header regex patterns.
  3. Each section = one candidate chunk (max STATUTE_MAX_TOKENS).
  4. Oversized sections are hard-split; undersized adjacent sections are NOT merged
     (legal context: one section = one discrete legal provision).
  5. Section heading is prepended to every chunk for context retention.
"""
from __future__ import annotations

import re
import uuid
from typing import TypedDict

from config import STATUTE_MAX_TOKENS
from utils.text_cleaner import clean_text, normalize_section_header
from utils.logger import get_logger

logger = get_logger(__name__)

# Matches: "Section 302", "SECTION 302A", "Article 21", "ARTICLE 370"
_SECTION_PATTERN = re.compile(
    r"(?m)^((?:SECTION|Section|ARTICLE|Article|CLAUSE|Clause)\s+\d+[\w\.\-]*(?:\s*[\.\-]\s*\w+)*)\s*[\.:\-]?\s*$"
)

# Approximate token count (1 token ≈ 4 chars for English legal text)
def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _hard_split(heading: str, text: str, max_tokens: int) -> list[str]:
    """Split oversized section text into sub-chunks of max_tokens."""
    words = text.split()
    chunks: list[str] = []
    chunk_words: list[str] = []
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
            chunks.append(heading + "\n" + " ".join(chunk_words))
            chunk_words = []
            token_estimate = 0
        chunk_words.append(word)
        token_estimate += word_tokens

    if chunk_words:
        chunks.append(heading + "\n" + " ".join(chunk_words))

    return chunks


class RawChunk(TypedDict):
    chunk_id: str
    text: str
    section_number: str
    section_heading: str
    chunk_index: int


def chunk_statute(
    pages: list[dict],
    act_name: str = "",
) -> list[RawChunk]:
    """
    Chunk a statute document into section-level chunks.

    Args:
        pages: List of PageData dicts from pdf_loader.load_pdf().
        act_name: Human-readable name of the Act (e.g. "Indian Penal Code").

    Returns:
        List of RawChunk dicts ready for metadata_builder.
    """
    # 1. Join all page text
    full_text = "\n".join(clean_text(p["raw_text"]) for p in pages)

    # 2. Find section split points
    splits = list(_SECTION_PATTERN.finditer(full_text))

    if not splits:
        logger.warning(f"No section headers found in '{act_name}'. Falling back to hard split.")
        return _fallback_chunk(full_text, act_name)

    raw_sections: list[tuple[str, str]] = []  # (heading, body)

    for i, match in enumerate(splits):
        heading = match.group(1).strip()
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(full_text)
        body = full_text[start:end].strip()
        raw_sections.append((heading, body))

    # 3. Build chunks respecting max token limit
    all_chunks: list[RawChunk] = []
    global_index = 0

    for heading, body in raw_sections:
        normalized_heading = normalize_section_header(heading)

        # Extract section number from heading e.g. "Section 302" → "302"
        sec_num_match = re.search(r"\d+[\w\.\-]*", heading)
        section_number = sec_num_match.group(0) if sec_num_match else heading

        full_chunk_text = f"{normalized_heading}\n{body}"

        if _approx_tokens(full_chunk_text) <= STATUTE_MAX_TOKENS:
            all_chunks.append(
                RawChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=full_chunk_text,
                    section_number=section_number,
                    section_heading=normalized_heading,
                    chunk_index=global_index,
                )
            )
            global_index += 1
        else:
            # Hard split oversized section
            sub_chunks = _hard_split(normalized_heading, body, STATUTE_MAX_TOKENS)
            for sub in sub_chunks:
                all_chunks.append(
                    RawChunk(
                        chunk_id=str(uuid.uuid4()),
                        text=sub,
                        section_number=section_number,
                        section_heading=normalized_heading,
                        chunk_index=global_index,
                    )
                )
                global_index += 1

    logger.info(f"Statute '{act_name}': {len(raw_sections)} sections → {len(all_chunks)} chunks")
    return all_chunks


def _fallback_chunk(full_text: str, act_name: str) -> list[RawChunk]:
    """Fallback: hard-split the entire text when no section headers found."""
    words = full_text.split()
    chunks: list[RawChunk] = []
    chunk_words: list[str] = []
    token_estimate = 0
    index = 0

    expanded_words = []
    for word in words:
        if len(word) > 2000:
            expanded_words.extend([word[i:i+2000] for i in range(0, len(word), 2000)])
        else:
            expanded_words.append(word)

    for word in expanded_words:
        word_tokens = max(1, len(word) // 4)
        if token_estimate + word_tokens > STATUTE_MAX_TOKENS and chunk_words:
            chunks.append(
                RawChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=" ".join(chunk_words),
                    section_number="",
                    section_heading=f"{act_name} (Part {index + 1})",
                    chunk_index=index,
                )
            )
            chunk_words = []
            token_estimate = 0
            index += 1
        chunk_words.append(word)
        token_estimate += word_tokens

    if chunk_words:
        chunks.append(
            RawChunk(
                chunk_id=str(uuid.uuid4()),
                text=" ".join(chunk_words),
                section_number="",
                section_heading=f"{act_name} (Part {index + 1})",
                chunk_index=index,
            )
        )

    return chunks
