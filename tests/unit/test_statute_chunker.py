"""tests/unit/test_statute_chunker.py"""
from __future__ import annotations

import pytest
from ingestion.statute_chunker import chunk_statute, _approx_tokens


SAMPLE_STATUTE_PAGES = [
    {
        "page_num": 1,
        "raw_text": (
            "THE INDIAN PENAL CODE, 1860\n\n"
            "Section 299\nCulpable homicide.\n"
            "Whoever causes death by doing an act with the intention of causing death, "
            "or with the intention of causing such bodily injury as is likely to cause death, "
            "or with the knowledge that he is likely by such act to cause death, commits the "
            "offence of culpable homicide.\n\n"
            "Section 300\nMurder.\n"
            "Except in the cases hereinafter excepted, culpable homicide is murder, if the act "
            "by which the death is caused is done with the intention of causing death, or "
            "if it is done with the intention of causing such bodily injury as the offender "
            "knows to be likely to cause the death of the person to whom the harm is caused.\n\n"
            "Section 302\nPunishment for murder.\n"
            "Whoever commits murder shall be punished with death, or imprisonment "
            "for life, and shall also be liable to fine."
        ),
        "source_path": "data/statutes/criminal/IPC_1860.pdf",
    }
]


def test_chunk_statute_produces_chunks():
    chunks = chunk_statute(SAMPLE_STATUTE_PAGES, act_name="Indian Penal Code")
    assert len(chunks) >= 3


def test_chunk_statute_ids_are_unique():
    chunks = chunk_statute(SAMPLE_STATUTE_PAGES, act_name="Indian Penal Code")
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_statute_section_numbers_extracted():
    chunks = chunk_statute(SAMPLE_STATUTE_PAGES, act_name="Indian Penal Code")
    section_numbers = [c["section_number"] for c in chunks]
    assert "299" in section_numbers
    assert "302" in section_numbers


def test_chunk_statute_heading_prepended():
    chunks = chunk_statute(SAMPLE_STATUTE_PAGES, act_name="Indian Penal Code")
    for chunk in chunks:
        assert chunk["section_heading"] in chunk["text"]


def test_chunk_statute_indices_sequential():
    chunks = chunk_statute(SAMPLE_STATUTE_PAGES, act_name="Indian Penal Code")
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i


def test_approx_tokens():
    assert _approx_tokens("hello world") > 0
    assert _approx_tokens("") == 1


def test_fallback_chunking_on_no_headers():
    pages = [{"page_num": 1, "raw_text": "word " * 600, "source_path": "test.pdf"}]
    chunks = chunk_statute(pages, act_name="Test Act")
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c["text"]) > 0
