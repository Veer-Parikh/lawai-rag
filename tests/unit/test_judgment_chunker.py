"""tests/unit/test_judgment_chunker.py"""
from __future__ import annotations

import pytest
from ingestion.judgment_chunker import (
    chunk_judgment,
    _extract_citation,
    _extract_case_name,
    _split_into_paragraphs,
)

SAMPLE_JUDGMENT_PAGES = [
    {
        "page_num": 1,
        "raw_text": (
            "State of Maharashtra v. Madhkar Narayan Mardikar\n\n"
            "AIR 1991 SC 207\n\n"
            "BENCH:\n"
            "K. Jagannatha Shetty, S.C. Agrawal\n\n"
            "HEADNOTE:\n"
            "Right to privacy — Prosecutrix in rape case — Entitlement to privacy — "
            "Even women of easy virtue have right to privacy. No one can examine them "
            "about their past sexual history to find out whether they consented.\n\n"
            "JUDGMENT:\n"
            "This appeal arises from the judgment of the High Court of Maharashtra. "
            "The accused was charged under Section 376 IPC.\n\n"
            "The Sessions Court had acquitted the accused. The State filed an appeal.\n\n"
            "HELD:\n"
            "Every woman is entitled to her sexual privacy and it is not open to any "
            "and every person to violate her privacy as and when he wishes."
        ),
        "source_path": "data/supreme_court_judgments/1991/madhkar_mardikar.pdf",
    }
]


def test_chunk_judgment_produces_chunks():
    chunks, case_name, citation = chunk_judgment(SAMPLE_JUDGMENT_PAGES)
    assert len(chunks) >= 1


def test_chunk_judgment_extracts_headnote():
    chunks, _, _ = chunk_judgment(SAMPLE_JUDGMENT_PAGES)
    headnote_chunks = [c for c in chunks if c["is_headnote"]]
    assert len(headnote_chunks) >= 1


def test_chunk_judgment_extracts_case_name():
    _, case_name, _ = chunk_judgment(SAMPLE_JUDGMENT_PAGES)
    assert "Madhkar" in case_name or "Maharashtra" in case_name


def test_chunk_judgment_extracts_citation():
    _, _, citation = chunk_judgment(SAMPLE_JUDGMENT_PAGES)
    assert "AIR 1991 SC 207" == citation


def test_chunk_judgment_ids_unique():
    chunks, _, _ = chunk_judgment(SAMPLE_JUDGMENT_PAGES)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_extract_citation_patterns():
    assert _extract_citation("Ref: AIR 2022 SC 100") == "AIR 2022 SC 100"
    assert _extract_citation("(2019) 5 SCC 200") == "(2019) 5 SCC 200"
    assert _extract_citation("no citation here") == ""


def test_split_into_paragraphs_filters_short():
    text = "Short.\n\nThis is a proper paragraph with enough content to be included.\n\nX"
    paras = _split_into_paragraphs(text)
    assert all(len(p) > 30 for p in paras)
