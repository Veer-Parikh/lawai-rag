"""tests/unit/test_retrieval_quality_assessor.py"""
from __future__ import annotations

import pytest
from retrieval.retrieval_quality_assessor import (
    ConfidenceTier,
    assess,
    build_low_confidence_response,
)

_CHUNK = lambda score, src: {"chunk_id": "abc", "score": score, "source_file": src}


def test_high_confidence():
    chunks = [_CHUNK(0.92, "ipc.pdf"), _CHUNK(0.85, "bns.pdf")]
    result = assess(chunks)
    assert result.tier == ConfidenceTier.HIGH
    assert result.top_score == pytest.approx(0.92)


def test_medium_confidence():
    chunks = [_CHUNK(0.72, "ipc.pdf"), _CHUNK(0.68, "bns.pdf")]
    result = assess(chunks)
    assert result.tier == ConfidenceTier.MEDIUM


def test_low_confidence():
    chunks = [_CHUNK(0.50, "ipc.pdf")]
    result = assess(chunks)
    assert result.tier == ConfidenceTier.LOW


def test_empty_chunks_returns_low():
    result = assess([])
    assert result.tier == ConfidenceTier.LOW
    assert result.top_score == 0.0


def test_narrow_context_detection():
    chunks = [_CHUNK(0.88, "ipc.pdf"), _CHUNK(0.82, "ipc.pdf"), _CHUNK(0.80, "ipc.pdf")]
    result = assess(chunks)
    assert result.is_narrow is True
    assert result.needs_diversity is True


def test_diverse_context_not_narrow():
    chunks = [_CHUNK(0.88, "ipc.pdf"), _CHUNK(0.85, "bns.pdf"), _CHUNK(0.80, "sc_2022.pdf")]
    result = assess(chunks)
    assert result.is_narrow is False


def test_low_confidence_response_structure():
    resp = build_low_confidence_response()
    assert "answer" in resp
    assert "relevant_sections" in resp
    assert resp["confidence_tier"] == ConfidenceTier.LOW.value
    assert resp["sources"] == []
