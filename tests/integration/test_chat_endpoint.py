"""tests/integration/test_chat_endpoint.py — Integration tests for POST /chat.

These tests mock Pinecone and Gemini to avoid real API calls.
Run with: pytest tests/integration/ -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app, raise_server_exceptions=False)

# ── Shared mock chunk ─────────────────────────────────────────────────────────
MOCK_CHUNK = {
    "chunk_id": "test-chunk-001",
    "score": 0.87,
    "text": (
        "Section 302\nPunishment for murder.\n"
        "Whoever commits murder shall be punished with death, "
        "or imprisonment for life."
    ),
    "doc_type": "statute",
    "domain": "criminal",
    "namespace": "statutes-criminal",
    "act_name": "Indian Penal Code",
    "section_number": "302",
    "section_heading": "Section 302",
    "case_name": None,
    "citation": None,
    "source_file": "data/statutes/criminal/IPC_1860.pdf",
    "cross_references": [],
    "is_headnote": False,
}

MOCK_GEMINI_RESPONSE = {
    "answer": "Murder under Section 302 IPC is punishable by death or life imprisonment.",
    "relevant_sections": ["Section 302, Indian Penal Code"],
    "legal_explanation": "Section 302 of the Indian Penal Code prescribes the punishment for murder.",
    "disclaimer": (
        "This response is for informational purposes only and does not "
        "constitute legal advice."
    ),
}


@patch("retrieval.pinecone_retriever._get_index")
@patch("ingestion.embedder._get_model")
@patch("retrieval.reranker._get_reranker")
@patch("llm.gemini_client.genai.GenerativeModel")
def test_chat_high_confidence_success(mock_genai, mock_reranker, mock_embedder, mock_index):
    # Mock embedding
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768]
    mock_embedder.return_value = mock_model

    # Mock Pinecone
    mock_match = MagicMock()
    mock_match.id = "test-chunk-001"
    mock_match.score = 0.87
    mock_match.metadata = {k: v for k, v in MOCK_CHUNK.items() if k != "chunk_id"}
    mock_index.return_value.query.return_value = MagicMock(matches=[mock_match])

    # Mock reranker
    mock_cross_enc = MagicMock()
    mock_cross_enc.predict.return_value = [0.87]
    mock_reranker.return_value = mock_cross_enc

    # Mock Gemini
    import json
    mock_gen_model = MagicMock()
    mock_gen_model.generate_content.return_value = MagicMock(
        text=json.dumps(MOCK_GEMINI_RESPONSE),
        usage_metadata=MagicMock(total_token_count=150),
    )
    mock_genai.return_value = mock_gen_model

    response = client.post("/chat", json={"query": "What is the punishment for murder under IPC?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "relevant_sections" in data
    assert "confidence_tier" in data
    assert "sources" in data
    assert data["confidence_tier"] in ("HIGH", "MEDIUM")


def test_chat_query_too_short():
    response = client.post("/chat", json={"query": "hi"})
    assert response.status_code == 422  # Pydantic min_length validation


def test_chat_missing_query():
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_health_endpoint():
    with patch("pinecone.Pinecone") as mock_pc, \
         patch("google.generativeai.list_models"):
        mock_pc.return_value.list_indexes.return_value = [MagicMock(name="bharat-law")]
        response = client.get("/health")
    # Either 200 or 503 depending on mock completeness
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data
    assert "checks" in data


def test_search_endpoint_returns_results():
    with patch("retrieval.pinecone_retriever._get_index") as mock_index, \
         patch("ingestion.embedder._get_model") as mock_embedder, \
         patch("retrieval.reranker._get_reranker") as mock_reranker:

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 768]
        mock_embedder.return_value = mock_model

        mock_match = MagicMock()
        mock_match.id = "test-chunk-001"
        mock_match.score = 0.80
        mock_match.metadata = {k: v for k, v in MOCK_CHUNK.items() if k != "chunk_id"}
        mock_index.return_value.query.return_value = MagicMock(matches=[mock_match])

        mock_cross_enc = MagicMock()
        mock_cross_enc.predict.return_value = [0.80]
        mock_reranker.return_value = mock_cross_enc

        response = client.get("/search?q=Section+302+IPC&domain=criminal")

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "total_results" in data
    assert "latency_ms" in data
