"""Feedback -> review queue -> eval case, over HTTP."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import config
from api.main import app

client = TestClient(app)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_PASSWORD", "correct horse")
    monkeypatch.setattr(config, "REVIEW_QUEUE_PATH", str(tmp_path / "queue.sqlite3"))
    monkeypatch.setattr(config, "FEEDBACK_LOG_PATH", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setattr(config, "REVIEWED_EVAL_DATASET", str(tmp_path / "evals" / "reviewed-cases.json"))
    return tmp_path


def _researcher() -> dict:
    return {"Authorization": f"Bearer {client.post('/auth/demo').json()['token']}"}


def _admin() -> dict:
    return {"Authorization": f"Bearer {client.post('/auth/admin', json={'password': 'correct horse'}).json()['token']}"}


def _report(share: bool) -> dict:
    return client.post("/feedback", headers=_researcher(), json={
        "query": "Is IPC 302 still in force?",
        "kind": "stale_status",
        "share_question": share,
        "answer": "IPC section 302 is in force.",
        "citations": ["IPC Section 302"],
        "sources": [{"title": "Indian Penal Code, 1860", "section": "302", "status": "in_force"}],
        "confidence_tier": "MEDIUM",
    }).json()


def test_helpful_feedback_does_not_enter_queue(isolated):
    body = client.post("/feedback", headers=_researcher(), json={"query": "q", "kind": "helpful"}).json()

    assert body["queued_for_review"] is False
    assert client.get("/admin/review", headers=_admin()).json()["counts"]["open"] == 0


def test_unshared_report_keeps_no_question_text(isolated):
    assert _report(share=False)["queued_for_review"] is True

    item = client.get("/admin/review", headers=_admin()).json()["items"][0]

    assert item["question"] is None and item["answer"] is None
    assert item["citations"] == ["IPC Section 302"]


def test_confirmed_shared_report_becomes_eval_case(isolated):
    _report(share=True)
    headers = _admin()
    item_id = client.get("/admin/review", headers=headers).json()["items"][0]["id"]

    early = client.post(f"/admin/review/{item_id}/promote", headers=headers)
    client.patch(f"/admin/review/{item_id}", headers=headers, json={
        "status": "confirmed", "reviewer_note": "IPC was repealed on 1 July 2024.",
        "expected_sections": ["BNS section 103"],
    })
    promoted = client.post(f"/admin/review/{item_id}/promote", headers=headers)

    assert early.status_code == 409
    assert promoted.status_code == 200
    case = json.loads((isolated / "evals" / "reviewed-cases.json").read_text())[0]
    assert case["id"] == f"review-{item_id}"
    assert case["category"] == "temporal"
    assert case["expected_anchors"] == ["BNS section 103"]
    assert client.get("/admin/review", params={"status": "promoted"}, headers=headers).json()["counts"]["promoted"] == 1
    datasets = {d["name"]: d for d in client.get("/admin/evals", headers=headers).json()["datasets"]}
    assert datasets["reviewed-cases.json"]["from_review"] == 1


def test_review_queue_requires_admin(isolated):
    assert client.get("/admin/review", headers=_researcher()).status_code == 403
