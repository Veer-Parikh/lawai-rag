"""Auth flow and admin role gating, with the data directory isolated to tmp_path."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import config
from api.main import app

client = TestClient(app)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_PASSWORD", "correct horse")
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(config, "INGESTION_LEDGER_PATH", str(tmp_path / "ledger.sqlite3"))
    monkeypatch.setattr(config, "EVAL_LOG_PATH", str(tmp_path / "eval.jsonl"))
    monkeypatch.setattr(config, "FEEDBACK_LOG_PATH", str(tmp_path / "feedback.jsonl"))
    (tmp_path / "data").mkdir()
    return tmp_path


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_demo_login_returns_researcher_session(isolated):
    session = client.post("/auth/demo").json()

    assert session["role"] == "researcher"
    me = client.get("/auth/me", headers=_bearer(session["token"])).json()
    assert me["name"] == "Demo researcher"


def test_admin_login_rejects_wrong_password(isolated):
    assert client.post("/auth/admin", json={"password": "nope"}).status_code == 401


def test_researcher_cannot_reach_admin_routes(isolated):
    token = client.post("/auth/demo").json()["token"]

    assert client.get("/admin/corpus").status_code == 401
    assert client.get("/admin/corpus", headers=_bearer(token)).status_code == 403


def test_admin_can_upload_list_and_read_analytics(isolated):
    token = client.post("/auth/admin", json={"password": "correct horse"}).json()["token"]

    upload = client.post(
        "/admin/corpus/upload",
        headers=_bearer(token),
        data={"destination": "mappings", "title": "IPC to BNS table"},
        files={"file": ("crosswalk.pdf", b"%PDF-1.7\n", "application/pdf")},
    )
    corpus = client.get("/admin/corpus", headers=_bearer(token)).json()
    analytics = client.get("/admin/analytics?days=7", headers=_bearer(token))

    assert upload.status_code == 200
    assert upload.json()["path"] == "mappings/crosswalk.pdf"
    assert corpus["summary"]["pending"] == 1
    assert corpus["files"][0]["title"] == "IPC to BNS table"
    assert analytics.status_code == 200
    assert len(analytics.json()["daily"]) == 7


def test_ingest_refuses_paths_outside_data_dir(isolated):
    token = client.post("/auth/admin", json={"password": "correct horse"}).json()["token"]

    response = client.post(
        "/admin/ingest",
        headers=_bearer(token),
        json={"target_path": "../../etc", "doc_type": "all"},
    )

    assert response.status_code == 400


def test_feedback_requires_sign_in_and_stores_hash_only(isolated):
    payload = {"query": "What replaced IPC 302?", "kind": "stale_status"}
    assert client.post("/feedback", json=payload).status_code == 401

    token = client.post("/auth/demo").json()["token"]
    assert client.post("/feedback", json=payload, headers=_bearer(token)).status_code == 200
    stored = (isolated / "feedback.jsonl").read_text()
    assert "IPC 302" not in stored
    assert "stale_status" in stored
