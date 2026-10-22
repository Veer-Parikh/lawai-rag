"""Matters: ownership, saved answers, memo export."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import config
from api.main import app

client = TestClient(app)

ANSWER = {
    "answer": "IPC section 302 is replaced by **BNS section 103**.",
    "confidence_tier": "HIGH",
    "citations": [
        {"text": "BNS Section 103", "verified": True, "source_urls": ["https://www.indiacode.nic.in/"]},
        {"text": "BNS Section 101", "verified": False, "source_urls": []},
    ],
    "sources": [{"act_name": "Bharatiya Nyaya Sanhita, 2023", "section": "103", "legal_status": "in_force",
                 "effective_date": "2024-07-01", "source_url": "https://www.indiacode.nic.in/"}],
    "citation_warnings": [],
}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_PATH", str(tmp_path / "workspace.sqlite3"))
    return tmp_path


def _login(device: str) -> dict:
    token = client.post("/auth/demo", json={"device_id": device}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_same_device_keeps_its_matters(isolated):
    first = _login("browser-one-1234")
    created = client.post("/matters", headers=first, json={"title": "State v. Sharma", "reference": "CR-17/2026"})
    again = _login("browser-one-1234")

    assert created.status_code == 201
    assert [m["title"] for m in client.get("/matters", headers=again).json()] == ["State v. Sharma"]


def test_other_researchers_cannot_see_or_change_a_matter(isolated):
    owner, stranger = _login("browser-owner-1"), _login("browser-stranger-1")
    matter_id = client.post("/matters", headers=owner, json={"title": "Bail strategy"}).json()["id"]

    assert client.get("/matters", headers=stranger).json() == []
    assert client.get(f"/matters/{matter_id}", headers=stranger).status_code == 404
    assert client.delete(f"/matters/{matter_id}", headers=stranger).status_code == 404
    assert client.get("/matters").status_code == 401


def test_saved_answers_and_memo(isolated):
    headers = _login("browser-memo-1")
    matter_id = client.post("/matters", headers=headers, json={"title": "Murder charge review"}).json()["id"]
    entry = client.post(f"/matters/{matter_id}/entries", headers=headers, json={
        "question": "Which BNS section replaced IPC 302?", "as_of": "2025-01-01", "answer": ANSWER,
    }).json()
    client.patch(f"/matters/{matter_id}/entries/{entry['id']}", headers=headers, json={"note": "Cite s.103(1)."})

    matter = client.get(f"/matters/{matter_id}", headers=headers).json()
    memo = client.get(f"/matters/{matter_id}/memo.md", headers=headers)

    assert matter["entry_count"] == 1 and matter["entries"][0]["note"] == "Cite s.103(1)."
    assert memo.status_code == 200
    assert 'filename="murder-charge-review-memo.md"' in memo.headers["content-disposition"]
    text = memo.text
    assert "# Research memo: Murder charge review" in text
    assert "Partly supported" in text  # one citation was not matched
    assert "BNS Section 101: NOT found in the retrieved sources" in text
    assert "| 1 | Bharatiya Nyaya Sanhita, 2023 | 103 | In force | 2024-07-01 |" in text
    assert "Cite s.103(1)." in text

    assert client.delete(f"/matters/{matter_id}/entries/{entry['id']}", headers=headers).status_code == 204
    assert client.delete(f"/matters/{matter_id}", headers=headers).status_code == 204
    assert client.get("/matters", headers=headers).json() == []
