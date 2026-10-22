"""Change feed and admin source checks over HTTP, with network fetches stubbed."""
from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

import config
from api.main import app
from ingestion import change_watch
from ingestion.refresh import RetrievedSource

client = TestClient(app)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "source_manifest.json").write_text(json.dumps([
        {"source_id": "bns-commencement", "title": "BNS commencement", "url": "https://egazette.gov.in/so850",
         "authority": "gazette", "document_type": "commencement_notification"},
        {"source_id": "pib", "title": "PIB", "url": "https://pib.gov.in/", "authority": "pib"},
    ]))
    monkeypatch.setattr(config, "ADMIN_PASSWORD", "correct horse")
    monkeypatch.setattr(config, "DATA_DIR", str(data))
    monkeypatch.setattr(config, "CHANGE_LOG_PATH", str(tmp_path / "changes.sqlite3"))
    monkeypatch.setattr(config, "SOURCE_SNAPSHOT_PATH", str(tmp_path / "snapshots.sqlite3"))
    monkeypatch.setattr(config, "INGESTION_LEDGER_PATH", str(tmp_path / "ledger.sqlite3"))
    monkeypatch.setattr(
        change_watch, "fetch_source",
        lambda source: RetrievedSource(source, b"<p>S.O. 850(E)</p>", "raw", "2026-10-21T00:00:00+00:00"),
    )
    return tmp_path


def _admin() -> dict:
    return {"Authorization": f"Bearer {client.post('/auth/admin', json={'password': 'correct horse'}).json()['token']}"}


def test_upload_appears_in_change_feed(isolated):
    client.post(
        "/admin/corpus/upload",
        headers=_admin(),
        data={"destination": "gazette/amendments", "title": "BNS (Amendment) Act, 2026"},
        files={"file": ("amendment.pdf", b"%PDF-1.7\n", "application/pdf")},
    )

    events = client.get("/changes").json()["events"]

    assert events[0]["outcome"] == "added"
    assert events[0]["title"] == "BNS (Amendment) Act, 2026"
    assert events[0]["authority"] == "gazette"


def test_admin_source_check_records_new_source(isolated):
    headers = _admin()
    listing = client.get("/admin/sources", headers=headers).json()
    assert [s["source_id"] for s in listing["sources"]] == ["bns-commencement"]
    assert listing["not_watched"][0]["source_id"] == "pib"

    assert client.post("/admin/sources/check", json={}, headers=headers).status_code == 200
    for _ in range(50):
        job = client.get("/admin/sources", headers=headers).json()["job"]
        if job["status"] != "running":
            break
        time.sleep(0.05)

    assert job["status"] == "completed" and job["result"]["new"] == 1
    assert client.get("/changes").json()["events"][0]["outcome"] == "new"
    assert client.post("/admin/sources/check", json={}).status_code == 401
