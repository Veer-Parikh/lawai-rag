"""Crosswalk, provision text, timelines and admin catalog actions over HTTP."""
from __future__ import annotations

import time
from datetime import date

import pytest
from fastapi.testclient import TestClient

import config
from api.main import app
from api.routers import legal as legal_router
from legal.documents import DocumentStatus, LegalDocument, ProvisionVersion
from legal.store import DocumentStore

client = TestClient(app)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_PASSWORD", "correct horse")
    monkeypatch.setattr(config, "CATALOG_PATH", str(tmp_path / "catalog.sqlite3"))
    monkeypatch.setattr(config, "CROSSWALK_PATH", str(tmp_path / "crosswalk.sqlite3"))
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path / "data"))
    legal_router.invalidate_catalog_cache()
    DocumentStore(config.CATALOG_PATH).upsert(LegalDocument(
        document_id="bns-2023", title="Bharatiya Nyaya Sanhita, 2023", act_code="BNS",
        source_url="https://www.indiacode.nic.in/", publisher="Legislative Department",
        status=DocumentStatus.IN_FORCE,
        provisions=[ProvisionVersion("bns-2023-s103", "103", "Punishment for murder",
                                     "(1) Whoever commits murder shall be punished with death.", date(2024, 7, 1),
                                     page_anchor="page 34")],
        source_authority="gazette",
    ))
    yield tmp_path
    legal_router.invalidate_catalog_cache()


def _admin() -> dict:
    token = client.post("/auth/admin", json={"password": "correct horse"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_acts_report_catalog_coverage(isolated):
    acts = {act["code"]: act for act in client.get("/acts").json()}

    assert acts["BNS"]["sections_in_catalog"] == 1
    assert acts["IPC"]["status"] == "repealed" and acts["IPC"]["repealed_by"] == "BNS, section 358"


def test_crosswalk_search_finds_seeded_rows(isolated):
    body = client.get("/crosswalk", params={"q": "IPC 302"}).json()

    assert body["total"] > 50 and body["verified"] == 0
    assert [(r["to_code"], r["to_section"]) for r in body["rows"]] == [("BNS", "103(1)")]


def test_provision_text_and_missing_catalog_message(isolated):
    found = client.get("/provisions/BNS/103(1)")
    missing = client.get("/provisions/BSA/23")

    assert found.status_code == 200
    assert found.json()["text"].startswith("(1) Whoever commits murder")
    assert found.json()["extracted_from"] == "Gazette of India PDF"
    assert missing.status_code == 404 and "not in the catalog" in missing.json()["detail"]


def test_timeline_includes_text_and_predecessor(isolated):
    body = client.get("/timeline/BNS/103").json()

    assert body["text"].startswith("(1) Whoever")
    assert any(e["kind"] == "predecessor" and e["code"] == "IPC" for e in body["events"])
    assert client.get("/timeline/XYZ/1").status_code == 404


def test_only_admins_can_verify_crosswalk_rows(isolated):
    row_id = client.get("/crosswalk", params={"q": "IPC 302"}).json()["rows"][0]["id"]
    demo = {"Authorization": f"Bearer {client.post('/auth/demo').json()['token']}"}

    assert client.patch(f"/admin/crosswalk/{row_id}", json={"verified": True}, headers=demo).status_code == 403
    verified = client.patch(f"/admin/crosswalk/{row_id}", json={"verified": True}, headers=_admin())
    assert verified.status_code == 200
    assert verified.json()["verified_by"] == "Administrator"


def test_catalog_build_job_reports_skipped_acts(isolated):
    headers = _admin()
    assert client.post("/admin/catalog/build", json={"codes": ["NOPE"]}, headers=headers).status_code == 400

    started = client.post("/admin/catalog/build", json={"codes": ["IPC"]}, headers=headers)
    assert started.status_code == 200
    for _ in range(50):
        job = client.get("/admin/catalog/build", headers=headers).json()["job"]
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "completed"
    assert job["result"][0]["status"] == "skipped"
