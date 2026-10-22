from __future__ import annotations

import json

import pytest

from ingestion.corpus import resolve_inside, safe_filename, scan_corpus, store_upload
from ingestion.ledger import IngestionLedger

PDF = b"%PDF-1.7\n%test\n"


def test_resolve_inside_rejects_traversal(tmp_path):
    (tmp_path / "gazette").mkdir()
    assert resolve_inside(tmp_path, "gazette") == (tmp_path / "gazette").resolve()
    with pytest.raises(ValueError):
        resolve_inside(tmp_path, "../secrets")
    with pytest.raises(ValueError):
        resolve_inside(tmp_path, "gazette/../../etc")


def test_safe_filename_normalizes_and_forces_pdf():
    assert safe_filename("S.O. 850(E) BNS.PDF") == "s.o._850_e_bns.pdf"
    with pytest.raises(ValueError):
        safe_filename("///.pdf")


def test_store_upload_writes_file_and_sidecar(tmp_path):
    stored = store_upload(
        tmp_path,
        destination="gazette/amendments",
        filename="Amendment 2025.pdf",
        content=PDF,
        title="BNS amendment",
        source_url="https://egazette.gov.in/x.pdf",
        publisher="Gazette of India",
        uploaded_by="Administrator",
    )
    sidecar = json.loads(stored.with_name(stored.name + ".metadata.json").read_text())

    assert stored.read_bytes() == PDF
    assert sidecar["authority"] == "gazette"
    assert sidecar["download_status"] == "uploaded"
    assert len(sidecar["sha256"]) == 64


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"destination": "../outside"}, "corpus folders"),
        ({"content": b"<html>"}, "only PDF"),
        ({"source_url": "http://insecure.test/a.pdf"}, "https"),
    ],
)
def test_store_upload_rejects_bad_input(tmp_path, kwargs, message):
    params = dict(
        destination="mappings", filename="a.pdf", content=PDF, title=None,
        source_url=None, publisher=None, uploaded_by="Administrator",
    )
    params.update(kwargs)
    with pytest.raises(ValueError, match=message):
        store_upload(tmp_path, **params)


def test_scan_joins_ledger_status(tmp_path):
    ingested = store_upload(
        tmp_path, destination="mappings", filename="a.pdf", content=PDF,
        title=None, source_url=None, publisher=None, uploaded_by="Administrator",
    )
    store_upload(
        tmp_path, destination="mappings", filename="b.pdf", content=PDF,
        title=None, source_url=None, publisher=None, uploaded_by="Administrator",
    )
    (tmp_path / "download_manifest.json").write_text(json.dumps({"records": [
        {"source_id": "x", "title": "Portal", "download_status": "failed", "error": "HTTP Error 403"},
    ]}))
    ledger = IngestionLedger(":memory:")
    ledger.record(
        path=ingested.resolve(), doc_type="mapping", chunks_created=4,
        chunks_uploaded=4, error=None, dry_run=False,
    )

    inventory = scan_corpus(tmp_path, ledger)

    assert inventory["summary"]["pdfs"] == 2
    assert inventory["summary"]["ingested"] == 1
    assert inventory["summary"]["pending"] == 1
    assert inventory["summary"]["chunks_uploaded"] == 4
    assert inventory["unavailable"][0]["error"] == "HTTP Error 403"


def test_scan_flags_empty_downloads_as_not_ingestible(tmp_path):
    folder = tmp_path / "primary_statutes" / "historical" / "ipc"
    folder.mkdir(parents=True)
    (folder / "ipc_1860.pdf").write_bytes(b"")

    inventory = scan_corpus(tmp_path, IngestionLedger(":memory:"))

    assert inventory["files"][0]["empty"] is True
    assert inventory["files"][0]["ingestible"] is False
    assert inventory["summary"] == {**inventory["summary"], "pdfs": 0, "pending": 0, "empty": 1}
