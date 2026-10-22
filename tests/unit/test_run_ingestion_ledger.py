from __future__ import annotations

from pathlib import Path

import config
from ingestion import run_ingestion
from ingestion.ledger import IngestionLedger


def _fake_ingest(pdf_path, doc_type, cross_ref_index, dry_run=False):
    return {"file": str(pdf_path), "chunks_created": 3, "chunks_uploaded": 0 if dry_run else 3, "error": None}


def _corpus(tmp_path: Path) -> Path:
    folder = tmp_path / "primary_statutes" / "current" / "bns"
    folder.mkdir(parents=True)
    (folder / "a.pdf").write_bytes(b"%PDF")
    (folder / "b.pdf").write_bytes(b"%PDF")
    return folder


def test_stable_chunk_id_is_deterministic_per_file_and_position(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    pdf = _corpus(tmp_path) / "a.pdf"

    assert run_ingestion.stable_chunk_id(pdf, 0) == run_ingestion.stable_chunk_id(pdf, 0)
    assert run_ingestion.stable_chunk_id(pdf, 0) != run_ingestion.stable_chunk_id(pdf, 1)


def test_run_records_each_file_and_reports_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(run_ingestion, "ingest_file", _fake_ingest)
    folder = _corpus(tmp_path)
    ledger = IngestionLedger(":memory:")
    progress = []

    stats = run_ingestion.run(str(folder), dry_run=True, ledger=ledger, on_progress=lambda *args: progress.append(args))

    assert stats["files_processed"] == 2
    assert {Path(path).name for path in ledger.latest_by_path()} == {"a.pdf", "b.pdf"}
    assert progress[-1] == (2, 2, "")


def test_single_file_target_and_skip_ingested(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(run_ingestion, "ensure_index_exists", lambda: None)
    monkeypatch.setattr(run_ingestion, "ingest_file", _fake_ingest)
    folder = _corpus(tmp_path)
    ledger = IngestionLedger(":memory:")

    first = run_ingestion.run(str(folder / "a.pdf"), ledger=ledger)
    second = run_ingestion.run(str(folder), skip_ingested=True, ledger=ledger)

    assert first["files_found"] == 1
    assert second["files_found"] == 1  # a.pdf was already ingested
