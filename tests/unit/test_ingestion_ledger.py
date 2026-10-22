from __future__ import annotations

from datetime import datetime, timezone

from ingestion.ledger import IngestionLedger


def _record(ledger, path, *, error=None, dry_run=False, minute=0):
    return ledger.record(
        path=path,
        doc_type="statute",
        chunks_created=10,
        chunks_uploaded=0 if (error or dry_run) else 10,
        error=error,
        dry_run=dry_run,
        now=datetime(2026, 10, 21, 10, minute, tzinfo=timezone.utc),
    )


def test_latest_attempt_wins_per_path():
    ledger = IngestionLedger(":memory:")
    _record(ledger, "data/a.pdf", error="boom", minute=1)
    _record(ledger, "data/a.pdf", minute=2)

    latest = ledger.latest_by_path()

    assert latest["data/a.pdf"].status == "ingested"
    assert latest["data/a.pdf"].chunks_uploaded == 10


def test_dry_run_does_not_mask_a_real_ingestion():
    ledger = IngestionLedger(":memory:")
    _record(ledger, "data/a.pdf", minute=1)
    _record(ledger, "data/a.pdf", dry_run=True, minute=2)

    assert ledger.latest_by_path()["data/a.pdf"].status == "ingested"


def test_history_is_newest_first():
    ledger = IngestionLedger(":memory:")
    _record(ledger, "data/a.pdf", minute=1)
    _record(ledger, "data/b.pdf", minute=2)

    assert [entry.path for entry in ledger.history()] == ["data/b.pdf", "data/a.pdf"]
