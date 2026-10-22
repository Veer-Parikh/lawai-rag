from __future__ import annotations

import json

from ingestion.change_watch import ChangeLog, check_sources, fingerprint, watched_sources
from ingestion.refresh import RetrievedSource, SourceSnapshotStore
from ingestion.source_registry import SourceSpec

SOURCE = SourceSpec("bns-gazette", "BNS Gazette", "https://egazette.gov.in/bns", "Gazette", "gazette", "statute")


def test_fingerprint_ignores_markup_churn_but_not_text_changes():
    first = b"<html><script>var t=1;</script><body>Notification S.O. 850(E)</body></html>"
    churned = b"<html><script>var t=2;</script><body>  Notification   S.O. 850(E) </body></html>"
    amended = b"<html><body>Notification S.O. 851(E)</body></html>"

    assert fingerprint(first) == fingerprint(churned)
    assert fingerprint(first) != fingerprint(amended)


def test_check_sources_reports_new_unchanged_changed_and_failed():
    snapshots, log = SourceSnapshotStore(":memory:"), ChangeLog(":memory:")
    bodies = iter([b"version one", b"version one", b"version two"])

    def fetch(source):
        return RetrievedSource(source, next(bodies), "raw", "2026-10-21T00:00:00+00:00")

    def failing(_source):
        raise OSError("HTTP Error 403: Forbidden")

    outcomes = [check_sources([SOURCE], snapshots=snapshots, log=log, fetch=fetch) for _ in range(3)]
    failed = check_sources([SOURCE], snapshots=snapshots, log=log, fetch=failing)

    assert [next(k for k, v in o.items() if v) for o in outcomes] == ["new", "unchanged", "changed"]
    assert failed["failed"] == 1
    assert [event.outcome for event in log.feed()] == ["changed", "new"]
    latest = log.latest_by_source()["bns-gazette"]
    assert latest["last_check"]["outcome"] == "failed"
    assert latest["last_change"]["outcome"] == "changed"


def test_watched_sources_keeps_authoritative_hosts_only(tmp_path):
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps([
        {"source_id": "a", "title": "Gazette", "url": "https://egazette.gov.in/", "authority": "gazette"},
        {"source_id": "b", "title": "PIB", "url": "https://pib.gov.in/", "authority": "pib"},
        {"source_id": "c", "title": "Fake", "url": "https://evil.example/", "authority": "gazette"},
    ]))

    watched, skipped = watched_sources(manifest)

    assert [s.source_id for s in watched] == ["a"]
    assert {row["source_id"] for row in skipped} == {"b", "c"}
