from __future__ import annotations

import json
from datetime import date

import pytest

from legal.acts import provision_status
from legal.catalog_builder import build_catalog
from legal.crosswalk import CrosswalkStore, base_section, row_id
from legal.store import DocumentStore
from legal.timeline import build_timeline


@pytest.fixture
def store(tmp_path):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"note": "test", "rows": [
        {"from_code": "IPC", "from_section": "302", "from_heading": "Punishment for murder",
         "to_code": "BNS", "to_section": "103(1)", "to_heading": "Punishment for murder",
         "relation": "equivalent", "note": ""},
        {"from_code": "IPC", "from_section": "377", "from_heading": "Unnatural offences",
         "to_code": None, "to_section": None, "to_heading": None, "relation": "omitted", "note": "No equivalent."},
        {"from_code": None, "from_section": None, "from_heading": None,
         "to_code": "BNS", "to_section": "111", "to_heading": "Organised crime", "relation": "new", "note": ""},
    ]}))
    return CrosswalkStore(":memory:", seed_path=seed)


def test_base_section_and_row_id():
    assert base_section("103(1)") == "103"
    assert base_section("65b") == "65B"
    assert row_id({"from_code": "IPC", "from_section": "302", "to_code": "BNS", "to_section": "103(1)"}) == "ipc-302-to-bns-103-1"


@pytest.mark.parametrize("query", ["IPC 302", "ipc s.302", "302", "section 302", "Punishment for murder"])
def test_search_by_section_or_heading(store, query):
    assert [row.id for row in store.search(query)] == ["ipc-302-to-bns-103-1"]


def test_search_by_new_code_section_matches_sub_sections(store):
    assert [row.to_section for row in store.search("BNS 103")] == ["103(1)"]


def test_seed_never_overwrites_verification(store, tmp_path):
    store.set_verified("ipc-302-to-bns-103-1", verified=True, by="Administrator")
    store.seed(tmp_path / "seed.json")

    row = store.get("ipc-302-to-bns-103-1")
    assert row.verified is True
    assert row.verified_by == "Administrator"


def test_provision_status_handles_repeal_and_exceptions():
    assert provision_status("IPC", "302", on=date(2024, 6, 30)) == "in_force"
    assert provision_status("IPC", "302", on=date(2024, 7, 1)) == "repealed"
    assert provision_status("BNS", "103", on=date(2024, 6, 30)) == "not_yet_in_force"
    assert provision_status("BNS", "106", on=date(2025, 1, 1)) == "partly_in_force"
    assert provision_status("BNS", "106(2)", on=date(2025, 1, 1)) == "not_in_force"


def test_timeline_links_old_and_new_provisions(store):
    timeline = build_timeline("BNS", "103", crosswalk=store, on=date(2025, 1, 1))

    kinds = [event["kind"] for event in timeline["events"]]
    assert timeline["status"] == "in_force"
    assert timeline["heading"] == "Punishment for murder"
    assert kinds[0] == "predecessor"
    assert "predecessor_repealed" in kinds and "notified" in kinds
    notified = next(event for event in timeline["events"] if event["kind"] == "notified")
    assert notified["instrument"] == "Notification S.O. 850(E)"
    assert [event["date"] for event in timeline["events"]] == sorted(event["date"] for event in timeline["events"])


def test_timeline_for_repealed_and_omitted_sections(store):
    replaced = build_timeline("IPC", "302", crosswalk=store, on=date(2025, 1, 1))
    omitted = build_timeline("IPC", "377", crosswalk=store, on=date(2025, 1, 1))

    assert replaced["status"] == "repealed"
    assert any(e["kind"] == "replaced" and e["section"] == "103(1)" for e in replaced["events"])
    assert any(e["kind"] == "omitted" for e in omitted["events"])


def test_timeline_marks_excluded_sub_section(store):
    timeline = build_timeline("BNS", "106", crosswalk=store, on=date(2025, 1, 1))

    assert timeline["status"] == "partly_in_force"
    assert any(e["kind"] == "exception" and e["section"] == "106(2)" for e in timeline["events"])


def test_build_catalog_skips_missing_and_empty_pdfs(tmp_path):
    bns = tmp_path / "primary_statutes" / "current" / "bns"
    bns.mkdir(parents=True)
    (bns / "bns_2023.pdf").write_bytes(b"%PDF")
    ipc = tmp_path / "primary_statutes" / "historical" / "ipc"
    ipc.mkdir(parents=True)
    (ipc / "ipc_1860.pdf").write_bytes(b"")
    pages = ["Murder. 1. Whoever. \nTheft. 2. Whoever.\nCheating. 3. Whoever.\nExtortion. 4. Whoever.\nRobbery. 5. Whoever."]
    catalog = DocumentStore(":memory:")

    results = {r.code: r for r in build_catalog(tmp_path, catalog, codes=["BNS", "IPC", "BSA"], read_pages=lambda _: pages)}

    assert results["BNS"].status == "built" and results["BNS"].sections == 5
    assert results["IPC"].status == "skipped" and "empty" in results["IPC"].reason
    assert results["BSA"].status == "skipped" and "missing" in results["BSA"].reason
    document = catalog.get("bns-2023")
    assert document.provisions[0].heading == "Murder"
    assert document.provisions[0].effective_from == date(2024, 7, 1)
