import tempfile
import unittest
from pathlib import Path

from ingestion.refresh import RetrievedSource, SourceSnapshotStore
from ingestion.source_registry import SourceSpec


class RefreshContractTests(unittest.TestCase):
    def test_in_memory_snapshot_store_round_trips_hash(self):
        source = SourceSpec(
            "bns",
            "BNS",
            "https://indiacode.nic.in/bns.pdf",
            "India Code",
            "india_code",
            "statute",
        )
        retrieved = RetrievedSource(source, b"source", "hash", "2026-10-21T00:00:00Z")
        store = SourceSnapshotStore(":memory:")

        store.record_if_changed(retrieved)

        self.assertEqual(store.latest_hash("bns"), "hash")

    def test_snapshot_store_deduplicates_unchanged_content(self):
        source = SourceSpec(
            "bns",
            "BNS",
            "https://indiacode.nic.in/bns.pdf",
            "India Code",
            "india_code",
            "statute",
        )
        retrieved = RetrievedSource(source, b"source", "hash", "2026-10-21T00:00:00Z")

        with tempfile.TemporaryDirectory() as directory:
            store = SourceSnapshotStore(Path(directory) / "refresh.sqlite3")

            self.assertTrue(store.record_if_changed(retrieved))
            self.assertFalse(store.record_if_changed(retrieved))
            self.assertEqual(store.latest_hash("bns"), "hash")

    def test_retrieved_source_has_content_hash(self):
        source = SourceSpec(
            "bns",
            "BNS",
            "https://indiacode.nic.in/bns.pdf",
            "India Code",
            "india_code",
            "statute",
        )
        retrieved = RetrievedSource(
            source=source,
            content=b"source",
            content_hash="hash",
            retrieved_at="2026-10-21T00:00:00Z",
        )

        self.assertEqual(retrieved.source.source_id, "bns")
