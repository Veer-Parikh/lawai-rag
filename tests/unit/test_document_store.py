import tempfile
import unittest
from datetime import date
from pathlib import Path

from legal.documents import DocumentStatus, LegalDocument, ProvisionVersion
from legal.store import DocumentStore


class DocumentStoreTests(unittest.TestCase):
    def test_in_memory_store_round_trips_documents(self):
        store = DocumentStore(":memory:")
        document = LegalDocument(
            document_id="memory",
            title="Memory document",
            act_code="MEM",
            source_url="https://example.test/memory",
            publisher="Test",
            status=DocumentStatus.IN_FORCE,
            provisions=[],
        )

        store.upsert(document)

        self.assertEqual(store.get("memory").title, document.title)

    def test_upsert_and_reload_document_from_sqlite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.sqlite3"
            document = LegalDocument(
                document_id="bns-2023",
                title="Bharatiya Nyaya Sanhita, 2023",
                act_code="BNS",
                source_url="https://example.test/bns",
                publisher="India Code",
                status=DocumentStatus.IN_FORCE,
                effective_date=date(2024, 7, 1),
                provisions=[
                    ProvisionVersion(
                        "bns-1", "1", "Short title", "text", date(2024, 7, 1)
                    )
                ],
            )

            DocumentStore(path).upsert(document)
            loaded = DocumentStore(path).get("bns-2023")

            self.assertEqual(loaded.title, document.title)
            self.assertEqual(loaded.provisions[0].section_number, "1")
            self.assertEqual(loaded.effective_date, date(2024, 7, 1))

    def test_missing_document_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyError):
                DocumentStore(Path(directory) / "catalog.sqlite3").get("missing")
