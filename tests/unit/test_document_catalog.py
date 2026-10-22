import unittest
from datetime import date

from legal.catalog import DocumentCatalog
from legal.documents import DocumentStatus, LegalDocument, ProvisionVersion


class DocumentCatalogTests(unittest.TestCase):
    def test_compare_returns_added_removed_and_changed_provisions(self):
        before = LegalDocument(
            document_id="old",
            title="Old Act",
            act_code="OLD",
            source_url="https://example.test/old",
            publisher="Gazette",
            status=DocumentStatus.SUPERSEDED,
            provisions=[
                ProvisionVersion("1", "1", "One", "old text", date(2020, 1, 1)),
                ProvisionVersion("2", "2", "Two", "removed", date(2020, 1, 1)),
            ],
        )
        after = LegalDocument(
            document_id="new",
            title="New Act",
            act_code="NEW",
            source_url="https://example.test/new",
            publisher="India Code",
            status=DocumentStatus.IN_FORCE,
            provisions=[
                ProvisionVersion("1-new", "1", "One", "new text", date(2024, 7, 1)),
                ProvisionVersion("3", "3", "Three", "added", date(2024, 7, 1)),
            ],
        )

        result = DocumentCatalog().compare(before, after)

        self.assertEqual(result.changed_sections, ["1"])
        self.assertEqual(result.removed_sections, ["2"])
        self.assertEqual(result.added_sections, ["3"])
