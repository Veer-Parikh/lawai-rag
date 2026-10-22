import unittest
from datetime import date

from legal.documents import (
    AmendmentRelation,
    DocumentStatus,
    LegalDocument,
    ProvisionVersion,
)


class LegalDocumentTests(unittest.TestCase):
    def setUp(self):
        self.document = LegalDocument(
            document_id="bns-2023",
            title="Bharatiya Nyaya Sanhita, 2023",
            act_code="BNS",
            source_url="https://example.test/bns",
            publisher="India Code",
            status=DocumentStatus.IN_FORCE,
            provisions=[
                ProvisionVersion(
                    provision_id="bns-1",
                    section_number="1",
                    heading="Short title",
                    text="This Act may be called the Bharatiya Nyaya Sanhita, 2023.",
                    effective_from=date(2024, 7, 1),
                )
            ],
        )

    def test_finds_provision_in_force_on_date(self):
        provision = self.document.provision_at("1", date(2024, 7, 1))

        self.assertEqual(provision.provision_id, "bns-1")

    def test_missing_provision_is_explicit(self):
        with self.assertRaises(KeyError):
            self.document.provision_at("302", date(2024, 7, 1))

    def test_amendment_relation_is_serializable(self):
        relation = AmendmentRelation(
            relation_id="amend-1",
            from_document_id="ipc-1860",
            to_document_id="bns-2023",
            relation_type="replaced_by",
            instrument_id="gazette-1",
        )

        self.assertEqual(relation.relation_type, "replaced_by")
