import unittest

from ingestion.source_registry import SourceRegistry, SourceSpec


class SourceRegistryTests(unittest.TestCase):
    def test_rejects_non_authoritative_host(self):
        registry = SourceRegistry(
            allowed_hosts={"indiacode.nic.in", "egazette.nic.in", "mha.gov.in"}
        )

        with self.assertRaises(ValueError):
            registry.validate_url("https://example.com/law.pdf")

    def test_normalizes_source_metadata(self):
        registry = SourceRegistry(allowed_hosts={"indiacode.nic.in"})
        source = registry.register(
            SourceSpec(
                source_id="bns",
                title="Bharatiya Nyaya Sanhita",
                url="https://indiacode.nic.in/bns.pdf",
                publisher="India Code",
                authority="india_code",
                document_type="statute",
            )
        )

        self.assertEqual(source.source_id, "bns")
        self.assertEqual(source.authority, "india_code")
