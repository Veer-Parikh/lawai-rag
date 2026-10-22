import unittest

from legal.graph import AmendmentGraph, LegalRelation


class AmendmentGraphTests(unittest.TestCase):
    def test_returns_related_versions_in_order(self):
        graph = AmendmentGraph()
        graph.add(
            LegalRelation(
                relation_id="r1",
                from_id="ipc-302",
                to_id="bns-103",
                relation_type="replaced_by",
                instrument_id="gazette-1",
            )
        )

        self.assertEqual(graph.related("ipc-302"), ["bns-103"])

    def test_rejects_duplicate_relation_ids(self):
        graph = AmendmentGraph()
        relation = LegalRelation("r1", "a", "b", "amended_by", "instrument")
        graph.add(relation)
        with self.assertRaises(ValueError):
            graph.add(relation)
