import unittest

from retrieval.hybrid import reciprocal_rank_fusion


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_fuses_rankings_and_preserves_metadata(self):
        lexical = [
            {"chunk_id": "section-302", "score": 12.0, "source": "bm25"},
            {"chunk_id": "section-304", "score": 9.0, "source": "bm25"},
        ]
        dense = [
            {"chunk_id": "section-304", "score": 0.91, "source": "hnsw"},
            {"chunk_id": "section-302", "score": 0.88, "source": "hnsw"},
        ]

        result = reciprocal_rank_fusion([lexical, dense], k=60, top_n=2)

        self.assertEqual([item["chunk_id"] for item in result], ["section-302", "section-304"])
        self.assertEqual(result[0]["retrieval_stages"], ["bm25", "hnsw"])

    def test_empty_rankings_return_empty_list(self):
        self.assertEqual(reciprocal_rank_fusion([], k=60, top_n=5), [])
