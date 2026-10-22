import unittest
from datetime import date

from retrieval.service import (
    HybridRetrievalService,
    RetrievalQuery,
    retrieve_hybrid_candidates,
)


class HybridRetrievalServiceTests(unittest.TestCase):
    def test_adapter_fuses_provider_candidates_with_lexical_results(self):
        candidates = [
            {"chunk_id": "lexical", "text": "BNS commencement notification", "score": 0.2},
            {"chunk_id": "dense", "text": "new criminal code", "score": 0.9},
        ]

        results = retrieve_hybrid_candidates("BNS commencement", candidates, top_n=2)

        self.assertEqual({item["chunk_id"] for item in results}, {"lexical", "dense"})
        self.assertGreaterEqual(results[0]["rrf_score"], results[1]["rrf_score"])

    def test_filters_stale_candidates_before_fusion(self):
        documents = [
            {
                "chunk_id": "current",
                "text": "BNS section 103 punishment",
                "legal_status": "in_force",
                "effective_date": "2024-07-01",
            },
            {
                "chunk_id": "stale",
                "text": "IPC section 302 punishment",
                "legal_status": "superseded",
                "effective_date": "1860-01-01",
            },
        ]
        service = HybridRetrievalService(
            documents,
            dense_search=lambda _query: [
                {"chunk_id": "stale", "score": 0.99, "source": "hnsw"},
                {"chunk_id": "current", "score": 0.80, "source": "hnsw"},
            ],
        )

        results = service.search(
            RetrievalQuery(text="punishment", as_of=date(2025, 1, 1), top_n=5)
        )

        self.assertEqual([item["chunk_id"] for item in results], ["current"])

    def test_query_rejects_invalid_top_n(self):
        with self.assertRaises(ValueError):
            RetrievalQuery(text="section 1", top_n=0)
