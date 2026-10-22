import unittest

from observability.eval_metrics import mean_reciprocal_rank, recall_at_k


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_at_k_counts_relevant_documents(self):
        value = recall_at_k(
            retrieved_ids=["a", "b", "c"],
            relevant_ids={"b", "d"},
            k=2,
        )

        self.assertEqual(value, 0.5)

    def test_mean_reciprocal_rank_counts_queries_without_hits_as_zero(self):
        value = mean_reciprocal_rank(
            retrieved_batches=[["x", "b"], ["z"], ["a", "b"]],
            relevant_batches=[{"b"}, {"q"}, {"a"}],
        )

        self.assertAlmostEqual(value, (0.5 + 0.0 + 1.0) / 3)
