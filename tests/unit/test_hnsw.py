import unittest

from retrieval.hnsw import HnswRetriever


class HnswAdapterTests(unittest.TestCase):
    def test_rejects_invalid_dimension_before_loading_provider(self):
        with self.assertRaises(ValueError):
            HnswRetriever("missing.index", dimension=0)
