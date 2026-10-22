import unittest

from retrieval.hybrid import BM25Retriever


class BM25Tests(unittest.TestCase):
    def test_exact_section_terms_rank_first(self):
        retriever = BM25Retriever(
            [
                {"chunk_id": "302", "text": "Section 302 punishment for murder"},
                {"chunk_id": "304", "text": "Section 304 culpable homicide"},
            ]
        )

        results = retriever.search("Section 302 murder", top_n=1)

        self.assertEqual(results[0]["chunk_id"], "302")
        self.assertEqual(results[0]["source"], "bm25")
