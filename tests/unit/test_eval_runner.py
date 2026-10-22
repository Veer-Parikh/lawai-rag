import unittest

from evals.run_eval import evaluate_case


class EvaluationRunnerTests(unittest.TestCase):
    def test_unanswerable_case_requires_abstention(self):
        result = evaluate_case(
            {
                "id": "unknown",
                "expected_source_ids": [],
                "must_abstain": True,
            },
            [],
            abstained=True,
        )

        self.assertTrue(result["passed"])

    def test_answerable_case_requires_expected_source(self):
        result = evaluate_case(
            {
                "id": "source",
                "expected_source_ids": ["source-1"],
                "must_abstain": False,
            },
            ["source-1"],
        )

        self.assertEqual(result["recall_at_5"], 1.0)
        self.assertTrue(result["passed"])
