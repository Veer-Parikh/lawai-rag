import unittest

from observability.tracing import TraceRecorder


class TraceRecorderTests(unittest.TestCase):
    def test_span_records_request_identity_and_duration(self):
        recorder = TraceRecorder()

        with recorder.span(
            "retrieval.bm25",
            trace_id="request-1",
            attributes={"corpus_snapshot": "2026-10-19"},
        ):
            pass

        [record] = recorder.snapshot()
        self.assertEqual(record.trace_id, "request-1")
        self.assertEqual(record.name, "retrieval.bm25")
        self.assertEqual(record.attributes["corpus_snapshot"], "2026-10-19")
        self.assertIsNotNone(record.duration_ms)
