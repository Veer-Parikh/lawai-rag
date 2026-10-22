import unittest

from observability.metrics import MetricsRegistry


class MetricsRegistryTests(unittest.TestCase):
    def test_prometheus_text_contains_counter_and_histogram(self):
        metrics = MetricsRegistry()
        metrics.increment("http_requests_total", {"route": "/chat"})
        metrics.observe("http_latency_ms", 42, {"route": "/chat"})

        output = metrics.prometheus_text()

        self.assertIn('http_requests_total{route="/chat"} 1', output)
        self.assertIn('http_latency_ms_count{route="/chat"} 1', output)
        self.assertIn('http_latency_ms_sum{route="/chat"} 42', output)
