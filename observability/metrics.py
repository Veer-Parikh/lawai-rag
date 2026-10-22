"""Dependency-free Prometheus text registry for local and container deployments."""
from __future__ import annotations

from collections import defaultdict
from threading import Lock


class MetricsRegistry:
    def __init__(self) -> None:
        self._counters: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._observations: defaultdict[
            tuple[str, tuple[tuple[str, str], ...]], list[float]
        ] = defaultdict(lambda: [0, 0.0])
        self._lock = Lock()

    @staticmethod
    def _key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((labels or {}).items()))

    def increment(self, name: str, labels: dict[str, str] | None = None, value: float = 1) -> None:
        with self._lock:
            self._counters[(name, self._key(labels))] += value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            observation = self._observations[(name, self._key(labels))]
            observation[0] += 1
            observation[1] += value

    def prometheus_text(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"{name}{self._format_labels(labels)} {value:g}")
            for (name, labels), observation in sorted(self._observations.items()):
                lines.append(f"{name}_count{self._format_labels(labels)} {int(observation[0])}")
                lines.append(f"{name}_sum{self._format_labels(labels)} {observation[1]:g}")
        return "\n".join(lines) + ("\n" if lines else "")

    def snapshot(self) -> dict[str, list[dict]]:
        """JSON-friendly view of every series since process start."""
        with self._lock:
            counters = [
                {"name": name, "labels": dict(labels), "value": value}
                for (name, labels), value in sorted(self._counters.items())
            ]
            observations = [
                {
                    "name": name,
                    "labels": dict(labels),
                    "count": int(observation[0]),
                    "mean": observation[1] / observation[0] if observation[0] else 0.0,
                }
                for (name, labels), observation in sorted(self._observations.items())
            ]
        return {"counters": counters, "observations": observations}

    @staticmethod
    def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
        if not labels:
            return ""
        encoded = ",".join(f'{key}="{value.replace(chr(34), chr(92) + chr(34))}"' for key, value in labels)
        return "{" + encoded + "}"
