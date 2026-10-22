"""Allowlisted authoritative-source registry and refresh metadata."""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    title: str
    url: str
    publisher: str
    authority: str
    document_type: str
    language: str = "en"


class SourceRegistry:
    def __init__(self, *, allowed_hosts: set[str]) -> None:
        self.allowed_hosts = {host.lower() for host in allowed_hosts}
        self._sources: dict[str, SourceSpec] = {}

    def validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in self.allowed_hosts:
            raise ValueError("source URL must be HTTPS and on an allowlisted authority host")

    def register(self, source: SourceSpec) -> SourceSpec:
        self.validate_url(source.url)
        if source.authority not in {"gazette", "india_code", "mha", "supreme_court"}:
            raise ValueError("unsupported source authority")
        self._sources[source.source_id] = source
        return source

    def get(self, source_id: str) -> SourceSpec:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise KeyError(f"source not registered: {source_id}") from exc

    def all(self) -> list[SourceSpec]:
        return list(self._sources.values())
