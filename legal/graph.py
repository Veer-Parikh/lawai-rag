"""Explicit amendment and predecessor/successor relationships."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegalRelation:
    relation_id: str
    from_id: str
    to_id: str
    relation_type: str
    instrument_id: str


class AmendmentGraph:
    def __init__(self) -> None:
        self._relations: dict[str, LegalRelation] = {}

    def add(self, relation: LegalRelation) -> None:
        if relation.relation_id in self._relations:
            raise ValueError(f"duplicate legal relation: {relation.relation_id}")
        if relation.relation_type not in {
            "amended_by",
            "replaced_by",
            "commenced_by",
            "repealed_by",
            "interpreted_by",
        }:
            raise ValueError("unsupported legal relation type")
        self._relations[relation.relation_id] = relation

    def related(self, document_id: str) -> list[str]:
        return [
            relation.to_id
            for relation in self._relations.values()
            if relation.from_id == document_id
        ]
