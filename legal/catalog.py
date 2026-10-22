"""Catalog operations for legal document versions and change timelines."""
from __future__ import annotations

from dataclasses import dataclass

from legal.documents import LegalDocument


@dataclass(frozen=True)
class DocumentDiff:
    from_document_id: str
    to_document_id: str
    changed_sections: list[str]
    added_sections: list[str]
    removed_sections: list[str]


class DocumentCatalog:
    @staticmethod
    def compare(before: LegalDocument, after: LegalDocument) -> DocumentDiff:
        old = {item.section_number: item for item in before.provisions}
        new = {item.section_number: item for item in after.provisions}
        changed = sorted(
            section
            for section in old.keys() & new.keys()
            if old[section].text != new[section].text
            or old[section].heading != new[section].heading
        )
        return DocumentDiff(
            from_document_id=before.document_id,
            to_document_id=after.document_id,
            changed_sections=changed,
            added_sections=sorted(new.keys() - old.keys()),
            removed_sections=sorted(old.keys() - new.keys()),
        )
