"""Build the legal catalog (one versioned document per Act) from the corpus PDFs.

Usage:
    python -m legal.catalog_builder            # build every Act whose PDF exists
    python -m legal.catalog_builder BNS IEA    # build selected Acts
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import config
from legal.acts import ACTS, Act
from legal.documents import DocumentStatus, LegalDocument, ProvisionVersion
from legal.provisions import parse_sections
from legal.store import DocumentStore
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BuildResult:
    code: str
    status: str            # "built" | "skipped" | "failed"
    sections: int = 0
    reason: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _read_pages(pdf_path: Path) -> list[str]:
    from ingestion.pdf_loader import load_pdf

    return [page["raw_text"] for page in load_pdf(pdf_path)]


def build_document(act: Act, pages: list[str], *, document_hash: str | None = None) -> LegalDocument:
    sections = parse_sections(pages)
    if not sections:
        raise ValueError("no numbered sections could be recognised in the text")
    provisions = [
        ProvisionVersion(
            provision_id=f"{act.document_id}-s{section.number}",
            section_number=section.number,
            heading=section.heading or "",
            text=section.text,
            effective_from=act.in_force_from,
            effective_until=act.repealed_on,
            status=DocumentStatus.IN_FORCE,
            page_anchor=f"page {section.page}",
        )
        for section in sections
    ]
    return LegalDocument(
        document_id=act.document_id,
        title=act.title,
        act_code=act.code,
        source_url=act.source_url,
        publisher=act.publisher,
        status=DocumentStatus.REPEALED if act.repealed_on else DocumentStatus.IN_FORCE,
        provisions=provisions,
        publication_date=act.assented_on,
        assent_date=act.assented_on,
        effective_date=act.in_force_from,
        document_hash=document_hash,
        source_authority="india_code" if "historical" in act.pdf_path else "gazette",
    )


def build_catalog(
    data_dir: str | Path,
    store: DocumentStore,
    *,
    codes: list[str] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    read_pages: Callable[[Path], list[str]] = _read_pages,
) -> list[BuildResult]:
    acts = [act for act in ACTS.values() if not codes or act.code in codes]
    results: list[BuildResult] = []
    for position, act in enumerate(acts):
        if on_progress:
            on_progress(position, len(acts), act.code)
        pdf_path = Path(data_dir) / act.pdf_path
        if not pdf_path.exists():
            results.append(BuildResult(act.code, "skipped", reason=f"{act.pdf_path} is missing"))
            continue
        if pdf_path.stat().st_size == 0:
            results.append(BuildResult(act.code, "skipped", reason=f"{act.pdf_path} is empty"))
            continue
        try:
            document = build_document(
                act,
                read_pages(pdf_path),
                document_hash=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
            )
            store.upsert(document)
            results.append(BuildResult(act.code, "built", sections=len(document.provisions)))
            logger.info(f"Catalog: {act.code} built with {len(document.provisions)} sections")
        except Exception as exc:
            logger.warning(f"Catalog: {act.code} failed: {exc}")
            results.append(BuildResult(act.code, "failed", reason=str(exc)))
    if on_progress:
        on_progress(len(acts), len(acts), "")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the LawAI legal catalog from corpus PDFs")
    parser.add_argument("codes", nargs="*", help="Act codes to build (default: all)")
    args = parser.parse_args()
    for result in build_catalog(config.DATA_DIR, DocumentStore(config.CATALOG_PATH), codes=args.codes or None):
        print(f"{result.code:5} {result.status:8} {result.sections:4} {result.reason or ''}")
