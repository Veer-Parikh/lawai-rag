"""
api/routers/legal.py — Crosswalk explorer, provision text and provision timelines.

Read-only and public, like /documents. Writes (verification, catalog builds)
live under /admin.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date

from fastapi import APIRouter, HTTPException, Query

import config
from api.schemas.legal_schema import (
    ActSummary,
    CrosswalkResponse,
    ProvisionResponse,
    TimelineResponse,
)
from legal.acts import ACTS, get_act, provision_status
from legal.crosswalk import SEED_PATH, CrosswalkStore, base_section
from legal.documents import LegalDocument
from legal.store import DocumentStore
from legal.timeline import STATUS_LABELS, build_timeline

router = APIRouter()

_lock = threading.Lock()
_crosswalk: CrosswalkStore | None = None
_catalog_cache: dict[str, LegalDocument | None] = {}


def crosswalk_store() -> CrosswalkStore:
    global _crosswalk
    with _lock:
        if _crosswalk is None or str(_crosswalk.path) != config.CROSSWALK_PATH:
            _crosswalk = CrosswalkStore(config.CROSSWALK_PATH)
        return _crosswalk


def catalog_store() -> DocumentStore:
    return DocumentStore(config.CATALOG_PATH)


def invalidate_catalog_cache() -> None:
    with _lock:
        _catalog_cache.clear()


def _catalog_document(document_id: str) -> LegalDocument | None:
    with _lock:
        if document_id in _catalog_cache:
            return _catalog_cache[document_id]
    try:
        document = catalog_store().get(document_id)
    except KeyError:
        document = None
    with _lock:
        _catalog_cache[document_id] = document
    return document


def _catalog_built_at(document_id: str) -> str | None:
    connection = sqlite3.connect(config.CATALOG_PATH)
    try:
        row = connection.execute(
            "SELECT updated_at FROM legal_documents WHERE document_id = ?", (document_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        row = None
    finally:
        connection.close()
    return row[0] if row else None


class _CachedCatalog:
    """Catalog reader backed by the in-process document cache."""

    def get(self, document_id: str) -> LegalDocument:
        document = _catalog_document(document_id)
        if document is None:
            raise KeyError(document_id)
        return document


def _act_or_404(code: str):
    try:
        return get_act(code)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown Act code '{code}'. Use one of: {', '.join(ACTS)}")


@router.get("/acts", response_model=list[ActSummary])
async def list_acts() -> list[ActSummary]:
    summaries = []
    for act in ACTS.values():
        document = _catalog_document(act.document_id)
        summaries.append(ActSummary(
            code=act.code,
            title=act.title,
            act_number=act.act_number,
            status=act.status,
            in_force_from=act.in_force_from.isoformat(),
            repealed_on=act.repealed_on.isoformat() if act.repealed_on else None,
            repealed_by=act.repealed_by,
            successor=act.successor,
            predecessor=act.predecessor,
            commencement=act.commencement.label if act.commencement else None,
            not_in_force=list(act.not_in_force),
            sections_in_catalog=len(document.provisions) if document else 0,
            catalog_built_at=_catalog_built_at(act.document_id) if document else None,
        ))
    return summaries


@router.get("/crosswalk", response_model=CrosswalkResponse)
async def crosswalk(
    q: str = Query("", max_length=120, description="Section ('IPC 302', '302') or words from a heading"),
    code: str | None = Query(None, description="Limit to rows touching this Act code"),
    relation: str | None = Query(None, description="equivalent, modified, omitted or new"),
) -> CrosswalkResponse:
    store = crosswalk_store()
    rows = store.search(q, code=code, relation=relation)
    everything = store.all()
    note = json.loads(SEED_PATH.read_text(encoding="utf-8")).get("note", "")
    return CrosswalkResponse(
        note=note,
        total=len(everything),
        verified=sum(1 for row in everything if row.verified),
        rows=[row.to_dict() for row in rows],
    )


@router.get("/provisions/{code}/{section}", response_model=ProvisionResponse)
async def provision(code: str, section: str) -> ProvisionResponse:
    act = _act_or_404(code)
    document = _catalog_document(act.document_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"The {act.code} text is not in the catalog yet. An administrator can build it from Corpus and ingestion.",
        )
    wanted = base_section(section)
    match = next((p for p in document.provisions if p.section_number.upper() == wanted), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Section {wanted} was not found in the {act.code} text.")
    status = provision_status(act.code, section.upper())
    return ProvisionResponse(
        code=act.code,
        section=match.section_number,
        act_title=act.title,
        heading=match.heading or None,
        text=match.text,
        page_anchor=match.page_anchor,
        status=status,
        status_label=STATUS_LABELS[status],
        source_url=act.source_url,
        extracted_from="Gazette of India PDF" if document.source_authority == "gazette" else "India Code PDF",
    )


@router.get("/timeline/{code}/{section}", response_model=TimelineResponse)
async def timeline(code: str, section: str, on: date | None = Query(None, description="Status as on this date")) -> TimelineResponse:
    act = _act_or_404(code)
    result = build_timeline(act.code, section, crosswalk=crosswalk_store(), catalog=_CachedCatalog(), on=on)
    return TimelineResponse(**result)
