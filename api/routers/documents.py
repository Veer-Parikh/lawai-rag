from __future__ import annotations

from fastapi import APIRouter, HTTPException

import config

from api.schemas.documents_schema import DocumentChangeResponse, DocumentSummary
from legal.catalog import DocumentCatalog
from legal.store import DocumentStore

router = APIRouter(prefix="/documents")
catalog = DocumentCatalog()
store = DocumentStore(config.CATALOG_PATH)


def register_document(document) -> None:
    store.upsert(document)


def _summary(document) -> DocumentSummary:
    return DocumentSummary(
        document_id=document.document_id,
        title=document.title,
        act_code=document.act_code,
        publisher=document.publisher,
        source_url=document.source_url,
        status=document.status.value,
        publication_date=document.publication_date,
        effective_date=document.effective_date,
        source_authority=document.source_authority,
    )


@router.get("", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    return [_summary(document) for document in store.list()]


@router.get("/{document_id}", response_model=DocumentSummary)
async def get_document(document_id: str) -> DocumentSummary:
    try:
        document = store.get(document_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="document version not found")
    return _summary(document)


@router.get(
    "/{from_document_id}/compare/{to_document_id}",
    response_model=DocumentChangeResponse,
)
async def compare_documents(
    from_document_id: str, to_document_id: str
) -> DocumentChangeResponse:
    try:
        before = store.get(from_document_id)
        after = store.get(to_document_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="document version not found")
    return DocumentChangeResponse(**catalog.compare(before, after).__dict__)
