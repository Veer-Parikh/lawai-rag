"""
api/routers/matters.py — A researcher's matters: saved answers, notes, and memo export.

Every route is scoped to the signed-in subject; one researcher never sees
another's matters.
"""
from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

import config
from api.auth import Principal, require_principal
from workspace.matters import MatterStore
from workspace.memo import render_markdown

router = APIRouter(prefix="/matters")


class MatterCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    reference: str = Field("", max_length=120)
    notes: str = Field("", max_length=5000)


class MatterUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=200)
    reference: str | None = Field(None, max_length=120)
    notes: str | None = Field(None, max_length=5000)


class EntryCreate(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    as_of: str | None = Field(None, max_length=10)
    answer: dict[str, Any]
    note: str = Field("", max_length=5000)


class EntryUpdate(BaseModel):
    note: str = Field(..., max_length=5000)


def _store() -> MatterStore:
    return MatterStore(config.WORKSPACE_PATH)


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Matter not found")


@router.get("")
async def list_matters(principal: Principal = Depends(require_principal)) -> list[dict]:
    return [matter.to_dict() for matter in _store().list(principal.subject)]


@router.post("", status_code=201)
async def create_matter(body: MatterCreate, principal: Principal = Depends(require_principal)) -> dict:
    try:
        return _store().create(principal.subject, title=body.title, reference=body.reference, notes=body.notes).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{matter_id}")
async def get_matter(matter_id: str, principal: Principal = Depends(require_principal)) -> dict:
    try:
        return _store().get(principal.subject, matter_id).to_dict()
    except KeyError:
        raise _not_found()


@router.patch("/{matter_id}")
async def update_matter(matter_id: str, body: MatterUpdate, principal: Principal = Depends(require_principal)) -> dict:
    try:
        return _store().update(principal.subject, matter_id, **body.model_dump()).to_dict()
    except KeyError:
        raise _not_found()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/{matter_id}", status_code=204)
async def delete_matter(matter_id: str, principal: Principal = Depends(require_principal)) -> None:
    try:
        _store().delete(principal.subject, matter_id)
    except KeyError:
        raise _not_found()


@router.post("/{matter_id}/entries", status_code=201)
async def add_entry(matter_id: str, body: EntryCreate, principal: Principal = Depends(require_principal)) -> dict:
    try:
        entry = _store().add_entry(principal.subject, matter_id, question=body.question, answer=body.answer,
                                   as_of=body.as_of, note=body.note)
    except KeyError:
        raise _not_found()
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    return asdict(entry)


@router.patch("/{matter_id}/entries/{entry_id}")
async def update_entry(matter_id: str, entry_id: str, body: EntryUpdate,
                       principal: Principal = Depends(require_principal)) -> dict:
    try:
        return asdict(_store().update_entry(principal.subject, matter_id, entry_id, note=body.note))
    except KeyError:
        raise HTTPException(status_code=404, detail="Saved answer not found")


@router.delete("/{matter_id}/entries/{entry_id}", status_code=204)
async def delete_entry(matter_id: str, entry_id: str, principal: Principal = Depends(require_principal)) -> None:
    try:
        _store().delete_entry(principal.subject, matter_id, entry_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Saved answer not found")


@router.get("/{matter_id}/memo.md", response_class=PlainTextResponse)
async def memo_markdown(matter_id: str, principal: Principal = Depends(require_principal)) -> PlainTextResponse:
    try:
        matter = _store().get(principal.subject, matter_id)
    except KeyError:
        raise _not_found()
    filename = re.sub(r"[^a-z0-9]+", "-", matter.title.lower()).strip("-") or "memo"
    return PlainTextResponse(
        render_markdown(matter, prepared_by=principal.name),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}-memo.md"'},
    )
