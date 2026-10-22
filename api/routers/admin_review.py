"""
api/routers/admin_review.py — Review queue triage and the eval datasets it feeds.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

import config
from api.auth import Principal, require_admin
from observability.review_queue import ReviewQueue

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


class ResolveRequest(BaseModel):
    status: Literal["open", "confirmed", "dismissed"]
    reviewer_note: str | None = Field(None, max_length=1000)
    expected_sections: list[str] | None = Field(None, max_length=20)
    must_abstain: bool | None = None


def _queue() -> ReviewQueue:
    return ReviewQueue(config.REVIEW_QUEUE_PATH)


@router.get("/review")
async def review_items(
    status: Literal["open", "confirmed", "dismissed", "promoted"] | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    queue = _queue()
    return {"counts": queue.counts(), "items": [item.to_dict() for item in queue.list(status, limit)]}


@router.patch("/review/{item_id}")
async def resolve_item(item_id: int, body: ResolveRequest, principal: Principal = Depends(require_admin)) -> dict:
    try:
        item = _queue().resolve(
            item_id, status=body.status, by=principal.name, note=body.reviewer_note,
            expected_sections=body.expected_sections, must_abstain=body.must_abstain,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Review item not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return item.to_dict()


@router.post("/review/{item_id}/promote")
async def promote_item(item_id: int, principal: Principal = Depends(require_admin)) -> dict:
    try:
        case = _queue().promote(item_id, dataset_path=config.REVIEWED_EVAL_DATASET, by=principal.name)
    except KeyError:
        raise HTTPException(status_code=404, detail="Review item not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"eval_case": case, "dataset": Path(config.REVIEWED_EVAL_DATASET).name}


@router.get("/evals")
async def eval_datasets() -> dict:
    """Versioned eval datasets with case counts by category."""
    folder = Path(config.REVIEWED_EVAL_DATASET).parent
    datasets = []
    for path in sorted(folder.glob("*.json")):
        try:
            cases = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(cases, list):
            continue
        datasets.append({
            "name": path.name,
            "cases": len(cases),
            "abstention_cases": sum(1 for case in cases if case.get("must_abstain")),
            "categories": dict(Counter(case.get("category", "uncategorised") for case in cases).most_common()),
            "from_review": sum(1 for case in cases if (case.get("provenance") or {}).get("source") == "review-queue"),
        })
    return {"datasets": datasets}
