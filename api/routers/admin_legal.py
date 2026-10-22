"""
api/routers/admin_legal.py — Admin actions on legal reference data:
verify crosswalk rows and build the provision catalog from corpus PDFs.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

import config
from api.auth import Principal, require_admin
from api.jobs import jobs
from api.routers.legal import catalog_store, crosswalk_store, invalidate_catalog_cache
from api.schemas.legal_schema import CatalogBuildRequest, CrosswalkRowModel, CrosswalkVerifyRequest
from legal.acts import ACTS
from legal.catalog_builder import build_catalog

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


@router.patch("/crosswalk/{row_id}", response_model=CrosswalkRowModel)
async def verify_crosswalk_row(
    row_id: str,
    body: CrosswalkVerifyRequest,
    principal: Principal = Depends(require_admin),
) -> CrosswalkRowModel:
    try:
        row = crosswalk_store().set_verified(row_id, verified=body.verified, by=principal.name, note=body.note)
    except KeyError:
        raise HTTPException(status_code=404, detail="Crosswalk row not found")
    return CrosswalkRowModel(**row.to_dict())


@router.post("/catalog/build")
async def start_catalog_build(
    body: CatalogBuildRequest,
    principal: Principal = Depends(require_admin),
) -> dict:
    unknown = [code for code in body.codes or [] if code not in ACTS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown Act codes: {', '.join(unknown)}")

    def work(progress):
        results = build_catalog(config.DATA_DIR, catalog_store(), codes=body.codes, on_progress=progress)
        invalidate_catalog_cache()
        return [result.to_dict() for result in results]

    try:
        job = jobs.start("catalog_build", work, requested_by=principal.name)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return job.to_dict()


@router.get("/catalog/build")
async def catalog_build_status() -> dict:
    job = jobs.latest("catalog_build")
    return {"job": job.to_dict() if job else None}
