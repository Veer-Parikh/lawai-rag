"""
api/routers/auth.py — One-click demo login, admin password login, session lookup.
"""
# No `from __future__ import annotations`: the slowapi decorator would hide
# this module's globals from FastAPI's annotation resolution.

import hashlib
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request

import config
from api.auth import Principal, check_admin_password, issue_token, require_principal
from api.middleware import limiter
from api.schemas.auth_schema import AdminLoginRequest, DemoLoginRequest, SessionResponse
from utils.logger import get_logger

router = APIRouter(prefix="/auth")
logger = get_logger(__name__)


def _session(token: str, principal: Principal) -> SessionResponse:
    return SessionResponse(
        token=token,
        name=principal.name,
        role=principal.role,
        expires_at=principal.expires_at,
    )


@router.post("/demo", response_model=SessionResponse)
async def demo_login(body: DemoLoginRequest | None = None) -> SessionResponse:
    if not config.DEMO_LOGIN_ENABLED:
        raise HTTPException(status_code=403, detail="Demo access is turned off")
    device = body.device_id if body and body.device_id else None
    suffix = hashlib.sha256(device.encode()).hexdigest()[:12] if device else secrets.token_hex(6)
    token, principal = issue_token(f"demo-{suffix}", "Demo researcher", "researcher")
    return _session(token, principal)


@router.post("/admin", response_model=SessionResponse)
@limiter.limit("5/minute")
async def admin_login(request: Request, body: AdminLoginRequest) -> SessionResponse:
    if not config.ADMIN_PASSWORD:
        raise HTTPException(
            status_code=503,
            detail="Admin sign-in is not configured. Set ADMIN_PASSWORD in the API environment.",
        )
    if not check_admin_password(body.password):
        logger.warning("Rejected admin sign-in attempt")
        raise HTTPException(status_code=401, detail="That password is incorrect")
    token, principal = issue_token("admin", "Administrator", "admin")
    return _session(token, principal)


@router.get("/me", response_model=SessionResponse)
async def me(principal: Principal = Depends(require_principal)) -> SessionResponse:
    return SessionResponse(
        token=None,
        name=principal.name,
        role=principal.role,
        expires_at=principal.expires_at,
    )
