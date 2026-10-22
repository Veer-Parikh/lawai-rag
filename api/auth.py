"""
api/auth.py — Demo-grade session tokens and role dependencies.

Tokens are HMAC-SHA256 signed JSON payloads (stdlib only). There is no user
database: the demo login issues a "researcher" token, and the admin login
issues an "admin" token after checking ADMIN_PASSWORD. Replace with a real
identity provider before multi-tenant use.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Literal

from fastapi import Depends, Header, HTTPException

import config
from utils.logger import get_logger

logger = get_logger(__name__)

Role = Literal["researcher", "admin"]

_SECRET = (config.AUTH_SECRET or secrets.token_hex(32)).encode()
if not config.AUTH_SECRET:
    logger.warning("AUTH_SECRET is not set; using a per-process secret (sessions reset on restart)")


@dataclass(frozen=True)
class Principal:
    subject: str
    name: str
    role: Role
    expires_at: int


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def issue_token(subject: str, name: str, role: Role, *, now: float | None = None) -> tuple[str, Principal]:
    issued = int(now if now is not None else time.time())
    principal = Principal(
        subject=subject,
        name=name,
        role=role,
        expires_at=issued + config.AUTH_TOKEN_TTL_HOURS * 3600,
    )
    payload = _b64(json.dumps(
        {"sub": subject, "name": name, "role": role, "exp": principal.expires_at},
        separators=(",", ":"),
    ).encode())
    signature = _b64(hmac.new(_SECRET, payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{signature}", principal


def verify_token(token: str, *, now: float | None = None) -> Principal:
    try:
        payload, signature = token.split(".", 1)
    except ValueError as exc:
        raise ValueError("malformed token") from exc
    expected = _b64(hmac.new(_SECRET, payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(expected, signature):
        raise ValueError("bad signature")
    claims = json.loads(_unb64(payload))
    if claims.get("role") not in ("researcher", "admin"):
        raise ValueError("unknown role")
    if int(claims["exp"]) < int(now if now is not None else time.time()):
        raise ValueError("token expired")
    return Principal(
        subject=claims["sub"],
        name=claims["name"],
        role=claims["role"],
        expires_at=int(claims["exp"]),
    )


def check_admin_password(candidate: str) -> bool:
    if not config.ADMIN_PASSWORD:
        return False
    return hmac.compare_digest(candidate.encode(), config.ADMIN_PASSWORD.encode())


def _bearer(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def optional_principal(
    authorization: str | None = Header(None),
) -> Principal | None:
    token = _bearer(authorization)
    if not token:
        return None
    try:
        return verify_token(token)
    except ValueError:
        return None


def require_principal(principal: Principal | None = Depends(optional_principal)) -> Principal:
    if principal is None:
        raise HTTPException(status_code=401, detail="Sign in to continue")
    return principal


def require_admin(
    authorization: str | None = Header(None),
    x_admin_key: str | None = Header(None, alias="X-Admin-Key"),
) -> Principal:
    """Accept an admin bearer token, or the legacy X-Admin-Key for scripts."""
    if x_admin_key is not None:
        if config.ADMIN_API_KEY and hmac.compare_digest(x_admin_key, config.ADMIN_API_KEY):
            return Principal(subject="api-key", name="API key", role="admin", expires_at=0)
        raise HTTPException(status_code=403, detail="Invalid admin key")
    principal = optional_principal(authorization)
    if principal is None:
        raise HTTPException(status_code=401, detail="Sign in as an administrator")
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    return principal
