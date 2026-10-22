"""
api/schemas/auth_schema.py — Pydantic models for /auth endpoints.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DemoLoginRequest(BaseModel):
    # A random ID the browser keeps, so a demo researcher gets their own
    # matters back on the next visit. It is hashed before use.
    device_id: str | None = Field(None, min_length=8, max_length=128)


class AdminLoginRequest(BaseModel):
    password: str = Field(..., min_length=1, max_length=256)


class SessionResponse(BaseModel):
    token: str | None = None
    name: str
    role: Literal["researcher", "admin"]
    expires_at: int
