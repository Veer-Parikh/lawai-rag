from __future__ import annotations

import pytest

from api.auth import issue_token, verify_token


def test_token_round_trip_preserves_role():
    token, principal = issue_token("demo-1", "Demo researcher", "researcher", now=1_000)
    verified = verify_token(token, now=1_001)
    assert verified == principal
    assert verified.role == "researcher"


def test_tampered_token_is_rejected():
    token, _ = issue_token("demo-1", "Demo researcher", "researcher", now=1_000)
    payload, signature = token.split(".")
    forged, _ = issue_token("admin", "Administrator", "admin", now=1_000)
    with pytest.raises(ValueError, match="bad signature"):
        verify_token(f"{forged.split('.')[0]}.{signature}", now=1_001)


def test_expired_token_is_rejected():
    token, principal = issue_token("demo-1", "Demo researcher", "researcher", now=1_000)
    with pytest.raises(ValueError, match="expired"):
        verify_token(token, now=principal.expires_at + 1)


def test_malformed_token_is_rejected():
    with pytest.raises(ValueError):
        verify_token("not-a-token")
