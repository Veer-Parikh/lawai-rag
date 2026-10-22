"""Version-aware legal evidence contracts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class EvidenceStatus(str, Enum):
    ENACTED = "enacted"
    NOTIFIED = "notified"
    IN_FORCE = "in_force"
    SUPERSEDED = "superseded"
    REPEALED = "repealed"
    UNKNOWN = "unknown"


class SourceAuthority(str, Enum):
    GAZETTE = "gazette"
    INDIA_CODE = "india_code"
    MHA_EXPLANATORY = "mha_explanatory"
    PIB_EXPLANATORY = "pib_explanatory"
    CURATED_JUDGMENT = "curated_judgment"


_AUTHORITY_RANK = {
    SourceAuthority.GAZETTE: 0,
    SourceAuthority.INDIA_CODE: 1,
    SourceAuthority.MHA_EXPLANATORY: 2,
    SourceAuthority.PIB_EXPLANATORY: 3,
    SourceAuthority.CURATED_JUDGMENT: 4,
}


@dataclass(frozen=True)
class LegalInterval:
    starts_on: date
    ends_on: date | None
    status: EvidenceStatus
    jurisdiction: str
    territory: str | None = None
    condition: str | None = None

    def applies_on(
        self,
        on_date: date,
        jurisdiction: str = "IN",
        territory: str | None = None,
        condition_satisfied: bool | None = None,
    ) -> bool:
        """Return true only for a known in-force interval at the requested date."""
        if self.status is not EvidenceStatus.IN_FORCE:
            return False
        if self.jurisdiction != jurisdiction:
            return False
        if self.territory is not None and self.territory != territory:
            return False
        if self.condition is not None and condition_satisfied is not True:
            return False
        if on_date < self.starts_on:
            return False
        return self.ends_on is None or on_date < self.ends_on


def resolve_source_conflict(sources: list[SourceAuthority]) -> SourceAuthority:
    """Choose the highest-authority source, rejecting ambiguous duplicates."""
    if not sources:
        raise ValueError("at least one source is required")
    if len(sources) != len(set(sources)):
        raise ValueError("duplicate source authorities cannot resolve a conflict")
    ranked = sorted(sources, key=_AUTHORITY_RANK.__getitem__)
    best_rank = _AUTHORITY_RANK[ranked[0]]
    equally_ranked = [source for source in ranked if _AUTHORITY_RANK[source] == best_rank]
    if len(equally_ranked) > 1:
        raise ValueError("equally authoritative sources conflict")
    return ranked[0]
