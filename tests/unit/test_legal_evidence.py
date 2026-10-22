import unittest
from datetime import date

from legal.evidence import (
    EvidenceStatus,
    LegalInterval,
    SourceAuthority,
    resolve_source_conflict,
)


class LegalIntervalTests(unittest.TestCase):
    def test_interval_is_in_force_on_inclusive_start_and_exclusive_end(self):
        interval = LegalInterval(
            starts_on=date(2024, 7, 1),
            ends_on=date(2025, 1, 1),
            status=EvidenceStatus.IN_FORCE,
            jurisdiction="IN",
        )

        self.assertTrue(interval.applies_on(date(2024, 7, 1)))
        self.assertTrue(interval.applies_on(date(2024, 12, 31)))
        self.assertFalse(interval.applies_on(date(2025, 1, 1)))

    def test_unknown_status_does_not_claim_to_be_in_force(self):
        interval = LegalInterval(
            starts_on=date(2024, 7, 1),
            ends_on=None,
            status=EvidenceStatus.UNKNOWN,
            jurisdiction="IN",
        )

        self.assertFalse(interval.applies_on(date(2025, 1, 1)))

    def test_territory_restriction_requires_matching_territory(self):
        interval = LegalInterval(
            starts_on=date(2024, 7, 1),
            ends_on=None,
            status=EvidenceStatus.IN_FORCE,
            jurisdiction="IN",
            territory="Delhi",
        )

        self.assertFalse(interval.applies_on(date(2025, 1, 1)))
        self.assertTrue(interval.applies_on(date(2025, 1, 1), territory="Delhi"))


class SourceConflictTests(unittest.TestCase):
    def test_primary_source_wins_over_explanatory_source(self):
        winner = resolve_source_conflict(
            [
                SourceAuthority.MHA_EXPLANATORY,
                SourceAuthority.GAZETTE,
                SourceAuthority.INDIA_CODE,
            ]
        )

        self.assertEqual(winner, SourceAuthority.GAZETTE)

    def test_equal_primary_sources_are_conflicting(self):
        with self.assertRaises(ValueError):
            resolve_source_conflict(
                [SourceAuthority.GAZETTE, SourceAuthority.GAZETTE]
            )
