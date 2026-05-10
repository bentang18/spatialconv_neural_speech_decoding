"""BrainTreebank cohort manifest constants.

The session lists mirror Neuroprobe's published BrainTreebank splits. v14 parcel
coverage filters remain separate: `TIER1_WHITELIST` is populated only after the
BT support caches are baked.
"""

from __future__ import annotations

BT_NANO_SESSIONS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 4),
    (3, 1),
    (4, 0),
    (7, 1),
    (10, 1),
)
"""Tiny BrainTreebank subset used by Neuroprobe for CI-scale smoke tests."""

BT_LITE_SESSIONS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 4),
    (3, 0),
    (3, 1),
    (4, 0),
    (4, 1),
    (7, 0),
    (7, 1),
    (10, 0),
    (10, 1),
)
"""Lite BrainTreebank sessions exposed by Neuroprobe."""

BT_FULL_SESSIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 0),
    (3, 1),
    (3, 2),
    (4, 0),
    (4, 1),
    (4, 2),
    (5, 0),
    (6, 0),
    (6, 1),
    (6, 4),
    (7, 0),
    (7, 1),
    (8, 0),
    (9, 0),
    (10, 0),
    (10, 1),
)
"""Full BrainTreebank subject/trial sessions used by Neuroprobe."""

TIER1_WHITELIST: tuple[tuple[int, int], ...] = ()
"""(subject_id, trial_id) pairs kept after Tier-1 parcel coverage filtering."""

TIER1_BNA_PARCEL_INDICES_1BASED: tuple[int, ...] = ()
"""BT-Tier-1 parcel index list (1-based BNA), populated by Stage-0 A0."""
