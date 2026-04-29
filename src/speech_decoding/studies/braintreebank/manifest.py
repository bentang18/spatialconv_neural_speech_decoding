"""BrainTreebank Tier-1 cohort manifest.

`TIER1_WHITELIST` is the (subject_num, trial) list that survives the v14
Tier-1-parcel coverage filter — populated by Stage-0 Block A0 once the BT
support caches are baked. Keeping the constant here so `BraintreebankStudy`
imports cleanly today.
"""

from __future__ import annotations

TIER1_WHITELIST: list[tuple[int, int]] = []
"""(subject_num, trial) pairs kept after Tier-1 parcel coverage filtering."""

TIER1_BNA_PARCEL_INDICES_1BASED: tuple[int, ...] = ()
"""BT-Tier-1 parcel index list (1-based BNA), populated by Stage-0 A0."""
