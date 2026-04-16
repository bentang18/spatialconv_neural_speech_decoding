"""Frozen atlas/subparcel token layout for Neural Field Perceiver v14 Phase 1.

Frozen 2026-04-14 by blocker #4 in ``docs/implementation_tasks.md``. Revised
later that day to the argmax-centric rule:

    parcel p becomes a Tier-1 token iff
        argmax_wins(p) >= 10

where ``argmax_wins(p)`` is the number of Phase-1 LH electrodes for which
``p`` is the dominant parcel across all 246 BNA ROIs, computed on the
frozen spatial pipeline (raw MNI + 8 mm PM dilation + σ=1.5 mm Gaussian).

The rule went through two earlier iterations in the same session:

1. "Top-12 by effective_N" — a quantity rule masquerading as a quality rule.
2. "effective_N >= 10 AND argmax_wins >= 1" — caught leakage sinks (A4t,
   A44op, etc.) but excluded primary auditory cortex (TE1.0/1.2) because
   its eff_N (7.14) fell below the arbitrary 10 floor despite 12 genuine
   argmax-winning electrodes across 3 patients.
3. "argmax_wins >= 10" (current) — directly measures "how many electrodes
   have this parcel as their true anatomical home," without using eff_N as
   a proxy that entangles sampling density with evidence quality. Cutoff
   is robust: argmax in the [5, 9] band is empty on current data, so any
   threshold from 5 to 12 gives the same 15-parcel list.

The ranking was computed by ``scripts/rank_parcels_by_support.py`` over
the 7 Phase-1 LH patients (S14, S16, S23, S26, S33, S39, S62) = 1280
electrodes. Uniform ``k_parcel = 1`` (no 2-token splits in Phase 1); the
2-token variant is a named ablation gated on ``parcel_frames.npz`` and a
future elongation metric; it is not part of the Phase-1 baseline.

``PROVISIONAL_TOKEN_SPEC`` is ``False``. ``assert_token_spec_frozen`` is a
no-op kept for API stability with code written against the earlier
provisional banner.
"""

from __future__ import annotations


PROVISIONAL_TOKEN_SPEC: bool = False


def assert_token_spec_frozen() -> None:
    """Assert that the Phase-1 token spec is frozen. No-op since 2026-04-14."""

    if PROVISIONAL_TOKEN_SPEC:
        raise RuntimeError(
            "token_spec is still provisional — blocker #4 must close before "
            "DEFAULT_BASE_PARCELS / DEFAULT_SPLIT_COUNTS may enter a v14 code "
            "path. See docs/implementation_tasks.md #4."
        )


# 15 LH Brainnetome parcels selected by argmax_wins >= 10 on the frozen
# Phase-1 spatial pipeline (raw + dilated_8mm + σ=1.5 mm, 2026-04-14).
# Ordered by argmax_wins high -> low (primary), eff_N high -> low (tiebreak).
# Tuple of (parcel_name, BNA_index_1based) pairs so name-to-index and
# ordering are both canonical in a single source.
DEFAULT_BASE_PARCELS: tuple[tuple[str, int], ...] = (
    ("A45c",        33),   # IFG_L_6_3   argmx=223  eff_N=202.43  pts=6
    ("A45r",        35),   # IFG_L_6_4   argmx=199  eff_N=179.17  pts=6
    ("IFS",         31),   # IFG_L_6_2   argmx=181  eff_N=188.61  pts=6
    ("A4hf",        53),   # PrG_L_6_1   argmx=168  eff_N=121.19  pts=5
    ("A44v",        39),   # IFG_L_6_6   argmx=123  eff_N=112.55  pts=6
    ("A9/46v",      21),   # MFG_L_7_4   argmx=100  eff_N= 77.02  pts=4
    ("A12/47l",     51),   # OrG_L_6_6   argmx= 48  eff_N= 32.08  pts=2
    ("A4tl",        61),   # PrG_L_6_5   argmx= 44  eff_N= 52.15  pts=5
    ("A38l",        77),   # STG_L_6_5   argmx= 40  eff_N= 15.60  pts=3
    ("A44d",        29),   # IFG_L_6_1   argmx= 34  eff_N= 48.13  pts=6
    ("A1/2/3tonIa", 157),  # PoG_L_4_2   argmx= 29  eff_N= 26.93  pts=2
    ("A1/2/3ulhf", 155),   # PoG_L_4_1   argmx= 28  eff_N= 30.10  pts=2
    ("IFJ",         17),   # MFG_L_7_2   argmx= 27  eff_N= 23.65  pts=4
    ("A6cvl",       63),   # PrG_L_6_6   argmx= 24  eff_N= 31.94  pts=5
    ("TE1.0/1.2",   73),   # STG_L_6_3   argmx= 12  eff_N=  7.14  pts=3  primary auditory (Heschl's)
)

# Uniform k_parcel = 1 in Phase 1. The 2-token variant is a named ablation
# and requires a future parcel elongation metric before it can be derived.
DEFAULT_SPLIT_COUNTS: dict[str, int] = {name: 1 for name, _ in DEFAULT_BASE_PARCELS}


def default_token_count() -> int:
    """Return the current default atlas/subparcel token count (N_tok = 15)."""

    return sum(DEFAULT_SPLIT_COUNTS.values())
