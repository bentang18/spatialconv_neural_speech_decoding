"""Provisional atlas/subparcel token layout for Neural Field Perceiver v14.

=============================================================================
PROVISIONAL — DISCUSS BEFORE USE (2026-04-13)
=============================================================================

The 16 base parcels and the 2-token splits below were inherited from the
pre-Phase-1 v14 draft. They are still explicitly listed as an open blocker
in `docs/implementation_tasks.md` (#4 "Decide the default parcel split
map"). Under the 2026-04-13 discussion-first rule (`CLAUDE.md` -> "Working
Principle: Discuss Before Code"), this file must not be treated as a
locked contract. Before any v14 component consumes `DEFAULT_BASE_PARCELS`
or `DEFAULT_SPLIT_COUNTS`, the following must be discussed and agreed:

- whether the 16-parcel base list is the right short list for Phase 1
- the canonical ordering of the parcels within the list (currently
  grouped by functional category, but the grouping was not an explicit
  agreement)
- which parcels actually get a 2-token split, and the anatomical
  justification for each
- whether A4tl should stay single-token or get a split
- the resulting `N_tok` and how it interacts with `token_mask` /
  `token_support`
- how the token ordering maps to the inter-region graph attention bias
  (parent parcels + sibling splits)

If you are reading this before the above has been discussed, surface the
discussion with Ben rather than importing these constants.
=============================================================================
"""

from __future__ import annotations

DEFAULT_BASE_PARCELS: tuple[str, ...] = (
    "A6cvl",
    "A4tl",
    "A4hf",
    "A1/2/3tonIa",
    "A1/2/3ulhf",
    "A2",
    "A44d",
    "A45c",
    "A44v",
    "A45i",
    "A45r",
    "A44op",
    "STGpp",
    "STGa",
    "INSa",
    "MFG",
)

# Current default: 16 base parcels with selective splits for the most
# spatially elongated / coverage-sensitive parcels, yielding 21 tokens total.
DEFAULT_SPLIT_COUNTS: dict[str, int] = {
    "A6cvl": 2,
    "A4tl": 1,
    "A4hf": 2,
    "A1/2/3tonIa": 2,
    "A1/2/3ulhf": 2,
    "A2": 2,
    "A44d": 1,
    "A45c": 1,
    "A44v": 1,
    "A45i": 1,
    "A45r": 1,
    "A44op": 1,
    "STGpp": 1,
    "STGa": 1,
    "INSa": 1,
    "MFG": 1,
}


def default_token_count() -> int:
    """Return the current default atlas/subparcel token count."""

    return sum(DEFAULT_SPLIT_COUNTS.values())
