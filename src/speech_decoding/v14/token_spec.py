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
