"""Pre-v14 CTC + articulatory helpers split out of `data/phoneme_map.py` (2026-04-13).

This file is PART OF THE QUARANTINE. It is not importable from active code:
the pytest CI guard `tests/v14/test_no_legacy_imports.py` rejects any v14
module that tries to import from `speech_decoding.archive.*`.

Why it was split out:

- v14 does not use CTC. The 3-query AR decoder has a different vocab and
  loss contract, so `CTC_BLANK`, `encode_ctc_label`, `decode_ctc_indices`,
  and the 1-indexed `phoneme_to_index` / `index_to_phoneme` pair (which
  reserves index 0 for the CTC blank) encode a supervision contract that
  does not transfer.
- The 9x15 `ARTICULATORY_MATRIX` and `ARTICULATORY_FEATURES` list were
  built for the v12 articulatory head. That head was explicitly demoted
  during the per-phoneme MFA sweep (2026-04-04) — flat head beat
  articulatory head by ~4 percentage points on single-phoneme
  classification — and v14 uses neither the head nor the 15-feature
  bottleneck.

Under the 2026-04-13 working principle, v14 will define its own
label -> integer index contract once the blocker in
`docs/implementation_tasks.md` (#16) is discussed and locked. This file
exists only as historical reference for what the previous contract was.
"""
from __future__ import annotations

import numpy as np

from speech_decoding.data.phoneme_map import (
    ARPA_PHONEMES,
    normalize_label,
)

# CTC blank token index (1-indexed mapping reserves 0 for the blank).
CTC_BLANK: int = 0

# Phoneme-to-CTC-index mapping (1-indexed; 0 = blank).
_PHON2IDX: dict[str, int] = {p: i + 1 for i, p in enumerate(ARPA_PHONEMES)}
_IDX2PHON: dict[int, str] = {v: k for k, v in _PHON2IDX.items()}

# Articulatory feature names (15 binary features across 6 groups).
ARTICULATORY_FEATURES: list[str] = [
    # CV (2)
    "consonant", "vowel",
    # Place (3)
    "bilabial", "labiodental", "velar",
    # Manner (2)
    "stop", "fricative",
    # Voicing (2)
    "voiced", "voiceless",
    # Height (3)
    "low", "mid", "high",
    # Backness (3)
    "front", "central", "back",
]

# A[i, j] = 1 iff phoneme i (in ARPA_PHONEMES order) has feature j.
# Order: AA, EH, IY, UH, B, P, V, G, K (matches ARPA_PHONEMES).
# Cols:  C  V  bil lab vel stp fri vcd vcl low mid hi  frt cen bck
ARTICULATORY_MATRIX: np.ndarray = np.array([
    # AA (/a/): vowel, low, central
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    # EH (/ae/): vowel, mid, front
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    # IY (/i/): vowel, high, front
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    # UH (/u/): vowel, high, back
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    # B: consonant, bilabial, stop, voiced
    [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # P: consonant, bilabial, stop, voiceless
    [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    # V: consonant, labiodental, fricative, voiced
    [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    # G: consonant, velar, stop, voiced
    [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # K: consonant, velar, stop, voiceless
    [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
], dtype=np.float32)


def phoneme_to_index(phoneme: str) -> int:
    """Map a canonical ARPA phoneme to its CTC index (1-9). Blank = 0."""
    return _PHON2IDX[phoneme]


def index_to_phoneme(idx: int) -> str:
    """Map a CTC index (1-9) back to its canonical ARPA phoneme."""
    return _IDX2PHON[idx]


def encode_ctc_label(seq: list[str]) -> list[int]:
    """Encode a phoneme sequence to CTC label indices.

    Args:
        seq: List of phoneme labels (any notation).

    Returns:
        List of integer indices (1-9). No blanks inserted.
    """
    return [phoneme_to_index(normalize_label(p)) for p in seq]


def decode_ctc_indices(indices: list[int]) -> list[str]:
    """Decode CTC output indices to phoneme labels, skipping blanks."""
    return [index_to_phoneme(i) for i in indices if i != CTC_BLANK]
