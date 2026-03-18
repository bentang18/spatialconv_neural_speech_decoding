"""Phoneme label mapping, CTC encoding, and articulatory feature matrix.

Handles the mixed-notation labels from the PS dataset (ARPAbet with stress
markers like AA1, EH1 AND lowercase like a, ae) and the full ARPABET labels
from the Lexical dataset.
"""
from __future__ import annotations

import numpy as np

# --- The 9 PS phonemes in canonical ARPABET notation ---
ARPA_PHONEMES: list[str] = ["AA", "EH", "IY", "UH", "B", "P", "V", "G", "K"]

# Lowercase PS notation → canonical ARPA
PS2ARPA: dict[str, str] = {
    "a": "AA", "ae": "EH", "i": "IY", "u": "UH",
    "b": "B", "p": "P", "v": "V", "g": "G", "k": "K",
}

# Inverse mapping
ARPA2PS: dict[str, str] = {v: k for k, v in PS2ARPA.items()}

# All valid ARPABET symbols (PS 9 + full English set for Lexical)
_ALL_ARPABET: set[str] = {
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
}

# CTC blank token index
CTC_BLANK: int = 0

# Phoneme-to-CTC-index mapping (1-indexed; 0 = blank)
_PHON2IDX: dict[str, int] = {p: i + 1 for i, p in enumerate(ARPA_PHONEMES)}
_IDX2PHON: dict[int, str] = {v: k for k, v in _PHON2IDX.items()}

# Articulatory feature names (15 binary features across 6 groups)
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

# A[i, j] = 1 iff phoneme i (in ARPA_PHONEMES order) has feature j
# Order: B, P, V, G, K, AA, EH, IY, UH (matching ARPA_PHONEMES)
# Cols:  C  V  bil lab vel stp fri vcd vcl low mid hi  frt cen bck
ARTICULATORY_MATRIX: np.ndarray = np.array([
    # AA (/a/): vowel, low, central
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    # EH (/æ/): vowel, mid, front
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


def normalize_label(label: str) -> str:
    """Normalize a phoneme label to canonical ARPABET (no stress markers).

    Handles three cases:
    1. Lowercase PS notation: 'a' → 'AA', 'ae' → 'EH'
    2. ARPABET with stress digits: 'AA1' → 'AA', 'IY0' → 'IY'
    3. Already-canonical ARPABET: 'AA' → 'AA', 'B' → 'B'
    """
    # Case 1: lowercase PS notation
    if label in PS2ARPA:
        return PS2ARPA[label]

    # Case 2: strip trailing stress digit
    stripped = label.rstrip("0123456789")

    # Case 3: check against known ARPABET
    if stripped in _ALL_ARPABET:
        return stripped

    raise ValueError(f"Unknown phoneme label: '{label}'")


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


def filter_to_ps_phonemes(labels: list[str]) -> list[bool]:
    """Return boolean mask: True if label normalizes to one of the 9 PS phonemes."""
    ps_set = set(ARPA_PHONEMES)
    result = []
    for label in labels:
        try:
            normed = normalize_label(label)
            result.append(normed in ps_set)
        except ValueError:
            result.append(False)
    return result
