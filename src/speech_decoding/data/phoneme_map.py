"""Phoneme label space for the PS task.

Contract-neutral label-space utilities only. This file must not grow into
a supervision-contract file. In particular:

- no label -> integer index mapping (that is a v14 decoder decision; see
  `docs/implementation_tasks.md` blocker #16)
- no CTC helpers (v14 does not use CTC)
- no articulatory feature matrix (v12 articulatory head is not used)

The CTC and articulatory helpers that used to live here were split out
on 2026-04-13 into
`src/speech_decoding/archive/legacy/data/phoneme_map_ctc_articulatory.py`
and are not importable from active code.

What this module provides:

- `ARPA_PHONEMES`: the 9 canonical PS phonemes in the frozen alphabetical
  ordering from `#16`. Position in this list is the label index
  (`AA=0, AE=1, B=2, G=3, IY=4, K=5, P=6, UW=7, V=8`).
- `PS2ARPA` / `ARPA2PS`: notation conversion between the lowercase PS
  convention (`a`, `ae`, `i`, `u`, `b`, `p`, `v`, `g`, `k`) and canonical
  ARPABET (`AA`, `AE`, `IY`, `UW`, `B`, `P`, `V`, `G`, `K`), frozen per
  `#17` (`ae → AE` for /æ/, `u → UW` for /u/).
- `normalize_label`: the single canonical path for going from whatever a
  raw upstream label looks like (lowercase PS, stressed ARPABET, or
  already-canonical ARPABET) to the canonical ARPA string.
- `filter_to_ps_phonemes`: boolean mask helper for cross-task label
  filtering. Does not assume the caller is doing CTC or anything else.

Any future v14 loader that needs a label index must either live in
`src/speech_decoding/v14/` or be added here with explicit discussion.
"""
from __future__ import annotations

# --- The 9 PS phonemes in canonical ARPABET notation ---
#
# Ordering: alphabetical, frozen per #16. Position in this list is the
# decoder label index (AA=0, AE=1, B=2, G=3, IY=4, K=5, P=6, UW=7, V=8).
ARPA_PHONEMES: list[str] = ["AA", "AE", "B", "G", "IY", "K", "P", "UW", "V"]

# Lowercase PS notation -> canonical ARPA.
PS2ARPA: dict[str, str] = {
    "a": "AA", "ae": "AE", "i": "IY", "u": "UW",
    "b": "B", "p": "P", "v": "V", "g": "G", "k": "K",
}

# Inverse mapping.
ARPA2PS: dict[str, str] = {v: k for k, v in PS2ARPA.items()}

# All valid ARPABET symbols (PS 9 + full English set for Lexical).
_ALL_ARPABET: set[str] = {
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
}


def normalize_label(label: str) -> str:
    """Normalize a phoneme label to canonical ARPABET (no stress markers).

    Three input cases are handled:

    1. Lowercase PS notation: `'a' -> 'AA'`, `'ae' -> 'AE'`.
    2. ARPABET with stress digits: `'AA1' -> 'AA'`, `'IY0' -> 'IY'`.
    3. Already-canonical ARPABET: `'AA' -> 'AA'`, `'B' -> 'B'`.

    Raises `ValueError` for any input that does not resolve to a known
    ARPABET symbol. This is intentional: silent passthrough of unknown
    labels would corrupt downstream stats.
    """
    # Case 1: lowercase PS notation.
    if label in PS2ARPA:
        return PS2ARPA[label]

    # Case 2: strip trailing stress digit.
    stripped = label.rstrip("0123456789")

    # Case 3: check against known ARPABET.
    if stripped in _ALL_ARPABET:
        return stripped

    raise ValueError(f"Unknown phoneme label: '{label}'")


def filter_to_ps_phonemes(labels: list[str]) -> list[bool]:
    """Return a boolean mask: True iff label normalizes to one of the 9 PS phonemes.

    Used for cross-task filtering when the upstream source includes
    non-PS ARPABET labels. Does not assume any particular downstream
    supervision contract.
    """
    ps_set = set(ARPA_PHONEMES)
    result: list[bool] = []
    for label in labels:
        try:
            normed = normalize_label(label)
            result.append(normed in ps_set)
        except ValueError:
            result.append(False)
    return result
