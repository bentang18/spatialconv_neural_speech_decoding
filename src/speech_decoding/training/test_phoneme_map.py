"""Tests for the contract-neutral phoneme label space.

Only the label space itself is tested here. The pre-v14 CTC helpers and
articulatory feature matrix were split out into
`src/speech_decoding/archive/legacy/data/phoneme_map_ctc_articulatory.py`
on 2026-04-13 and are out of the active import path.

The previous (pre-split) test file is preserved at
`tests/archive/legacy/test_phoneme_map_pre_v14.py` for historical
reference. It is auto-excluded from pytest collection.
"""
from __future__ import annotations

import pytest

from speech_decoding.training.phoneme_map import (
    ARPA_PHONEMES,
    ARPA2PS,
    PS2ARPA,
    filter_to_ps_phonemes,
    normalize_label,
)


class TestPhonemeConstants:
    def test_nine_phonemes(self) -> None:
        assert len(ARPA_PHONEMES) == 9

    def test_arpa_phonemes_are_unique(self) -> None:
        assert len(set(ARPA_PHONEMES)) == 9

    def test_arpa_phonemes_are_canonical_arpabet(self) -> None:
        """Every entry should be a valid ARPABET string with no stress digit."""
        for phon in ARPA_PHONEMES:
            assert phon == phon.upper()
            assert not phon[-1].isdigit()

    def test_ps2arpa_covers_all_lowercase(self) -> None:
        ps_labels = {"a", "ae", "i", "u", "b", "p", "v", "g", "k"}
        assert set(PS2ARPA.keys()) == ps_labels

    def test_ps2arpa_values_match_arpa_phonemes(self) -> None:
        assert set(PS2ARPA.values()) == set(ARPA_PHONEMES)

    def test_arpa2ps_is_exact_inverse(self) -> None:
        for ps, arpa in PS2ARPA.items():
            assert ARPA2PS[arpa] == ps
        assert set(ARPA2PS.keys()) == set(ARPA_PHONEMES)


class TestNormalizeLabel:
    """Normalization of mixed PS / ARPABET labels to canonical ARPA."""

    def test_lowercase_ps_labels(self) -> None:
        assert normalize_label("a") == "AA"
        assert normalize_label("ae") == "AE"
        assert normalize_label("i") == "IY"
        assert normalize_label("u") == "UW"
        assert normalize_label("b") == "B"
        assert normalize_label("p") == "P"
        assert normalize_label("v") == "V"
        assert normalize_label("g") == "G"
        assert normalize_label("k") == "K"

    def test_arpabet_with_stress(self) -> None:
        """Trailing digits (stress markers) are stripped."""
        assert normalize_label("AA1") == "AA"
        assert normalize_label("EH1") == "EH"
        assert normalize_label("IY0") == "IY"
        assert normalize_label("UH1") == "UH"

    def test_arpabet_without_stress(self) -> None:
        """Already-normalized ARPA labels pass through unchanged."""
        assert normalize_label("AA") == "AA"
        assert normalize_label("B") == "B"
        assert normalize_label("EH") == "EH"

    def test_full_arpabet_lexical_labels(self) -> None:
        """Full-ARPABET labels from the Lexical task also normalize."""
        assert normalize_label("HH") == "HH"
        assert normalize_label("EY1") == "EY"
        assert normalize_label("AH0") == "AH"
        assert normalize_label("Z") == "Z"

    def test_unknown_label_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown phoneme label"):
            normalize_label("ZZ")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown phoneme label"):
            normalize_label("")


class TestFilterToPS:
    def test_filter_keeps_ps_phonemes(self) -> None:
        """Per #17, EH is no longer a PS phoneme (that slot moved to AE)."""
        labels = ["AA", "B", "HH", "EH", "Z", "IY"]
        mask = filter_to_ps_phonemes(labels)
        assert mask == [True, True, False, False, False, True]

    def test_filter_normalizes_first(self) -> None:
        labels = ["AA1", "b", "HH", "ae"]
        mask = filter_to_ps_phonemes(labels)
        assert mask == [True, True, False, True]

    def test_filter_handles_unknown_gracefully(self) -> None:
        """Unknown labels become False, not an exception."""
        labels = ["AA", "ZZ", "nonsense", "b"]
        mask = filter_to_ps_phonemes(labels)
        assert mask == [True, False, False, True]
