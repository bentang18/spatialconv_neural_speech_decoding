"""Tests for phoneme label mapping and articulatory feature matrix."""
import numpy as np
import pytest

from speech_decoding.data.phoneme_map import (
    ARPA_PHONEMES,
    PS2ARPA,
    ARPA2PS,
    CTC_BLANK,
    ARTICULATORY_MATRIX,
    ARTICULATORY_FEATURES,
    normalize_label,
    encode_ctc_label,
    decode_ctc_indices,
    phoneme_to_index,
    index_to_phoneme,
    filter_to_ps_phonemes,
)


class TestPhonemeConstants:
    def test_nine_phonemes(self):
        assert len(ARPA_PHONEMES) == 9

    def test_ps2arpa_covers_all_lowercase(self):
        """All lowercase PS labels map to ARPA."""
        ps_labels = {"a", "ae", "i", "u", "b", "p", "v", "g", "k"}
        assert set(PS2ARPA.keys()) == ps_labels

    def test_arpa2ps_inverse(self):
        """ARPA2PS is the exact inverse of PS2ARPA."""
        for ps, arpa in PS2ARPA.items():
            assert ARPA2PS[arpa] == ps

    def test_ctc_blank_is_zero(self):
        assert CTC_BLANK == 0

    def test_phoneme_indices_are_1_to_9(self):
        indices = [phoneme_to_index(p) for p in ARPA_PHONEMES]
        assert sorted(indices) == list(range(1, 10))


class TestNormalizeLabel:
    """Test normalization of mixed PS/ARPABET labels to canonical ARPA."""

    def test_lowercase_ps_labels(self):
        assert normalize_label("a") == "AA"
        assert normalize_label("ae") == "EH"
        assert normalize_label("i") == "IY"
        assert normalize_label("u") == "UH"
        assert normalize_label("b") == "B"
        assert normalize_label("p") == "P"
        assert normalize_label("v") == "V"
        assert normalize_label("g") == "G"
        assert normalize_label("k") == "K"

    def test_arpabet_with_stress(self):
        """Strip stress markers (digits) from ARPABET labels."""
        assert normalize_label("AA1") == "AA"
        assert normalize_label("EH1") == "EH"
        assert normalize_label("IY0") == "IY"
        assert normalize_label("UH1") == "UH"

    def test_arpabet_without_stress(self):
        """Already-normalized ARPA labels pass through."""
        assert normalize_label("AA") == "AA"
        assert normalize_label("B") == "B"
        assert normalize_label("EH") == "EH"

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError, match="Unknown phoneme label"):
            normalize_label("ZZ")

    def test_full_arpabet_lexical_labels(self):
        """Lexical dataset uses full ARPABET (HH, EY1, Z, AH0, etc.)."""
        assert normalize_label("HH") == "HH"
        assert normalize_label("EY1") == "EY"
        assert normalize_label("AH0") == "AH"
        assert normalize_label("Z") == "Z"


class TestCTCEncoding:
    def test_encode_three_phoneme_sequence(self):
        """CTC label for /bak/ = [B, AA, K] → [idx(B), idx(AA), idx(K)]."""
        seq = ["B", "AA", "K"]
        encoded = encode_ctc_label(seq)
        assert len(encoded) == 3
        assert all(isinstance(i, int) for i in encoded)
        assert all(1 <= i <= 9 for i in encoded)

    def test_encode_with_normalization(self):
        """Encoding handles mixed labels."""
        seq_mixed = ["b", "AA1", "k"]
        seq_clean = ["B", "AA", "K"]
        assert encode_ctc_label(seq_mixed) == encode_ctc_label(seq_clean)

    def test_decode_round_trip(self):
        """encode → decode is identity."""
        seq = ["B", "AA", "K"]
        encoded = encode_ctc_label(seq)
        decoded = decode_ctc_indices(encoded)
        assert decoded == seq

    def test_decode_skips_blanks(self):
        """Decode with blanks interspersed."""
        encoded = [0, phoneme_to_index("B"), 0, phoneme_to_index("AA"), 0]
        decoded = decode_ctc_indices(encoded)
        assert decoded == ["B", "AA"]


class TestArticulatoryMatrix:
    def test_shape(self):
        """9 phonemes × 6 feature groups (15 binary features total)."""
        assert ARTICULATORY_MATRIX.shape == (9, 15)

    def test_binary(self):
        """All entries are 0 or 1."""
        assert set(np.unique(ARTICULATORY_MATRIX)) == {0, 1}

    def test_consonants_have_cv_place_manner_voicing(self):
        """Consonants (b,p,v,g,k) have C=1, V=0, and place+manner+voicing."""
        consonants = ["B", "P", "V", "G", "K"]
        for phon in consonants:
            idx = phoneme_to_index(phon) - 1  # 0-indexed into matrix
            row = ARTICULATORY_MATRIX[idx]
            assert row[0] == 1, f"{phon} should be consonant"
            assert row[1] == 0, f"{phon} should not be vowel"
            # Should have exactly one place, one manner, one voicing
            assert sum(row[2:5]) == 1, f"{phon} should have one place"
            assert sum(row[5:7]) == 1, f"{phon} should have one manner"
            assert sum(row[7:9]) == 1, f"{phon} should have one voicing"
            # No height/backness
            assert sum(row[9:]) == 0, f"{phon} should have no height/backness"

    def test_vowels_have_cv_height_backness(self):
        """Vowels (a,ae,i,u) have V=1, C=0, and height+backness."""
        vowels = ["AA", "EH", "IY", "UH"]
        for phon in vowels:
            idx = phoneme_to_index(phon) - 1
            row = ARTICULATORY_MATRIX[idx]
            assert row[0] == 0, f"{phon} should not be consonant"
            assert row[1] == 1, f"{phon} should be vowel"
            # No place/manner/voicing
            assert sum(row[2:9]) == 0, f"{phon} should have no place/manner/voicing"
            # Should have exactly one height, one backness
            assert sum(row[9:12]) == 1, f"{phon} should have one height"
            assert sum(row[12:15]) == 1, f"{phon} should have one backness"

    def test_b_vs_p_differ_only_in_voicing(self):
        """b and p differ only in voicing (voiced vs voiceless)."""
        b_idx = phoneme_to_index("B") - 1
        p_idx = phoneme_to_index("P") - 1
        b_row = ARTICULATORY_MATRIX[b_idx]
        p_row = ARTICULATORY_MATRIX[p_idx]
        diff = np.where(b_row != p_row)[0]
        assert len(diff) == 2  # voiced and voiceless columns
        assert set(diff) == {7, 8}  # voicing columns

    def test_feature_names(self):
        assert len(ARTICULATORY_FEATURES) == 15
        assert ARTICULATORY_FEATURES[0] == "consonant"
        assert ARTICULATORY_FEATURES[1] == "vowel"

    def test_each_phoneme_has_features(self):
        """No all-zero rows."""
        for i in range(9):
            assert ARTICULATORY_MATRIX[i].sum() >= 3


class TestFilterToPS:
    def test_filter_keeps_ps_phonemes(self):
        labels = ["AA", "B", "HH", "EH", "Z", "IY"]
        mask = filter_to_ps_phonemes(labels)
        assert mask == [True, True, False, True, False, True]

    def test_filter_normalizes_first(self):
        labels = ["AA1", "b", "HH", "ae"]
        mask = filter_to_ps_phonemes(labels)
        assert mask == [True, True, False, True]
