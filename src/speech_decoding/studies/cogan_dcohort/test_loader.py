"""Tests for the Cogan D-cohort voltage loader core (channel-select + resample)."""

from __future__ import annotations

import numpy as np
import pytest

from speech_decoding.studies.cogan_dcohort.loader import (
    TARGET_RATE_HZ,
    parse_contact,
    resample_to,
    select_neural,
)


def _synth(n_ch: int, n_samp: int) -> np.ndarray:
    # Deterministic ramps, distinct per channel — order-preservation is checkable.
    return np.stack([np.arange(n_samp) + 1000 * c for c in range(n_ch)]).astype(
        np.float32
    )


def test_select_keeps_neural_drops_trig_and_preserves_order():
    ch_names = ["ROG1", "TRIG", "ROG2", "EKG"]
    data = _synth(4, 50)
    types = {"ROG1": "ECOG", "TRIG": "TRIG", "ROG2": "ECOG", "EKG": "ECG"}
    out, kept = select_neural(ch_names, data, types)
    assert kept == ["ROG1", "ROG2"]
    # rows follow kept order, values intact
    np.testing.assert_array_equal(out[0], data[0])
    np.testing.assert_array_equal(out[1], data[2])


def test_select_accepts_seeg_type_too():
    ch_names = ["A1", "A2"]
    out, kept = select_neural(ch_names, _synth(2, 10), {"A1": "SEEG", "A2": "ECOG"})
    assert kept == ["A1", "A2"]
    assert out.shape == (2, 10)


def test_select_drops_extra_bad():
    ch_names = ["ROG1", "ROG2", "ROG3"]
    types = {c: "ECOG" for c in ch_names}
    _, kept = select_neural(ch_names, _synth(3, 10), types, extra_bad=["ROG2"])
    assert kept == ["ROG1", "ROG3"]


def test_select_unknown_type_is_dropped_failsafe():
    ch_names = ["ROG1", "MYSTERY"]
    _, kept = select_neural(ch_names, _synth(2, 10), {"ROG1": "ECOG"})
    assert kept == ["ROG1"]


def test_select_raises_when_nothing_neural():
    with pytest.raises(ValueError, match="no neural channels"):
        select_neural(["TRIG"], _synth(1, 10), {"TRIG": "TRIG"})


def test_select_row_count_mismatch_raises():
    with pytest.raises(ValueError, match="!= n ch_names"):
        select_neural(["A", "B"], _synth(1, 10), {"A": "ECOG", "B": "ECOG"})


def test_resample_identity_when_equal():
    d = _synth(3, 128)
    out = resample_to(d, 2048.0, 2048.0)
    np.testing.assert_array_equal(out, d)


def test_resample_1024_to_2048_doubles_length():
    d = _synth(2, 100)
    out = resample_to(d, 1024.0, 2048.0)
    assert out.shape == (2, 200)  # up=2, down=1


def test_resample_2000_to_2048_ratio():
    d = _synth(1, 2000)
    out = resample_to(d, 2000.0, TARGET_RATE_HZ)
    # up=128, down=125 → 2000 * 128/125 = 2048
    assert out.shape == (1, 2048)


def test_resample_dtype_is_float32():
    out = resample_to(_synth(2, 64).astype(np.float64), 1000.0, 2048.0)
    assert out.dtype == np.float32


def test_parse_contact_letters_then_digits():
    assert parse_contact("ROG13") == ("ROG", 13)
    assert parse_contact("RFOP1") == ("RFOP", 1)


def test_parse_contact_embedded_digit_prefix():
    # L1IF10: the L1 implant-grid prefix keeps its digit; shaft is L1IF, depth 10.
    assert parse_contact("L1IF10") == ("L1IF", 10)
    assert parse_contact("R2SF3") == ("R2SF", 3)


def test_parse_contact_no_trailing_digit_is_none():
    for nm in ["EKGL", "EKGR", "LEMG", "REMG", "Cz", "Fz", "Pz", ""]:
        assert parse_contact(nm) is None


def test_select_drops_physio_mistyped_as_neural():
    # EKGL / Cz typed ECOG/SEEG in some Cogan tsv → dropped (no contact index),
    # even though the type whitelist would keep them.
    ch_names = ["ROG1", "EKGL", "ROG2", "Cz"]
    types = {"ROG1": "ECOG", "EKGL": "ECOG", "ROG2": "ECOG", "Cz": "SEEG"}
    _, kept = select_neural(ch_names, _synth(4, 10), types)
    assert kept == ["ROG1", "ROG2"]
