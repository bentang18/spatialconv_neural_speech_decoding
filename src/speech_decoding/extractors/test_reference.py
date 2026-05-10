from __future__ import annotations

import mne
import numpy as np
import pytest

from speech_decoding.extractors.reference import CARIeegExtractor, parse_shaft


def _make_raw(ch_names: list[str], data: np.ndarray, sfreq: float = 200.0) -> mne.io.RawArray:
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")
    return mne.io.RawArray(data, info, verbose=False)


def test_parse_shaft_basic_cases():
    assert parse_shaft("F1") == ("F", 1)
    assert parse_shaft("F12") == ("F", 12)
    assert parse_shaft("OFa1") == ("OFa", 1)
    assert parse_shaft("LH-3") == ("LH-", 3)


def test_parse_shaft_no_suffix():
    assert parse_shaft("noisy") == ("noisy", None)
    assert parse_shaft("") == ("", None)


def test_global_car_zeros_channel_mean():
    n_ch, n_samples = 4, 100
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_ch, n_samples)).astype(np.float64)
    data += np.array([10.0, 20.0, 30.0, 40.0])[:, None]  # per-channel offsets
    raw = _make_raw(["A1", "A2", "A3", "A4"], data.copy())

    extractor = CARIeegExtractor(car="global")
    out = extractor._apply_car(raw)
    assert np.allclose(out._data.mean(axis=0), 0.0)


def test_shaft_car_zeros_per_shaft_mean():
    rng = np.random.default_rng(1)
    n_samples = 50
    data = rng.standard_normal((6, n_samples)).astype(np.float64)
    data[:3] += 5.0  # shaft A common-mode
    data[3:] -= 7.0  # shaft B common-mode
    ch_names = ["A1", "A2", "A3", "B1", "B2", "B3"]
    raw = _make_raw(ch_names, data.copy())

    extractor = CARIeegExtractor(car="shaft")
    out = extractor._apply_car(raw)
    assert np.allclose(out._data[:3].mean(axis=0), 0.0)
    assert np.allclose(out._data[3:].mean(axis=0), 0.0)


def test_shaft_car_singleton_shaft_zeros_signal():
    data = np.array([[1.0, 2.0, 3.0]])
    raw = _make_raw(["X1"], data.copy())
    extractor = CARIeegExtractor(car="shaft")
    out = extractor._apply_car(raw)
    assert np.allclose(out._data, 0.0)


def test_car_and_bipolar_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        CARIeegExtractor(car="global", reference="bipolar", picks=("seeg",))


def test_car_and_explicit_bipolar_ref_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        CARIeegExtractor(car="global", bipolar_ref=(["A1"], ["A2"]))


def test_car_none_is_passthrough_construction():
    extractor = CARIeegExtractor()
    assert extractor.car is None
