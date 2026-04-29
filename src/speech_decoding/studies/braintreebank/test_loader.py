"""Tests for `bt_load_raw()` against a `BrainTreebankSubject`-shaped stub.

Real h5 reads only happen on DCC; the laptop test exercises the wrapping
contract — shape, dtype, channel-name ↔ row alignment, sfreq passthrough.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from speech_decoding.studies.braintreebank.loader import bt_load_raw


@dataclass
class _FakeBT:
    data: np.ndarray
    electrode_labels: list[str]
    sampling_rate: float

    def get_electrode_data(self) -> np.ndarray:
        return self.data


def test_bt_load_raw_returns_raw_voltage_at_native_rate() -> None:
    n_ch, n_t = 4, 2048
    rng = np.random.default_rng(0)
    fake = _FakeBT(
        data=rng.standard_normal((n_ch, n_t)).astype(np.float64),
        electrode_labels=[f"E{i:03d}" for i in range(n_ch)],
        sampling_rate=2048.0,
    )

    data, ch_names, sfreq = bt_load_raw(fake)

    assert data.shape == (n_ch, n_t)
    assert data.dtype == np.float32  # cast from float64
    assert ch_names == [f"E{i:03d}" for i in range(n_ch)]
    assert sfreq == 2048.0


def test_bt_load_raw_rejects_label_count_mismatch() -> None:
    fake = _FakeBT(
        data=np.zeros((4, 100), dtype=np.float32),
        electrode_labels=["only_two", "labels"],
        sampling_rate=2048.0,
    )
    with pytest.raises(ValueError, match="electrode_labels len"):
        bt_load_raw(fake)


def test_bt_load_raw_rejects_non_2d_voltage() -> None:
    fake = _FakeBT(
        data=np.zeros((4,), dtype=np.float32),
        electrode_labels=["x", "y", "z", "w"],
        sampling_rate=2048.0,
    )
    with pytest.raises(ValueError, match="expected"):
        bt_load_raw(fake)
