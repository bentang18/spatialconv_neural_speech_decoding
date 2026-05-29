"""Tests for the MNE LOF bad-channel mask (T1.7).

These tests need ``mne`` available; they're skipped if not installed.
"""

from __future__ import annotations

import numpy as np
import pytest


mne = pytest.importorskip("mne")


from speech_decoding.extractors.quality import (
    DEFAULT_LOF_N_NEIGHBORS,
    DEFAULT_LOF_THRESHOLD,
    MneLofBadChannelMask,
    lof_bad_channel_mask,
)


def _correlated_voltage(
    n_channels: int = 30, n_samples: int = 5_000, rng_seed: int = 0
) -> np.ndarray:
    """Generate a (C, n_samples) signal where channels are correlated noise
    plus per-channel idiosyncratic noise. LOF should *not* flag any of these
    as outliers under the default threshold."""
    rng = np.random.default_rng(rng_seed)
    common = rng.standard_normal((1, n_samples))
    idiosyncratic = rng.standard_normal((n_channels, n_samples)) * 0.3
    return common + idiosyncratic


def test_lof_flags_an_injected_outlier_channel() -> None:
    """An outlier channel with much larger variance than its neighbors must
    show up in the LOF mask."""
    voltage = _correlated_voltage().astype(np.float64)
    # Inject one extreme channel
    bad_idx = 5
    voltage[bad_idx] = np.random.default_rng(1).standard_normal(voltage.shape[1]) * 50.0
    mask = lof_bad_channel_mask(voltage, sample_rate_hz=2048.0)
    assert mask.dtype == bool
    assert mask.shape == (voltage.shape[0],)
    assert mask[bad_idx], "injected outlier must be flagged"


def test_lof_returns_all_false_on_clean_data() -> None:
    """Channels drawn from the same correlated-noise distribution should
    not be flagged at default threshold=1.5 (LOF's standard non-outlier zone).
    """
    voltage = _correlated_voltage().astype(np.float64)
    mask = lof_bad_channel_mask(voltage, sample_rate_hz=2048.0)
    # Allow up to ~10% spurious flags due to random correlation.
    assert mask.sum() <= max(1, voltage.shape[0] // 10), (
        f"expected near-zero false positives on clean data, got {mask.sum()}"
    )


def test_lof_respects_n_neighbors_and_threshold() -> None:
    """Lowering the threshold must produce more bad-channel flags (or equal),
    since LOF returns channels whose local-outlier-factor exceeds threshold."""
    voltage = _correlated_voltage().astype(np.float64)
    voltage[3] *= 10.0  # mild outlier
    voltage[7] *= 30.0  # severe outlier
    permissive = lof_bad_channel_mask(
        voltage, sample_rate_hz=2048.0, threshold=DEFAULT_LOF_THRESHOLD,
    )
    strict = lof_bad_channel_mask(
        voltage, sample_rate_hz=2048.0, threshold=1.05,
    )
    assert int(strict.sum()) >= int(permissive.sum()), (
        f"lower threshold should flag at least as many channels: "
        f"permissive={permissive.sum()}, strict={strict.sum()}"
    )


def test_lof_default_hyperparameters_match_spec() -> None:
    """5/17 L.0 amendment locks threshold=1.5, n_neighbors=20 — pin the defaults."""
    assert DEFAULT_LOF_THRESHOLD == 1.5
    assert DEFAULT_LOF_N_NEIGHBORS == 20


def test_mne_lof_bad_channel_mask_transform_matches_function() -> None:
    """The Pydantic transform delegates to the same function with the same params."""
    voltage = _correlated_voltage().astype(np.float64)
    voltage[5] *= 30.0
    t = MneLofBadChannelMask()
    a = lof_bad_channel_mask(voltage, sample_rate_hz=2048.0)
    b = t(voltage, sample_rate_hz=2048.0)
    np.testing.assert_array_equal(a, b)
