"""SCAFFOLD-06: variable-T (RoPE-compatible) + variable-C (per-corpus)
collate function.

The Phase-1 mix spans 4 corpora with different ``C_MAX`` and ``T_max``;
the collate pads to the per-batch max so the RoPE table indexes
correctly and the per-batch ``valid_mask`` lets the encoder ignore the
padded electrodes.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.extractors.collate import (
    v14_variable_tc_collate,
)


def _sample(C: int, T: int, F: int, K: int, d_freq: int = 30) -> dict:
    return {
        "electrode_tokens": torch.randn(C, T, F),
        "support": torch.zeros(C, K),
        "valid_mask": torch.ones(C, dtype=torch.bool),
    }


def test_collate_pads_C_to_batch_max() -> None:
    """Two samples with different C → batched tensor has C = max."""
    s1 = _sample(C=4, T=10, F=30, K=80)
    s2 = _sample(C=7, T=10, F=30, K=80)
    batch = v14_variable_tc_collate([s1, s2])
    assert batch["electrode_tokens"].shape[1] == 7  # C
    assert batch["support"].shape[1] == 7
    assert batch["valid_mask"].shape == (2, 7)


def test_collate_pads_T_to_batch_max() -> None:
    """Different T per sample → padded to per-batch max."""
    s1 = _sample(C=4, T=8, F=30, K=80)
    s2 = _sample(C=4, T=12, F=30, K=80)
    batch = v14_variable_tc_collate([s1, s2])
    assert batch["electrode_tokens"].shape[2] == 12  # T
    # Sample-1 trailing T positions are zero-padded.
    assert torch.all(batch["electrode_tokens"][0, :, 8:, :] == 0)


def test_collate_valid_mask_marks_padded_electrodes_false() -> None:
    """``valid_mask`` is False at padded electrode positions."""
    s1 = _sample(C=2, T=10, F=30, K=80)
    s2 = _sample(C=5, T=10, F=30, K=80)
    batch = v14_variable_tc_collate([s1, s2])
    # Sample-1: positions 0..1 True, 2..4 False (padded).
    assert torch.equal(batch["valid_mask"][0], torch.tensor([True, True, False, False, False]))
    # Sample-2: all True.
    assert batch["valid_mask"][1].all()


def test_collate_zero_pads_support_for_padded_electrodes() -> None:
    """Padded electrodes get zero-support rows."""
    K = 6
    s1 = _sample(C=2, T=4, F=4, K=K)
    s1["support"][0, 0] = 1.0
    s2 = _sample(C=4, T=4, F=4, K=K)
    s2["support"][2, 1] = 1.0
    batch = v14_variable_tc_collate([s1, s2])
    assert torch.equal(batch["support"][0, 0], torch.tensor([1.0, 0, 0, 0, 0, 0]))
    # Padded rows (2, 3) on sample-1 are zero.
    assert torch.all(batch["support"][0, 2:] == 0)


def test_collate_preserves_target_field_unmodified() -> None:
    """Scalar / fixed-shape fields (e.g. ``target``) round-trip
    untouched."""
    s1 = _sample(C=2, T=4, F=4, K=6)
    s2 = _sample(C=3, T=4, F=4, K=6)
    s1["target"] = torch.tensor(0)
    s2["target"] = torch.tensor(1)
    batch = v14_variable_tc_collate([s1, s2])
    assert torch.equal(batch["target"], torch.tensor([0, 1]))


def test_collate_handles_single_sample_batch() -> None:
    """Batch of 1: collate must not error (degenerate case during
    fast_dev_run)."""
    s1 = _sample(C=3, T=4, F=4, K=6)
    batch = v14_variable_tc_collate([s1])
    assert batch["electrode_tokens"].shape == (1, 3, 4, 4)


def test_collate_rejects_freq_bin_mismatch() -> None:
    """Per-corpus F should be fixed; mixing different F values within a
    batch is a config error."""
    s1 = _sample(C=2, T=4, F=30, K=6)
    s2 = _sample(C=2, T=4, F=29, K=6)
    with pytest.raises(ValueError, match="freq"):
        v14_variable_tc_collate([s1, s2])


def test_collate_rejects_parcel_count_mismatch() -> None:
    """K_parcels must match across samples."""
    s1 = _sample(C=2, T=4, F=4, K=6)
    s2 = _sample(C=2, T=4, F=4, K=8)
    with pytest.raises(ValueError, match="parcel"):
        v14_variable_tc_collate([s1, s2])


# --- 2STFT dual-band (electrode_tokens_high) -------------------------------


def _dual_sample(C: int, *, t_low: int = 8, t_high: int = 16, K: int = 6) -> dict:
    """Freq-major dual-band sample: low (C, F_low=14, t_low), high (C, F_high=9,
    t_high). Same electrodes/support/valid_mask across both bands."""
    return {
        "electrode_tokens": torch.randn(C, 14, t_low),
        "electrode_tokens_high": torch.randn(C, 9, t_high),
        "support": torch.zeros(C, K),
        "valid_mask": torch.ones(C, dtype=torch.bool),
    }


def test_collate_dual_band_pads_high_to_same_C_max() -> None:
    """High band is padded to the SAME C_max as the low band (shared electrodes),
    keeping its own freq/time axes."""
    s1 = _dual_sample(C=4)
    s2 = _dual_sample(C=7)
    batch = v14_variable_tc_collate([s1, s2])
    assert batch["electrode_tokens"].shape == (2, 7, 14, 8)
    assert batch["electrode_tokens_high"].shape == (2, 7, 9, 16)
    # Sample-1's padded electrodes (4..6) on the high band are zero.
    assert torch.all(batch["electrode_tokens_high"][0, 4:] == 0)
    # Real electrodes preserved.
    assert torch.equal(batch["electrode_tokens_high"][0, :4], s1["electrode_tokens_high"])


def test_collate_dual_band_absent_high_is_noop() -> None:
    """Non-2STFT batches (no high band) never grow an electrode_tokens_high key."""
    s1 = _sample(C=2, T=4, F=4, K=6)
    batch = v14_variable_tc_collate([s1])
    assert "electrode_tokens_high" not in batch


def test_collate_dual_band_rejects_high_freq_mismatch() -> None:
    """High-band F mismatch within a batch is a config error."""
    s1 = _dual_sample(C=2)
    s2 = _dual_sample(C=2)
    s2["electrode_tokens_high"] = torch.randn(2, 9, 15)  # t_high differs (last axis)
    with pytest.raises(ValueError, match="high-band freq"):
        v14_variable_tc_collate([s1, s2])


def test_collate_dual_band_rejects_electrode_count_mismatch() -> None:
    """The two bands must share electrodes (same per-sample C)."""
    s1 = _dual_sample(C=3)
    s1["electrode_tokens_high"] = torch.randn(2, 9, 16)  # C=2 != low-band C=3
    with pytest.raises(ValueError, match="share electrodes"):
        v14_variable_tc_collate([s1])
