"""Tests for the raw-feature linear-probe baseline (online_probe_raw_baseline).

Covers the only genuinely new logic — raw |STFT| token construction on the
tokenizer grid (band order, intra-band freq-major/time-minor order, slow Re/Im →
magnitude), the un-pooled WS feature matrix, electrode→parcel CS pooling — plus a
small end-to-end run over a synthetic ProbeDataset (keys, ranges, determinism).
The ridge/AUROC primitives themselves are already covered in test_online_probe.py.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from speech_decoding.experiments.online_probe import SubjectProbeData
from speech_decoding.experiments.online_probe_raw_baseline import (
    feature_matrix_per_electrode,
    pool_electrodes_to_parcels,
    raw_tokens_from_bands,
    run_raw_baseline,
)
from speech_decoding.models.v14_converged import BANDS, N_TOKENS, token_metadata


def _bands(B: int, C: int, slow_v=1.0, beta_v=2.0, hg_v=3.0):
    """Uniform-per-band synthetic tensors on the 1 s grid (slow Re=slow_v, Im=0)."""
    slow = torch.zeros(B, C, 2, BANDS[0].n_freq_bins, BANDS[0].n_time_frames)
    slow[:, :, 0] = slow_v
    beta = torch.full((B, C, BANDS[1].n_freq_bins, BANDS[1].n_time_frames), beta_v)
    hg = torch.full((B, C, BANDS[2].n_freq_bins, BANDS[2].n_time_frames), hg_v)
    return slow, beta, hg


def test_raw_tokens_shape_is_30_d1():
    slow, beta, hg = _bands(2, 3)
    toks = raw_tokens_from_bands(slow, beta, hg)
    assert toks.shape == (2, 3, N_TOKENS, 1) == (2, 3, 30, 1)


def test_raw_tokens_band_order_and_counts():
    # Each band uniform but distinct -> token vector is slow×6 then beta×8 then hg×16.
    slow, beta, hg = _bands(1, 1, slow_v=1.0, beta_v=2.0, hg_v=3.0)
    toks = raw_tokens_from_bands(slow, beta, hg)[0, 0, :, 0]
    band_id, _, _ = token_metadata()
    expected = torch.where(
        band_id == 0, 1.0, torch.where(band_id == 1, 2.0, 3.0)
    )
    assert torch.allclose(toks, expected)
    # counts per band match the tokenizer geometry
    assert int((band_id == 0).sum()) == 6
    assert int((band_id == 1).sum()) == 8
    assert int((band_id == 2).sum()) == 16


def test_slow_band_magnitude_from_re_im():
    # Re=3, Im=4 everywhere -> |STFT| = 5 for every slow token.
    slow = torch.zeros(1, 1, 2, BANDS[0].n_freq_bins, BANDS[0].n_time_frames)
    slow[:, :, 0] = 3.0
    slow[:, :, 1] = 4.0
    beta = torch.zeros(1, 1, BANDS[1].n_freq_bins, BANDS[1].n_time_frames)
    hg = torch.zeros(1, 1, BANDS[2].n_freq_bins, BANDS[2].n_time_frames)
    toks = raw_tokens_from_bands(slow, beta, hg)[0, 0, :, 0]
    band_id, _, _ = token_metadata()
    assert torch.allclose(toks[band_id == 0], torch.full((6,), 5.0))


def test_slow_intra_band_freq_major_time_minor_order():
    # Give the slow band a distinct constant per (freq-patch, time-patch); the
    # recovered 6 slow tokens must be freq-patch-major, time-minor.
    b = BANDS[0]
    slow = torch.zeros(1, 1, 2, b.n_freq_bins, b.n_time_frames)
    expected = []
    for fp in range(b.n_freq_patches):       # 3
        for tp in range(b.n_time_patches):   # 2
            val = 10.0 * fp + (tp + 1)       # distinct: 1,2, 11,12, 21,22
            fr = slice(fp * b.kernel_freq, fp * b.kernel_freq + b.kernel_freq)
            tr = slice(tp * b.kernel_time, tp * b.kernel_time + b.kernel_time)
            slow[:, :, 0, fr, tr] = val
            expected.append(val)
    beta = torch.zeros(1, 1, BANDS[1].n_freq_bins, BANDS[1].n_time_frames)
    hg = torch.zeros(1, 1, BANDS[2].n_freq_bins, BANDS[2].n_time_frames)
    toks = raw_tokens_from_bands(slow, beta, hg)[0, 0, :6, 0]
    assert torch.allclose(toks, torch.tensor(expected))


def test_patch_pool_averages_within_patch():
    # A single slow freq-patch (rows 0-1) with a 2x2 time-patch of values 1,2,3,4
    # must average to 2.5 (avg-pool, not max/first).
    b = BANDS[0]
    slow = torch.zeros(1, 1, 2, b.n_freq_bins, b.n_time_frames)
    slow[:, :, 0, 0, 0] = 1.0
    slow[:, :, 0, 0, 1] = 2.0
    slow[:, :, 0, 1, 0] = 3.0
    slow[:, :, 0, 1, 1] = 4.0
    beta = torch.zeros(1, 1, BANDS[1].n_freq_bins, BANDS[1].n_time_frames)
    hg = torch.zeros(1, 1, BANDS[2].n_freq_bins, BANDS[2].n_time_frames)
    tok0 = raw_tokens_from_bands(slow, beta, hg)[0, 0, 0, 0]
    assert float(tok0) == 2.5


def test_feature_matrix_per_electrode_masks_and_flattens():
    toks = torch.arange(1 * 4 * 30 * 1, dtype=torch.float32).reshape(1, 4, 30, 1)
    mask = torch.tensor([True, False, True, True])
    fm = feature_matrix_per_electrode(toks, mask)
    assert fm.shape == (1, 3 * 30)              # 3 valid electrodes × 30 tokens
    # first 30 cols are electrode 0's tokens (electrode 1 dropped, not zeroed)
    assert torch.allclose(fm[0, :30], toks[0, 0, :, 0])
    assert torch.allclose(fm[0, 30:60], toks[0, 2, :, 0])


def test_pool_electrodes_to_parcels_mean_and_present():
    # 3 electrodes -> parcels [0,0,1]; n_parcels=4. Parcel 0 = mean(elec0,elec1),
    # parcel 1 = elec2, parcels 2,3 empty/absent.
    raw = torch.zeros(1, 3, 30, 1)
    raw[0, 0] = 2.0
    raw[0, 1] = 4.0
    raw[0, 2] = 9.0
    ppe = torch.tensor([0, 0, 1])
    mask = torch.ones(3, dtype=torch.bool)
    pooled, present = pool_electrodes_to_parcels(raw, ppe, mask, n_parcels=4)
    assert pooled.shape == (1, 4, 30, 1)
    assert torch.allclose(pooled[0, 0], torch.full((30, 1), 3.0))   # (2+4)/2
    assert torch.allclose(pooled[0, 1], torch.full((30, 1), 9.0))
    assert present.tolist() == [True, True, False, False]


def test_pool_electrodes_to_parcels_respects_electrode_mask():
    # Masking electrode 1 (the value-4 one) -> parcel 0 = elec0 only = 2.0.
    raw = torch.zeros(1, 3, 30, 1)
    raw[0, 0] = 2.0
    raw[0, 1] = 4.0
    raw[0, 2] = 9.0
    ppe = torch.tensor([0, 0, 1])
    mask = torch.tensor([True, False, True])
    pooled, present = pool_electrodes_to_parcels(raw, ppe, mask, n_parcels=4)
    assert torch.allclose(pooled[0, 0], torch.full((30, 1), 2.0))
    assert present.tolist() == [True, True, False, False]


# ----------------------------------------------------------------- end-to-end


@dataclasses.dataclass
class _FakeDataset:
    """Minimal ProbeDataset with structured features so AUROC is well-defined."""

    ws_subjects: list[int]
    cs_anchor: int
    cs_test_subjects: list[int]
    tasks: list[str]
    n_parcels: int
    _data: dict[int, SubjectProbeData]

    def subject_data(self, subject_id: int) -> SubjectProbeData:
        return self._data[subject_id]


def _subject(subject_id: int, n: int, c: int, n_parcels: int, *, seed: int):
    g = torch.Generator().manual_seed(seed)
    # balanced ±1 labels in two contiguous halves so each WS fold has both classes
    half = np.array([1.0, -1.0] * (n // 2), dtype=np.float64)[:n]
    labels = {"delta_volume": half.copy()}
    slow = torch.randn(n, c, 2, BANDS[0].n_freq_bins, BANDS[0].n_time_frames, generator=g)
    # inject label signal into the slow magnitude so the probe is non-degenerate
    slow[:, :, 0] += torch.tensor(half, dtype=torch.float32)[:, None, None, None]
    beta = torch.randn(n, c, BANDS[1].n_freq_bins, BANDS[1].n_time_frames, generator=g).abs()
    hg = torch.randn(n, c, BANDS[2].n_freq_bins, BANDS[2].n_time_frames, generator=g).abs()
    ppe = torch.arange(c) % n_parcels
    mask = torch.ones(c, dtype=torch.bool)
    return SubjectProbeData(
        subject_id=subject_id,
        slow=slow,
        beta=beta,
        hg=hg,
        parcel_per_electrode=ppe,
        electrode_mask=mask,
        labels=labels,
        sessions=[(subject_id, 0)],
    )


def _fake_dataset():
    n_parcels = 4
    data = {
        2: _subject(2, 8, 6, n_parcels, seed=0),    # CS anchor
        10: _subject(10, 8, 5, n_parcels, seed=1),  # WS + CS test
        11: _subject(11, 8, 5, n_parcels, seed=2),  # WS only
    }
    return _FakeDataset(
        ws_subjects=[10, 11],
        cs_anchor=2,
        cs_test_subjects=[10],
        tasks=["delta_volume"],
        n_parcels=n_parcels,
        _data=data,
    )


def test_run_raw_baseline_keys_and_ranges():
    out = run_raw_baseline(_fake_dataset())
    assert set(out) == {
        "val_probe/raw/ws/delta_volume",
        "val_probe/raw/cs/delta_volume",
        "val_probe/raw/gap/delta_volume",
    }
    ws = out["val_probe/raw/ws/delta_volume"]
    cs = out["val_probe/raw/cs/delta_volume"]
    assert 0.0 <= ws <= 1.0
    assert 0.0 <= cs <= 1.0
    assert out["val_probe/raw/gap/delta_volume"] == ws - cs


def test_run_raw_baseline_deterministic():
    a = run_raw_baseline(_fake_dataset())
    b = run_raw_baseline(_fake_dataset())
    assert a == b
