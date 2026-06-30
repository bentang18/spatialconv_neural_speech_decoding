"""Tests for the attentive per-cell readouts (Stage 2b)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.experiments.pretrain_probe_attentive import (
    attentive_cs_cell_auroc,
    attentive_ws_cell_auroc,
)
from speech_decoding.experiments.v2_attentive_probe import AttentiveProbeHead
from speech_decoding.experiments.v2_attentive_train import HeadTrainConfig

N_PARCELS = 4
S = 2
K = 2
D = 12


def _cfg() -> HeadTrainConfig:
    # Small/fast; strong-signal synthetic data needs few steps. n_time_frames is
    # overridden to S by the readout, so leave it 0 here.
    return HeadTrainConfig(
        d_model=D, n_heads=6, parcel_dropout=0.2, attn_dropout=0.1, mlp_dropout=0.1,
        residual_dropout=0.1, lr=3e-3, weight_decay=0.1, batch_size=64,
        max_steps=400, eval_every=25, patience=8, swad_warmup=50, seed=0,
    )


def _rows(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cut = n // 2
    held = np.arange(cut, n)
    vcut = len(held) // 2
    return np.arange(cut), held[:vcut], held[vcut:]


def _labels(rng, n: int) -> np.ndarray:
    return rng.choice([0.0, 1.0], size=n)


def _electrode_grid(rng, y, c, signal_elec, *, snr=4.0):
    """(N,C,S,d) grid; the signal electrode's frame-0 channel-0 encodes the label."""
    n = len(y)
    g = rng.normal(size=(n, c, S, D)).astype(np.float32)
    g[:, signal_elec, 0, 0] = ((y * 2 - 1) * snr + 0.2 * rng.normal(size=n)).astype(np.float32)
    return torch.from_numpy(g)


def _parcel_grid(rng, y, p, signal_parcel, *, snr=4.0):
    """(N,P,k,S,d) grid; the signal parcel's seed-0 frame-0 channel-0 encodes the label."""
    n = len(y)
    g = rng.normal(size=(n, p, K, S, D)).astype(np.float32)
    g[:, signal_parcel, 0, 0, 0] = ((y * 2 - 1) * snr + 0.2 * rng.normal(size=n)).astype(np.float32)
    return torch.from_numpy(g)


def test_group_dropout_runs_and_keeps_rows_nonempty():
    """The irregular-group parcel-dropout path drops whole anatomical parcels and
    never leaves a clip with zero tokens."""
    torch.manual_seed(0)
    head = AttentiveProbeHead(D, n_heads=6, parcel_dropout=0.5).train()
    x = torch.randn(8, 6, D)                         # 6 tokens
    pids = torch.tensor([0, 0, 0, 1, 1, 2])          # irregular groups {3,2,1}
    out = head(x, token_parcel_ids=pids)
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()


def test_ws_electrode_recovers_signal():
    rng = np.random.default_rng(0)
    n, c = 120, 6
    y = _labels(rng, n)
    grid = _electrode_grid(rng, y, c, signal_elec=2)
    tr, va, te = _rows(n)
    pe = torch.tensor([0, 0, 1, 1, 2, 2])
    em = torch.ones(c, dtype=torch.bool)
    a = attentive_ws_cell_auroc(
        grid, y, train_rows=tr, val_rows=va, test_rows=te, tap_space="electrode",
        parcel_per_electrode=pe, electrode_mask=em, n_parcels=N_PARCELS, cfg=_cfg(),
    )
    assert a > 0.8


def test_ws_parcel_recovers_signal():
    rng = np.random.default_rng(1)
    n = 120
    y = _labels(rng, n)
    grid = _parcel_grid(rng, y, N_PARCELS, signal_parcel=1)
    tr, va, te = _rows(n)
    pe = torch.tensor([0, 0, 1, 1, 2, 2])  # parcels 0,1,2 supported; 3 not (masked)
    em = torch.ones(6, dtype=torch.bool)
    a = attentive_ws_cell_auroc(
        grid, y, train_rows=tr, val_rows=va, test_rows=te, tap_space="parcel",
        parcel_per_electrode=pe, electrode_mask=em, n_parcels=N_PARCELS,
        parcel_labels=torch.arange(N_PARCELS), cfg=_cfg(),
    )
    assert a > 0.8


def test_cs_electrode_full_grid_no_intersect_recovers_signal():
    """CS electrode: train on the anchor's full electrode set, test on a different
    montage's full set (no pool, no intersect) — shared signal electrode transfers."""
    rng = np.random.default_rng(2)
    na, nt = 100, 100
    pe_a = torch.tensor([0, 0, 1, 1, 2, 2])
    pe_t = torch.tensor([0, 0, 1, 1, 2, 2, 3])  # test has an extra electrode (diff T)
    ya, yt = _labels(rng, na), _labels(rng, nt)
    ga = _electrode_grid(rng, ya, 6, signal_elec=0)
    gt = _electrode_grid(rng, yt, 7, signal_elec=0)
    _, va, te = _rows(nt)
    a = attentive_cs_cell_auroc(
        ga, ya, gt, yt, val_rows=va, test_rows=te, tap_space="electrode",
        pe_anchor=pe_a, em_anchor=torch.ones(6, dtype=torch.bool),
        pe_test=pe_t, em_test=torch.ones(7, dtype=torch.bool),
        n_parcels=N_PARCELS, cfg=_cfg(),
    )
    assert a > 0.75


def test_cs_parcel_full_grid_no_intersect_recovers_signal():
    rng = np.random.default_rng(3)
    na, nt = 100, 100
    pe = torch.tensor([0, 0, 1, 1, 2, 2])
    em = torch.ones(6, dtype=torch.bool)
    ya, yt = _labels(rng, na), _labels(rng, nt)
    ga = _parcel_grid(rng, ya, N_PARCELS, signal_parcel=1)
    gt = _parcel_grid(rng, yt, N_PARCELS, signal_parcel=1)
    _, va, te = _rows(nt)
    a = attentive_cs_cell_auroc(
        ga, ya, gt, yt, val_rows=va, test_rows=te, tap_space="parcel",
        pe_anchor=pe, em_anchor=em, pe_test=pe, em_test=em,
        n_parcels=N_PARCELS, parcel_labels_anchor=torch.arange(N_PARCELS),
        parcel_labels_test=torch.arange(N_PARCELS), cfg=_cfg(),
    )
    assert a > 0.75


def test_nan_labels_dropped():
    rng = np.random.default_rng(4)
    n, c = 120, 6
    y = _labels(rng, n)
    y[::4] = np.nan
    grid = _electrode_grid(rng, np.nan_to_num(y, nan=1.0), c, signal_elec=2)
    tr, va, te = _rows(n)
    pe = torch.tensor([0, 0, 1, 1, 2, 2])
    a = attentive_ws_cell_auroc(
        grid, y, train_rows=tr, val_rows=va, test_rows=te, tap_space="electrode",
        parcel_per_electrode=pe, electrode_mask=torch.ones(c, dtype=torch.bool),
        n_parcels=N_PARCELS, cfg=_cfg(),
    )
    assert np.isfinite(a)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
