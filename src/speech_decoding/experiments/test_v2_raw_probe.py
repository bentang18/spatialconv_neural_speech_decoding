"""Laptop TDD for the v2 raw-feature probe floor (:mod:`v2_raw_probe`).

Pure torch/numpy/sklearn — no model, no DCC. Pins the RAW RAW contract: every
``|STFT|`` bin exposed (no patch pooling), bands concatenated, electrode→parcel
mean taken at every bin, and the WS/CS logistic protocol end-to-end.
"""

from __future__ import annotations

import types

import numpy as np
import torch

from speech_decoding.experiments.v2_raw_probe import (
    pool_electrodes_to_parcels,
    per_electrode_features,
    raw_bins_from_bands,
    raw_ws_cs_auroc,
    run_v2_raw_baseline,
)


def test_raw_bins_no_pooling_shape_and_order():
    # two bands on DIFFERENT F×T grids (LFS-like 28×5, HGA-like 7×33 shrunk to 4×3 / 2×5).
    n, c = 3, 6
    lfs = torch.arange(n * c * 4 * 3, dtype=torch.float32).reshape(n, c, 4, 3)
    hga = torch.arange(n * c * 2 * 5, dtype=torch.float32).reshape(n, c, 2, 5) + 1000.0
    raw = raw_bins_from_bands([lfs, hga])

    d_raw = 4 * 3 + 2 * 5  # = 22, every bin kept (no kernel pooling)
    assert raw.shape == (n, c, d_raw, 1)
    # band concat: first 12 cols == LFS flattened freq-major/time-minor, next 10 == HGA.
    assert torch.equal(raw[:, :, :12, 0], lfs.reshape(n, c, 12))
    assert torch.equal(raw[:, :, 12:, 0], hga.reshape(n, c, 10))


def test_raw_bins_rejects_non_4d_or_mismatched_nc():
    bad = torch.zeros(3, 6, 2, 4, 5)  # 5-D (e.g. a Cartesian band) — v2 is magnitude only
    try:
        raw_bins_from_bands([bad])
        assert False, "expected ValueError on non-(N,C,F,T) band"
    except ValueError:
        pass
    a = torch.zeros(3, 6, 4, 3)
    b = torch.zeros(3, 5, 2, 5)  # different C
    try:
        raw_bins_from_bands([a, b])
        assert False, "expected ValueError on mismatched (N,C)"
    except ValueError:
        pass


def test_per_electrode_features_drops_invalid_and_flattens():
    n, c, d_raw = 4, 5, 7
    raw = torch.randn(n, c, d_raw, 1)
    mask = torch.tensor([1, 0, 1, 1, 0])  # 3 valid electrodes
    z = per_electrode_features(raw, mask)
    assert z.shape == (n, 3 * d_raw)
    # first valid electrode's bins lead the row.
    assert torch.allclose(z[:, :d_raw], raw[:, 0, :, 0])


def test_pool_to_parcels_mean_at_every_bin():
    # electrodes 0,1 -> parcel 0; electrode 2 -> parcel 1; parcel 2 empty.
    n, c, d_raw, n_parcels = 2, 3, 4, 3
    raw = torch.randn(n, c, d_raw, 1)
    ppe = torch.tensor([0, 0, 1])
    mask = torch.ones(c)
    pooled, present = pool_electrodes_to_parcels(raw, ppe, mask, n_parcels)
    assert pooled.shape == (n, n_parcels, d_raw, 1)
    assert present.tolist() == [True, True, False]
    # parcel 0 == per-bin mean of electrodes 0,1; parcel 1 == electrode 2.
    assert torch.allclose(pooled[:, 0], (raw[:, 0] + raw[:, 1]) / 2)
    assert torch.allclose(pooled[:, 1], raw[:, 2])
    assert torch.allclose(pooled[:, 2], torch.zeros(n, d_raw, 1))


def test_pool_respects_electrode_mask():
    n, c, d_raw, n_parcels = 2, 3, 4, 2
    raw = torch.randn(n, c, d_raw, 1)
    ppe = torch.tensor([0, 0, 1])
    mask = torch.tensor([1.0, 0.0, 1.0])  # electrode 1 invalid -> parcel 0 == electrode 0 only
    pooled, present = pool_electrodes_to_parcels(raw, ppe, mask, n_parcels)
    assert torch.allclose(pooled[:, 0], raw[:, 0])
    assert present.tolist() == [True, True]


def _fake_dataset(seed: int, *, separable: bool):
    """A tiny pure-tensor v2 probe dataset: 3 subjects (anchor 0, test 1, ws {0,1,2}),
    one task. When ``separable`` the +1/-1 classes carry a constant feature offset so
    a linear probe scores AUROC≈1; otherwise labels are random → AUROC≈0.5."""
    rng = np.random.default_rng(seed)
    n, c, n_parcels = 60, 4, 3
    bands_shapes = [(3, 2), (2, 4)]  # two magnitude bands, different F×T
    ppe = torch.tensor([0, 0, 1, 2])  # parcels 0,1,2 all present (shared across subjects)
    mask = torch.ones(c)

    subjects = {}
    for sid in (0, 1, 2):
        y = rng.choice([-1.0, 1.0], size=n)
        bands = []
        for (f, t) in bands_shapes:
            base = torch.from_numpy(rng.standard_normal((n, c, f, t)).astype(np.float32))
            if separable:
                base += torch.from_numpy((y[:, None, None, None] * 3.0).astype(np.float32))
            bands.append(base)
        labels = {"delta_volume": (y if separable else rng.choice([-1.0, 1.0], size=n))}
        subjects[sid] = types.SimpleNamespace(
            subject_id=sid, bands=bands, parcel_per_electrode=ppe,
            electrode_mask=mask, labels=labels,
        )

    return types.SimpleNamespace(
        ws_subjects=[0, 1, 2], cs_anchor=0, cs_test_subjects=[1],
        tasks=["delta_volume"], n_parcels=n_parcels,
        subject_data=lambda s: subjects[s],
    )


def test_end_to_end_separable_scores_high():
    out = run_v2_raw_baseline(_fake_dataset(0, separable=True), max_iter=2000)
    assert set(out) == {
        "val_probe/raw/ws/delta_volume",
        "val_probe/raw/cs/delta_volume",
        "val_probe/raw/gap/delta_volume",
    }
    assert out["val_probe/raw/ws/delta_volume"] > 0.9
    assert out["val_probe/raw/cs/delta_volume"] > 0.9


def test_end_to_end_random_scores_near_chance():
    out = run_v2_raw_baseline(_fake_dataset(1, separable=False), max_iter=2000)
    assert 0.3 < out["val_probe/raw/ws/delta_volume"] < 0.7


def test_raw_ws_cs_auroc_matches_driver():
    ds = _fake_dataset(2, separable=True)
    needed = sorted({ds.cs_anchor, *ds.ws_subjects, *ds.cs_test_subjects})
    sd = {s: ds.subject_data(s) for s in needed}
    raw = {s: raw_bins_from_bands(sd[s].bands) for s in needed}
    direct = raw_ws_cs_auroc(
        raw, sd, ws_subjects=ds.ws_subjects, cs_anchor=ds.cs_anchor,
        cs_test_subjects=ds.cs_test_subjects, tasks=ds.tasks,
        n_parcels=ds.n_parcels, max_iter=2000,
    )
    via_driver = run_v2_raw_baseline(ds, max_iter=2000)
    assert direct == via_driver
