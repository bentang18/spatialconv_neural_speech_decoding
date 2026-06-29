"""Laptop TDD for the v2 encoder-tap bench (:mod:`v2_probe_bench`).

Two layers, no DCC / no real ckpt:
  - the pure per-parcel latent WS/CS logistic (:func:`latent_ws_cs_auroc`) on
    synthetic features (separable → AUROC≈1; random → ≈0.5);
  - an end-to-end smoke on a TINY randomly-initialised :class:`V14ConvergedV2`:
    :meth:`encode_clip_taps` → :func:`encode_subject_taps` → :func:`run_v2_encoder_taps`,
    pinning the tap shapes + that every ``val_probe/{frontend,latent}/...`` metric is
    emitted and finite (a random net only scores ≈chance — this is a wiring test, not
    a performance claim).
"""

from __future__ import annotations

import types

import numpy as np
import torch

from speech_decoding.experiments.v2_probe_bench import (
    encode_subject_taps,
    latent_ws_cs_auroc,
    run_v2_encoder_taps,
)
from speech_decoding.models.v14_converged_v2 import (
    V14ConvergedV2,
    V14ConvergedV2Config,
    bands_for_clip_len,
)


def _separable_latent(seed: int, *, separable: bool):
    """3 subjects, parcels {5,9,20} present in all; one task. Latent (N,P,d)."""
    rng = np.random.default_rng(seed)
    n, d = 60, 8
    labels_ids = torch.tensor([5, 9, 20])
    subjects, latent, lab = {}, {}, {}
    for sid in (0, 1, 2):
        y = rng.choice([-1.0, 1.0], size=n)
        feat = rng.standard_normal((n, 3, d)).astype(np.float32)
        if separable:
            feat += (y[:, None, None] * 3.0).astype(np.float32)
        latent[sid] = torch.from_numpy(feat)
        lab[sid] = labels_ids
        yv = y if separable else rng.choice([-1.0, 1.0], size=n)
        subjects[sid] = types.SimpleNamespace(labels={"delta_volume": yv})
    return latent, lab, subjects


def test_latent_ws_cs_separable_scores_high():
    latent, lab, sd = _separable_latent(0, separable=True)
    out = latent_ws_cs_auroc(
        latent, lab, sd, ws_subjects=[0, 1, 2], cs_anchor=0, cs_test_subjects=[1, 2],
        tasks=["delta_volume"], n_parcels=62, max_iter=2000,
    )
    assert out["val_probe/latent/ws/delta_volume"] > 0.9
    assert out["val_probe/latent/cs/delta_volume"] > 0.9


def test_latent_ws_cs_random_near_chance():
    latent, lab, sd = _separable_latent(1, separable=False)
    out = latent_ws_cs_auroc(
        latent, lab, sd, ws_subjects=[0, 1, 2], cs_anchor=0, cs_test_subjects=[1, 2],
        tasks=["delta_volume"], n_parcels=62, max_iter=2000,
    )
    assert 0.3 < out["val_probe/latent/ws/delta_volume"] < 0.7


def _tiny_model():
    cfg = V14ConvergedV2Config(
        d_model=16, n_heads=4, frontend_layers=2, latent_layers=2,
        m2_pred_layers=2, m4_pred_layers=2, pred_dim=16, n_parcels=62, k=2,
        tube_ratio=0.35,
    )
    torch.manual_seed(0)
    return V14ConvergedV2(cfg).eval()


def _bands(n, c, clip_s=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = bands_for_clip_len(clip_s)
    lfs = torch.randn(n, c, b[0].n_freq_bins, b[0].n_time_frames, generator=g)
    hga = torch.randn(n, c, b[1].n_freq_bins, b[1].n_time_frames, generator=g)
    return [lfs, hga]


def test_encode_subject_taps_shapes():
    model = _tiny_model()
    n, c = 20, 4
    poe = torch.tensor([5, 5, 9, 20])  # parcels {5,9,20}
    mask = torch.ones(c)
    front, front_keepS, latent, latent_keepS, labels = encode_subject_taps(
        model, _bands(n, c), poe, mask, 62,
        clip_len_s=1.0, device=torch.device("cpu"), batch_size=8,
    )
    assert front.shape == (n, c, 16)          # token-pooled per-electrode (N,C,d)
    assert latent.shape == (n, 3, 16)         # token-pooled per-parcel (N,P,d)
    assert latent_keepS.shape[:2] == (n, 3)   # keep-S per-parcel (N,P,k·S·d)
    assert latent_keepS.shape[2] % 16 == 0    # multiple of d
    assert latent_keepS.shape[2] > 16         # cells kept → wider than pooled d
    assert front_keepS.shape[:2] == (n, 3)    # keep-S frontend, parcel-pooled (N,P,S·d)
    assert front_keepS.shape[2] % 16 == 0     # multiple of d
    assert front_keepS.shape[2] == latent_keepS.shape[2] // 2  # S·d == (k·S·d)/k, k=2
    assert labels.tolist() == [5, 9, 20]


def test_run_v2_encoder_taps_emits_all_metrics():
    model = _tiny_model()
    n, c = 30, 4
    poe = torch.tensor([5, 5, 9, 20])
    rng = np.random.default_rng(3)

    def subject(sid):
        y = rng.choice([-1.0, 1.0], size=n)
        return types.SimpleNamespace(
            bands=_bands(n, c, seed=sid), parcel_per_electrode=poe,
            electrode_mask=torch.ones(c), labels={"delta_volume": y},
        )

    subjects = {s: subject(s) for s in (0, 1, 2)}
    dataset = types.SimpleNamespace(
        ws_subjects=[0, 1, 2], cs_anchor=0, cs_test_subjects=[1, 2],
        tasks=["delta_volume"], n_parcels=62, subject_data=lambda s: subjects[s],
    )
    out = run_v2_encoder_taps(
        dataset, model, clip_len_s=1.0, device=torch.device("cpu"), max_iter=300,
    )
    for tap in ("frontend", "frontend_keepS", "latent", "latent_keepS"):
        for split in ("ws", "cs", "gap"):
            assert f"val_probe/{tap}/{split}/delta_volume" in out
    assert all(np.isfinite(v) for v in out.values())
