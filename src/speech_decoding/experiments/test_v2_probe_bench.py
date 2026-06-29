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
    encode_subject_tokens,
    latent_ws_cs_auroc,
    run_v2_attentive_bench,
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
    front, front_keepS, latent, latent_keepS, pool_keepS, labels = encode_subject_taps(
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
    # pool tap (M3 surface) shares the latent's (P,k,S,d) layout → same keep-S width
    assert pool_keepS.shape == latent_keepS.shape
    assert labels.tolist() == [5, 9, 20]


def test_pool_tagged_is_pool_plus_latent_parcel_embed():
    """M3 attentive surface = bare pool + the latent's OWN parcel embed (no new table)."""
    model = _tiny_model()
    n, c = 6, 4
    poe = torch.tensor([5, 5, 9, 20])
    lfs, hga = _bands(n, c)
    taps = model.encode_clip_taps(lfs, hga, poe, clip_len_s=1.0)
    labels = taps["labels"]                                        # (P,)
    tag = model.latent.parcel_embed(labels)[None, :, None, None, :]
    assert torch.allclose(taps["pool_tagged"], taps["pool"] + tag, atol=1e-6)
    # bare pool is left UNtagged (the ridge rung depends on it)
    assert not torch.allclose(taps["pool_tagged"], taps["pool"])


def test_encode_subject_tokens_shapes_and_teacher():
    model = _tiny_model()
    n, c = 12, 4
    poe = torch.tensor([5, 5, 9, 20])  # parcels {5,9,20}
    bands = _bands(n, c)
    m3, m4, labels = encode_subject_tokens(
        model, bands, poe, clip_len_s=1.0, device=torch.device("cpu"), batch_size=8,
    )
    # full per-token grid kept (N,P,k,S,d): P=3 active parcels, k=2, d=16
    assert m3.shape[0] == n and m3.shape[1] == 3 and m3.shape[2] == 2 and m3.shape[-1] == 16
    assert m4.shape == m3.shape
    assert labels.tolist() == [5, 9, 20]
    # teacher towers are deepcopies at init ⇒ teacher tap == student tap (path works)
    m3_t, m4_t, lab_t = encode_subject_tokens(
        model, bands, poe, clip_len_s=1.0, device=torch.device("cpu"),
        use_teacher=True, batch_size=8,
    )
    assert torch.allclose(m3_t, m3, atol=1e-6)
    assert torch.allclose(m4_t, m4, atol=1e-6)
    assert lab_t.tolist() == labels.tolist()


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
    for tap in ("frontend", "frontend_keepS", "latent", "latent_keepS", "pool_keepS"):
        for split in ("ws", "cs", "gap"):
            assert f"val_probe/{tap}/{split}/delta_volume" in out
    assert all(np.isfinite(v) for v in out.values())

    # taps subset → only the requested families are scored (memory-bounded path).
    sub = run_v2_encoder_taps(
        dataset, model, clip_len_s=1.0, device=torch.device("cpu"), max_iter=300,
        taps=("latent_keepS", "pool_keepS"),
    )
    assert {k.split("/")[1] for k in sub} == {"latent_keepS", "pool_keepS"}
    for tap in ("latent_keepS", "pool_keepS"):
        for split in ("ws", "cs", "gap"):
            assert f"val_probe/{tap}/{split}/delta_volume" in sub
    assert all(np.isfinite(v) for v in sub.values())


def test_run_v2_attentive_bench_smoke():
    """Tiny model + 3 subjects → attentive M3/M4 CS metrics emitted and finite."""
    model = _tiny_model()
    n, c = 24, 4
    poe = torch.tensor([5, 5, 9, 20])
    rng = np.random.default_rng(7)

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
    out = run_v2_attentive_bench(
        dataset, model, clip_len_s=1.0, device=torch.device("cpu"),
        wd_grid=(0.1,), dropout_grid=(0.1,), ls_grid=(0.0,), n_diwa_seeds=1,
        n_heads=4, max_steps=120, eval_every=30, swad_warmup=30, patience=3,
        batch_size=16,
    )
    for surface in ("m3", "m4"):
        assert f"val_probe/attn_{surface}_student/cs/delta_volume" in out
        assert f"val_probe/attn_{surface}_student/cs_std/delta_volume" in out
    assert all(np.isfinite(v) for v in out.values())
