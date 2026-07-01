"""Laptop TDD for the attentive-head training core (:mod:`v2_attentive_train`).

Synthetic separable token sets: train_head should reach high val AUROC, early-stop
before max_steps when val plateaus, produce a valid SWAD mean, apply label smoothing
without breaking, and diwa_average should fuse independent heads into a working one.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from speech_decoding.experiments.v2_attentive_probe import AttentiveProbeHead
from speech_decoding.experiments.v2_attentive_train import (
    HeadTrainConfig,
    diwa_average,
    loso_pooled_cs,
    pad_concat,
    train_head,
)


def _separable(seed: int, *, n=240, t=8, d=16, separable=True):
    rng = np.random.default_rng(seed)
    y = rng.choice([0.0, 1.0], size=n).astype(np.float32)
    x = rng.standard_normal((n, t, d)).astype(np.float32)
    if separable:
        # Shift HALF the dims by the label — a direction (first-half vs second-half
        # contrast) that survives the head's pre-norm LN(X). A uniform all-dim shift is a
        # pure per-token mean offset LN would remove exactly, collapsing the task to chance.
        x[..., : d // 2] += (2.0 * y[:, None, None] - 1.0) * 1.3
    return torch.from_numpy(x), torch.from_numpy(y)


def _split(x, y, frac=0.7):
    n = x.shape[0]
    cut = int(n * frac)
    return x[:cut], y[:cut], x[cut:], y[cut:]


def test_train_head_learns_and_swad_valid():
    x, y = _separable(0)
    xtr, ytr, xv, yv = _split(x, y)
    cfg = HeadTrainConfig(d_model=16, n_heads=4, attn_dropout=0.0, mlp_dropout=0.0,
                          residual_dropout=0.0, token_dropout=0.0, lr=3e-3,
                          weight_decay=0.05, max_steps=1500, eval_every=50,
                          swad_warmup=100, patience=8, seed=0)
    out = train_head(xtr, ytr, xv, yv, cfg)
    assert out["best_val"] > 0.9
    assert out["swad_val"] > 0.85          # averaged head also generalizes
    # the swad state loads into a fresh head and scores high on its own
    head = AttentiveProbeHead(16, n_heads=4).eval()
    head.load_state_dict(out["swad_state"])
    with torch.no_grad():
        s = head(xv).squeeze(-1).numpy()
    assert roc_auc_score(yv.numpy(), s) > 0.85


def test_train_head_early_stops_on_plateau():
    """Random data → val never improves → stop well before max_steps."""
    x, y = _separable(1, separable=False)
    xtr, ytr, xv, yv = _split(x, y)
    cfg = HeadTrainConfig(d_model=16, n_heads=4, max_steps=5000, eval_every=20,
                          swad_warmup=40, patience=5, seed=1)
    out = train_head(xtr, ytr, xv, yv, cfg)
    assert out["steps"] < 5000             # early stop fired


def test_label_smoothing_runs_and_is_ranking_neutral_ish():
    x, y = _separable(2)
    xtr, ytr, xv, yv = _split(x, y)
    base = HeadTrainConfig(d_model=16, n_heads=4, attn_dropout=0.0, mlp_dropout=0.0,
                           residual_dropout=0.0, token_dropout=0.0, lr=3e-3,
                           max_steps=1200, eval_every=50, swad_warmup=100, seed=3)
    out0 = train_head(xtr, ytr, xv, yv, base)
    ls = HeadTrainConfig(**{**base.__dict__, "label_smoothing": 0.1})
    out_ls = train_head(xtr, ytr, xv, yv, ls)
    # both reach high AUROC; LS doesn't wreck ranking on a separable binary task
    assert out0["best_val"] > 0.9 and out_ls["best_val"] > 0.9


def test_diwa_average_fuses_heads():
    x, y = _separable(4)
    xtr, ytr, xv, yv = _split(x, y)
    states = []
    for sd in (0, 1, 2):
        cfg = HeadTrainConfig(d_model=16, n_heads=4, attn_dropout=0.0, mlp_dropout=0.0,
                              residual_dropout=0.0, token_dropout=0.0, lr=3e-3,
                              max_steps=1000, eval_every=50, swad_warmup=100, seed=sd)
        states.append(train_head(xtr, ytr, xv, yv, cfg)["swad_state"])
    fused = diwa_average(states)
    head = AttentiveProbeHead(16, n_heads=4).eval()
    head.load_state_dict(fused)
    with torch.no_grad():
        s = head(xv).squeeze(-1).numpy()
    assert roc_auc_score(yv.numpy(), s) > 0.85


def test_pad_concat_shapes_and_mask():
    a = torch.randn(3, 4, 8)
    b = torch.randn(2, 7, 8)
    x, m = pad_concat([a, b])
    assert x.shape == (5, 7, 8)
    assert m.shape == (5, 7)
    assert m[:3, :4].all() and not m[:3, 4:].any()   # subject A: 4 valid, 3 pad
    assert m[3:].all()                                # subject B: all 7 valid
    assert torch.allclose(x[:3, :4], a)              # real tokens preserved
    assert torch.allclose(x[:3, 4:], torch.zeros(3, 3, 8))  # pad is zero


def _multi_subject(seed, *, separable, n_sub=5, d=12):
    """Subjects with DIFFERENT token counts; token-mean encodes a SHARED label rule."""
    rng = np.random.default_rng(seed)
    tokens, labels = {}, {}
    for s in range(n_sub):
        n = 120
        t = 5 + s * 2                                # 5,7,9,11,13 tokens → padding path
        y = rng.choice([0.0, 1.0], size=n).astype(np.float32)
        x = rng.standard_normal((n, t, d)).astype(np.float32)
        if separable:
            x[:, :, 0] += (2.0 * y - 1.0)[:, None] * 1.6   # shared discriminative dim
        tokens[s] = torch.from_numpy(x)
        labels[s] = y
    return tokens, labels, list(range(n_sub))


def test_loso_pooled_cs_separable_beats_chance():
    tokens, labels, subs = _multi_subject(0, separable=True)
    base = HeadTrainConfig(d_model=12, n_heads=4, attn_dropout=0.0, mlp_dropout=0.0,
                           residual_dropout=0.0, token_dropout=0.0, lr=3e-3,
                           max_steps=600, eval_every=50, swad_warmup=100, patience=6)
    grid = [{"weight_decay": 0.05, "label_smoothing": 0.0}]
    out = loso_pooled_cs(tokens, labels, subs, base, hp_grid=grid, n_diwa_seeds=1)
    assert len(out["folds"]) == 5
    assert out["cs_mean"] > 0.8
    # firewall: the test subject is never in the fit pool (selection or final refit)
    for f in out["folds"]:
        assert f["test_s"] not in f["fit_subjects"]
        assert set(f["fit_subjects"]) == set(subs) - {f["test_s"]}


def test_loso_pooled_cs_random_near_chance():
    tokens, labels, subs = _multi_subject(1, separable=False)
    base = HeadTrainConfig(d_model=12, n_heads=4, max_steps=400, eval_every=40,
                           swad_warmup=80, patience=4)
    grid = [{"weight_decay": 0.1, "label_smoothing": 0.0}]
    out = loso_pooled_cs(tokens, labels, subs, base, hp_grid=grid, n_diwa_seeds=1)
    assert 0.3 < out["cs_mean"] < 0.7


def test_loso_pooled_cs_hp_selection_picks_by_val():
    """Two-HP grid runs end-to-end and records the selected hp per fold."""
    tokens, labels, subs = _multi_subject(2, separable=True, n_sub=4)
    base = HeadTrainConfig(d_model=12, n_heads=4, attn_dropout=0.0, mlp_dropout=0.0,
                           residual_dropout=0.0, token_dropout=0.0, lr=3e-3,
                           max_steps=400, eval_every=50, swad_warmup=80, patience=5)
    grid = [{"weight_decay": 0.01, "label_smoothing": 0.0},
            {"weight_decay": 1.0, "label_smoothing": 0.1}]
    out = loso_pooled_cs(tokens, labels, subs, base, hp_grid=grid, n_diwa_seeds=1)
    assert len(out["folds"]) == 4
    for f in out["folds"]:
        assert f["hp"] in grid


def test_diwa_average_is_parameter_mean():
    a = {"w": torch.zeros(3), "b": torch.ones(2)}
    b = {"w": torch.ones(3) * 2, "b": torch.ones(2) * 3}
    out = diwa_average([a, b])
    assert torch.allclose(out["w"], torch.ones(3))      # mean(0,2)=1
    assert torch.allclose(out["b"], torch.ones(2) * 2)  # mean(1,3)=2
