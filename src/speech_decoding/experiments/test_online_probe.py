"""Quantitative TDD for the online linear probe's pure core + callback gating.

Spec: ``reports/online_probe_spec_2026_06_18.md``. Everything here is numpy/torch
only — the load-bearing decision logic (time-binning on the real 38-token clock,
electrode→parcel pooling, dual-form ridge, contiguous 2-fold, parcel intersection,
firewall) is checked with synthetic features against hand-computed expectations,
no BT / DCC. ``run_probe`` + the callback are exercised with a fake tap-model so
the orchestration (WS/CS/gap keys, eval→train restore, OFF-by-default gate) is
verified without building a real encoder.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.experiments import online_probe as op
from speech_decoding.models.v14_converged import token_metadata


# --------------------------------------------------------------- time binning
def test_time_bin_matches_real_converged_clock_k2() -> None:
    """On the real 38-token clock (slow slots {0,8} / beta {0,2,…,14} / hg {0,…,15}),
    K=2 splits at 0.5 s = slot//8. Per-token counts scale by each band's freq-patch
    count: slow 3 freq × (1+1) = (3,3), beta 2 × (4+4) = (8,8), hg 1 × (8+8) = (8,8)
    → 19 early / 19 late of the 38 tokens."""
    band_id, freq_id, time_slot = token_metadata()
    assert int(time_slot.max()) == 15  # span = 16 slots = 1 s on the HG clock
    tb = op.time_bin_of_slot(time_slot, 2)
    for bi, (early, late) in {0: (3, 3), 1: (8, 8), 2: (8, 8)}.items():
        sel = tb[band_id == bi]
        assert int((sel == 0).sum()) == early and int((sel == 1).sum()) == late
    assert int((tb == 0).sum()) == 19 and int((tb == 1).sum()) == 19


def test_time_bin_k1_collapses_all() -> None:
    _, _, time_slot = token_metadata()
    tb = op.time_bin_of_slot(time_slot, 1)
    assert torch.equal(tb, torch.zeros_like(tb))


def test_n_freq_is_six_slow3_beta2_hg1() -> None:
    """The 6 freq-patches the probe keeps (spec §4): slow 3, beta 2, HG 1."""
    band_id, freq_id, _ = token_metadata()
    assert int(freq_id.max()) + 1 == 6
    # freq ids are band-contiguous: slow {0,1,2}, beta {3,4}, hg {5}
    assert set(freq_id[band_id == 0].tolist()) == {0, 1, 2}
    assert set(freq_id[band_id == 1].tolist()) == {3, 4}
    assert set(freq_id[band_id == 2].tolist()) == {5}


# ----------------------------------------------------------------- pooling
def test_pool_electrode_to_parcel_mean_and_present() -> None:
    """Two parcels, 3 electrodes (parcels [0,0,1]), one electrode masked out.
    Parcel 0 = mean of its two VALID electrodes; an absent parcel is present=False."""
    n_freq, k, d = 6, 2, 4
    band_id, freq_id, time_slot = token_metadata()
    n = freq_id.numel()
    feats = torch.zeros(1, 3, n, d)
    feats[0, 0] = 1.0
    feats[0, 1] = 3.0   # masked -> must NOT count
    feats[0, 2] = 5.0
    ppe = torch.tensor([0, 0, 1])
    mask = torch.tensor([True, False, True])
    pooled, present = op.pool_window_features(
        feats, freq_id, time_slot, ppe, mask, n_parcels=4, k=k
    )
    assert pooled.shape == (1, 4, n_freq, k, d)
    assert torch.allclose(pooled[0, 0], torch.ones(n_freq, k, d))      # only elec 0 (elec1 masked)
    assert torch.allclose(pooled[0, 1], torch.full((n_freq, k, d), 5.0))
    assert present.tolist() == [True, True, False, False]
    assert torch.allclose(pooled[0, 2], torch.zeros(n_freq, k, d))     # empty parcel = 0


def test_pool_time_bin_averages_within_half() -> None:
    """A token feature that increases with physical time: pooled K=2 early-bin mean
    < late-bin mean, and K=1 equals the all-token mean (binning is the only diff)."""
    band_id, freq_id, time_slot = token_metadata()
    n = freq_id.numel()
    # feature[token] = its physical slot (so later tokens are larger), one channel
    feats = time_slot.float().reshape(1, 1, n, 1).clone()
    ppe, mask = torch.tensor([0]), torch.tensor([True])
    pooled2, _ = op.pool_window_features(feats, freq_id, time_slot, ppe, mask, 1, 2)
    pooled1, _ = op.pool_window_features(feats, freq_id, time_slot, ppe, mask, 1, 1)
    # hg freq-patch (freq id 5) has slots 0..15: early mean = mean(0..7)=3.5, late=11.5
    hg_fi = 5
    assert pooled2[0, 0, hg_fi, 0, 0].item() == pytest.approx(3.5)
    assert pooled2[0, 0, hg_fi, 1, 0].item() == pytest.approx(11.5)
    assert pooled1[0, 0, hg_fi, 0, 0].item() == pytest.approx(7.5)  # mean(0..15)


# ----------------------------------------------------------- feature matrix / inter
def test_feature_matrix_selects_and_flattens() -> None:
    pooled = torch.arange(1 * 4 * 6 * 2 * 3, dtype=torch.float32).reshape(1, 4, 6, 2, 3)
    z = op.feature_matrix(pooled, [1, 3])
    assert z.shape == (1, 2 * 6 * 2 * 3)
    assert torch.allclose(z[0, : 6 * 2 * 3], pooled[0, 1].reshape(-1))


def test_parcel_intersection() -> None:
    a = torch.tensor([True, True, False, True])
    b = torch.tensor([False, True, True, True])
    assert op.parcel_intersection(a, b).tolist() == [1, 3]


# --------------------------------------------------------------- dual ridge
def test_dual_ridge_default_lambda_is_trace_over_n() -> None:
    """λ defaults to trace(G)/N (spec §5): an explicit λ=trace(G)/N call must match
    the default-λ call exactly."""
    rng = np.random.default_rng(0)
    ztr = rng.standard_normal((8, 20))
    ytr = rng.choice([-1.0, 1.0], 8)
    zte = rng.standard_normal((5, 20))
    g = ztr @ ztr.T
    lam = np.trace(g) / 8
    a = op.dual_ridge_scores(ztr, ytr, zte)
    b = op.dual_ridge_scores(ztr, ytr, zte, lam=lam)
    assert np.allclose(a, b)


def test_dual_ridge_separates_linear_signal() -> None:
    """A label-carrying feature dim → train-set scores rank-order the labels (AUROC
    1.0 on the training points), confirming the dual solve actually fits."""
    rng = np.random.default_rng(1)
    y = np.array([-1.0] * 25 + [1.0] * 25)
    z = 0.05 * rng.standard_normal((50, 10))
    z[:, 0] += y                         # dim 0 carries the label
    pred = op.dual_ridge_scores(z, y, z)
    assert op.auroc(pred, y) > 0.99


def test_auroc_perfect_and_single_class() -> None:
    assert op.auroc(np.array([0.1, 0.2, 0.9, 1.0]), np.array([-1, -1, 1, 1])) == 1.0
    assert np.isnan(op.auroc(np.array([0.1, 0.2]), np.array([1, 1])))  # single-class -> nan


# -------------------------------------------------------------- contiguous folds
def test_contiguous_folds_match_sklearn_kfold() -> None:
    """Parity with upstream's ``KFold(n_splits=2, shuffle=False)`` (spec §5)."""
    from sklearn.model_selection import KFold

    for n in (10, 11, 7):
        ours = op.contiguous_folds(n, 2)
        ref = list(KFold(n_splits=2, shuffle=False).split(np.arange(n)))
        for (tr, te), (rtr, rte) in zip(ours, ref):
            assert np.array_equal(np.sort(tr), np.sort(rtr))
            assert np.array_equal(te, rte)
        # test blocks are contiguous stretches
        assert np.array_equal(ours[0][1], np.arange(0, n - n // 2))


def test_ws_2fold_auroc_high_on_separable() -> None:
    rng = np.random.default_rng(2)
    y = np.concatenate([np.full(20, -1.0), np.full(20, 1.0)])
    rng.shuffle(y)                       # de-correlate label from time order
    z = 0.1 * rng.standard_normal((40, 8))
    z[:, 0] += y
    assert op.ws_auroc_2fold(z, y) > 0.9


# ------------------------------------------------------------------- firewall
def test_firewall_raises_on_lite_overlap() -> None:
    with pytest.raises(AssertionError, match="firewall"):
        op.assert_firewall([(2, 1), (2, 0)], lite_sessions=[(2, 0), (2, 4)])


def test_firewall_passes_clean() -> None:
    op.assert_firewall([(2, 1), (1, 0)], lite_sessions=[(2, 0), (2, 4)])  # no raise


# --------------------------------------------------- fake model + run_probe / callback
class _FakeTokenizer:
    def __init__(self) -> None:
        band_id, freq_id, time_slot = token_metadata()
        self.band_id, self.freq_global_id, self.time_slot = band_id, freq_id, time_slot


class _FakeFrontend:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()


class _FakeModel:
    """Mimics the converged tap interface. ``encode_latent`` writes the per-window
    label (smuggled in ``slow[:,:,0,0,0]``) into feature channel 0 so the latent
    tap is linearly decodable; ``encode_frontend`` returns noise → ~chance. This
    lets us assert latent > frontend and that gap is finite — pure plumbing."""

    def __init__(self) -> None:
        self.student_frontend = _FakeFrontend()
        self.training = True
        self.n_tokens = self.student_frontend.tokenizer.time_slot.numel()

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def encode_frontend(self, slow, beta, hg):
        b, c = slow.shape[0], slow.shape[1]
        g = torch.Generator().manual_seed(int(slow.sum().abs().item()) % 99991)
        return torch.randn(b, c, self.n_tokens, 4, generator=g)

    def encode_latent(self, slow, beta, hg, ppe, *, electrode_mask=None):
        b, c = slow.shape[0], slow.shape[1]
        sig = slow[:, :, 0, 0, 0]                      # (B,C) per-window label, all electrodes
        feats = 0.01 * torch.randn(b, c, self.n_tokens, 4)
        feats[..., 0] += sig[:, :, None]
        return feats


class _FakeDataset:
    def __init__(self) -> None:
        self.ws_subjects = [1, 2, 3]
        self.cs_anchor = 2
        self.cs_test_subjects = [1, 3]
        self.tasks = ["delta_volume"]
        self.n_parcels = 4

    def subject_data(self, sid: int) -> op.SubjectProbeData:
        rng = np.random.default_rng(sid)
        nwin, c = 60, 3
        y = np.concatenate([np.full(30, -1.0), np.full(30, 1.0)])
        rng.shuffle(y)
        slow = torch.zeros(nwin, c, 2, 6, 5)
        slow[:, :, 0, 0, 0] = torch.from_numpy(y).float()[:, None]   # smuggle label
        beta = torch.zeros(nwin, c, 6, 17)
        hg = torch.zeros(nwin, c, 9, 33)
        ppe = torch.tensor([0, 1, 2])         # parcels 0,1,2 present; 3 absent
        mask = torch.ones(c, dtype=torch.bool)
        return op.SubjectProbeData(
            subject_id=sid, slow=slow, beta=beta, hg=hg,
            parcel_per_electrode=ppe, electrode_mask=mask,
            labels={"delta_volume": y}, sessions=[(sid, 1)],
        )


def test_run_probe_keys_and_latent_beats_frontend() -> None:
    m = _FakeModel().eval()
    metrics = op.run_probe(m, _FakeDataset(), k_list=(2, 1))
    # all headline keys present for both taps, both k, the one task
    for tap in ("frontend", "latent"):
        for sfx in ("", "_k1"):
            for kind in ("ws", "cs", "gap"):
                assert f"val_probe/{tap}/{kind}/delta_volume{sfx}" in metrics
    # gap = ws - cs exactly
    g = metrics["val_probe/latent/gap/delta_volume"]
    assert g == pytest.approx(
        metrics["val_probe/latent/ws/delta_volume"]
        - metrics["val_probe/latent/cs/delta_volume"]
    )
    # latent carries the label, frontend is noise: latent WS clearly decodable
    assert metrics["val_probe/latent/ws/delta_volume"] > 0.9
    assert metrics["val_probe/latent/ws/delta_volume"] > metrics["val_probe/frontend/ws/delta_volume"]


def test_callback_off_by_default_is_inert() -> None:
    cb = op.OnlineLinearProbe(_FakeDataset())          # enabled defaults False
    trainer = type("T", (), {"global_rank": 0, "global_step": 1000})()
    assert cb._should_fire(trainer) is False


def test_callback_gating_rank_and_cadence() -> None:
    cb = op.OnlineLinearProbe(_FakeDataset(), enabled=True, cadence=1000)
    fire = lambda rank, step: cb._should_fire(  # noqa: E731
        type("T", (), {"global_rank": rank, "global_step": step})()
    )
    assert fire(0, 1000) is True
    assert fire(0, 2000) is True
    assert fire(0, 0) is False        # step 0 excluded
    assert fire(0, 1500) is False     # off-cadence
    assert fire(1, 1000) is False     # non-zero rank no-op


def test_callback_fire_restores_train_and_logs() -> None:
    logged: dict[str, float] = {}

    class _PL:
        def __init__(self, model):
            self.model = model

        def log(self, name, value, **kw):
            logged[name] = float(value)

    m = _FakeModel()
    m.train()
    pl_mod = _PL(m)
    cb = op.OnlineLinearProbe(_FakeDataset(), enabled=True)
    cb.fire(trainer=None, pl_module=pl_mod)
    assert m.training is True                          # train() restored after eval()
    assert any(k.startswith("val_probe/latent/ws/") for k in logged)
    assert all(np.isfinite(v) for v in logged.values())  # only finite metrics logged


def test_callback_fire_is_fail_soft_and_disables() -> None:
    """A forward error inside the probe must NOT crash the run (spec §1): fire logs,
    returns {} (logs nothing), restores train(), and DISABLES itself so it never
    re-fires every cadence. The diagnostic is never load-bearing."""
    logged: dict[str, float] = {}

    class _BoomModel(_FakeModel):
        def encode_latent(self, *a, **kw):
            raise RuntimeError("simulated forward failure")

    class _PL:
        def __init__(self, model):
            self.model = model

        def log(self, name, value, **kw):
            logged[name] = float(value)

    m = _BoomModel()
    m.train()
    cb = op.OnlineLinearProbe(_FakeDataset(), enabled=True)
    out = cb.fire(trainer=None, pl_module=_PL(m))
    assert out == {}                                   # no metrics returned
    assert logged == {}                                # nothing logged
    assert m.training is True                          # train() restored despite the error
    assert cb.enabled is False                         # disabled → won't re-fire
