"""TDD for the converged-v2 neuraltrain wiring: the ``V14ConvergedV2Net`` config
build() + the thin ``V14ConvergedV2BrainModule`` seams (2-band ingest with the
homogeneity/no-padding contracts, the loss path, grad isolation student-vs-teacher,
the EMA tick, optimiser construction with the teacher excluded)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from neuraltrain.optimizers import LightningOptimizer

from speech_decoding.experiments.v14_converged_v2_module import (
    V14ConvergedV2BrainModule,
)
from speech_decoding.models.v14_converged_v2 import (
    V14ConvergedV2,
    bands_for_clip_len,
)
from speech_decoding.models.v14_converged_v2_config import V14ConvergedV2Net

N_PARCELS = 62


def _tiny_model() -> V14ConvergedV2:
    return V14ConvergedV2Net(
        d_model=16,
        n_heads=4,
        frontend_layers=1,
        latent_layers=1,
        m2_pred_layers=1,
        m4_pred_layers=1,
        pred_dim=16,
        n_parcels=N_PARCELS,
    ).build(0, 0)


def _optim(weight_decay: float = 0.0) -> LightningOptimizer:
    return LightningOptimizer(
        optimizer={"name": "AdamW", "lr": 1e-3,
                   "kwargs": {"weight_decay": weight_decay}}
    )


def _module(clip_len_s: float = 5.0, **kw) -> V14ConvergedV2BrainModule:
    return V14ConvergedV2BrainModule(
        model=_tiny_model(), optim_config=_optim(kw.pop("wd", 0.0)),
        clip_len_s=clip_len_s, **kw,
    )


def _batch(B: int = 3, *, clip_len_s: float = 5.0, valid: torch.Tensor | None = None,
           heterogeneous: bool = False):
    """Synthetic session-homogeneous 2-band batch on the locked geometry. 6
    electrodes → 3 parcels (2 each) via a one-hot support CONSTANT across clips."""
    torch.manual_seed(0)
    bands = bands_for_clip_len(clip_len_s)
    lfs_b, hga_b = bands
    C, K = 6, N_PARCELS
    lfs = torch.randn(B, C, lfs_b.n_freq_bins, lfs_b.n_time_frames)
    hga = torch.randn(B, C, hga_b.n_freq_bins, hga_b.n_time_frames)
    support = torch.zeros(B, C, K)
    labels = [5, 5, 9, 9, 20, 20]
    for e in range(C):
        support[:, e, labels[e]] = 1.0
    if heterogeneous:
        support[1, 0] = 0.0
        support[1, 0, 41] = 1.0          # clip 1 electrode 0 → different parcel
    data = {
        "electrode_tokens_lfs": lfs,
        "electrode_tokens_hga": hga,
        "support": support,
    }
    if valid is not None:
        data["valid_mask"] = valid
    return SimpleNamespace(data=data)


# ------------------------------------------------------------------ config build
def test_config_build_returns_model_that_forwards():
    model = _tiny_model()
    assert isinstance(model, V14ConvergedV2)
    assert model.cfg.k == 2 and model.cfg.tube_ratio == 0.25  # locked defaults


def test_config_shape_fields_required():
    with pytest.raises(Exception):                            # missing d_model etc.
        V14ConvergedV2Net(n_parcels=N_PARCELS).build(0, 0)


# ------------------------------------------------------------------ batch ingest
def test_v2_inputs_derive_homogeneous_parcel():
    m = _module()
    lfs, hga, poe = m._v2_inputs(_batch(B=3).data)
    assert lfs.shape[1] == 6 and hga.shape[1] == 6
    assert poe.shape == (6,)
    assert poe.tolist() == [5, 5, 9, 9, 20, 20]


def test_v2_inputs_reject_heterogeneous_batch():
    m = _module()
    with pytest.raises(ValueError, match="session-homogeneous"):
        m._v2_inputs(_batch(B=3, heterogeneous=True).data)


def test_v2_inputs_drops_unmapped_electrode():
    """An unmapped electrode (valid_mask False across ALL clips) is DROPPED at
    ingest, not rejected — v2 has no masked-electrode path, and the
    session-homogeneous batch drops the same row for every clip (uniform C kept)."""
    m = _module()
    valid = torch.ones(3, 6, dtype=torch.bool)
    valid[:, 5] = False                                       # electrode 5 unmapped
    lfs, hga, poe = m._v2_inputs(_batch(B=3, valid=valid).data)
    assert lfs.shape[1] == 5 and hga.shape[1] == 5
    assert poe.tolist() == [5, 5, 9, 9, 20]                   # electrode 5 (parcel 20) gone


def test_v2_inputs_reject_nonconstant_valid_mask():
    """valid_mask varying across clips violates session-homogeneity → fail loud
    (a single (C,) drop set would otherwise be ambiguous)."""
    m = _module()
    valid = torch.ones(3, 6, dtype=torch.bool)
    valid[0, 5] = False                                       # only clip 0 differs
    with pytest.raises(ValueError, match="constant across clips"):
        m._v2_inputs(_batch(B=3, valid=valid).data)


# -------------------------------------------------------------------- loss path
def test_step_returns_finite_loss_dict():
    m = _module()
    out = m._step(_batch(B=3).data)
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    for key in ("loss_m2", "loss_m4", "loss_m4_tubed", "loss_m4_untubed"):
        assert key in out


def test_step_1s_clip():
    m = _module(clip_len_s=1.0)
    out = m._step(_batch(B=2, clip_len_s=1.0).data)
    assert torch.isfinite(out["loss"])


def test_mask_timing_probe_finite_and_flushes(monkeypatch):
    """V14_MASK_TIMING arms the kineto-free per-section probe without breaking the
    loss path: finite loss every step, and the buffer flushes (clears) once the
    step count hits the cadence."""
    monkeypatch.setenv("V14_MASK_TIMING", "2")
    m = _module()
    m.train()
    assert m._mask_timing_every == 2
    for _ in range(2):
        out = m._step(_batch(B=2).data)
        assert torch.isfinite(out["loss"])
    assert all(len(v) == 0 for v in m._mask_timing_buf.values())  # flushed at cadence


def test_mask_timing_off_by_default():
    """No env ⇒ zero overhead: the probe never arms and the buffer stays empty."""
    m = _module()
    m.train()
    assert m._mask_timing_every == 0 and m._step_timing is False
    m._step(_batch(B=2).data)
    assert all(len(v) == 0 for v in m._mask_timing_buf.values())


def test_grad_flows_to_student_not_teacher():
    m = _module()
    m._step(_batch(B=2).data)["loss"].backward()
    sg = [p.grad for p in m.model.frontend.parameters() if p.grad is not None]
    assert sg and all(torch.isfinite(g).all() for g in sg)
    assert all(p.grad is None for p in m.model.teacher_frontend.parameters())


def test_ema_tick_moves_teacher():
    m = _module()
    with torch.no_grad():
        for p in m.model.frontend.parameters():
            p.add_(1.0)
    tp_ = next(m.model.teacher_frontend.parameters())
    sp = next(m.model.frontend.parameters())
    before = (tp_ - sp).abs().mean().item()
    m.on_before_zero_grad(optimizer=None)
    assert (tp_ - sp).abs().mean().item() < before


def test_configure_optimizers_excludes_teacher():
    m = _module(wd=0.1)
    opt = m.configure_optimizers()
    opt_obj = opt["optimizer"] if isinstance(opt, dict) else opt
    n_opt = sum(p.numel() for g in opt_obj.param_groups for p in g["params"])
    n_student = sum(p.numel() for p in m.model.parameters() if p.requires_grad)
    assert n_opt == n_student                                # teacher excluded


def test_overfit_one_batch():
    """The thin shell can drive the model down on a fixed batch (smoke for the
    full ingest→forward→backward→opt→ema loop)."""
    m = _module()
    opt = torch.optim.AdamW(
        [p for p in m.model.parameters() if p.requires_grad], lr=3e-3
    )
    batch = _batch(B=3)
    first = m._step(batch.data)["loss"].item()
    for _ in range(15):
        loss = m._step(batch.data)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        m.model.ema_step()
    assert m._step(batch.data)["loss"].item() < first
