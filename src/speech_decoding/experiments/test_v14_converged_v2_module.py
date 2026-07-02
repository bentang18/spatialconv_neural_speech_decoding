"""TDD for the converged-v2 neuraltrain wiring: the ``V14ConvergedV2Net`` config
build() + the thin ``V14ConvergedV2BrainModule`` seams (2-band ingest with the
homogeneity/no-padding contracts, the loss path, grad isolation student-vs-teacher,
the EMA tick, optimiser construction with the teacher excluded)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from neuraltrain.optimizers import LightningOptimizer

# Import registers WarmupCosine with the BaseLRScheduler discriminated union so
# LightningOptimizer(scheduler={"name": "WarmupCosine", ...}) validates (same
# registration path dispatch_v14 relies on).
from speech_decoding.experiments.lr_schedule import WarmupCosine  # noqa: F401
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


def _tiny_run_b_model() -> V14ConvergedV2:
    return V14ConvergedV2Net(
        d_model=16, n_heads=4, frontend_layers=1, latent_layers=1,
        m2_pred_layers=1, m4_pred_layers=1, pred_dim=16, n_parcels=N_PARCELS,
        m3_drop_frac=0.4, m3_min_keep=1,
    ).build(0, 0)


def _batch(B: int = 3, *, clip_len_s: float = 5.0, valid: torch.Tensor | None = None,
           heterogeneous: bool = False, coords: bool = False):
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
    if coords:
        c = torch.randn(C, 3) * 10.0                 # session-static native-RAS mm
        data["electrode_coords"] = c[None].expand(B, C, 3).contiguous()
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
    lfs, hga, poe, _coords = m._v2_inputs(_batch(B=3).data)
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
    lfs, hga, poe, _coords = m._v2_inputs(_batch(B=3, valid=valid).data)
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


# --------------------------------------------------------------- Run-B wiring
def test_v2_inputs_returns_coords_when_present():
    m = V14ConvergedV2BrainModule(
        model=_tiny_run_b_model(), optim_config=_optim(), clip_len_s=5.0,
    )
    lfs, hga, poe, coords = m._v2_inputs(_batch(B=3, coords=True).data)
    assert coords is not None and coords.shape == (6, 3)


def test_v2_inputs_rejects_misaligned_coords():
    m = _module()
    data = _batch(B=2, coords=True).data
    data["electrode_coords"] = data["electrode_coords"][:, :5]   # C mismatch
    with pytest.raises(ValueError, match="align with support"):
        m._v2_inputs(data)


def test_run_b_step_produces_melec_loss():
    m = V14ConvergedV2BrainModule(
        model=_tiny_run_b_model(), optim_config=_optim(), clip_len_s=5.0,
    )
    out = m._step(_batch(B=3, coords=True).data)
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    assert "loss_melec" in out and torch.isfinite(out["loss_melec"])


def test_run_b_step_requires_coords():
    m = V14ConvergedV2BrainModule(
        model=_tiny_run_b_model(), optim_config=_optim(), clip_len_s=5.0,
    )
    with pytest.raises(ValueError, match="requires electrode_coords"):
        m._step(_batch(B=2, coords=False).data)


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


def _find_warmup_scheduler(opt):
    """Dig the _WarmupCosineLR out of whatever configure_optimizers returned."""
    if isinstance(opt, dict):
        sc = opt.get("lr_scheduler")
        if isinstance(sc, dict):
            sc = sc.get("scheduler")
        return sc
    return None


def test_configure_optimizers_threads_warmup_horizon():
    """Regression for the dropped-total_steps bug: WarmupCosine must receive the
    training horizon from the trainer, else warmup_steps clamps to 0 and the LR is
    a flat constant (the --warmup-steps no-op that caused the Run-B smoke storm)."""
    warmup, total = 50, 200
    m = _module()
    m.optim_config = LightningOptimizer(
        optimizer={"name": "AdamW", "lr": 1e-3, "kwargs": {"weight_decay": 0.0}},
        scheduler={"name": "WarmupCosine", "warmup_steps": warmup,
                   "min_lr_ratio": 1.0},
        interval="step",
    )
    m._trainer = SimpleNamespace(estimated_stepping_batches=total)  # noqa: SLF001

    opt = m.configure_optimizers()
    sched = _find_warmup_scheduler(opt)
    assert sched is not None, "scheduler missing from configure_optimizers"
    # The bug: total_steps=None -> warmup_steps clamped to 0. The fix threads the
    # horizon so the warmup window is preserved.
    assert sched.warmup_steps == warmup, sched.warmup_steps
    assert sched.total_steps == total

    opt_obj = opt["optimizer"] if isinstance(opt, dict) else opt
    peak = 1e-3
    lr0 = opt_obj.param_groups[0]["lr"]
    assert lr0 < peak, f"LR should ramp from ~0, got {lr0}"     # not flat at peak
    for _ in range(warmup):
        opt_obj.step()
        sched.step()
    lr_after = opt_obj.param_groups[0]["lr"]
    assert abs(lr_after - peak) < 1e-9, lr_after                # reached peak


def test_configure_optimizers_no_trainer_is_safe():
    """No attached trainer (unit path) must not crash: horizon resolves to None
    and the build degrades gracefully rather than raising."""
    m = _module()
    m.optim_config = LightningOptimizer(
        optimizer={"name": "AdamW", "lr": 1e-3, "kwargs": {"weight_decay": 0.0}},
        scheduler={"name": "WarmupCosine", "warmup_steps": 50, "min_lr_ratio": 1.0},
        interval="step",
    )
    opt = m.configure_optimizers()      # no _trainer set
    assert opt is not None


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


# ----------------------------------------------- science-neutral speedup wiring
def test_speedups_off_by_default_eager_bytewise():
    """No env ⇒ no compile spec, empty _compiled_fwd, _call_model IS self.model."""
    m = _module()
    assert m._compile_spec is None
    assert m._compiled_fwd == {}
    assert m._sdpa_backend_name is None
    # _call_model with no compile spec routes straight to self.model.
    out = m._step(_batch(B=2).data)
    assert torch.isfinite(out["loss"])


def test_sdpa_backend_read_from_env(monkeypatch):
    """V14_SDPA_BACKEND is read once at construction; the forward still runs (on
    CPU the cuDNN context degrades to nullcontext — identical math)."""
    monkeypatch.setenv("V14_SDPA_BACKEND", "cudnn")
    m = _module()
    assert m._sdpa_backend_name == "cudnn"
    out = m._step(_batch(B=2).data)
    assert torch.isfinite(out["loss"])


def test_compile_spec_resolves_from_env(monkeypatch):
    """--compile --no-compile-dynamic ⇒ V14_COMPILE=1, V14_COMPILE_DYNAMIC=0 ⇒
    _compile_spec = ("default", False) (fully static, the required v2 setting)."""
    monkeypatch.setenv("V14_COMPILE", "1")
    monkeypatch.setenv("V14_COMPILE_DYNAMIC", "0")
    m = _module()
    assert m._compile_spec == ("default", False)
    # the OptimizedModule is a plain-dict entry, NOT a registered submodule ⇒ no
    # `_orig_mod.` prefix leaks into the checkpoint / param set.
    assert "model" in m._compiled_fwd
    assert not any("_orig_mod" in k for k in m.state_dict())


def test_getstate_drops_optimized_module(monkeypatch):
    """__getstate__ returns a pickle-safe copy with _compiled_fwd cleared (dynamo
    guard weakrefs are unpicklable); _compile_spec survives for the lazy rebuild."""
    monkeypatch.setenv("V14_COMPILE", "1")
    monkeypatch.setenv("V14_COMPILE_DYNAMIC", "0")
    m = _module()
    state = m.__getstate__()
    assert state["_compiled_fwd"] == {}
    assert state["_compile_spec"] == ("default", False)
    assert m._compiled_fwd != {}                          # original untouched (a copy)


def test_compiled_forward_runs_and_matches_eager():
    """End-to-end: the compiled forward traces + runs on CPU and its loss matches
    the SAME model's eager forward (compile is math-neutral). dynamic=False keeps
    dims concrete so the data-dependent gathers don't storm the symbolic solver.
    Compares one instance (identical weights) with the mask generator reset before
    each call so both draw the same masks; eval() silences dropout."""
    m = _module()
    m.eval()

    def _loss() -> float:
        m._mask_gen.manual_seed(m._mask_seed)
        with torch.no_grad():
            return m._step(_batch(B=2).data)["loss"].item()

    eager = _loss()
    m._compile_spec = ("default", False)
    m._build_compiled_model()
    try:
        compiled = _loss()
    finally:
        torch._dynamo.reset()
    assert m._compiled_fwd != {}                          # the OptimizedModule built
    assert abs(compiled - eager) < 1e-4
