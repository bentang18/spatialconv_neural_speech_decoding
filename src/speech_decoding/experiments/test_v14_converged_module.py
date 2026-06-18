"""TDD for V14ConvergedBrainModule — the Philosophy-B Lightning shell around the
self-contained V14ConvergedSSL. Quantitative checks of every seam: batch ingest
(support→parcel, valid→electrode_mask), the loss path, grad isolation
student-vs-teacher, the EMA tick, optimizer construction (teacher excluded +
no-WD split), the encoder phase-handoff round-trip, and overfit-one-batch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from neuraltrain.optimizers import LightningOptimizer

from speech_decoding.experiments.v14_converged_module import V14ConvergedBrainModule
from speech_decoding.models.v14_converged import V14ConvergedSSL


# --------------------------------------------------------------------- fixtures
def _tiny_model(n_parcels: int = 3) -> V14ConvergedSSL:
    return V14ConvergedSSL(
        16, n_parcels,
        n_heads=4,
        frontend_layers=1,
        latent_layers=1,
        m2_pred_dim=16, m2_pred_layers=1,
        m4_pred_dim=16, m4_pred_layers=1,
    )


def _optim(weight_decay: float = 0.0) -> LightningOptimizer:
    return LightningOptimizer(
        optimizer={"name": "AdamW", "lr": 1e-3,
                   "kwargs": {"weight_decay": weight_decay}}
    )


def _module(n_parcels: int = 3, **kw) -> V14ConvergedBrainModule:
    return V14ConvergedBrainModule(
        model=_tiny_model(n_parcels), optim_config=_optim(kw.pop("wd", 0.0)),
        ema_tau=kw.pop("ema_tau", 0.99), **kw,
    )


def _batch(B: int = 2, C: int = 6, K: int = 3, *, valid: torch.Tensor | None = None):
    """Synthetic 3-band batch on the locked BandSpec geometry. ``support`` is a
    one-hot assigning 2 electrodes to each of K parcels (so M4 tubes a real
    multi-electrode parcel and the electrode-mean is non-trivial)."""
    torch.manual_seed(0)
    slow = torch.randn(B, C, 2, 6, 5)
    beta = torch.randn(B, C, 6, 17)
    hg = torch.randn(B, C, 9, 33)
    support = torch.zeros(B, C, K)
    for e in range(C):
        support[:, e, e % K] = 1.0
    data = {
        "electrode_tokens_slow": slow,
        "electrode_tokens_beta": beta,
        "electrode_tokens_hg": hg,
        "support": support,
    }
    if valid is not None:
        data["valid_mask"] = valid
    return SimpleNamespace(data=data)


# ----------------------------------------------------------------- batch ingest
def test_converged_inputs_derive_parcel_and_mask() -> None:
    m = _module()
    data = _batch(B=2, C=6, K=3).data
    slow, beta, hg, ppe, emask = m._converged_inputs(data)
    assert slow.shape == (2, 6, 2, 6, 5)
    assert beta.shape == (2, 6, 6, 17) and hg.shape == (2, 6, 9, 33)
    # support one-hot e%K → parcel ids [0,1,2,0,1,2]
    assert ppe[0].tolist() == [0, 1, 2, 0, 1, 2]
    # no valid_mask → all electrodes real
    assert emask.dtype == torch.bool and bool(emask.all())


def test_converged_inputs_honor_valid_mask() -> None:
    m = _module()
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[0, 5] = False  # one padded electrode
    _, _, _, _, emask = m._converged_inputs(_batch(valid=valid).data)
    assert not bool(emask[0, 5]) and bool(emask[1, 5])


# -------------------------------------------------------------------- loss path
def test_step_returns_finite_nonneg_losses() -> None:
    m = _module()
    out = m._step(_batch().data)
    for k in ("loss", "l_m2", "l_m4"):
        assert torch.isfinite(out[k]) and float(out[k]) >= 0.0


def test_step_emits_per_band_diagnostics() -> None:
    # Ben 2026-06-18: per-band M2 (beta/hg) + M4 (slow/beta/hg) loss breakdowns
    # ride in the _step output so _log_losses can surface them.
    m = _module()
    out = m._step(_batch().data)
    for k in ("l_m2_beta", "l_m2_hg", "l_m4_slow", "l_m4_beta", "l_m4_hg"):
        assert k in out, f"missing per-band diagnostic {k}"
        assert torch.isfinite(out[k]) and float(out[k]) >= 0.0
    assert "l_m2_slow" not in out          # slow is M2-exempt


def test_training_step_returns_scalar_loss() -> None:
    m = _module()
    loss = m.training_step(_batch(), 0)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_grad_flows_to_student_not_teacher() -> None:
    m = _module()
    out = m._step(_batch().data)
    out["loss"].backward()
    teacher_ids = {id(p) for p in m.model.teacher_frontend.parameters()}
    got_student_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for _, p in m.model.named_parameters()
        if id(p) not in teacher_ids and p.requires_grad
    )
    assert got_student_grad, "student must receive gradient"
    assert all(p.grad is None for p in m.model.teacher_frontend.parameters()), \
        "frozen EMA teacher must never accumulate gradient"


# --------------------------------------------------------------------- EMA tick
def test_ema_tick_moves_teacher_toward_student() -> None:
    m = _module(ema_tau=0.5)
    # perturb the student frontend so it differs from the (cloned) teacher
    with torch.no_grad():
        for p in m.model.student_frontend.parameters():
            p.add_(torch.randn_like(p))
    t_before = next(iter(m.model.teacher_frontend.parameters())).clone()
    s = next(iter(m.model.student_frontend.parameters())).clone()
    m.on_before_zero_grad(optimizer=None)
    t_after = next(iter(m.model.teacher_frontend.parameters()))
    # τ=0.5 ⇒ teacher = 0.5·teacher + 0.5·student → strictly between, toward student
    assert not torch.allclose(t_after, t_before)
    assert (t_after - s).abs().sum() < (t_before - s).abs().sum()


# ------------------------------------------------------------------- optimizer
def _groups(m: V14ConvergedBrainModule):
    out = m.configure_optimizers()
    opt = out["optimizer"] if isinstance(out, dict) else out
    return opt.param_groups


def test_configure_optimizers_excludes_frozen_teacher() -> None:
    m = _module()
    optimized = {id(p) for g in _groups(m) for p in g["params"]}
    teacher_ids = {id(p) for p in m.model.teacher_frontend.parameters()}
    assert optimized.isdisjoint(teacher_ids), "EMA teacher must not be optimised"
    trainable = {id(p) for p in m.model.parameters() if p.requires_grad}
    assert optimized == trainable


def test_configure_optimizers_no_wd_split_at_positive_wd() -> None:
    m = _module(wd=0.1)
    groups = _groups(m)
    wds = {g["weight_decay"] for g in groups}
    assert 0.0 in wds and 0.1 in wds
    from speech_decoding.experiments.optim_param_groups import no_decay_param_ids
    exempt = no_decay_param_ids(m.model)
    for g in groups:
        for p in g["params"]:
            assert g["weight_decay"] == (0.0 if id(p) in exempt else 0.1)


def test_configure_optimizers_single_group_at_zero_wd() -> None:
    m = _module(wd=0.0)
    groups = _groups(m)
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0.0


# -------------------------------------------------------------- phase handoff
def test_transferable_state_roundtrips_encoder() -> None:
    src = _module()
    # train the source a touch so its encoder weights are non-trivial
    out = src._step(_batch().data)
    out["loss"].backward()
    with torch.no_grad():
        for p in src.model.student_frontend.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-0.1)
    state = src.transferable_state()
    assert set(state) == {"student_frontend", "latent"}

    dst = _module()
    dst.load_transferable_state(state)
    for (na, pa), (_, pb) in zip(
        src.model.student_frontend.named_parameters(),
        dst.model.student_frontend.named_parameters(),
    ):
        assert torch.allclose(pa, pb), f"frontend {na} not transferred"
    # teacher re-synced to the loaded student frontend
    for (_, ps), (nt, pt) in zip(
        dst.model.student_frontend.named_parameters(),
        dst.model.teacher_frontend.named_parameters(),
    ):
        assert torch.allclose(ps, pt), f"teacher not re-synced at {nt}"


def test_load_transferable_state_missing_component_raises() -> None:
    m = _module()
    with pytest.raises(KeyError):
        m.load_transferable_state({"latent": {}})


# ---------------------------------------------------------------- overfit check
def test_overfits_one_batch_fixed_masks() -> None:
    """The decisive veracity check: with the mask draw frozen (re-seed each step)
    the shipped default config (M4 precision weight ON) must drive a single
    batch's loss steeply down.

    Seed BEFORE the build so the random init is fixed — otherwise the ratio is
    flaky run-to-run. The M4 precision weight downweights the synthetic montage's
    n=2 parcels to ((2-1)/(11-1))^1 = 0.1, so M4 contributes a 10×-attenuated
    signal and the batch overfits mostly through M2; 150 steps clears 0.5 with a
    comfortable margin (ratio ≈ 0.37)."""
    torch.manual_seed(0)
    m = _module()
    batch = _batch()
    opt = torch.optim.Adam(
        [p for p in m.model.parameters() if p.requires_grad], lr=3e-3,
    )
    losses = []
    for _ in range(150):
        m._mask_gen.manual_seed(0)  # freeze the mask so the target is stationary
        opt.zero_grad()
        out = m._step(batch.data)
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"]))
    assert losses[-1] < 0.5 * losses[0], (
        f"one-batch loss did not overfit: {losses[0]:.4f} → {losses[-1]:.4f}"
    )
