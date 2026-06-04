from __future__ import annotations

import math

import torch

from neuraltrain.optimizers.base import LightningOptimizer

from speech_decoding.experiments.lr_schedule import WarmupCosine, _WarmupCosineLR


def _opt(*lrs: float) -> torch.optim.Optimizer:
    groups = [{"params": [torch.zeros(1, requires_grad=True)], "lr": lr} for lr in lrs]
    return torch.optim.SGD(groups, lr=lrs[0])


def _lrs_over(sched: _WarmupCosineLR, steps: int) -> list[list[float]]:
    out = [list(sched.get_last_lr())]
    for _ in range(steps):
        sched.step()
        out.append(list(sched.get_last_lr()))
    return out


def test_warmup_then_cosine_shape_single_group() -> None:
    """Linear ramp 0→peak over warmup, cosine→0 after — at the documented steps."""
    sched = _WarmupCosineLR(_opt(1.0), warmup_steps=10, total_steps=110, min_lr_ratio=0.0)
    lrs = _lrs_over(sched, 110)  # last_epoch 0..110

    assert math.isclose(lrs[0][0], 0.1, rel_tol=1e-6)   # step 0 → 1/warmup
    assert math.isclose(lrs[9][0], 1.0, rel_tol=1e-6)   # warmup end → peak
    assert math.isclose(lrs[10][0], 1.0, rel_tol=1e-6)  # cosine start → peak
    assert math.isclose(lrs[60][0], 0.5, abs_tol=1e-6)  # cosine midpoint → 0.5·peak
    assert math.isclose(lrs[110][0], 0.0, abs_tol=1e-9)  # end → 0


def test_no_warmup_is_pure_cosine() -> None:
    sched = _WarmupCosineLR(_opt(2.0), warmup_steps=0, total_steps=100, min_lr_ratio=0.0)
    lrs = _lrs_over(sched, 100)
    assert math.isclose(lrs[0][0], 2.0, rel_tol=1e-6)     # starts at peak
    assert math.isclose(lrs[50][0], 1.0, abs_tol=1e-6)    # half-cosine → 0.5·peak
    assert math.isclose(lrs[100][0], 0.0, abs_tol=1e-9)


def test_min_lr_ratio_preserves_discriminative_ratio() -> None:
    """A non-zero cosine floor must scale EACH group by its own base LR, so the
    P2/P3 two-group discriminative-LR ratio survives (the bug an absolute
    CosineAnnealingLR ``eta_min`` would introduce)."""
    base_hi, base_lo = 3e-4, 3e-5  # ratio 0.1 (front-end = base/10)
    sched = _WarmupCosineLR(
        _opt(base_hi, base_lo), warmup_steps=5, total_steps=50, min_lr_ratio=0.1
    )
    lrs = _lrs_over(sched, 50)
    # End of cosine: multiplier == min_lr_ratio for BOTH groups → ratio intact.
    assert math.isclose(lrs[50][0], base_hi * 0.1, rel_tol=1e-6)
    assert math.isclose(lrs[50][1], base_lo * 0.1, rel_tol=1e-6)
    for hi, lo in lrs:  # ratio is the same at every step
        assert math.isclose(lo / hi, 0.1, rel_tol=1e-6)


def test_warmup_clamped_below_total() -> None:
    """A short smoke run can request warmup >= total; the cosine window stays
    >= 1 step rather than dividing by zero."""
    sched = _WarmupCosineLR(_opt(1.0), warmup_steps=200, total_steps=5, min_lr_ratio=0.0)
    assert sched.warmup_steps == 4  # clamped to total-1
    _lrs_over(sched, 5)  # does not raise


def test_config_build_with_total_steps() -> None:
    cfg = WarmupCosine(warmup_steps=10, min_lr_ratio=0.0)
    sched = cfg.build(_opt(1.0), total_steps=110)
    assert isinstance(sched, _WarmupCosineLR)
    assert sched.warmup_steps == 10 and sched.total_steps == 110


def test_config_build_without_total_steps_is_constant() -> None:
    """No trainer attached (unit-test / no-budget path) → constant LR, no crash."""
    cfg = WarmupCosine(warmup_steps=10)
    sched = cfg.build(_opt(1.0))  # total_steps absent, field None
    lrs = _lrs_over(sched, 20)
    assert all(math.isclose(g[0], 1.0, rel_tol=1e-6) for g in lrs)


def test_config_field_total_steps_fallback() -> None:
    cfg = WarmupCosine(warmup_steps=2, total_steps=20)
    sched = cfg.build(_opt(1.0))  # build kwarg absent → use the field
    assert sched.total_steps == 20


def test_discriminator_resolves_warmup_cosine() -> None:
    """``{"name": "WarmupCosine"}`` resolves through exca's DiscriminatedModel
    (by class __name__), and the LightningOptimizer threads total_steps through
    to the scheduler build."""
    lopt = LightningOptimizer.model_validate(
        {
            "optimizer": {"name": "AdamW", "lr": 5e-4, "kwargs": {"weight_decay": 0.05, "betas": [0.9, 0.95]}},
            "scheduler": {"name": "WarmupCosine", "warmup_steps": 5, "min_lr_ratio": 0.0},
            "interval": "step",
        }
    )
    assert isinstance(lopt.scheduler, WarmupCosine)
    params = [torch.zeros(1, requires_grad=True)]
    out = lopt.build(params, total_steps=100)
    assert "lr_scheduler" in out
    assert isinstance(out["lr_scheduler"]["scheduler"], _WarmupCosineLR)
    assert out["lr_scheduler"]["interval"] == "step"
    assert out["lr_scheduler"]["scheduler"].total_steps == 100


def test_constant_schedule_omits_scheduler() -> None:
    """A LightningOptimizer with no scheduler key builds optimizer-only (the
    prior constant-LR default — confirms the warmup path is strictly opt-in)."""
    lopt = LightningOptimizer.model_validate(
        {"optimizer": {"name": "Adam", "lr": 1e-3, "kwargs": {"fused": False}}}
    )
    assert lopt.scheduler is None
    out = lopt.build([torch.zeros(1, requires_grad=True)], total_steps=100)
    assert "lr_scheduler" not in out
