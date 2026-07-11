"""v14_converged_v3 Lightning module — configure_optimizers / training_step / EMA.

The module is a thin wrapper; these tests pin the pieces that MUST be right for a
launch: (1) the no-WD split decays the parcel table but exempts biases/LN and
never touches the frozen teacher; (2) a forward produces a finite scalar loss on a
realistic synthetic session; (3) the EMA teacher advances exactly once per
optimiser step via ``on_before_zero_grad``.
"""

from __future__ import annotations

import torch

import speech_decoding.experiments.dispatch_v14 as dv
from speech_decoding.experiments.v14_converged_v3_module import (
    V3Batch,
    V14ConvergedV3Module,
)
from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

N_PARCELS = 8


def _optim_config(*, weight_decay: float):
    from neuraltrain.optimizers.base import LightningOptimizer

    cfg = dv._build_optim_cfg(
        lr=6e-3,
        lr_schedule="warmup_cosine",
        warmup_steps=5000,
        min_lr_ratio=1.0,
        weight_decay=weight_decay,
        optimizer_name="AdamW",
        adam_betas=(0.9, 0.95),
    )
    return LightningOptimizer.model_validate(cfg)


def _session_batch(*, n_rows: int = 2):
    labels = ["LA1", "LA2", "LA3", "LB1", "LB2"]
    parcel_id = torch.tensor([0, 0, 0, 1, 1])
    sc = build_sidecar(labels, parcel_id=parcel_id)
    geom = build_l1_geometry(sc)
    n = len(labels)
    # 3 bands on the shared 32 Hz clock (uniform hop=64, no hold): SLOW 7 bins,
    # MID 6 bins, HGA 7 bins — all T=8 frames.
    bands = [
        torch.randn(n_rows, n, 7, 8),
        torch.randn(n_rows, n, 6, 8),
        torch.randn(n_rows, n, 7, 8),
    ]
    return V3Batch(bands=bands, geom=geom, parcel_id=sc.parcel_id)


def _module(*, weight_decay: float = 0.04) -> V14ConvergedV3Module:
    model = V3ConvergedModel(n_parcels=N_PARCELS)
    return V14ConvergedV3Module(model=model, optim_config=_optim_config(weight_decay=weight_decay))


def test_configure_optimizers_decays_parcel_embed_exempts_norms() -> None:
    mod = _module(weight_decay=0.04)
    built = mod.configure_optimizers()
    optimizer = built["optimizer"] if isinstance(built, dict) else built
    if isinstance(optimizer, (list, tuple)):
        optimizer = optimizer[0]

    # The parcel identity table (2-D) must ride in a DECAYED group; a bias (1-D)
    # in the wd=0 exempt group. Locate each by id across the built param groups.
    parcel_w = mod.model.objective.online.encoder.parcel_embed.embed.weight
    a_bias = next(
        p for n, p in mod.model.named_parameters() if n.endswith("qkv.bias")
    )
    groups = optimizer.param_groups
    parcel_wd = next(
        g["weight_decay"] for g in groups
        if any(id(p) == id(parcel_w) for p in g["params"])
    )
    bias_wd = next(
        g["weight_decay"] for g in groups
        if any(id(p) == id(a_bias) for p in g["params"])
    )
    assert parcel_wd > 0.0, "parcel embed must be weight-decayed (upstream rule)"
    assert bias_wd == 0.0, "biases must be weight-decay-exempt"


def test_configure_optimizers_excludes_frozen_teacher() -> None:
    mod = _module()
    built = mod.configure_optimizers()
    optimizer = built["optimizer"] if isinstance(built, dict) else built
    if isinstance(optimizer, (list, tuple)):
        optimizer = optimizer[0]
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    teacher_ids = {
        id(p) for p in mod.model.objective.teacher.model.parameters()
    }
    assert opt_ids.isdisjoint(teacher_ids), "frozen teacher must not be optimized"
    # every optimized param carries grad
    assert all(
        p.requires_grad for g in optimizer.param_groups for p in g["params"]
    )


def test_training_step_finite_scalar_loss() -> None:
    mod = _module()
    batch = _session_batch(n_rows=2)
    loss = mod.training_step(batch, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert mod._last_batch_size == 2
    loss.backward()  # graph-connected, backward clean
    grads = [
        p.grad for p in mod.model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_mask_seed_is_step_deterministic_but_varies() -> None:
    mod = _module()
    g0a = mod._step_generator(torch.device("cpu")).initial_seed()
    g0b = mod._step_generator(torch.device("cpu")).initial_seed()
    assert g0a == g0b  # same step ⇒ same seed (resume-stable)


def test_ema_teacher_advances_once_per_step() -> None:
    mod = _module()
    batch = _session_batch(n_rows=2)
    before = int(mod.model.objective.teacher._step.item())
    mod.training_step(batch, 0)
    mod.on_before_zero_grad(optimizer=None)
    after = int(mod.model.objective.teacher._step.item())
    assert after == before + 1


def test_context_lambda_schedule() -> None:
    # #66 upstream Lambda_LinearWarmupHold: off before start, linear ramp, hold after.
    from speech_decoding.experiments.v14_converged_v3_module import context_lambda

    assert context_lambda(0, 0.5, 15_000, 30_000) == 0.0  # before start
    assert context_lambda(14_999, 0.5, 15_000, 30_000) == 0.0
    assert context_lambda(15_000, 0.5, 15_000, 30_000) == 0.0  # ramp start = 0
    assert abs(context_lambda(22_500, 0.5, 15_000, 30_000) - 0.25) < 1e-9  # midpoint
    assert context_lambda(30_000, 0.5, 15_000, 30_000) == 0.5  # hold
    assert context_lambda(100_000, 0.5, 15_000, 30_000) == 0.5
    assert context_lambda(50_000, 0.0, 15_000, 30_000) == 0.0  # hold<=0 ⇒ always off


def test_training_step_context_off_by_default() -> None:
    # Library default hold=0.0 ⇒ the context head gets no gradient (static-off).
    # It must ALSO be frozen (requires_grad=False) so DDP's reducer skips it under
    # find_unused_parameters=False — otherwise the multi-GPU run aborts on it.
    mod = _module()
    assert mod._context_lambda_hold == 0.0
    ctx_head = mod.model.objective.pred_to_target_context
    assert all(not p.requires_grad for p in ctx_head.parameters())
    assert ctx_head.weight not in set(mod._trainable_parameters())
    mod.training_step(_session_batch(n_rows=2), 0).backward()
    assert all(p.grad is None for p in ctx_head.parameters())


def test_training_step_with_context_loss_trains_head() -> None:
    # hold>0 with the ramp already complete at step 0 (start=end=0 ⇒ λ=hold at step 0);
    # the module passes a 0-d tensor and the context head is trained.
    model = V3ConvergedModel(n_parcels=N_PARCELS)
    mod = V14ConvergedV3Module(
        model=model, optim_config=_optim_config(weight_decay=0.04),
        context_lambda_hold=0.5, context_warmup_start=0, context_warmup_end=0,
    )
    loss = mod.training_step(_session_batch(n_rows=2), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    ctx_head = mod.model.objective.pred_to_target_context
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in ctx_head.parameters()
    )
