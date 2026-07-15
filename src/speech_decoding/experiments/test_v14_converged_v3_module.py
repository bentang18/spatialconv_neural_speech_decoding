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


def _session_batch_with_stats(*, n_rows: int = 2):
    # Same synthetic session as _session_batch, plus per-parcel frozen z-score stats
    # ALREADY gathered to (P, 6) for the two present parcels (as build_session_setup
    # would produce). std=1 ⇒ z-score is identity, so the target stays finite.
    batch = _session_batch(n_rows=n_rows)
    n_parcels_present = int(batch.parcel_id.unique().numel())
    return V3Batch(
        bands=batch.bands,
        geom=batch.geom,
        parcel_id=batch.parcel_id,
        stat_mean=torch.zeros(n_parcels_present, 6),
        stat_std=torch.ones(n_parcels_present, 6),
    )


def test_secondary_off_by_default_freezes_perceiver() -> None:
    # Default secondary_active=False ⇒ no stats ever reach forward, so the write-only
    # Perceiver head gets no gradient. It must be FROZEN (requires_grad=False) so DDP's
    # reducer skips it under find_unused_parameters=False — else the multi-GPU run aborts.
    mod = _module()
    assert mod._secondary_active is False
    perceiver = mod.model.objective.perceiver
    assert perceiver is not None  # deep_sup default ON ⇒ the head exists
    assert all(not p.requires_grad for p in perceiver.parameters())
    trainable = set(mod._trainable_parameters())
    assert all(p not in trainable for p in perceiver.parameters())
    out_loss = mod.training_step(_session_batch(n_rows=2), 0)  # JEPA-only, no stats
    out_loss.backward()
    assert all(p.grad is None for p in perceiver.parameters())


def test_secondary_active_trains_perceiver_on_stats() -> None:
    # secondary_active=True + per-session stats in the batch ⇒ the Gaussian-NLL fires and
    # the write-only Perceiver receives gradient. The total is finite and graph-connected.
    model = V3ConvergedModel(n_parcels=N_PARCELS)  # deep_sup default ON
    mod = V14ConvergedV3Module(
        model=model, optim_config=_optim_config(weight_decay=0.04),
        secondary_active=True,
    )
    perceiver = mod.model.objective.perceiver
    assert all(p.requires_grad for p in perceiver.parameters())  # NOT frozen
    loss = mod.training_step(_session_batch_with_stats(n_rows=2), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in perceiver.parameters()
    )


def test_grad_ratio_off_by_default_logs_nothing() -> None:
    # grad_ratio_every_n_steps default 0 ⇒ the live loss-balance readout never fires,
    # even with the secondary active. The launch config relies on this default.
    model = V3ConvergedModel(n_parcels=N_PARCELS)
    mod = V14ConvergedV3Module(
        model=model, optim_config=_optim_config(weight_decay=0.04),
        secondary_active=True,
    )
    assert mod.grad_ratio_every_n_steps == 0
    logged: dict[str, object] = {}
    mod.log = lambda name, value, **_: logged.__setitem__(name, value)  # type: ignore[method-assign]
    loss = mod.training_step(_session_batch_with_stats(n_rows=2), 0)
    loss.backward()  # graph still intact (nothing consumed it)
    assert not any(k.startswith("train_mon_grad") or k == "train_mon_g_jepa" for k in logged)


def test_grad_ratio_logs_balance_and_retains_graph_single_process() -> None:
    # grad_ratio_every_n_steps=1 + secondary active + no trainer (world_size==1) ⇒ the
    # readout fires: ‖g_nll‖/‖g_jepa‖ and λ·ratio are logged, both grad norms > 0, and the
    # graph SURVIVES for Lightning's own backward (retain_graph in the helper). Invariant
    # printed (feedback-build-the-invariant-into-the-probe).
    model = V3ConvergedModel(n_parcels=N_PARCELS)
    mod = V14ConvergedV3Module(
        model=model, optim_config=_optim_config(weight_decay=0.04),
        secondary_active=True, grad_ratio_every_n_steps=1,
    )
    logged: dict[str, float] = {}
    mod.log = lambda name, value, **_: logged.__setitem__(name, float(value))  # type: ignore[method-assign]
    loss = mod.training_step(_session_batch_with_stats(n_rows=2), 0)
    loss.backward()  # must not raise — the probe retained the graph
    online = list(mod.model.objective.online.parameters())
    grads_flow = any(p.grad is not None and p.grad.abs().sum() > 0 for p in online)

    r = logged["train_mon_grad_ratio"]
    rw = logged["train_mon_grad_ratio_weighted"]
    lam = float(mod.model.objective.lambda_nll)
    ok = (
        logged["train_mon_g_jepa"] > 0.0
        and logged["train_mon_g_nll"] > 0.0
        and r > 0.0
        and abs(rw - lam * r) < 1e-5
        and grads_flow
    )
    print(
        f"[check] wired grad-ratio: ‖g_jepa‖={logged['train_mon_g_jepa']:.4e} "
        f"‖g_nll‖={logged['train_mon_g_nll']:.4e} ratio={r:.3f} λ·ratio={rw:.3f} "
        f"(λ={lam}) backward-after-probe={grads_flow} → {'OK' if ok else 'VIOLATED'}"
    )
    assert ok
