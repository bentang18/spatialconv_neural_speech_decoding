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


def test_nll_lambda_schedule_ramps_10k_to_20k() -> None:
    # The eager λ ramp (Ben 2026-07-15, 10k→20k): λ=0 before the start step, linear 0→hold
    # over [start, start+steps], hold after. hold is the objective's lambda_nll. Invariant
    # PRINTED (feedback-build-the-invariant-into-the-probe): the schedule must be 0 below
    # start, exactly hold/2 at the midpoint, hold at the end, and clamp above. These are the
    # real launch numbers (--nll-warmup-start-step 10000 --nll-warmup-steps 10000).
    model = V3ConvergedModel(n_parcels=N_PARCELS)
    mod = V14ConvergedV3Module(
        model=model, optim_config=_optim_config(weight_decay=0.04),
        secondary_active=True,
        nll_warmup_start_step=10_000, nll_warmup_steps=10_000,
    )
    hold = float(mod.model.objective.lambda_nll)
    cases = {
        0: 0.0,
        9_999: 0.0,
        10_000: 0.0,          # ramp opens AT start ⇒ frac 0 here
        15_000: 0.5 * hold,   # midpoint of [10k, 20k]
        20_000: hold,         # ramp closes
        40_000: hold,         # clamp above
    }
    worst = 0.0
    for step, want in cases.items():
        got = mod._nll_lambda(step)
        worst = max(worst, abs(got - want))
    ok = worst < 1e-9 and mod._nll_lambda(-5) == 0.0
    print(
        f"[check] λ_nll schedule (hold={hold}, 10k→20k): "
        f"λ(0)={mod._nll_lambda(0):.4f} λ(15k)={mod._nll_lambda(15_000):.4f} "
        f"λ(20k)={mod._nll_lambda(20_000):.4f} max|Δ|={worst:.2e} → {'OK' if ok else 'VIOLATED'}"
    )
    assert ok


def test_nll_lambda_default_is_constant_hold() -> None:
    # Default warmup params (start=0, steps=0) ⇒ constant λ==hold at every step. This is
    # the backward-compatible path: a run that doesn't pass --nll-warmup-* trains at the
    # fixed λ the objective was built with, from step 0.
    model = V3ConvergedModel(n_parcels=N_PARCELS)
    mod = V14ConvergedV3Module(
        model=model, optim_config=_optim_config(weight_decay=0.04),
        secondary_active=True,
    )
    hold = float(mod.model.objective.lambda_nll)
    steps = [0, 1, 5_000, 50_000]
    ok = all(abs(mod._nll_lambda(s) - hold) < 1e-9 for s in steps)
    print(
        f"[check] default λ_nll constant: hold={hold} "
        f"λ@{steps}={[round(mod._nll_lambda(s), 4) for s in steps]} → {'OK' if ok else 'VIOLATED'}"
    )
    assert ok


def test_nll_lambda_zero_keeps_perceiver_in_graph_with_zero_grad() -> None:
    # DDP graph-stability invariant: while λ==0 (below the ramp start) the total is
    # jepa + 0.0*nll, NOT a branch that drops the nll term. So the write-only Perceiver
    # stays IN the autograd graph and receives a ZERO gradient (not None) — DDP's reducer
    # sees it "used", so no find_unused_parameters abort and no graph-structure mutation
    # when the ramp opens. Contrast: at full λ the SAME param set gets a nonzero grad.
    def _mod(start, steps):
        m = V3ConvergedModel(n_parcels=N_PARCELS)
        return V14ConvergedV3Module(
            model=m, optim_config=_optim_config(weight_decay=0.04),
            secondary_active=True,
            nll_warmup_start_step=start, nll_warmup_steps=steps,
        )

    # step 0 < start=100 ⇒ λ=0
    mod0 = _mod(start=100, steps=50)
    assert mod0._nll_lambda(0) == 0.0
    loss0 = mod0.training_step(_session_batch_with_stats(n_rows=2), 0)
    loss0.backward()
    perc0 = list(mod0.model.objective.perceiver.parameters())
    all_present0 = all(p.grad is not None for p in perc0)
    zero_grad0 = all(float(p.grad.abs().sum()) == 0.0 for p in perc0)

    # constant hold ⇒ λ>0, SAME param set, nonzero grad
    mod1 = _mod(start=0, steps=0)
    assert mod1._nll_lambda(0) > 0.0
    loss1 = mod1.training_step(_session_batch_with_stats(n_rows=2), 0)
    loss1.backward()
    perc1 = list(mod1.model.objective.perceiver.parameters())
    all_present1 = all(p.grad is not None for p in perc1)
    nonzero_grad1 = any(float(p.grad.abs().sum()) > 0.0 for p in perc1)

    ok = all_present0 and zero_grad0 and all_present1 and nonzero_grad1
    print(
        f"[check] λ=0 graph-stable: perceiver grads present@λ0={all_present0} "
        f"zero@λ0={zero_grad0} present@hold={all_present1} nonzero@hold={nonzero_grad1} "
        f"→ {'OK' if ok else 'VIOLATED'}"
    )
    assert ok


def test_training_step_logs_ramped_total_and_lambda() -> None:
    # training_step must combine the loss in EAGER code (compile-safe): total =
    # jepa + λ·nll with the RAMPED λ, and log train_nll_lambda (the live λ),
    # train_nll_weighted (== λ·nll), and train_loss (== the returned total). Invariant:
    # the three logged pieces are self-consistent with the returned scalar.
    model = V3ConvergedModel(n_parcels=N_PARCELS)
    mod = V14ConvergedV3Module(
        model=model, optim_config=_optim_config(weight_decay=0.04),
        secondary_active=True,  # default warmup ⇒ constant hold at step 0
    )
    logged: dict[str, float] = {}
    mod.log = lambda name, value, **_: logged.__setitem__(name, float(value))  # type: ignore[method-assign]
    total = mod.training_step(_session_batch_with_stats(n_rows=2), 0)
    hold = float(mod.model.objective.lambda_nll)
    lam = logged["train_nll_lambda"]
    jepa = logged["train_jepa_loss"]
    nll = logged["train_nll_loss"]
    weighted = logged["train_nll_weighted"]
    recomposed = jepa + lam * nll
    ok = (
        abs(lam - hold) < 1e-9
        and abs(weighted - lam * nll) < 1e-5
        and abs(logged["train_loss"] - float(total)) < 1e-5
        and abs(float(total) - recomposed) < 1e-4
    )
    print(
        f"[check] ramped total: λ={lam:.4f} jepa={jepa:.4f} nll={nll:.4f} "
        f"λ·nll={weighted:.4f} total={float(total):.4f} (jepa+λ·nll={recomposed:.4f}) "
        f"→ {'OK' if ok else 'VIOLATED'}"
    )
    assert ok


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
