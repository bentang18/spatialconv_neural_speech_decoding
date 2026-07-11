"""SSL-health monitor for v14_converged_v3 — a lean Lightning ``Callback``.

The v2 ``SSLHealthMonitor`` is welded to the v2 schema (``model.frontend/latent/
m2_predictor`` groups, M2/M4 tap keys, ``electrode_tokens_lfs/hga`` bands). v3 has a
different structure (online stem+encoder / predictor towers, DKT-tag identity,
whole-vs-intra mask tiers), so this is a v3-bound sibling that REUSES the stateless
helpers — :func:`grad_spike_monitor` and :func:`teacher_rank_monitor` — and v3's own
``_ln_target``. Nothing is copied from the v2 callback beyond those shared helpers.

Family A — model-agnostic (grads/params), in ``on_before_optimizer_step`` at the
monitor cadence (true_update_ratio + update_cos run every step to span their
consecutive-step windows):
  * grad spike        — total grad-L2 over trainable params.
  * ema_weight_gap    — ``‖θ_online − θ_teacher‖₂ / ‖θ_online‖₂`` (collapse guard).
  * true_update_ratio — per group (online / predictor) ``‖Δθ‖₂/‖θ_prev‖₂`` across two
    consecutive optimiser steps (LR-calibration; ~1e-3 healthy).
  * update_cos        — per group ``cos(Δθ₁, Δθ₂)`` over consecutive updates (HVP-free
    edge-of-stability: → −1 too-hot, → +1 smooth descent).
  * grad_noise_scale  — McCandlish B_simple from AdamW's moment EMAs.

Family B — tap-dependent (detached forward taps), in ``on_train_batch_end``, reading
``pl_module._last_taps`` (present ⇒ this is a cadence step; the module already
decided, so no off-by-one re-derivation):
  * rankme + feat_std at encoder block 12 (terminal representation) — effective-rank
    / feature-std collapse guard. (The block-3 pre-L2 tap was removed 2026-07-10.)
  * explained_var + pred_target_var_ratio + L1, split whole-sensor vs intra-sensor —
    is L2 actually learning the cross-sensor task, or is the encoder cheating locally?

Keys follow the ``train_mon_<name>`` convention. Off-cost when not due: the module
only produces taps on cadence steps, and the snapshots are cadence-gated.
"""

from __future__ import annotations

import torch
from lightning import pytorch as pl
from torch import Tensor

from speech_decoding.experiments.monitors.grad_spike import grad_spike_monitor
from speech_decoding.experiments.monitors.teacher_rank import teacher_rank_monitor

# v3 routing groups for update-ratio / cosine — the two trainable towers.
_ROUTING_GROUPS: tuple[str, ...] = ("online", "predictor")
_STATS_EPS: float = 1e-8


def _group(pl_module: pl.LightningModule, name: str):
    """The named v3 trainable tower (``online`` = stem+encoder, ``predictor``)."""
    obj = pl_module.model.objective
    return {"online": obj.online, "predictor": obj.predictor}.get(name)


class SSLHealthMonitorV3(pl.Callback):
    def __init__(self, *, every_n_steps: int = 1) -> None:
        super().__init__()
        self.every_n_steps = max(int(every_n_steps), 1)
        self._grad_ema_l2: float = 0.0
        self._update_snapshot: dict[str, list[Tensor]] | None = None
        self._cos_snap: dict[str, list[Tensor]] | None = None
        self._cos_delta_a: dict[str, list[Tensor]] | None = None

    def _due(self, step: int) -> bool:
        return step % self.every_n_steps == 0

    # --------------------------------------------------- Family A (grads/params)
    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer
    ) -> None:
        step = int(pl_module.global_step)
        self._maybe_log_true_update_ratio(pl_module, step)
        self._maybe_log_update_cosine(pl_module, step)
        if not self._due(step):
            return
        grad_sq = torch.zeros((), dtype=torch.float32)
        for p in pl_module.model.parameters():
            if p.requires_grad and p.grad is not None:
                grad_sq = grad_sq + p.grad.detach().to(torch.float32).pow(2).sum()
        verdict = grad_spike_monitor(
            grad_l2=float(grad_sq.sqrt().item()), prior_ema_l2=self._grad_ema_l2
        )
        self._grad_ema_l2 = verdict.new_grad_ema_l2
        pl_module.log("train_mon_grad_l2", verdict.grad_l2, on_step=True)
        pl_module.log("train_mon_grad_ema_l2", verdict.grad_ema_l2, on_step=True)
        pl_module.log("train_mon_grad_spike_ratio", verdict.spike_ratio, on_step=True)
        pl_module.log(
            "train_mon_grad_spike", 1.0 if verdict.is_spike else 0.0, on_step=True
        )
        gap = self._ema_weight_gap(pl_module)
        if gap is not None:
            pl_module.log("train_mon_ema_weight_gap", gap, on_step=True)
        self._grad_noise_scale(trainer, pl_module, optimizer)

    def _ema_weight_gap(self, pl_module: pl.LightningModule) -> float | None:
        obj = getattr(pl_module.model, "objective", None)
        if obj is None:
            return None
        student, teacher = obj.online, obj.teacher.model
        num = torch.zeros((), dtype=torch.float32)
        den = torch.zeros((), dtype=torch.float32)
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            s = ps.detach().to(torch.float32)
            num = num + (s - pt.detach().to(torch.float32)).pow(2).sum()
            den = den + s.pow(2).sum()
        return float((num.sqrt() / (den.sqrt() + 1e-12)).item())

    def _grad_noise_scale(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer
    ) -> None:
        """McCandlish 2018 B_simple from AdamW's moment EMAs — two reductions over
        ``optimizer.state``, no extra forward/backward."""
        if optimizer is None:
            return
        # Accumulate on-GPU in a running float64 scalar and sync ONCE at the end,
        # instead of two blocking .item() per parameter tensor (hundreds of
        # device→host stalls/step). Bit-identical: both paths are a double-precision
        # SEQUENTIAL sum (same iteration order) of the same float32 per-tensor
        # reductions — float32→float64 promotion is exact, IEEE add is deterministic.
        sig_t: torch.Tensor | None = None
        v_t: torch.Tensor | None = None
        for group in optimizer.param_groups:
            beta1, beta2 = group.get("betas", (0.9, 0.999))
            for p in group["params"]:
                st = optimizer.state.get(p)
                if not st or "exp_avg" not in st or "exp_avg_sq" not in st:
                    continue
                t = float(st["step"]) if "step" in st else 0.0
                bc1 = 1.0 - beta1 ** t if t > 0 else 1.0
                bc2 = 1.0 - beta2 ** t if t > 0 else 1.0
                m = st["exp_avg"].detach().to(torch.float32) / bc1
                v = st["exp_avg_sq"].detach().to(torch.float32) / bc2
                s = m.pow(2).sum().double()
                vv = v.sum().double()
                sig_t = s if sig_t is None else sig_t + s
                v_t = vv if v_t is None else v_t + vv
        sig = float(sig_t.item()) if sig_t is not None else 0.0
        v_sum = float(v_t.item()) if v_t is not None else 0.0
        if sig <= 0.0:
            return
        var = max(0.0, v_sum - sig)
        b_eff = (
            max(1, int(getattr(trainer, "world_size", 1) or 1))
            * max(1, int(getattr(trainer, "accumulate_grad_batches", 1) or 1))
            * max(1, int(getattr(pl_module, "_last_batch_size", 1) or 1))
        )
        pl_module.log("train_mon_grad_noise_signal", sig, on_step=True)
        pl_module.log("train_mon_grad_noise_var", var, on_step=True)
        pl_module.log("train_mon_grad_noise_ratio", var / sig, on_step=True)
        pl_module.log("train_mon_grad_noise_scale", (var / sig) * b_eff, on_step=True)

    def _maybe_log_true_update_ratio(
        self, pl_module: pl.LightningModule, step: int
    ) -> None:
        snap = self._update_snapshot
        if snap is not None:
            for name in _ROUTING_GROUPS:
                prev = snap.get(name)
                mod = _group(pl_module, name)
                if prev is None or mod is None:
                    continue
                delta_sq = torch.zeros((), dtype=torch.float32)
                base_sq = torch.zeros((), dtype=torch.float32)
                for p_now, p_prev in zip(mod.parameters(), prev):
                    a = p_now.detach().to(torch.float32)
                    b = p_prev.to(torch.float32)
                    delta_sq = delta_sq + (a - b).pow(2).sum()
                    base_sq = base_sq + b.pow(2).sum()
                bn = float(base_sq.sqrt().item())
                if bn > 0.0:
                    pl_module.log(
                        f"train_mon_true_update_ratio_{name}",
                        float(delta_sq.sqrt().item()) / (bn + 1e-12),
                        on_step=True,
                    )
            self._update_snapshot = None
        if self._due(step):
            self._update_snapshot = {
                name: [p.detach().clone() for p in mod.parameters()]
                for name in _ROUTING_GROUPS
                if (mod := _group(pl_module, name)) is not None
            }

    def _maybe_log_update_cosine(
        self, pl_module: pl.LightningModule, step: int
    ) -> None:
        groups = [n for n in _ROUTING_GROUPS if _group(pl_module, n) is not None]
        snap = self._cos_snap
        if snap is None:
            if self._due(step):
                self._cos_snap = {
                    n: [p.detach().clone() for p in _group(pl_module, n).parameters()]
                    for n in groups
                }
            return
        cur_delta: dict[str, list[Tensor]] = {}
        for n in groups:
            prev = snap.get(n)
            mod = _group(pl_module, n)
            if prev is None or mod is None:
                continue
            cur_delta[n] = [
                p.detach().to(torch.float32) - b.to(torch.float32)
                for p, b in zip(mod.parameters(), prev)
            ]
        if self._cos_delta_a is None:
            self._cos_delta_a = cur_delta
            self._cos_snap = {
                n: [p.detach().clone() for p in _group(pl_module, n).parameters()]
                for n in groups
            }
            return
        for n in groups:
            da, db = self._cos_delta_a.get(n), cur_delta.get(n)
            if da is None or db is None:
                continue
            dot = torch.zeros((), dtype=torch.float32)
            na = torch.zeros((), dtype=torch.float32)
            nb = torch.zeros((), dtype=torch.float32)
            for a, b in zip(da, db):
                dot = dot + (a * b).sum()
                na = na + a.pow(2).sum()
                nb = nb + b.pow(2).sum()
            na_v, nb_v = float(na.sqrt().item()), float(nb.sqrt().item())
            if na_v > 0.0 and nb_v > 0.0:
                pl_module.log(
                    f"train_mon_update_cos_{n}",
                    float(dot.item()) / (na_v * nb_v),
                    on_step=True,
                )
        self._cos_snap = None
        self._cos_delta_a = None

    # ------------------------------------------------------------- Family B (taps)
    @torch.no_grad()
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
        outputs, batch, batch_idx: int,  # noqa: ARG002
    ) -> None:
        taps = getattr(pl_module, "_last_taps", None)
        if not taps:
            return
        accum = max(1, int(getattr(trainer, "accumulate_grad_batches", 1) or 1))
        if (batch_idx + 1) % accum != 0:
            return
        # rankme + feat_std at the encoder's final block (12). The block-3 tap was
        # removed (Ben 2026-07-10): the pre-first-L2 rank/std comparison is no longer
        # tracked — only the terminal encoder representation.
        tap = taps.get("enc12")
        if isinstance(tap, Tensor):
            self._rank_and_std(pl_module, tap, key="enc12_")
        # tier-split explained-var / var-ratio / L1.
        self._tier_stats(pl_module, taps, "whole")
        self._tier_stats(pl_module, taps, "intra")

    def _rank_and_std(
        self, pl_module: pl.LightningModule, tap: Tensor, *, key: str
    ) -> None:
        flat = tap.detach().reshape(-1, tap.shape[-1]).to(torch.float32)
        if flat.shape[0] < 2:
            return
        prefix = f"train_mon_{key}"
        verdict = teacher_rank_monitor(flat)
        pl_module.log(f"{prefix}rankme", verdict.rankme, on_step=True)
        pl_module.log(
            f"{prefix}rankme_normalised", verdict.rankme_normalised, on_step=True
        )
        std = flat.std(dim=0, unbiased=False)
        pl_module.log(f"{prefix}feat_std_mean", std.mean(), on_step=True)
        pl_module.log(f"{prefix}feat_std_min", std.min(), on_step=True)

    def _tier_stats(
        self, pl_module: pl.LightningModule, taps: dict[str, Tensor], tier: str
    ) -> None:
        """explained_var / pred_target_var_ratio / L1 on one mask tier. The target
        rows are already ``target_ln``'d in the objective (matching the loss), so
        the L1 here equals the loss contribution of this tier."""
        pred = taps.get(f"pred_{tier}")
        target = taps.get(f"tgt_{tier}")
        if not isinstance(pred, Tensor) or not isinstance(target, Tensor):
            return
        p = pred.detach().to(torch.float32)
        t = target.detach().to(torch.float32)
        if p.shape[0] < 2:
            return
        target_var = t.var(dim=0, unbiased=False).mean()
        resid_var = (t - p).var(dim=0, unbiased=False).mean()
        pred_var = p.var(dim=0, unbiased=False).mean()
        pl_module.log(
            f"train_mon_{tier}_explained_var",
            1.0 - resid_var / (target_var + _STATS_EPS), on_step=True,
        )
        pl_module.log(
            f"train_mon_{tier}_pred_target_var_ratio",
            pred_var / (target_var + _STATS_EPS), on_step=True,
        )
        pl_module.log(f"train_mon_{tier}_l1", (t - p).abs().mean(), on_step=True)


__all__ = ["SSLHealthMonitorV3"]
