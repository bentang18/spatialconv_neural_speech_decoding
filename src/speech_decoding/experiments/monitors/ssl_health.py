"""SSL-health monitor — a LEAN Lightning ``Callback`` for the converged-v2 arch.

One shared callback that carries the SSL-health diagnostics the 3STFT
``V14ConvergedBrainModule`` logs inline, but only the LEAN set Ben named (no
grad-noise-scale, no parcel-coverage, no immediate-update-ratio):

Family A — MODEL-AGNOSTIC (grads/params, no forward taps), in
``on_before_optimizer_step``:
  * grad spike       — total grad-L2 → :func:`grad_spike_monitor` (shared helper).
  * ema_weight_gap   — ``‖θ_s − θ_t‖₂ / ‖θ_s‖₂`` over the frontend EMA pair.
  * true_update_ratio — per routing group ``‖Δθ‖₂/‖θ_prev‖₂`` across two
    consecutive optimiser steps, measured only at the monitor cadence.

Family B — TAP-DEPENDENT (detached forward taps), in ``on_train_batch_end`` at
the monitor cadence, reading ``pl_module._last_taps``:
  * rankme           — frontend + latent effective rank (:func:`teacher_rank_monitor`).
  * feat_std         — per-dim std (mean/min) on the frontend + latent rows.
  * explained_var + pred_target_var_ratio — for M2 and M4, on the LN-normalised
    targets (the loss applies ``_ln_target`` before the abs error, so the monitor
    must too).

Input-stats — TAP-FREE, reads ``batch.data`` directly every cadence: per-band
nonfinite-frac / absmax / mean / std (finite values only) — the bad-electrode
input-transient tripwire.

All log keys follow the 3STFT ``train_mon_<name>`` convention so the dashboards
stay consistent. The callback is OFF-cost when not due: Family B reads taps that
the module only produces on cadence steps, and Family A's true_update_ratio only
snapshots params at the cadence.
"""

from __future__ import annotations

import torch
from lightning import pytorch as pl
from torch import Tensor

from speech_decoding.experiments.monitors.grad_spike import grad_spike_monitor
from speech_decoding.experiments.monitors.teacher_rank import teacher_rank_monitor
from speech_decoding.models.v14_converged_v2 import _ln_target

# Routing groups for the true_update_ratio readback (v2 sub-modules).
# ``m3_predictor`` is None in legacy mode — both consumers below skip absent
# modules so the group is a no-op there and live only under Run-A.
_ROUTING_GROUPS: tuple[str, ...] = (
    "frontend", "latent", "m2_predictor", "m3_predictor", "m4_predictor",
)

# EMA student/teacher towers for ema_weight_gap. v2 has THREE EMA towers; the
# pool/latent pair is the one whose target path can collapse (the frontend gap
# alone — the legacy single metric — can't see it).
_EMA_TOWERS: tuple[str, ...] = ("frontend", "pool", "latent")

_STATS_EPS: float = 1e-8


class SSLHealthMonitor(pl.Callback):
    """LEAN SSL-health callback (see module docstring). One ``every_n_steps``
    cadence gates every diagnostic; the grad spike + ema_weight_gap fire on that
    cadence too (cheap, model-agnostic)."""

    def __init__(self, *, every_n_steps: int = 50) -> None:
        super().__init__()
        self.every_n_steps = max(int(every_n_steps), 1)
        # Persistent grad-L2 EMA buffer (plain float so it survives fit-loop
        # restarts the same way the 3STFT module's tensor buffer does).
        self._grad_ema_l2: float = 0.0
        # Per-group param snapshot for the 2-step true_update_ratio measurement.
        self._update_snapshot: dict[str, list[Tensor]] | None = None

    # ----------------------------------------------------------------- cadence
    def _due(self, step: int) -> bool:
        return step % self.every_n_steps == 0

    # --------------------------------------------------- Family A (grads/params)
    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer
    ) -> None:  # noqa: ARG002
        step = int(pl_module.global_step)
        # true_update_ratio: consume a prior snapshot (logged whenever one exists),
        # then take a fresh one only at the cadence. Run every step so the snapshot
        # taken at step N is consumed at step N+1.
        self._maybe_log_true_update_ratio(pl_module, step)
        if not self._due(step):
            return
        # grad spike: total grad-L2 over trainable params.
        grad_sq = torch.zeros((), dtype=torch.float32)
        for p in pl_module.model.parameters():
            if p.requires_grad and p.grad is not None:
                grad_sq = grad_sq + p.grad.detach().to(torch.float32).pow(2).sum()
        grad_l2 = float(grad_sq.sqrt().item())
        verdict = grad_spike_monitor(grad_l2=grad_l2, prior_ema_l2=self._grad_ema_l2)
        self._grad_ema_l2 = verdict.new_grad_ema_l2
        pl_module.log("train_mon_grad_l2", verdict.grad_l2, on_step=True)
        pl_module.log("train_mon_grad_ema_l2", verdict.grad_ema_l2, on_step=True)
        pl_module.log("train_mon_grad_spike_ratio", verdict.spike_ratio, on_step=True)
        pl_module.log(
            "train_mon_grad_spike", 1.0 if verdict.is_spike else 0.0, on_step=True
        )
        # ema_weight_gap per EMA tower. The frontend keeps the legacy key for chart
        # continuity; pool/latent expose the towers whose target path collapses.
        for tower in _EMA_TOWERS:
            gap = self._ema_weight_gap(pl_module, tower)
            if gap is None:
                continue
            key = (
                "train_mon_ema_weight_gap"
                if tower == "frontend"
                else f"train_mon_ema_weight_gap_{tower}"
            )
            pl_module.log(key, gap, on_step=True)
        # grad_noise_scale (critical-batch estimate) from AdamW's existing moments.
        self._grad_noise_scale(trainer, pl_module, optimizer)

    def _grad_noise_scale(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer
    ) -> None:
        """McCandlish 2018 gradient-noise-scale via AdamW's existing moment EMAs —
        NO extra forward/backward, just two reductions over ``optimizer.state``.

        AdamW keeps a bias-corrected first moment ``m̂ ≈ G`` (true gradient) and
        second moment ``v̂ ≈ E[|g|²]`` of the global-batch gradient ``g``. The
        gradient the optimiser sees is the DDP+accum mean over the effective batch
        ``B_eff = world · accum · per_rank_bs``, so ``Σ(v̂ − m̂²) ≈ tr(Σ)/B_eff``
        (variance of that mean). Then::

            signal = Σ m̂²                       # |G|²
            var    = max(0, Σ v̂ − Σ m̂²)        # tr(Σ)/B_eff
            B_simple = B_eff · var / signal      # critical batch (examples)

        Below ``B_simple`` a bigger batch ≈ proportionally fewer steps; above it the
        gradient is already clean ⇒ wasted compute. ``var`` can dip slightly negative
        from EMA noise ⇒ clamp the sum (not per-element)."""
        if optimizer is None:
            return
        sig = 0.0   # Σ m̂²
        v_sum = 0.0  # Σ v̂
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
                sig += float(m.pow(2).sum().item())
                v_sum += float(v.sum().item())
        if sig <= 0.0:
            return
        var = max(0.0, v_sum - sig)
        ratio = var / sig                       # = B_simple / B_eff
        b_eff = (
            max(1, int(getattr(trainer, "world_size", 1) or 1))
            * max(1, int(getattr(trainer, "accumulate_grad_batches", 1) or 1))
            * max(1, int(getattr(pl_module, "_last_batch_size", 1) or 1))
        )
        pl_module.log("train_mon_grad_noise_signal", sig, on_step=True)
        pl_module.log("train_mon_grad_noise_var", var, on_step=True)
        pl_module.log("train_mon_grad_noise_ratio", ratio, on_step=True)
        pl_module.log("train_mon_grad_noise_scale", ratio * b_eff, on_step=True)

    def _ema_weight_gap(
        self, pl_module: pl.LightningModule, tower: str = "frontend"
    ) -> float | None:
        """``‖θ_student − θ_teacher‖₂ / ‖θ_student‖₂`` over the named EMA tower's
        student/teacher pair (``<tower>`` vs ``teacher_<tower>``). Returns ``None``
        when the model lacks that tower (e.g. a model with no ``teacher_pool``)."""
        student = getattr(pl_module.model, tower, None)
        teacher = getattr(pl_module.model, f"teacher_{tower}", None)
        if student is None or teacher is None:
            return None
        num = torch.zeros((), dtype=torch.float32)
        den = torch.zeros((), dtype=torch.float32)
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            s = ps.detach().to(torch.float32)
            num = num + (s - pt.detach().to(torch.float32)).pow(2).sum()
            den = den + s.pow(2).sum()
        return float((num.sqrt() / (den.sqrt() + 1e-12)).item())

    def _maybe_log_true_update_ratio(
        self, pl_module: pl.LightningModule, step: int
    ) -> None:
        """Per-group ``‖θ_now − θ_prev‖₂ / ‖θ_prev‖₂`` across two consecutive
        optimiser steps — taken only at the cadence to stay cheap."""
        snap = self._update_snapshot
        if snap is not None:
            for group in _ROUTING_GROUPS:
                prev = snap.get(group)
                mod = getattr(pl_module.model, group, None)
                if prev is None or mod is None:
                    continue
                now = list(mod.parameters())
                delta_sq = torch.zeros((), dtype=torch.float32)
                base_sq = torch.zeros((), dtype=torch.float32)
                for p_now, p_prev in zip(now, prev):
                    a = p_now.detach().to(torch.float32)
                    b = p_prev.to(torch.float32)
                    delta_sq = delta_sq + (a - b).pow(2).sum()
                    base_sq = base_sq + b.pow(2).sum()
                bn = float(base_sq.sqrt().item())
                if bn > 0.0:
                    pl_module.log(
                        f"train_mon_true_update_ratio_{group}",
                        float(delta_sq.sqrt().item()) / (bn + 1e-12),
                        on_step=True,
                    )
            self._update_snapshot = None
        if self._due(step):
            self._update_snapshot = {
                group: [
                    p.detach().clone()
                    for p in getattr(pl_module.model, group).parameters()
                ]
                for group in _ROUTING_GROUPS
                if getattr(pl_module.model, group, None) is not None
            }

    # ------------------------------------------------- Family B + input-stats
    @torch.no_grad()
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
        outputs, batch, batch_idx: int,  # noqa: ARG002
    ) -> None:
        # The module stashes ``_last_taps`` inside ``training_step`` (pre-optimiser
        # ``global_step`` = k) on EXACTLY its monitor-cadence batches — the same
        # cadence Family A uses in ``on_before_optimizer_step`` (also pre-step). By
        # the time THIS hook runs the optimiser has stepped and ``global_step`` is
        # k+1, so re-deriving the cadence here is off-by-one and silently dropped
        # every tap monitor. Presence of ``_last_taps`` IS the cadence signal — the
        # module already decided. Run input-stats on the same gate so Family B +
        # input-stats fire together with Family A.
        taps = getattr(pl_module, "_last_taps", None)
        if not taps:
            return
        # Under grad-accum the module stashes taps on EVERY micro-batch (all share
        # the pre-step global_step), but one logical training step = one optimiser
        # step = the LAST micro-batch of the accum window. Fire once there so an
        # accum-K config doesn't recompute the rankme SVD K× per step (Family A in
        # on_before_optimizer_step is already once-per-step). accum defaults to 1 ⇒
        # every batch (unchanged).
        accum = max(1, int(getattr(trainer, "accumulate_grad_batches", 1) or 1))
        if (batch_idx + 1) % accum != 0:
            return
        self._run_input_stats(pl_module, batch)
        self._run_tap_monitors(pl_module, taps)

    # -- input stats (tap-free; reads the batch) --------------------------------
    def _run_input_stats(self, pl_module: pl.LightningModule, batch) -> None:
        """Per-band nonfinite-frac / absmax / mean / std over FINITE values only —
        the bad-electrode input-transient tripwire. v2 bands are lfs / hga."""
        data = batch.data

        def _band(name: str) -> None:
            tensor = data.get(name)
            if not isinstance(tensor, Tensor) or tensor.numel() == 0:
                return
            f = tensor.detach().to(torch.float32)
            finite = torch.isfinite(f)
            n = f.numel()
            n_finite = int(finite.sum().item())
            pre = f"train_mon_input_{name}"
            pl_module.log(f"{pre}_nonfinite_frac", float((n - n_finite) / n),
                          on_step=True)
            if n_finite < 1:
                return
            ff = f[finite]
            pl_module.log(f"{pre}_absmax", ff.abs().max(), on_step=True)
            if n_finite >= 2:
                pl_module.log(f"{pre}_mean", ff.mean(), on_step=True)
                pl_module.log(f"{pre}_std", ff.std(unbiased=False), on_step=True)

        _band("electrode_tokens_lfs")
        _band("electrode_tokens_hga")

    # -- forward-tap monitors ---------------------------------------------------
    def _run_tap_monitors(
        self, pl_module: pl.LightningModule, taps: dict[str, Tensor]
    ) -> None:
        # rankme + feat_std on the frontend (teacher) tap.
        front = taps.get("_tap_teacher_frontend")
        if isinstance(front, Tensor):
            self._rank_and_std(pl_module, front, key="frontend_")
        # rankme + feat_std on the student-latent tap (bare prefix, matching 3STFT).
        latent = taps.get("_tap_student_latent")
        if isinstance(latent, Tensor):
            self._rank_and_std(pl_module, latent, key="")
        # Run-A only: rankme + feat_std on the teacher-pool tap (the M3 target).
        pool = taps.get("_tap_teacher_pool")
        if isinstance(pool, Tensor):
            self._rank_and_std(pl_module, pool, key="pool_")
        # explained_var + pred_target_var_ratio per head. Run-A regresses RAW
        # targets (no _ln_target) — the model flags this via ``_tap_raw_targets``
        # so the metric matches the loss. The M3 tap is absent in legacy mode ⇒
        # _jepa_stats returns early (no-op).
        rt = taps.get("_tap_raw_targets")
        raw = bool(rt.item()) if isinstance(rt, Tensor) else False
        self._jepa_stats(pl_module, taps, "_tap_m2_pred", "_tap_m2_target", "m2", raw)
        self._jepa_stats(pl_module, taps, "_tap_m3_pred", "_tap_m3_target", "m3", raw)
        self._jepa_stats(pl_module, taps, "_tap_m4_pred", "_tap_m4_target", "m4", raw)

    def _rank_and_std(
        self, pl_module: pl.LightningModule, tap: Tensor, *, key: str
    ) -> None:
        flat = tap.detach().reshape(-1, tap.shape[-1]).to(torch.float32)
        if flat.shape[0] < 2:
            return
        prefix = f"train_mon_{key}"
        verdict = teacher_rank_monitor(flat)
        pl_module.log(f"{prefix}rankme", verdict.rankme, on_step=True)
        pl_module.log(f"{prefix}rankme_normalised", verdict.rankme_normalised,
                      on_step=True)
        std = flat.std(dim=0, unbiased=False)
        pl_module.log(f"{prefix}feat_std_mean", std.mean(), on_step=True)
        pl_module.log(f"{prefix}feat_std_min", std.min(), on_step=True)

    def _jepa_stats(
        self, pl_module: pl.LightningModule, taps: dict[str, Tensor],
        pred_key: str, target_key: str, head: str, raw_targets: bool = False,
    ) -> None:
        """``explained_var = 1 − Var(target − pred) / Var(target)`` and
        ``pred_target_var_ratio = Var(pred) / Var(target)``. The target is
        LN-normalised (legacy shared-denom loss uses ``_ln_target``) UNLESS
        ``raw_targets`` (Run-A per-head loss regresses raw post-``ln_out``
        targets) — keeping the metric consistent with the loss either way.
        Variance is the mean of the per-dim variances over the (Q, d) rows."""
        pred = taps.get(pred_key)
        target = taps.get(target_key)
        if not isinstance(pred, Tensor) or not isinstance(target, Tensor):
            return
        p = pred.detach().to(torch.float32)
        td = target.detach().to(torch.float32)
        t = td if raw_targets else _ln_target(td)
        if p.shape[0] < 2:
            return
        target_var = t.var(dim=0, unbiased=False).mean()
        resid_var = (t - p).var(dim=0, unbiased=False).mean()
        pred_var = p.var(dim=0, unbiased=False).mean()
        explained_var = 1.0 - resid_var / (target_var + _STATS_EPS)
        ratio = pred_var / (target_var + _STATS_EPS)
        pl_module.log(f"train_mon_{head}_explained_var", explained_var, on_step=True)
        pl_module.log(f"train_mon_{head}_pred_target_var_ratio", ratio, on_step=True)


__all__ = ["SSLHealthMonitor"]
