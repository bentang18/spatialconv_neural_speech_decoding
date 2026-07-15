"""SSL-health monitor for v14_converged_v3 — a lean Lightning ``Callback``.

The v2 ``SSLHealthMonitor`` is welded to the v2 schema (``model.frontend/latent/
m2_predictor`` groups, M2/M4 tap keys, ``electrode_tokens_lfs/hga`` bands). v3 has a
different structure (online stem+encoder / predictor towers, DKT-tag identity,
whole-vs-intra mask tiers), so this is a v3-bound sibling that REUSES the stateless
helpers — :func:`grad_spike_monitor` and :func:`teacher_rank_monitor` — and v3's own
``_ln_target``. Nothing is copied from the v2 callback beyond those shared helpers.

Family A — model-agnostic (grads/params), in ``on_before_optimizer_step`` at the
monitor cadence:
  * grad spike        — total grad-L2 over trainable params.
  * ema_weight_gap    — ``‖θ_online − θ_teacher‖₂ / ‖θ_online‖₂`` (collapse guard).

  Three LR-health monitors were REMOVED 2026-07-12 (all log-only ⇒ science-neutral),
  after a per-monitor GH200 profile (job 2648659) showed they dominated the per-step
  budget: grad_noise_scale (McCandlish B_simple, 28.2 ms), update_cos (25.3 ms) and
  true_update_ratio (23.2 ms) — 76.7 of the 97 ms Family-A cost. They answered the
  "is the LR too hot?" question, which is now settled; grad-spike (paired with the
  3.0 grad-clip) remains as the divergence guard.

Family B — tap-dependent (detached forward taps), in ``on_train_batch_end``, reading
``pl_module._last_taps`` (present ⇒ this is a cadence step; the module already
decided, so no off-by-one re-derivation):
  * rankme + feat_std at encoder block 12 (terminal representation) — effective-rank
    / feature-std collapse guard. (The block-3 pre-L2 tap was removed 2026-07-10.)
  * explained_var + pred_target_var_ratio + L1, split whole-sensor vs intra-sensor —
    is L2 actually learning the cross-sensor task, or is the encoder cheating locally?

Keys follow the ``train_mon_<name>`` convention. Off-cost when not due: the module
only produces taps on cadence steps, so Family B is skipped entirely off-cadence.
"""

from __future__ import annotations

import torch
from lightning import pytorch as pl
from torch import Tensor

from speech_decoding.experiments.monitors.grad_spike import grad_spike_monitor
from speech_decoding.experiments.monitors.teacher_rank import teacher_rank_monitor

_STATS_EPS: float = 1e-8


class SSLHealthMonitorV3(pl.Callback):
    def __init__(self, *, every_n_steps: int = 1) -> None:
        super().__init__()
        self.every_n_steps = max(int(every_n_steps), 1)
        self._grad_ema_l2: float = 0.0

    def _due(self, step: int) -> bool:
        return step % self.every_n_steps == 0

    # --------------------------------------------------- Family A (grads/params)
    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer
    ) -> None:
        step = int(pl_module.global_step)
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

    # ------------------------------------------------------------- Family B (taps)
    @torch.no_grad()
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
        outputs, batch, batch_idx: int,  # noqa: ARG002
    ) -> None:
        accum = max(1, int(getattr(trainer, "accumulate_grad_batches", 1) or 1))
        if (batch_idx + 1) % accum != 0:
            return
        # input tripwire: per-band raw-token health BEFORE any tap read — catches a
        # corrupt/NaN band cache or a robust-z blow-up at the door, independent of taps.
        if self._due(int(getattr(pl_module, "global_step", 0))):
            self._input_tripwire(pl_module, batch)
        taps = getattr(pl_module, "_last_taps", None)
        if not taps:
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

    _BAND_NAMES: tuple[str, ...] = ("slow", "mid", "hga")

    def _input_tripwire(self, pl_module: pl.LightningModule, batch) -> None:
        """Per-band raw-input |STFT| token health (the v3 3-band analog of r2's
        ``input_electrode_tokens_lfs/hga`` tripwire): fraction non-finite, |max|, mean,
        std over the batch's band tensors. A NaN/inf cache or a normalization blow-up
        shows here before it silently poisons the loss. No-op if the batch is absent."""
        bands = getattr(batch, "bands", None)
        if not bands:
            return
        for x, name in zip(bands, self._BAND_NAMES):
            x = x.detach().to(torch.float32)
            prefix = f"train_mon_input_{name}_"
            # Single-pass reductions only — NO boolean-mask allocation (the costly part).
            # A NaN makes mean/std/absmax NaN, which nonfinite_frac already flags; both
            # are visible in the log, so masking buys nothing but a full-tensor copy.
            pl_module.log(
                f"{prefix}nonfinite_frac",
                (~torch.isfinite(x)).to(torch.float32).mean(), on_step=True,
            )
            pl_module.log(f"{prefix}absmax", x.abs().amax(), on_step=True)
            pl_module.log(f"{prefix}mean", x.mean(), on_step=True)
            pl_module.log(f"{prefix}std", x.std(unbiased=False), on_step=True)

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
