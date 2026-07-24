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
  * true_update_ratio — per group ``‖Δθ‖₂/‖θ_prev‖₂`` across two consecutive steps
    (LR calibration; ~1e-3 healthy). Runs every step to span its window.
  * grad_noise_scale  — McCandlish B_simple critical-batch estimate from AdamW's
    moment EMAs (opt-in, own coarse cadence). REMOVED 2026-07-12 with update_cos when
    the "is LR too hot?" question closed; RESTORED 2026-07-23 for the batch-vs-critical
    sizing question — a different axis: ``train_mon_grad_noise_scale`` is B_simple in
    SAMPLES, compared to tokens/step (≫ ⇒ below critical batch). ~28 ms, gated by
    --grad-noise-every-n-steps (0 = off) so it costs ~nothing amortized. update_cos
    stays removed (LR-health, settled).

Family B — tap-dependent (detached forward taps), in ``on_train_batch_end``, reading
``pl_module._last_taps`` (present ⇒ this is a cadence step; the module already
decided, so no off-by-one re-derivation):
  * rankme + feat_std at encoder block 12 (terminal representation) — effective-rank
    / feature-std collapse guard. (The block-3 pre-L2 tap was removed 2026-07-10.)
  * per-band scalar taps reduced INSIDE the objective (r4): JEPA explained_var /
    pred_target_var_ratio / L1 per band (#40), per-band Gaussian-NLL (#41), and
    predicted-cov entropy vs the noise-floor ceiling (#42). The r4 flat path has no
    whole/intra sensor tiers — the split that carried signal is now per-BAND (SLOW /
    MID / HGA), each reduced weighted over the margin-gated scored tokens.

Keys follow the ``train_mon_<name>`` convention. Off-cost when not due: the module
only produces taps on cadence steps, so Family B is skipped entirely off-cadence.
"""

from __future__ import annotations

import torch
from lightning import pytorch as pl
from torch import Tensor

from speech_decoding.experiments.monitors.grad_spike import grad_spike_monitor
from speech_decoding.experiments.monitors.teacher_rank import teacher_rank_monitor

# The named trainable towers whose per-group update-to-weight ratio we track. ``online`` =
# stem+encoder, ``predictor`` = predictor tower, ``mae_head`` = the r5 MAE linear recon decoder,
# ``mae_head_hga``/``mae_head_lfs`` = the v3r5nf per-stream recon decoders. Each is None on the
# arms that lack it ⇒ skipped (r5 has only mae_head; v3r5nf has only the two per-stream heads).
# Under AdamW raw grad-L2 is scale-free w.r.t. the effective step, so ``‖Δθ‖/‖θ‖`` (target ≈1e-3)
# is the genuinely-informative LR-health readout — per group so a frozen (→0) or runaway (≫1e-2)
# tower shows even when the global looks fine (and the two nf heads show independently, catching a
# stream the co-located mask-query fails to separate).
_ROUTING_GROUPS: tuple[str, ...] = (
    "online", "predictor", "mae_head", "mae_head_hga", "mae_head_lfs",
)


def _group(pl_module: pl.LightningModule, name: str):
    obj = getattr(pl_module.model, "objective", None)
    if obj is None:
        return None
    return {
        "online": getattr(obj, "online", None),
        "predictor": getattr(obj, "predictor", None),
        "mae_head": getattr(obj, "mae_head_r5", None),
        "mae_head_hga": getattr(obj, "mae_head_hga", None),
        "mae_head_lfs": getattr(obj, "mae_head_lfs", None),
    }.get(name)


class SSLHealthMonitorV3(pl.Callback):
    def __init__(self, *, every_n_steps: int = 1, per_stream_enc12: bool = False,
                 grad_noise_every_n_steps: int = 0) -> None:
        super().__init__()
        self.every_n_steps = max(int(every_n_steps), 1)
        # grad_noise_scale (McCandlish B_simple, critical-batch estimate) runs on its OWN coarse
        # cadence, decoupled from the tap monitors: it costs ~28 ms (two GPU reductions over the
        # AdamW moment buffers, no extra fwd/bwd), so at 0 it is OFF and at e.g. 25 it amortizes to
        # ~1 ms/step. Opt in (--grad-noise-every-n-steps) when sizing the batch vs critical batch.
        self.grad_noise_every_n_steps = max(int(grad_noise_every_n_steps), 0)
        # v3r5nf per-stream enc12 rankme/feat_std split. OFF by default (Ben 2026-07-22): the two
        # extra SVDs it runs per monitor step were the ~0.15 s/step "comb" on the fast arm's large
        # feature matrix — pure diagnostic overhead that cost the run its 50k target. Opt in with
        # --per-stream-enc12 for a diagnostic run.
        self.per_stream_enc12 = bool(per_stream_enc12)
        self._grad_ema_l2: float = 0.0
        # θ snapshot taken at a cadence step, read one step later to form the single-step Δθ.
        self._update_snapshot: dict[str, list[Tensor]] | None = None

    def _due(self, step: int) -> bool:
        return step % self.every_n_steps == 0

    # --------------------------------------------------- Family A (grads/params)
    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer
    ) -> None:
        step = int(pl_module.global_step)
        # update-to-weight ratio spans ONE optimizer step: snapshot θ at a cadence step, read
        # Δθ on the next call. Runs every step (the read is a no-op until a snapshot exists).
        self._maybe_log_true_update_ratio(pl_module, step)
        # grad_noise_scale on its own coarse cadence (independent of the tap-monitor cadence): the
        # critical-batch read costs ~28 ms, so gate it separately to keep throughput clean.
        if self.grad_noise_every_n_steps and step % self.grad_noise_every_n_steps == 0:
            self._grad_noise_scale(trainer, pl_module, optimizer)
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
        # clip-hit tracker: does the pre-clip norm exceed the ``gradient_clip_val`` rail (3.0)?
        # ``clip_hit`` is 0/1 on the cadence step (its wandb mean = the sampled clip-hit RATE — an
        # unbiased estimate, each cadence step a fair Bernoulli draw); ``clip_ratio`` = grad_l2 /
        # clip_val is the continuous headroom (≥1 ⇒ clip fired). Skipped when clipping is off.
        clip_val = float(getattr(trainer, "gradient_clip_val", 0.0) or 0.0)
        if clip_val > 0.0:
            pl_module.log(
                "train_mon_grad_clip_hit",
                1.0 if verdict.grad_l2 > clip_val else 0.0, on_step=True,
            )
            pl_module.log(
                "train_mon_grad_clip_ratio", verdict.grad_l2 / clip_val, on_step=True
            )
        gap = self._ema_weight_gap(pl_module)
        if gap is not None:
            pl_module.log("train_mon_ema_weight_gap", gap, on_step=True)

    def _ema_weight_gap(self, pl_module: pl.LightningModule) -> float | None:
        obj = getattr(pl_module.model, "objective", None)
        # MAE arm has no EMA teacher (obj.teacher is None) ⇒ no student-teacher weight gap
        # to report; skip the metric rather than dereference a None teacher.
        if obj is None or getattr(obj, "teacher", None) is None:
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
        """McCandlish 2018 B_simple = trΣ/‖G‖² from AdamW's moment EMAs — two reductions over
        ``optimizer.state``, no extra forward/backward. ``m`` (bias-corrected exp_avg) estimates
        the true gradient G, ``v`` (bias-corrected exp_avg_sq) estimates E[g²]; per coordinate
        Var(g_B) = v − m², so ‖G‖² = Σm² (signal) and trΣ = B_eff·Σ(v − m²) (per-example noise).
        ``grad_noise_scale`` = B_simple in SAMPLE units, directly comparable to B_eff (tokens/step):
        B_simple ≫ B_eff ⇒ below critical batch (raising batch buys ~linear step-efficiency);
        B_simple ≈ B_eff ⇒ at the knee. The moments are EMAs (β2 window), so this is the smoothed
        McCandlish estimate — a directional read, not the exact per-step accum estimator."""
        if optimizer is None:
            return
        # Accumulate on-GPU in a running float64 scalar and sync ONCE at the end, instead of two
        # blocking .item() per parameter tensor (hundreds of device→host stalls/step). Bit-
        # identical: both paths are a double-precision SEQUENTIAL sum (same iteration order) of the
        # same float32 per-tensor reductions — float32→float64 promotion is exact.
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
        """Per-group ``‖Δθ‖₂/‖θ_prev‖₂`` over ONE optimizer step — the AdamW effective-step
        readout (target ≈1e-3). Snapshot θ at a cadence step; on the next call the params have
        moved by exactly one ``optimizer.step()``, so the diff is the true single-step update."""
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
            # v3r5nf: split the pooled enc12 by token STREAM (0=HGA 1=LFS) and emit each
            # stream's OWN rankme + feat_std. The boolean gather + per-stream SVD run here in
            # the eager callback (off the compiled graph). Tests whether r5nf's low pooled
            # rankme is the between-stream identity axis (per-stream rankme ≫ pooled) or a
            # genuinely low per-stream spectrum. Absent for r4/r5-fused (no enc12_band tap).
            # OFF by default — the 2 extra SVDs are the fast-arm "comb"; opt in via --per-stream-enc12.
            band = taps.get("enc12_band") if self.per_stream_enc12 else None
            if isinstance(band, Tensor):
                flat = tap.detach().reshape(-1, tap.shape[-1])
                b = band.detach().reshape(-1)
                if b.shape[0] == flat.shape[0]:
                    for bid, bkey in ((0, "enc12_hga_"), (1, "enc12_lfs_")):
                        self._rank_and_std(pl_module, flat[b == bid], key=bkey)
        # scalar monitor taps reduced INSIDE the objective (r4): per-band JEPA
        # explained-var / var-ratio / L1 (#40), per-band NLL (#41), predicted-cov
        # entropy vs floor (#42). Each is a 0-dim tensor keyed by its own name.
        self._log_scalar_taps(pl_module, taps)

    _BAND_NAMES: tuple[str, ...] = ("slow", "mid", "hga")
    _BAND_NAMES_R5: tuple[str, ...] = ("hga", "lfs")  # r5 early-fusion input caches, in arg order

    def _band_names(self, pl_module: pl.LightningModule) -> tuple[str, ...]:
        obj = getattr(pl_module.model, "objective", None)
        if obj is not None and getattr(obj, "early_fusion", False):
            return self._BAND_NAMES_R5
        return self._BAND_NAMES

    def _input_tripwire(self, pl_module: pl.LightningModule, batch) -> None:
        """Per-band raw-input |STFT| token health (the v3 3-band analog of r2's
        ``input_electrode_tokens_lfs/hga`` tripwire): fraction non-finite, |max|, mean,
        std over the batch's band tensors. A NaN/inf cache or a normalization blow-up
        shows here before it silently poisons the loss. No-op if the batch is absent."""
        bands = getattr(batch, "bands", None)
        if not bands:
            return
        for x, name in zip(bands, self._band_names(pl_module)):
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

    def _log_scalar_taps(
        self, pl_module: pl.LightningModule, taps: dict[str, Tensor]
    ) -> None:
        """Log every 0-dim scalar tap the objective reduced on this cadence step
        (``jepa_*`` #40, ``nll_*`` #41, ``cov_entropy*`` #42). ``enc12`` — the only
        non-scalar tap — is consumed by ``_rank_and_std`` and skipped here."""
        for key, val in taps.items():
            if key == "enc12" or not isinstance(val, Tensor) or val.ndim != 0:
                continue
            pl_module.log(f"train_mon_{key}", val.detach().to(torch.float32), on_step=True)


__all__ = ["SSLHealthMonitorV3"]
