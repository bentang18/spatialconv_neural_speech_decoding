"""Collapse / divergence guard (#54): an ACTIVE training-run kill switch.

The B36 joint module already *computes and logs* a full panel of
collapse/divergence alarms — RankMe dimensional collapse
(``*_mon_rankme_alarm`` / ``*_mon_frontend_rankme_alarm`` and their softer
``*_warn`` siblings), latent-valid coverage degeneracy
(``*_mon_coverage_alarm``), gradient NaN/Inf (``train_mon_grad_diverged``),
and the masked-cell count (``*_n_masked``). But every one of those is
**passive**: logged and forgotten. Nothing stops a run that has gone bad.

This callback is the missing active layer. It watches those exact
already-logged keys (it recomputes nothing) and **aborts the phase** on a
catastrophic signal, after writing a diagnosis JSON.

Two tripwires, two clocks:
  * **Per optimizer step** (`on_train_batch_end`): a NaN/Inf loss or
    gradient never recovers, so it aborts immediately. ``train_mon_grad_
    diverged`` is logged in ``on_before_optimizer_step``, i.e. once per
    *optimizer* step — so under ``accumulate_grad_batches=K`` the live
    grad-divergence read advances once every K micro-batches, not every
    micro-batch (a NaN grad is still caught at the step that closes its
    accumulation window, at most K−1 micro-batches late). This is the only
    acute-failure catch that does not depend on the validation cadence.
  * **Per validation check** (`on_validation_epoch_end`): the soft alarm
    panel (RankMe collapse, coverage degeneracy, no-masking, loss
    blow-up) is debounced — it must persist across consecutive validation
    checks before it is believed. NOTE: on a ``max_steps``-budgeted SSL
    phase that ends mid-epoch, epoch-boundary validation never runs — so
    the run MUST set ``val_check_interval`` (a step count) for this panel
    to evaluate at all. The dispatch wires a default for the SSL phases;
    without it only the per-batch NaN/grad catch is active.

Abort semantics — why raise, not ``trainer.should_stop``:
    A diverged/collapsed model must NOT be promoted to the next phase.
    The phase handoff (``_snapshot`` → ``phase_N.ckpt``) runs only after
    ``trainer.fit`` returns *normally*; ``trainer.should_stop`` exits the
    loop normally and would therefore hand a broken model downstream.
    Raising :class:`CollapseStop` instead propagates out of
    ``trainer.fit`` so the snapshot is never written: the next phase's
    ``pretrained_ckpt`` (= the un-written ``phase_N.ckpt``) is absent and
    the chain stops with an actionable error rather than warm-starting off
    a broken model. ModelCheckpoint's ``last.ckpt`` lives under
    ``checkpoints/`` — a *different* path from the handoff ``snapshot_ckpt_
    to`` — so it remains for diagnosis but is never consumed downstream.

    Exca-cache caveat (#54 audit C1): the raised ``CollapseStop`` IS stored
    by exca as a job *failure*. Under the default ``infra.mode="cached"`` a
    same-config relaunch RE-RAISES that stored failure instead of
    recomputing — so a fix that does not change the config-derived uid
    (e.g. a *code* change) would never actually re-run. The chain dispatch
    therefore defaults ``--exca-mode`` to ``retry`` (recompute a cached
    *error*, keep cached *successes*) so the abort → diagnose → fix →
    relaunch loop works without the operator remembering the flag. The
    diagnosis JSON is the durable record of *why* the run died.

DDP safety (the capstone is 4-GPU DDP):
    The watched metrics are logged WITHOUT ``sync_dist``, so under DDP the
    soft-alarm fractions are rank-LOCAL — one rank can trip while another
    does not. If only one rank raised, the survivors would hang at the
    next NCCL collective until walltime. So the per-validation soft
    decision is reduced across ranks with ``reduce_boolean_decision`` (OR
    — any rank seeing collapse aborts all). The per-batch grad-divergence
    flag is computed from DDP-averaged gradients and is therefore already
    identical on every rank (a NaN on one rank → NaN-averaged grad on all
    → all abort together), so it needs no reduction; the rank-LOCAL
    ``outputs`` loss is only trusted on a single device. The rank-identity
    of the grad flag rests on ``on_before_optimizer_step`` firing *after*
    ``backward()``/DDP grad-sync (Lightning ``Precision._after_closure``) —
    re-verify on a Lightning bump or if the run switches to manual
    optimization.

    Precision caveat (not wired-relevant): under fp16 ``16-mixed`` the
    GradScaler produces transient ``inf`` grads on an overflow it is
    *designed* to recover from (skip step, lower scale) — the per-batch
    grad abort would false-trip on that. The wired config is ``bf16-mixed``
    (no scaler), where an ``inf`` grad is genuine divergence, so this does
    not bite; if ``16-mixed`` is ever used, debounce the grad abort.

Known limitation (documented, not a bug): a high-loss *plateau* — a run
that fails to learn but stays finite, full-rank, and below the blow-up
factor — trips nothing here. That is a "did-not-learn", not a
"diverged/collapsed", and is better read off the downstream probe metric
(``best_val_probe``); watch it on a long run.

Threshold philosophy:
    The *verdict* thresholds (RankMe normalised < 0.25 alarm / < 0.5 warn,
    coverage degenerate-fraction > 0.10) are already locked in the monitor
    modules (5/28 P0) — this guard does not re-derive them, it consumes
    their 0/1 flags. The only knobs this guard owns are the
    **debounce/response** parameters, deliberately LOOSE: firing one check
    late costs a little compute; firing on a transient is a false abort of a
    healthy multi-day run. So non-finite aborts immediately; the hard alarms
    need ``sustain_checks`` consecutive bad validations after a
    ``warmup_checks`` (and optional ``warmup_min_step``) grace.

    Soft-warn is advisory by default (2026-06-05). The ``*_warn`` tier
    (RankMe normalised < 0.5) originally aborted on a longer
    ``warn_sustain_checks`` streak — to catch a slow decline before it
    crossed the 0.25 alarm. But its 0.5 line is a DINOv3 image-ViT value
    (§3.3), and this iEEG |STFT| front-end's *healthy* born-state sits at
    ~0.32 normalised rank (gate-47722655 diagnosis: rank flat at 0.319 from
    init while val_loss fell — a floor, not a decline). So ``warn_aborts``
    defaults False: the warn tier logs + streaks for trend visibility but
    does NOT abort; the hard < 0.25 alarm (below the born-state) is the kill.
    This changes only the guard's *response* to the soft tier — the
    threshold VALUES (0.25/0.5) stay locked in the monitor modules (5/28
    P0). Recalibrating those values to this modality is a separate,
    lock-owning decision.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import json
import math
import typing as tp
from pathlib import Path

import lightning.pytorch as pl
from torch import Tensor


LOSS_BLOWUP_FACTOR: float = 10.0
"""A validation loss above ``factor × best-so-far`` is treated as
divergence. 10× is far outside healthy SSL-loss fluctuation (which lives
within ~2× of its running floor) — chosen loose so normal warm-up
transients never trip it."""

ALARM_FRACTION: float = 0.5
"""A 0/1 alarm logged ``on_epoch=True`` reduces to its *mean over the
check's steps* — the fraction of probed steps that alarmed. A validation
check counts as alarmed when a majority of its probed steps alarmed."""

BLOWUP_MIN_FLOOR: float = 1e-3
"""Absolute floor on the running-best val loss below which the loss-blow-up
*ratio* test is suppressed. The watched SSL losses are L1/Smooth-L1 on
LayerNorm'd (unit-scale) targets, so a healthy floor is O(0.1–1); a best
below 1e-3 would let a trivially small absolute increase manufacture a 10×
ratio (false abort). Below the floor the ratio test sleeps — RankMe and
no-masking already cover a true near-zero-loss target collapse."""

SUSTAIN_CHECKS: int = 2
"""Consecutive alarmed validation checks required before a *hard* alarm
(RankMe <0.25 / coverage / no-masking / loss-blow-up) aborts."""

WARN_SUSTAIN_CHECKS: int = 4
"""Consecutive checks for the softer ``*_rankme_warn`` (<0.5) streak —
longer than the hard sustain so it only fires on a sustained slow decline,
catching the slope before it reaches the 0.25 alarm."""

WARMUP_CHECKS: int = 1
"""Validation checks to ignore for soft criteria at the start of a phase
(random-init representations legitimately look low-rank / high-loss
early). Non-finite losses/grads are NOT subject to warm-up."""

_HISTORY: int = 12
"""Recent per-check metric snapshots retained for the diagnosis JSON."""


class CollapseStop(RuntimeError):
    """Raised to abort a training phase on catastrophic collapse/divergence.

    Propagates out of ``trainer.fit`` so the phase snapshot is never
    promoted to the next phase (see module docstring). The accompanying
    diagnosis JSON is the durable record.
    """


@dataclasses.dataclass(frozen=True)
class CollapseVerdict:
    """What tripped the guard — serialised into the diagnosis JSON."""

    criterion: str
    detail: str
    value: float
    global_step: int
    epoch: int
    suggested_action: str


def _as_float(value: tp.Any) -> float | None:
    """Coerce a logged metric (tensor or python scalar) to ``float``.

    Returns ``None`` when the key was absent or not a scalar so callers
    can ``if v is None: skip`` — a phase only logs the rank key for the
    representation it trains, so missing keys are normal, not errors. A
    non-scalar tensor (e.g. an un-reduced per-rank vector) also yields
    ``None`` rather than crashing.
    """
    if value is None:
        return None
    if isinstance(value, Tensor):
        if value.numel() != 1:
            return None
        return float(value.item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_loss(outputs: tp.Any) -> Tensor | None:
    """Pull the loss tensor out of a ``training_step`` output.

    The B36 module returns ``breakdown.total`` (a bare tensor); Lightning
    wraps it as ``{"loss": tensor}`` by the time it reaches
    ``on_train_batch_end``. Handle both; return ``None`` for anything else
    (the grad-divergence flag is the authoritative NaN backstop, so this
    is best-effort)."""
    if isinstance(outputs, Tensor):
        return outputs
    if isinstance(outputs, dict):
        loss = outputs.get("loss")
        return loss if isinstance(loss, Tensor) else None
    return None


class CollapseGuardCallback(pl.Callback):
    """Abort a phase on collapse/divergence; write a diagnosis. DDP-safe.

    Parameters
    ----------
    artifact_root:
        Directory to write ``collapse_diagnosis.json`` into (rank-suffixed
        under DDP). ``None`` (e.g. an infra-less smoke run) → the guard
        still aborts, it just skips the file write.
    loss_blowup_factor, alarm_fraction, sustain_checks, warmup_checks,
    warn_sustain_checks:
        Debounce knobs (see module constants for the rationale). Exposed
        for tests; the production defaults are the module constants.
    warmup_min_step:
        Optimizer-step count of LR warmup. Soft criteria are ignored while
        ``trainer.global_step < warmup_min_step`` (on top of the check-count
        grace), so a still-warming low-rank representation can't trip them.
        Default 0 → check-count grace only.
    warn_aborts:
        Whether a sustained soft ``*_warn`` (RankMe < 0.5) aborts. Default
        False — warn is advisory (logs + streaks, no abort); the hard < 0.25
        alarm is the kill. See the module "Threshold philosophy".
    """

    def __init__(
        self,
        *,
        artifact_root: Path | str | None = None,
        loss_blowup_factor: float = LOSS_BLOWUP_FACTOR,
        loss_blowup_min_floor: float = BLOWUP_MIN_FLOOR,
        alarm_fraction: float = ALARM_FRACTION,
        sustain_checks: int = SUSTAIN_CHECKS,
        warmup_checks: int = WARMUP_CHECKS,
        warn_sustain_checks: int = WARN_SUSTAIN_CHECKS,
        warmup_min_step: int = 0,
        warn_aborts: bool = False,
    ) -> None:
        super().__init__()
        self._artifact_root = Path(artifact_root) if artifact_root is not None else None
        self._blowup_factor = loss_blowup_factor
        self._blowup_min_floor = loss_blowup_min_floor
        self._alarm_fraction = alarm_fraction
        self._sustain = sustain_checks
        self._warmup = warmup_checks
        self._warn_sustain = warn_sustain_checks
        # Optimizer-step count of LR warmup. Soft criteria are not believed
        # until ``trainer.global_step >= warmup_min_step`` (in addition to the
        # check-count grace) — a still-warming representation legitimately
        # looks low-rank. 0 → check-count grace only (prior behaviour).
        self._warmup_min_step = warmup_min_step
        # Whether a sustained soft ``*_warn`` (RankMe < 0.5) aborts. Default
        # False: the 0.5 warn threshold is an image-ViT value (DINOv3 §3.3),
        # not calibrated to this iEEG front-end whose healthy born-state is
        # ~0.32 normalised rank — so warn is ADVISORY (logs + streaks for
        # trend visibility) and the hard < 0.25 alarm is the kill. The
        # threshold VALUES stay locked (5/28 P0); this only governs the guard's
        # *response* to the soft tier (a guard-owned debounce/policy knob).
        self._warn_aborts = warn_aborts
        # Per-criterion consecutive-bad-check counters.
        self._streak: dict[str, int] = collections.defaultdict(int)
        self._best_val_loss: float = math.inf
        self._n_checks: int = 0
        self._history: collections.deque[dict[str, float]] = collections.deque(
            maxlen=_HISTORY
        )

    # -- per-batch: immediate non-finite catch (rank-safe) --------------

    def on_train_batch_end(  # noqa: D401
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,  # noqa: ARG002 — Lightning hook signature
        outputs: tp.Any,
        batch: tp.Any,  # noqa: ARG002 — Lightning hook signature
        batch_idx: int,  # noqa: ARG002 — Lightning hook signature
    ) -> None:
        """Non-finite gradient or loss → abort now (a NaN never recovers).

        ``train_mon_grad_diverged`` is computed from the DDP-averaged
        gradient, so it is identical on every rank (a NaN on one rank
        averages to NaN on all) — every rank aborts together, no
        reduction needed. The ``outputs`` loss is rank-LOCAL, so it is
        only trusted on a single device; under DDP the grad flag (one
        optimiser step later) is the rank-safe catch.
        """
        diverged = _as_float(trainer.callback_metrics.get("train_mon_grad_diverged"))
        if diverged is not None and diverged >= 1.0:
            self._abort(
                trainer,
                self._verdict(
                    trainer, "grad_diverged",
                    "MON-GRAD-SPIKE-DIVERGENCE flagged a NaN/Inf gradient",
                    diverged,
                    "Lower the peak LR, enable/verify grad-clip (B01 "
                    "default 1.0), and inspect train_mon_grad_spike_ratio "
                    "for the step before divergence.",
                ),
            )
        if trainer.world_size == 1:
            loss = _extract_loss(outputs)
            if loss is not None and not bool(loss.isfinite().all()):
                self._abort(
                    trainer,
                    self._verdict(
                        trainer, "nonfinite_train_loss",
                        "training_step loss is NaN/Inf", float("nan"),
                        "Lower the peak LR / lengthen warm-up, verify "
                        "grad-clip is active, and check the front-end for "
                        "an un-normalised input spike.",
                    ),
                )

    # -- per-validation-check: sustained soft collapse (rank-consensus) -

    def on_validation_epoch_end(  # noqa: D401
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,  # noqa: ARG002 — Lightning hook signature
    ) -> None:
        """Debounced checks over the epoch-reduced alarm panel.

        Computes a rank-LOCAL verdict, then reduces the abort decision
        across ranks (OR) so a divergent per-rank alarm fraction can't
        deadlock DDP. ``reduce_boolean_decision`` is a collective — it is
        called exactly once here on every rank, regardless of the local
        decision, to stay collective-symmetric.
        """
        if trainer.sanity_checking:
            return  # the pre-train sanity val is not a real check
        metrics = trainer.callback_metrics
        self._n_checks += 1
        snapshot: dict[str, float] = {"epoch": float(trainer.current_epoch)}
        local_verdict: CollapseVerdict | None = None

        val_loss = _as_float(metrics.get("val_loss"))
        if val_loss is not None:
            snapshot["val_loss"] = val_loss
            if not math.isfinite(val_loss):
                # Not debounced — a NaN val loss is acute.
                local_verdict = self._verdict(
                    trainer, "nonfinite_val_loss", "val_loss is NaN/Inf",
                    val_loss,
                    "Same as a non-finite train loss: LR / warm-up / "
                    "grad-clip / input normalisation.",
                )
            else:
                self._best_val_loss = min(self._best_val_loss, val_loss)

        # Update every criterion's streak (so they accumulate/reset), and
        # take the first that reaches its sustain as the local verdict. Soft
        # criteria are believed only after BOTH the check-count grace AND the
        # LR-warmup window — ``global_step`` and ``warmup_min_step`` are both
        # in optimizer steps. (Default warmup_min_step=0 → check-count only.)
        past_warmup = (
            self._n_checks > self._warmup
            and int(trainer.global_step) >= self._warmup_min_step
        )
        for name, (is_bad, value, detail, action, sustain) in self._soft_checks(
            metrics, val_loss, snapshot
        ).items():
            if is_bad and past_warmup:
                self._streak[name] += 1
            else:
                self._streak[name] = 0
            # Soft ``*_warn`` (RankMe < 0.5) is ADVISORY by default: its 0.5
            # threshold is an image-ViT value, not calibrated to this iEEG
            # front-end whose healthy born-state is ~0.32 (2026-06-05 gate
            # diagnosis: rank flat at 0.319 from init while val_loss fell — a
            # floor, not a decline). It streaks + logs for trend visibility but
            # does not abort; the hard < 0.25 alarm (below the born-state) is
            # the kill. ``warn_aborts=True`` restores the abort.
            if name.endswith("_warn") and not self._warn_aborts:
                if self._streak[name] == self._warn_sustain:
                    trainer.print(
                        f"[CollapseGuard] ADVISORY: {name} sustained "
                        f"{self._warn_sustain} checks (value={value:.3g}); "
                        "soft warn is NOT aborting (hard <0.25 alarm armed)."
                    )
                continue
            if local_verdict is None and self._streak[name] >= sustain:
                local_verdict = self._verdict(trainer, name, detail, value, action)

        self._history.append(snapshot)
        self._consensus_abort(trainer, local_verdict)

    # -- internals ------------------------------------------------------

    def _soft_checks(
        self,
        metrics: tp.Mapping[str, tp.Any],
        val_loss: float | None,
        snapshot: dict[str, float],
    ) -> dict[str, tuple[bool, float, str, str, int]]:
        """Build the soft-criterion table for this validation check.

        Each entry is ``(is_bad_this_check, value, detail, action,
        sustain)``. Keys absent from ``metrics`` are skipped (a phase logs
        only the rank key for the representation it trains, so P1 has
        ``frontend_rankme`` and P2 has the M4 ``rankme``)."""
        out: dict[str, tuple[bool, float, str, str, int]] = {}

        # 1. loss blow-up vs running best (divergence that stays finite).
        #    Suppressed when best is below BLOWUP_MIN_FLOOR so a near-zero
        #    floor can't manufacture a large ratio from a trivial increase.
        if (
            val_loss is not None
            and math.isfinite(val_loss)
            and math.isfinite(self._best_val_loss)
            and self._best_val_loss > self._blowup_min_floor
        ):
            ratio = val_loss / self._best_val_loss
            out["loss_blowup"] = (
                ratio > self._blowup_factor,
                ratio,
                f"val_loss {val_loss:.4g} is {ratio:.1f}× the best "
                f"{self._best_val_loss:.4g} seen this phase",
                "Loss is climbing, not NaN — lower the LR or revert to the "
                "last healthy checkpoint and resume with a smaller step.",
                self._sustain,
            )

        # 2. RankMe collapse (phase-scoped). Hard alarm (<0.25) at the
        #    base sustain; soft warn (<0.5) at the longer warn-sustain to
        #    catch a slow decline before it crosses the alarm line.
        for base, label in (
            ("val_mon_rankme", "M4 parcel-token"),
            ("val_mon_frontend_rankme", "front-end M2"),
        ):
            alarm = _as_float(metrics.get(f"{base}_alarm"))
            if alarm is not None:
                snapshot[f"{base}_alarm"] = alarm
                out[f"{base}_alarm"] = (
                    alarm >= self._alarm_fraction,
                    alarm,
                    f"RankMe ALARM on the {label} representation "
                    f"(normalised effective rank < 0.25 on "
                    f"{alarm * 100:.0f}% of probed steps)",
                    "Representation is collapsing to a low-rank subspace — "
                    "check the EMA τ, the predictor warm-start, and that "
                    "the mask leaves targets (see no-masking).",
                    self._sustain,
                )
            warn = _as_float(metrics.get(f"{base}_warn"))
            if warn is not None:
                snapshot[f"{base}_warn"] = warn
                out[f"{base}_warn"] = (
                    warn >= self._alarm_fraction,
                    warn,
                    f"RankMe WARN on the {label} representation "
                    f"(normalised effective rank < 0.5 on "
                    f"{warn * 100:.0f}% of probed steps) sustained — a slow "
                    "rank decline before the 0.25 alarm",
                    "Same as the rank alarm but caught earlier; inspect the "
                    "rank trend and EMA τ before it hard-collapses.",
                    self._warn_sustain,
                )

        # 3. latent-valid coverage degeneracy.
        cov = _as_float(metrics.get("val_mon_coverage_alarm"))
        if cov is not None:
            snapshot["val_mon_coverage_alarm"] = cov
            out["coverage"] = (
                cov >= self._alarm_fraction,
                cov,
                f"Coverage alarm on {cov * 100:.0f}% of probed steps "
                "(>10% of clips have ≤1 active parcel slot)",
                "Too many clips collapse to a single parcel — check the "
                "ref-aug / DK support join and valid_mask threading.",
                self._sustain,
            )

        # 4. masking produced no targets (no SSL signal at all).
        n_masked = _as_float(metrics.get("val_n_masked"))
        if n_masked is not None:
            snapshot["val_n_masked"] = n_masked
            out["no_masking"] = (
                n_masked == 0.0,
                n_masked,
                "val_n_masked is 0 — the mask sampler scored no cells, so "
                "the JEPA loss is identically 0 (no learning signal)",
                "Inspect the M2/M4 mask sampler: ratio, clamp bounds, and "
                "that valid bins/parcels exist for this corpus.",
                self._sustain,
            )
        return out

    def _verdict(
        self,
        trainer: pl.Trainer,
        criterion: str,
        detail: str,
        value: float,
        action: str,
    ) -> CollapseVerdict:
        return CollapseVerdict(
            criterion=criterion,
            detail=detail,
            value=value,
            global_step=int(trainer.global_step),
            epoch=int(trainer.current_epoch),
            suggested_action=action,
        )

    def _consensus_abort(
        self, trainer: pl.Trainer, local_verdict: CollapseVerdict | None,
    ) -> None:
        """Reduce the soft-path abort decision across ranks, then raise.

        ``reduce_boolean_decision(..., all=False)`` is OR: if ANY rank
        found collapse, every rank raises — so DDP can't deadlock with one
        rank dead and the rest blocked on the next collective. Called once
        per validation check on every rank (collective-symmetric).
        """
        should = trainer.strategy.reduce_boolean_decision(
            local_verdict is not None, all=False,
        )
        if not should:
            return
        if local_verdict is not None:
            self._abort(trainer, local_verdict)
        # This rank saw nothing but a peer did — abort anyway, in lock-step.
        raise CollapseStop(
            "[CollapseGuard] consensus abort: a peer rank detected "
            f"collapse at step {int(trainer.global_step)}."
        )

    def _abort(self, trainer: pl.Trainer, verdict: CollapseVerdict) -> None:
        self._write_diagnosis(trainer, verdict)
        raise CollapseStop(
            f"[CollapseGuard] {verdict.criterion} at "
            f"step {verdict.global_step} / epoch {verdict.epoch}: "
            f"{verdict.detail} (value={verdict.value}). "
            f"Suggested: {verdict.suggested_action}"
        )

    def _write_diagnosis(
        self, trainer: pl.Trainer, verdict: CollapseVerdict,
    ) -> None:
        if self._artifact_root is None:
            return
        payload = {
            "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "global_rank": int(trainer.global_rank),
            "world_size": int(trainer.world_size),
            "verdict": dataclasses.asdict(verdict),
            "best_val_loss": (
                None if not math.isfinite(self._best_val_loss) else self._best_val_loss
            ),
            "n_checks": self._n_checks,
            "streaks": dict(self._streak),
            "recent_history": list(self._history),
        }
        # Rank-suffix under DDP so concurrent writers don't race the file.
        suffix = f"_rank{trainer.global_rank}" if trainer.world_size > 1 else ""
        try:
            self._artifact_root.mkdir(parents=True, exist_ok=True)
            out = self._artifact_root / f"collapse_diagnosis{suffix}.json"
            out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except OSError:
            # Never let a disk hiccup swallow the abort — the raise still
            # fires; the JSON is a convenience, not the kill mechanism.
            pass


__all__ = [
    "ALARM_FRACTION",
    "BLOWUP_MIN_FLOOR",
    "CollapseGuardCallback",
    "CollapseStop",
    "CollapseVerdict",
    "LOSS_BLOWUP_FACTOR",
    "SUSTAIN_CHECKS",
    "WARMUP_CHECKS",
    "WARN_SUSTAIN_CHECKS",
]
