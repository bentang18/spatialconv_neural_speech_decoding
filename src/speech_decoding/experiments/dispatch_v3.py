"""v14_converged_v3 — standalone pretrain launcher (Phase E1/E2).

A fresh, lean entrypoint that wires the v3 stack end-to-end; it does NOT reuse the
v2 ``dispatch_v14`` / exca ``Experiment`` machinery, which is welded to the v2
encoder config, pool taps and staged P1/P2/P3/P4 phases none of which v3 has (memo
project-v3-pipeline-build-contract-2026-07-10). v3 is a single plain-JEPA pretrain:
one model, one optimizer, one train loop, no eval cell.

    caches ─▶ load_v3_sessions ─▶ V3DataModule
                                   V3ConvergedModel ─▶ V14ConvergedV3Module ─▶ Trainer.fit
                                   LightningOptimizer (WarmupCosine, non-fused AdamW)

LOCKED launch hyperparameters (Ben 2026-07-10, memo project-v3-locked-intentional-
divergences-2026-07-10): lr 6e-3, wd 0.04, grad-clip 3.0, warmup_cosine /
min_lr_ratio 1.0, warmup 5000, ema-tau 0.99925, β₂ 0.95, seed 33, clip 3.0 s / 96
slots @ 32 fps, bs 32 × accum 4, monitor every step. These are the argparse
DEFAULTS here (unlike ``dispatch_v14`` whose v2 defaults differ), so a bare launch is
the locked recipe.

E2 trainer/precision knobs (audit project-v3-flop-throughput-audit-2026-07-10) live
in ``_build_trainer``: bf16-mixed, non-fused AdamW (fused ⊥ Lightning bf16 grad-clip),
matmul-high, activation-ckpt OFF (model default), DDP find_unused=False (v3 has no
dormant grad params — every param gets a gradient each step), monitor every_n_steps
=1, reload_dataloaders_every_n_epochs=1 (fresh windows/epoch under persistent
workers). torch.compile is an opt-in lever (``--compile``), applied LAST per the
audit ordering (varlen → bf16 → compile).

The braintreebank parcel lookup is the one seam validated on real data at F2; it is
injected into ``load_v3_sessions`` so the launcher assembly is unit-testable with a
stub (F1 local smoke).
"""

from __future__ import annotations

import argparse
import typing as tp

import lightning.pytorch as pl
import torch

# Importing lr_schedule registers the custom ``WarmupCosine`` scheduler on the
# neuraltrain discriminated-union so ``LightningOptimizer(**cfg)`` can resolve it.
import speech_decoding.experiments.lr_schedule  # noqa: F401
from neuraltrain import LightningOptimizer
from speech_decoding.experiments.monitors.ssl_health_v3 import SSLHealthMonitorV3
from speech_decoding.experiments.v14_converged_v3_module import V14ConvergedV3Module
from speech_decoding.models.v14_converged_v3.datamodule import V3DataModule
from speech_decoding.models.v14_converged_v3.dataset import V3SessionSpec
from speech_decoding.models.v14_converged_v3.masking import V3MaskConfig
from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel
from speech_decoding.models.v14_converged_v3.objective import LAMBDA_CTX, LAMBDA_NLL
from speech_decoding.models.v14_converged_v3.secondary_head import NLL_FLOOR_JITTER
from speech_decoding.models.v14_converged_v3.session_loader import (
    ParcelFn,
    load_v3_sessions,
)

_FPS = 32.0  # the v3 uniform 32 Hz frame clock (hop=64 @ native rate)


class _EpochSeedCallback(pl.Callback):
    """Fan the epoch out to the datamodule so windows re-draw + the sampler
    reshuffles each epoch. Lightning does not call a datamodule's custom
    ``set_epoch``; without this the per-epoch seed is frozen and every epoch draws
    the identical windows (``reload_dataloaders_every_n_epochs=1`` only rebuilds the
    loader, it does not advance the seed)."""

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        dm = getattr(trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "set_epoch"):
            dm.set_epoch(int(trainer.current_epoch))


class _StepTimeCallback(pl.Callback):
    """Log wall-clock ``train_sec_per_step`` every step — FREE (a CPU perf_counter
    diff between consecutive batch-ends + one scalar log; no CUDA sync, no extra
    kernel, and loss already logs each step). Measures true step-to-step throughput
    incl. dataloader, so compile spikes show as tall bars and the steady floor is
    obvious. First step is skipped (no prior mark); ``sync_dist=False`` ⇒ no
    cross-rank barrier.

    ALSO prints a per-rank HEARTBEAT to stdout every ``heartbeat_every`` optimizer
    steps (2026-07-16). The wandb scalar above is invisible in the .log file, which
    is why r4's NCCL-abort post-mortem could not name the step it died at, nor which
    rank ran ahead: the whole 14 h log held zero step lines. EVERY rank prints (not
    just rank 0) because the failure mode this exists to catch is rank DIVERGENCE —
    ranks 0/2/3 stalled in python at 266865 while rank 1 sat alone in a broadcast.
    Per-rank lines make that read straight off the log instead of being inferred from
    NCCL sequence numbers. Cost: one f-string + one write per rank per N steps, no
    CUDA sync (``global_step``/``current_epoch`` are host ints) ⇒ throughput-neutral.
    Deduped on global_step: ``on_train_batch_end`` fires once per MICRO-batch, so
    under grad-accum the same optimizer step arrives ``accumulate_grad_batches`` times.
    """

    def __init__(self, heartbeat_every: int = 100) -> None:
        self._t: float | None = None
        self._heartbeat_every = int(heartbeat_every)
        self._t0: float | None = None
        self._last_hb: int | None = None

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *_: object) -> None:  # noqa: ARG002
        import time

        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        sec = None if self._t is None else now - self._t
        if sec is not None:
            pl_module.log(
                "train_sec_per_step", sec,
                on_step=True, on_epoch=False, prog_bar=True, sync_dist=False,
            )
        self._t = now
        step = int(trainer.global_step)
        if (
            self._heartbeat_every > 0
            and step % self._heartbeat_every == 0
            and step != self._last_hb
        ):
            self._last_hb = step
            print(
                f"[hb] rank={trainer.global_rank} step={step} "
                f"epoch={int(trainer.current_epoch)} "
                f"sec_per_step={'nan' if sec is None else format(sec, '.3f')} "
                f"elapsed={now - self._t0:.0f}s",
                flush=True,
            )


# --------------------------------------------------------------------- optimizer
def build_v3_optim_cfg(
    *,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    min_lr_ratio: float,
    adam_beta2: float,
) -> LightningOptimizer:
    """The locked v3 ``LightningOptimizer``: non-fused AdamW + WarmupCosine.

    NON-fused is mandatory: a fused torch optimizer sets
    ``_step_supports_amp_scaling=True``, which makes Lightning's bf16-mixed AMP
    plugin REFUSE gradient clipping — and both grad-clip 3.0 and bf16-mixed are
    locked. An empty ``kwargs`` optimizer is the standard non-fused path. β₁ is the
    torch default 0.9; β₂ is the locked 0.95.
    """
    cfg: dict[str, tp.Any] = {
        "optimizer": {
            "name": "AdamW",
            "lr": lr,
            "kwargs": {"weight_decay": weight_decay, "betas": [0.9, adam_beta2]},
        },
        "scheduler": {
            "name": "WarmupCosine",
            "warmup_steps": warmup_steps,
            "min_lr_ratio": min_lr_ratio,
        },
        "interval": "step",
    }
    return LightningOptimizer(**cfg)


# ------------------------------------------------------------------ parcel seam
def make_bt_parcel_fn(
    bt_root: str,
    *,
    atlas: str = "dkt",
) -> ParcelFn:
    """Default ``parcel_fn``: DKT hard tag per electrode from BT anatomy (F2 seam).

    Maps each cache channel label to its one-hot DKT parcel via
    ``aligned_voltage_support`` (row-aligned to the voltage order). Electrodes with
    NO parcel support (all-zero row — outside every DKT parcel) get the RESERVED
    "unknown" id = ``len(parcel_labels)`` (=74), an identity distinct from every real
    parcel 0..73 that never collides with ``parcel_labels[0]``. ``_n_parcels``
    reserves the matching +1 identity-table row. Atlas name threads
    ``anatomy.atlas_spec`` so the DKT CSV column and the K=74 vocabulary can never
    desync. Lazily imported so the launcher is importable (and unit-testable with a
    stub parcel_fn) without the BT anatomy stack.
    """
    from speech_decoding.studies.braintreebank.anatomy import (
        aligned_voltage_support,
        atlas_spec,
    )

    lcol, plabels = atlas_spec(atlas)  # "dkt" → ("DKT", V14_DKT_PARCEL_LABELS/K=74)
    unknown_id = len(plabels)  # reserved id, distinct from every real parcel 0..K-1

    def parcel_fn(subject_id: int, trial_id: int, labels: tp.Sequence[str]) -> torch.Tensor:
        hs = aligned_voltage_support(
            bt_root, subject_id, trial_id=trial_id,
            parcel_labels=plabels, unmapped_policy="zero", label_column=lcol,
        )
        by_label = {
            lab: (int(hs.support[c].argmax()) if bool(hs.support[c].any()) else unknown_id)
            for c, lab in enumerate(hs.electrode_labels)
        }
        missing = [lab for lab in labels if lab not in by_label]
        if missing:
            raise KeyError(
                f"subject {subject_id} trial {trial_id}: cache labels absent from the "
                f"voltage order {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return torch.tensor([by_label[lab] for lab in labels], dtype=torch.long)

    return parcel_fn


# ------------------------------------------------------------------- assembly
def _n_parcels(sessions: tp.Sequence[V3SessionSpec]) -> int:
    """Identity-table size = DKT vocab (74) + 1 reserved 'unknown' row (id 74) = 75, FIXED.

    Not data-derived: the reserved unknown identity (id 74) must exist even when no
    electrode in the current cohort is unmapped, and the table must be identical
    across cohorts so a checkpoint's parcel-embedding rows mean the same thing
    everywhere. Asserts no realized id escapes the table.
    """
    from speech_decoding.studies.braintreebank.anatomy import V14_DKT_PARCEL_LABELS

    n = len(V14_DKT_PARCEL_LABELS) + 1  # 74 real (0..73) + 1 reserved unknown (74)
    realized = max(int(s.setup.parcel_id.max()) for s in sessions)
    if realized >= n:
        raise ValueError(
            f"realized parcel id {realized} exceeds DKT identity table size {n}"
        )
    return n


def build_v3_training(
    sessions: tp.Sequence[V3SessionSpec],
    args: argparse.Namespace,
) -> tuple[V14ConvergedV3Module, V3DataModule, pl.Trainer]:
    """Assemble (module, datamodule, trainer) from already-loaded sessions.

    Split out from ``main`` so F1 can drive the whole stack locally with synthetic
    sessions + a stub parcel_fn (no caches, no wandb, CPU).
    """
    mask_cfg = V3MaskConfig()  # locked two-tier config (its own defaults)
    mae = getattr(args, "objective", "jepa") == "mae"
    model = V3ConvergedModel(
        n_parcels=_n_parcels(sessions), mask_cfg=mask_cfg,
        deep_sup=getattr(args, "deep_sup", True),
        lambda_nll=getattr(args, "lambda_nll", LAMBDA_NLL),
        nll_floor=getattr(args, "nll_floor", True),
        secondary_loss=getattr(args, "secondary_loss", "nll"),
        context_loss=getattr(args, "context_loss", False),
        lambda_ctx=getattr(args, "lambda_ctx", LAMBDA_CTX),
        mae=mae,
    )
    optim = build_v3_optim_cfg(
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio,
        adam_beta2=args.adam_beta2,
    )
    module = V14ConvergedV3Module(
        model=model, optim_config=optim, seed=args.seed,
        monitor_every_n_steps=args.monitor_every_n_steps,
        secondary_active=_secondary_active(args),
        grad_ratio_every_n_steps=args.grad_ratio_every_n_steps,
        nll_warmup_start_step=getattr(args, "nll_warmup_start_step", 0),
        nll_warmup_steps=getattr(args, "nll_warmup_steps", 0),
        ctx_warmup_start_step=getattr(args, "ctx_warmup_start_step", 0),
        ctx_warmup_steps=getattr(args, "ctx_warmup_steps", 0),
    )
    dm = V3DataModule(
        sessions,
        batch_size=args.batch_size,
        clips_per_session=args.clips_per_session,
        clip_frames=round(args.clip_len * _FPS),
        fps=_FPS,
        num_workers=args.num_workers,
        seed=args.seed,
        same_session=args.same_session_ranks,
    )
    trainer = _build_trainer(args)
    return module, dm, trainer


def _secondary_active(args: argparse.Namespace) -> bool:
    """The secondary Gaussian-NLL is ON iff a state-stats dir is configured (the
    datamodule then supplies per-session stats and the write-only Perceiver receives
    gradient). deep_sup is required for the Perceiver to exist at all."""
    return bool(getattr(args, "state_stats_dir", None)) and bool(
        getattr(args, "deep_sup", True)
    )


def _enabled_optional_modules(args: argparse.Namespace) -> list[str]:
    """Optional modules that join the TRAINABLE set beyond the r1/r2 plain-JEPA
    baseline (encoder + EMA teacher + predictor + ``pred_to_target``).

    Each entry is ``"<module> (<flag that enabled it>)"``. The r3 gate below fires on a
    NON-EMPTY list under multi-GPU static_graph — the crash surface is "the trainable
    parameter set grew", not any one head, so a new optional module must be listed HERE
    or it walks into the same crash. r4's secondary write-only Perceiver Gaussian-NLL is
    exactly such a module: when --state-stats-dir is set the Perceiver receives gradient
    every step (same shape as r3's always-connected context head). r3 died on this under
    static_graph, so the launch must run --no-ddp-static-graph (the default) or ack it."""
    enabled: list[str] = []
    if _secondary_active(args):
        enabled.append("secondary Perceiver Gaussian-NLL (--state-stats-dir)")
    if getattr(args, "context_loss", False):
        enabled.append("V-JEPA 2.1 context head (--context-loss)")
    return enabled


def _build_trainer(args: argparse.Namespace) -> pl.Trainer:
    """The locked v3 Trainer (E2 knobs). Single-device unless ``--devices>1``."""
    torch.set_float32_matmul_precision("high")  # audit lever, numerics-safe

    callbacks: list[pl.Callback] = [
        _EpochSeedCallback(),
        _StepTimeCallback(),
        SSLHealthMonitorV3(every_n_steps=args.monitor_every_n_steps),
    ]
    if args.wandb_project:  # LearningRateMonitor needs a logger; verifies the 5k warmup ramp
        from lightning.pytorch.callbacks import LearningRateMonitor

        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if args.ckpt_ladder_every > 0 and args.ckpt_dir:
        from lightning.pytorch.callbacks import ModelCheckpoint

        callbacks.append(
            ModelCheckpoint(
                dirpath=args.ckpt_dir,
                filename="ladder-{step}",
                every_n_train_steps=args.ckpt_ladder_every,
                save_top_k=-1,  # keep the whole ladder for probing
                save_last=True,
            )
        )

    logger: pl.loggers.Logger | bool = False
    if args.wandb_project:
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(
            project=args.wandb_project, name=args.run_name, save_dir=args.ckpt_dir or ".",
        )

    # Ben 2026-07-13: constant LR (min_lr_ratio=1.0, V-JEPA-2 style) ⇒ max_steps is a pure
    # STOP-POINT, not a schedule parameter, so it can be whatever horizon a run wants.
    # --ssl-max-steps binds AS WRITTEN (it is required, so always explicit). The old
    # max(_, 100_000) floor was a one-time hack so the long-finished r2/r3 queue ran the full
    # 100k horizon without a resubmit; it has no live dependants and actively fought explicit
    # short-horizon arms (e.g. the 20k MAE-vs-JEPA board runs), so it is removed. Early
    # stopping = kill + ladder-pick, unchanged.
    ssl_max_steps = args.ssl_max_steps
    kwargs: dict[str, tp.Any] = dict(
        max_steps=ssl_max_steps,
        max_epochs=-1,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,  # pure pretrain: no val loop
        reload_dataloaders_every_n_epochs=1,  # fresh windows/epoch (persistent workers)
        enable_checkpointing=bool(args.ckpt_dir),
        use_distributed_sampler=False,  # the grouped batch sampler self-shards
    )
    if args.devices and args.devices != 1:
        from lightning.pytorch.strategies import DDPStrategy

        # v3 has NO dormant requires_grad params (single predictor, one loss ⇒ every
        # trainable param gets a gradient each step), so find_unused=False is safe.
        # static_graph (v2 `--ddp-static-graph`, default True): the per-session
        # electrode count varies TENSOR SHAPES, but the autograd graph STRUCTURE is
        # identical every step (same blocks, same params all participate) — which is
        # what static_graph requires, not fixed shapes. v2 ran static_graph=True with
        # the same session-grouped ragged batching, and it composes with torch.compile
        # (fewer DDP reducer re-analyses). Only active on multi-GPU (single-GPU launch
        # never builds a DDPStrategy).
        optional = _enabled_optional_modules(args)
        if (
            optional
            and args.ddp_static_graph
            and not getattr(args, "ack_r3_static_graph", False)
        ):
            raise SystemExit(
                "REFUSING TO LAUNCH: optional module(s) join the trainable set under\n"
                "multi-GPU DDP static_graph:\n"
                + "".join(f"    {m}\n" for m in optional)
                + "\n"
                "This is the r3 shape. r3 (DeltaAI 2653154, 2026-07-14) enabled exactly one\n"
                "such module (the context head) and all 4 ranks died at the FIRST backward with\n"
                "    expect_autograd_hooks_ INTERNAL ASSERT FAILED (c10d/reducer.cpp:1703)\n"
                "after ~5 min of compile warmup — 0 usable steps, GPU hours burned.\n"
                "\n"
                "ROOT CAUSE IS NOT ESTABLISHED. Two plausible stories were tested and REFUTED:\n"
                "  - 'context head is unused during the lambda warmup'  -> refuted: _static_off\n"
                "    is a TYPE test, so the 0-d tensor lambda keeps the head graph-connected;\n"
                "    it gets zero-valued grads, not no grads.\n"
                "  - 'the head joining DDP's buckets is the trigger'    -> refuted: a CPU/gloo\n"
                "    repro asserts identically with the head frozen (r2) and live (r3).\n"
                "What IS established (CPU/gloo, zero GPU, committed as\n"
                "experiments/test_ddp_static_graph_repro.py): static_graph=False survives every\n"
                "configuration tested.\n"
                "\n"
                "The gate is therefore NOT about which module it is — it fires on ANY growth of\n"
                "the trainable set beyond the r1/r2 baseline (r1 ran ~43k steps on that baseline\n"
                "with static_graph=True; nothing else about static_graph has been cleared).\n"
                "\n"
                "DO NOT 'fix' this with find_unused_parameters=True — the module is not unused.\n"
                "DO NOT freeze/unfreeze it at warmup_start — that mutates graph structure\n"
                "mid-run, which static_graph forbids; it trades a step-0 crash for a step-15k one.\n"
                "\n"
                "Relaunch with --no-ddp-static-graph (DDP bookkeeping only: it does not touch\n"
                "gradients, the loss, or any locked hyperparameter, so it is NOT a contract\n"
                "change). Costs some throughput.\n"
                "\n"
                "To override deliberately, pass --ack-r3-static-graph.\n"
                "See memory: project-v3-r3-ddp-static-graph-crash-2026-07-14"
            )
        kwargs["strategy"] = DDPStrategy(
            find_unused_parameters=False, static_graph=args.ddp_static_graph
        )
    return pl.Trainer(**kwargs)


# ----------------------------------------------------------------------- argv
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v14_converged_v3 pretrain launcher")
    # --- data ---
    p.add_argument("--bt-root", required=True, help="BrainTreebank root (anatomy)")
    p.add_argument("--band-cache-dir", dest="band_cache_dirs", action="append",
                   required=True,
                   help="one per band in v3 concat order (slow, mid, hga); pass 3×")
    p.add_argument("--span-dir", required=True, help="guard-2 bad-window span JSONs")
    p.add_argument("--lof-report-path", default=None, help="guard-1 LOF report (opt)")
    p.add_argument("--session", dest="sessions", action="append", required=True,
                   metavar="S:T", help="subject:trial, repeatable (e.g. 6:4)")
    p.add_argument("--clips-per-session", type=int, default=40_000,
                   help="per-epoch clip budget/session (operational; epoch length + "
                        "window re-draw cadence, NOT a model hyperparameter)")
    # Ben 2026-07-16: raised 2000 -> 40000. Science-neutral BY CONSTRUCTION — the dataset is
    # virtual (``V3SessionDataset.__getitem__`` draws a uniform-random t0 seeded by
    # (seed, epoch, index)), so a step consumes the same iid random windows at any epoch
    # length; only the epoch BOUNDARY moves. What it buys: at 2000 an epoch is ~52 opt-steps
    # (13 sessions x 2000 / (bs32 x accum4 x 4 ranks)) and ``reload_dataloaders_every_n_epochs
    # =1`` rebuilds the loader + respawns workers EVERY 52 steps — r4 did that 507 times before
    # dying at 26414 on an undiagnosed stall whose leading read is Lustre/IO. 40000 makes it
    # ~1016 steps/epoch, 20x less mmap/worker churn. Landed as the DEFAULT (not a floor: the
    # tests and the smoke launch pass small explicit values) so the already-queued r4b — which
    # passes no --clips-per-session and would otherwise inherit r4's exact IO exposure while
    # arms 1/2/4 pass 40000 explicitly — picks it up at RUN time, no resubmit, no forfeited
    # queue age. Same lever as SSL_MAX_STEPS_STD above; slurm stores the sbatch at SUBMIT, but
    # ``.venv/bin/python -m`` reads this module when the job starts.
    p.add_argument("--clip-len", type=float, default=3.0,
                   help="clip seconds (adjustable HP; r4 uses 2.0 — shorter clips give "
                        "more opt-steps + mask diversity per GPU-hour at a ~7% raw "
                        "contact-frame throughput cost, worth it in our step-bound regime)")
    # Per-band robust-z winsor |z| cap (v2 --session-z-winsor-{lfs,hga} port). SLOW+MID
    # = the old sub-HGA content → 15; HGA's heavier tails get the looser 20 (Ben
    # 2026-07-10). Applied at transform inside SessionRobustZNormalizer.
    p.add_argument("--session-z-winsor-slow", dest="winsor_slow", type=float, default=15.0)
    p.add_argument("--session-z-winsor-mid", dest="winsor_mid", type=float, default=15.0)
    p.add_argument("--session-z-winsor-hga", dest="winsor_hga", type=float, default=20.0)
    p.add_argument("--num-workers", type=int, default=4)
    # --- locked optimizer/schedule ---
    p.add_argument("--lr", type=float, default=6e-3)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.04)
    p.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=5000)
    p.add_argument("--min-lr-ratio", dest="min_lr_ratio", type=float, default=1.0)
    p.add_argument("--adam-beta2", dest="adam_beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=3.0)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--accumulate-grad-batches", dest="accumulate_grad_batches",
                   type=int, default=4)
    p.add_argument("--seed", type=int, default=33)
    # --- objective ---
    p.add_argument("--objective", dest="objective", choices=("jepa", "mae"),
                   default="jepa",
                   help="prediction TARGET: 'jepa' (default, EMA-teacher latent, the arm0 "
                        "recipe) or 'mae' (Masked Autoencoder, He 2021 / AudioMAE — "
                        "reconstruct each masked token's OWN norm_pix'd input |STFT| bins, "
                        "no EMA teacher). ONLY the target changes; the visible-only encoder, "
                        "predictor, mask query, margin-gated in_loss, and all locked HPs are "
                        "identical. 'mae' forbids --state-stats-dir / --context-loss.")
    # --- trainer/precision (E2) ---
    p.add_argument("--ssl-max-steps", dest="ssl_max_steps", type=int, required=True)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--accelerator", default="gpu")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--monitor-every-n-steps", dest="monitor_every_n_steps",
                   type=int, default=1)
    p.add_argument("--grad-ratio-every-n-steps", dest="grad_ratio_every_n_steps",
                   type=int, default=0,
                   help="live loss-balance readout (#43): every N opt-steps log "
                        "‖g_nll‖/‖g_jepa‖ on the shared online tower. 0 = OFF (the launch "
                        "default). SINGLE-PROCESS ONLY (--devices 1): the extra autograd.grad "
                        "passes re-enter the DDP reducer (the r3 static_graph×grad-accum crash "
                        "surface), so the module hard-refuses world_size>1. 1-GPU diagnostic only.")
    p.add_argument("--log-every-n-steps", dest="log_every_n_steps", type=int, default=1,
                   help="wandb flush cadence; 1 = per-step resolution (Ben 2026-07-11) "
                        "so update_cos/grad-spike/feat_std are not window-averaged away")
    p.add_argument("--state-stats-dir", dest="state_stats_dir", default=None,
                   help="dir of per-subject frozen state-norm tables (sub-<id>.npz with "
                        "stat_mean/stat_std, each (n_parcels, 6) indexed by parcel id "
                        "VALUE). Present ⇒ the secondary write-only Perceiver Gaussian-NLL "
                        "is ON (total = JEPA_L1 + λ·NLL); omit ⇒ JEPA-only. Requires "
                        "--deep-sup (the Perceiver reads the deep-sup taps).")
    p.add_argument("--lambda-nll", dest="lambda_nll", type=float, default=LAMBDA_NLL,
                   help="secondary Gaussian-NLL weight λ in total = JEPA_L1 + λ·NLL "
                        f"(contract §5 open knob; default {LAMBDA_NLL}). This is the HOLD value "
                        "the ramp climbs to. No effect without --state-stats-dir.")
    p.add_argument("--nll-floor", dest="nll_floor",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="ON (default, r1–r4): Sigma = L Lᵀ + the measured count-dependent "
                        "noise floor. --no-nll-floor is r5 Arm 2 (floor-off): the head learns "
                        f"Sigma with only a {NLL_FLOOR_JITTER:g} conditioning jitter, no "
                        "measured floor. No effect without --state-stats-dir.")
    p.add_argument("--secondary-loss", dest="secondary_loss",
                   choices=("nll", "l1", "diag_nll"), default="nll",
                   help="'diag_nll' (r5-mod, the CURRENT 5-dim objective): point (mu-only) "
                        "head scored by frozen-diagonal Gaussian NLL over the 5-dim state "
                        "[slow_mu, mid_mu, hga_mu, relmod48, relmod816] — sigma^2 is the "
                        "measured count-dependent noise floor, FROZEN, so the loss drops only "
                        "via a better mu (closes the r4 free-sigma flatline hatch). Requires "
                        "--nll-floor. 'nll' (r1–r4, RETIRED for the 5-dim layout — its D//2 "
                        "mean/std split mis-scores the 3/2 state): full-covariance Gaussian "
                        "NLL. 'l1' is r5 Arm 3 (POINT loss): mu-only head scored by mean "
                        "per-position sum|x-mu| over present dims. No effect without "
                        "--state-stats-dir.")
    p.add_argument("--nll-warmup-start-step", dest="nll_warmup_start_step",
                   type=int, default=0,
                   help="opt-step at which the secondary-NLL λ ramp OPENS. Before it, λ=0 "
                        "(the Perceiver head stays in-graph with zero grad — DDP-safe). "
                        "Default 0 ⇒ ramp opens immediately.")
    p.add_argument("--nll-warmup-steps", dest="nll_warmup_steps",
                   type=int, default=0,
                   help="length (opt-steps) of the linear 0→--lambda-nll ramp starting at "
                        "--nll-warmup-start-step; λ holds at --lambda-nll after. Default 0 ⇒ "
                        "no ramp (constant λ==--lambda-nll from --nll-warmup-start-step).")
    p.add_argument("--context-loss", dest="context_loss", action="store_true",
                   help="V-JEPA 2.1 §2.3.1 context loss: the predictor also predicts the "
                        "per-level-normed teacher target at the VISIBLE (context) tokens via a "
                        "SEPARATE projection head, scored by the same L1 at ~masked positions, "
                        "λ_ctx-weighted (total += λ_ctx·L_ctx). Off ⇒ zero new params. Requires "
                        "--deep-sup.")
    p.add_argument("--lambda-ctx", dest="lambda_ctx", type=float, default=LAMBDA_CTX,
                   help=f"context-loss weight λ_ctx — the HOLD the ramp climbs to (default "
                        f"{LAMBDA_CTX}, V-JEPA 2.1 Eq 2 peak). No effect without --context-loss.")
    p.add_argument("--ctx-warmup-start-step", dest="ctx_warmup_start_step",
                   type=int, default=0,
                   help="opt-step at which the context-loss λ_ctx ramp OPENS. Before it, λ_ctx=0 "
                        "(the context head stays in-graph with zero grad — DDP-safe). Default 0 "
                        "⇒ ramp opens immediately.")
    p.add_argument("--ctx-warmup-steps", dest="ctx_warmup_steps",
                   type=int, default=0,
                   help="length (opt-steps) of the linear 0→--lambda-ctx ramp starting at "
                        "--ctx-warmup-start-step; λ_ctx holds at --lambda-ctx after. Default 0 ⇒ "
                        "no ramp (constant λ_ctx==--lambda-ctx from --ctx-warmup-start-step).")
    p.add_argument("--deep-sup", dest="deep_sup",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="deep self-supervision (#61, V-JEPA 2.1 copy-exactly): tap "
                        "encoder blocks {3,6,9,12}, per-level affine LN, concat → "
                        "fusion-MLP → predictor → wide proj → per-level double-normed "
                        "targets. --no-deep-sup = the single-tap ablation arm")
    p.add_argument("--compile", dest="compile", action="store_true",
                   help="torch.compile the model, dynamic=False (audit lever; apply "
                        "LAST). v2 measured >1.5x; validate the flex-nesting on G4")
    p.add_argument("--ddp-static-graph", dest="ddp_static_graph",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="DDP static_graph (multi-GPU only). DEFAULT FLIPPED TO FALSE "
                        "2026-07-15: static_graph killed r3 at the first backward, and it "
                        "is the ONLY thing established across every sighting (a gloo CPU "
                        "repro asserts under grad-accum even with the optional module "
                        "FROZEN -- see test_ddp_static_graph_repro.py). r4 runs optional "
                        "modules, so it needs static_graph=False regardless; defaulting it "
                        "on was a landmine with nothing to gain. Pass --ddp-static-graph "
                        "explicitly to opt back in on a plain-JEPA baseline.")
    p.add_argument("--ack-r3-static-graph", dest="ack_r3_static_graph",
                   action="store_true",
                   help="override the r3 crash gate and launch an optional-module run "
                        "(e.g. --state-stats-dir set) with multi-GPU static_graph anyway "
                        "(this killed r3 at the first backward)")
    p.add_argument("--same-session-ranks", dest="same_session_ranks",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="All DDP ranks step the SAME session each step (v2 "
                        "--same-session-ranks): identical shape across ranks ⇒ no "
                        "recompile-desync stall under --compile + no straggler. "
                        "DEFAULT ON (max-throughput); --no-same-session-ranks to disable. "
                        "Multi-GPU only; no-op at devices=1.")
    p.add_argument("--sdpa-backend", dest="sdpa_backend", default="auto",
                   choices=["auto", "cudnn", "flash", "efficient", "math"],
                   help="Preferred SDPA backend (v2 --sdpa-backend cudnn). Sets a "
                        "PRIORITY list (chosen first, then flash/efficient/math) so "
                        "an unusable kernel gracefully falls back — never hard-errors. "
                        "'auto' (default) leaves torch's dispatcher untouched.")
    # --- io ---
    p.add_argument("--ckpt-dir", default=None, help="checkpoint + wandb save dir")
    p.add_argument("--ckpt-ladder-every", dest="ckpt_ladder_every", type=int,
                   default=1000, help="save a ladder ckpt every N steps (0=off); "
                                      "1000 = v2 probe-ladder cadence (Ben 2026-07-10)")
    p.add_argument("--resume-ckpt", dest="resume_ckpt", default=None,
                   help="Lightning ckpt_path to restore before fit (model + optimizer + "
                        "EMA + global_step + loop state). Use to FORK a run: point at a "
                        "prior ladder-<step>.ckpt to branch from that step with a changed "
                        "config (e.g. r4b = r4's step-10000 ckpt with --lambda-nll 0). The "
                        "resumed trainable-param set must match the ckpt's, so keep "
                        "--state-stats-dir when the ckpt was trained with it. Default None "
                        "= fresh run from step 0.")
    p.add_argument("--wandb-project", dest="wandb_project", default=None,
                   help="wandb project (live); omit to disable logging")
    p.add_argument("--run-name", dest="run_name", default=None)
    return p


def _parse_sessions(raw: tp.Sequence[str]) -> list[tuple[int, int]]:
    out = []
    for item in raw:
        s, _, t = item.partition(":")
        if not t:
            raise ValueError(f"--session must be S:T (subject:trial), got {item!r}")
        out.append((int(s), int(t)))
    return out


def main(argv: tp.Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if len(args.band_cache_dirs) != 3:
        raise ValueError(
            f"need 3 --band-cache-dir (slow, mid, hga), got {len(args.band_cache_dirs)}"
        )
    if args.grad_ratio_every_n_steps > 0 and args.devices and args.devices != 1:
        raise SystemExit(
            "REFUSING TO LAUNCH: --grad-ratio-every-n-steps > 0 requires --devices 1.\n"
            "The live grad-ratio does two extra autograd.grad passes over the shared online\n"
            "tower; under multi-GPU DDP those re-enter the reducer over the shared graph — the\n"
            "r3 static_graph×grad-accum crash surface. It is a 1-GPU diagnostic lever only."
        )
    if args.objective == "mae" and (args.state_stats_dir or args.context_loss):
        raise SystemExit(
            "REFUSING TO LAUNCH: --objective mae has no secondary NLL or context loss.\n"
            "Drop --state-stats-dir / --context-loss (MAE reconstructs the raw input target)."
        )
    pl.seed_everything(args.seed, workers=True)

    sessions = load_v3_sessions(
        sessions=_parse_sessions(args.sessions),
        band_cache_dirs=args.band_cache_dirs,
        span_dir=args.span_dir,
        parcel_fn=make_bt_parcel_fn(args.bt_root),
        lof_report_path=args.lof_report_path,
        winsor=(args.winsor_slow, args.winsor_mid, args.winsor_hga),
        state_stats_dir=args.state_stats_dir,
    )
    module, dm, trainer = build_v3_training(sessions, args)
    if args.compile:
        # dynamic=False (STATIC graph per shape) = v2's --no-compile-dynamic, the config
        # that hit 1-2 s/step. Static shapes give the best-fused kernels; the EARLIER
        # G4 probe's thrash (3001 grad_mode + 2900 size recompiles, flat ~15 s/step) was
        # NOT dynamic=False's fault — it was cache EVICTION: v3 cycles 13 sessions whose
        # ONLINE M_vis and TEACHER/predictor N differ, ×grad_mode (online grad vs EMA
        # teacher no_grad) ×bf16/fp32 autocast ⇒ ~150 specialisations across the towers,
        # far above cache_size_limit=64. Once the working set exceeds the cache, Dynamo
        # evicts a variant and recompiles it next step ⇒ perpetual ping-pong, warmup
        # never ends. dynamic=None "fixed" the shape axis but broke build_pack_plan with
        # symbolic-value guards (packing.py:84 recompiled every step). The real fix:
        # keep dynamic=False and size the cache ABOVE the working set so the ~500-step
        # warmup COMPLETES and every step thereafter hits a cached static kernel. Warmup
        # is longer than v2 (13 session shapes vs 1 padded) but one-time and bounded.
        import torch._dynamo as _dynamo

        # DDPOptimizer OFF (v2 parity, v14_converged_v2_module.py:240): with it ON
        # (torch default) Dynamo splits the compiled graph at DDP bucket boundaries and
        # inserts its own allreduce, which (a) bypasses DDP's per-param autograd hooks —
        # the mechanism behind the static_graph `expect_autograd_hooks_` assert — and
        # (b) adds a graph break per bucket, worsening our already-heavy compile warmup.
        # For 11.95M params on a single NVLink node with accum4 (allreduce every 4th
        # step) the overlap it buys is negligible; one clean graph + DDP's native hook
        # overlap is the better trade. FLOP/numerics-neutral.
        _dynamo.config.optimize_ddp = False
        _dynamo.config.cache_size_limit = max(256, len(sessions) * 16)
        _dynamo.config.accumulated_cache_size_limit = max(512, len(sessions) * 32)
        module.model = torch.compile(module.model, dynamic=False)  # type: ignore[assignment]
    with _sdpa_ctx(args.sdpa_backend):
        trainer.fit(
            module, datamodule=dm,
            ckpt_path=getattr(args, "resume_ckpt", None) or None,
        )


def _sdpa_ctx(name: str):
    """Prefer one SDPA backend, keep the rest as fallback so an unusable kernel
    (e.g. cuDNN rejecting our additive pad-key bias) degrades instead of raising.
    Under ``--compile`` the context is active when inductor lowers SDPA on the first
    step, so the preference is baked into the compiled kernel. ``auto`` = no-op."""
    import contextlib

    if name == "auto":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    order = {
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    rest = [b for b in (SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH)
            if b is not order[name]]
    return sdpa_kernel([order[name], *rest], set_priority=True)


if __name__ == "__main__":
    main()
