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
    model = V3ConvergedModel(n_parcels=_n_parcels(sessions), mask_cfg=mask_cfg)
    optim = build_v3_optim_cfg(
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio,
        adam_beta2=args.adam_beta2,
    )
    module = V14ConvergedV3Module(
        model=model, optim_config=optim, seed=args.seed,
        monitor_every_n_steps=args.monitor_every_n_steps,
    )
    dm = V3DataModule(
        sessions,
        batch_size=args.batch_size,
        clips_per_session=args.clips_per_session,
        clip_frames=round(args.clip_len * _FPS),
        fps=_FPS,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    trainer = _build_trainer(args)
    return module, dm, trainer


def _build_trainer(args: argparse.Namespace) -> pl.Trainer:
    """The locked v3 Trainer (E2 knobs). Single-device unless ``--devices>1``."""
    torch.set_float32_matmul_precision("high")  # audit lever, numerics-safe

    callbacks: list[pl.Callback] = [
        _EpochSeedCallback(),
        SSLHealthMonitorV3(every_n_steps=args.monitor_every_n_steps),
    ]
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

    kwargs: dict[str, tp.Any] = dict(
        max_steps=args.ssl_max_steps,
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
    p.add_argument("--clips-per-session", type=int, default=2000,
                   help="per-epoch clip budget/session (operational; epoch length + "
                        "window re-draw cadence, NOT a model hyperparameter)")
    p.add_argument("--clip-len", type=float, default=3.0, help="clip seconds (locked 3.0)")
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
    # --- trainer/precision (E2) ---
    p.add_argument("--ssl-max-steps", dest="ssl_max_steps", type=int, required=True)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--accelerator", default="gpu")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--monitor-every-n-steps", dest="monitor_every_n_steps",
                   type=int, default=1)
    p.add_argument("--log-every-n-steps", dest="log_every_n_steps", type=int, default=50)
    p.add_argument("--compile", dest="compile", action="store_true",
                   help="torch.compile the model, dynamic=False (audit lever; apply "
                        "LAST). v2 measured >1.5x; validate the flex-nesting on G4")
    p.add_argument("--ddp-static-graph", dest="ddp_static_graph",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="DDP static_graph (multi-GPU only; v2 --ddp-static-graph)")
    # --- io ---
    p.add_argument("--ckpt-dir", default=None, help="checkpoint + wandb save dir")
    p.add_argument("--ckpt-ladder-every", dest="ckpt_ladder_every", type=int,
                   default=5000, help="save a ladder ckpt every N steps (0=off)")
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
    pl.seed_everything(args.seed, workers=True)

    sessions = load_v3_sessions(
        sessions=_parse_sessions(args.sessions),
        band_cache_dirs=args.band_cache_dirs,
        span_dir=args.span_dir,
        parcel_fn=make_bt_parcel_fn(args.bt_root),
        lof_report_path=args.lof_report_path,
    )
    module, dm, trainer = build_v3_training(sessions, args)
    if args.compile:
        # dynamic=False (v2 `--no-compile-dynamic`): every shape v3 sees is STATIC
        # per session — session-homogeneous batching + the varlen constant-masked-count
        # invariant fix M_vis (online) and N (teacher/predictor) per session. Letting
        # Dynamo specialize each shape gives the fast static graphs v2 measured >1.5×
        # from; the dynamic path would trace slower symbolic-shape graphs and can
        # graph-break. Cost: one recompile per distinct session shape ⇒ raise the
        # recompile cap above the cohort's shape count (online + full-N per session,
        # ×headroom) so we specialize instead of falling back to eager.
        #
        # OPEN (G4 measures): v3's L1 uses FlexAttention which SELF-compiles
        # (varlen.py `_flex_fn`), so torch.compile(model) wraps an already-compiled
        # inner HOP — v2 used SDPA-cudnn, no nesting. FlexAttention is designed to be
        # called inside a compiled region so this should lower natively, but whether
        # the varlen backend-dispatch / create_block_mask graph-breaks is exactly what
        # the G4 probe pins ("steps/sec AFTER compile"). If it under-delivers, the fix
        # is to drop the inner _flex_fn compile and let the outer compile own flex.
        import torch._dynamo as _dynamo

        _dynamo.config.cache_size_limit = max(64, len(sessions) * 4)
        module.model = torch.compile(module.model, dynamic=False)  # type: ignore[assignment]
    trainer.fit(module, datamodule=dm)


if __name__ == "__main__":
    main()
