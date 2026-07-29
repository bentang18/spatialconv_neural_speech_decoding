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
from dataclasses import replace

import lightning.pytorch as pl
import torch

# Importing lr_schedule registers the custom ``WarmupCosine`` scheduler on the
# neuraltrain discriminated-union so ``LightningOptimizer(**cfg)`` can resolve it.
import speech_decoding.experiments.lr_schedule  # noqa: F401
from neuraltrain import LightningOptimizer
from speech_decoding.experiments.monitors.ssl_health_v3 import SSLHealthMonitorV3
from speech_decoding.experiments.v14_converged_v3_module import V14ConvergedV3Module
from speech_decoding.models.v14_converged_v3.datamodule import V3DataModule
from speech_decoding.models.v14_converged_v3.dataset import (
    NATIVE_FINE_BAND_RATES,
    R5_BAND_RATES,
    UNIFORM_BAND_RATES,
    V3SessionSpec,
)
from speech_decoding.models.v14_converged_v3.masking import V3MaskConfig
from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel
from speech_decoding.models.v14_converged_v3.stem import NOFUSION_DECIMATE
from speech_decoding.models.v14_converged_v3.session_loader import (
    ParcelFn,
    load_v3_sessions,
)

_FPS = 32.0  # the v3 uniform 32 Hz frame clock (hop=64 @ native rate)

# v3r5nf (no-fusion) reads the SAME two 64 Hz caches as r5-fused (v3hga, v3lfs) — the ONLY
# change is fusion → full separation in the stem, not the per-band read offsets.
R5NF_BAND_RATES = R5_BAND_RATES


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
    stable_steps: int = 0,
    decay_shape: str = "cosine",
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
            "stable_steps": stable_steps,
            "decay_shape": decay_shape,
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


def _frontend_config(
    args: argparse.Namespace,
) -> tuple[bool, bool, bool, bool, tuple[tuple[int, int], ...]]:
    """(native_fine_hga, early_fusion, no_fusion, r6, band_rates) from ``--frontend``.

    ``v3`` (default) = uniform-32Hz PerBandStem (arm0, byte-identical to every prior run).
    ``v3fine`` = FineHgaStem on native-rate caches (SLOW 4Hz / MID 16Hz / HGA 128Hz).
    ``v3r5`` = EarlyFusionStem (Chang 2-stream): 2 caches (v3hga, v3lfs) both @64 Hz, fused
    5ch→256→stride-2 to 32 Hz tokens.
    ``v3r5nf`` = NoFusionStem: the SAME 2 caches, but each conv-pooled by its OWN stem into its
    OWN 32 Hz token stream (2 streams, 2-way band embed, independent masks/heads, MAE-only).
    ``v3r6`` = arm0's frontend VERBATIM — the SAME 3-band 32 Hz |STFT| caches, the SAME
    PerBandStem (decimating 8/2/1 internally), hence ``UNIFORM_BAND_RATES``. r6's deltas are the
    shaft-batched data regime and the loss (no norm_pix, no margin gate, per-sensor masks, a
    predictor band embed), NOT the frontend. It read ``R6_BAND_RATES`` until 2026-07-23, which
    declared a native 4/16/32 Hz bake that was never made and silently mis-windowed SLOW/MID
    (memo project-r6-band-rates-cache-rate-bug-2026-07-23); ``assert_band_rates_match_cache``
    now fails loud on any such mismatch.
    The SAME band_rates must reach the dataset (per-band read offsets), the loader (32Hz
    reference n_frames) AND the model (stem + T derivation), so all three read this one fn.
    (native_fine_hga, early_fusion, no_fusion, r6) are mutually exclusive.
    """
    frontend = getattr(args, "frontend", "v3")
    if frontend == "v3fine":
        return True, False, False, False, NATIVE_FINE_BAND_RATES
    if frontend == "v3r5":
        return False, True, False, False, R5_BAND_RATES
    if frontend in ("v3r5nf", "v3r5nffast"):
        # v3r5nffast: the no-fusion path with the first stem conv at stride 2 (net 4× → 16 Hz
        # tokens). SAME 2×64 Hz caches ⇒ SAME band_rates; the decimate differs (see _nf_decimate).
        return False, False, True, False, R5NF_BAND_RATES
    if frontend == "v3r6":
        return False, False, False, True, UNIFORM_BAND_RATES
    return False, False, False, False, UNIFORM_BAND_RATES


def _nf_decimate(args: argparse.Namespace) -> int:
    """NoFusionStem net decimation from ``--frontend``: 4 for v3r5nffast (16 Hz tokens), else 2
    (v3r5nf, 32 Hz — byte-identical). Separate from ``_frontend_config`` so that fn's return
    arity + the r5nf/fast shared (no_fusion, band_rates) tuple stay unchanged."""
    return 4 if getattr(args, "frontend", "v3") == "v3r5nffast" else NOFUSION_DECIMATE


def build_v3_training(
    sessions: tp.Sequence[V3SessionSpec],
    args: argparse.Namespace,
) -> tuple[V14ConvergedV3Module, V3DataModule, pl.Trainer]:
    """Assemble (module, datamodule, trainer) from already-loaded sessions.

    Split out from ``main`` so F1 can drive the whole stack locally with synthetic
    sessions + a stub parcel_fn (no caches, no wandb, CPU).
    """
    mae = getattr(args, "objective", "jepa") == "mae"
    native_fine_hga, early_fusion, no_fusion, r6, band_rates = _frontend_config(args)
    nf_decimate = _nf_decimate(args)
    # Temporal mask block width(s) in TOKENS — the τ-anchored SSL difficulty knob (masking.py:77).
    # v3r5nf masks HGA and LFS on SEPARATE grids, so each carries its OWN width: --hga-block-w /
    # --lfs-block-w (default HGA 3, LFS 5 — LFS wider to force extrapolation past the 83 ms LFS
    # decorrelation horizon; a narrow LFS block was trivially in-fillable at EV ~0.6). --temporal-
    # block-w sets BOTH streams (back-compat) unless a per-stream flag wins. Every non-no-fusion arm
    # (r5-fused / r4) uses --temporal-block-w for its single grid; unset ⇒ V3MaskConfig() default
    # (byte-identical to the locked config).
    tbw = getattr(args, "temporal_block_w", None)
    if no_fusion:
        hga_bw = getattr(args, "hga_block_w", None)
        lfs_bw = getattr(args, "lfs_block_w", None)
        if hga_bw is None:
            hga_bw = tbw
        if lfs_bw is None:
            lfs_bw = tbw
        kw: dict[str, int] = {}
        if hga_bw is not None:
            kw["hga_block_w"] = hga_bw
        if lfs_bw is not None:
            kw["lfs_block_w"] = lfs_bw
        mask_cfg = V3MaskConfig(**kw)
    else:
        mask_cfg = V3MaskConfig() if tbw is None else V3MaskConfig(temporal_block_w=tbw)
    # --mask-space-frac: HARD-MASKING OFAT knob (applied last, over any frontend's mask_cfg). r6
    # masks 0.75 of tokens at the default 0.50 space ∪ 0.50 time; raising space to 0.80 with time
    # held at 0.50 lands ~0.90 total and aims the hard masking at the load-bearing spatial (cross-
    # subject) axis. None ⇒ unchanged (byte-identical to the locked config).
    space_frac = getattr(args, "mask_space_frac", None)
    if space_frac is not None:
        mask_cfg = replace(mask_cfg, space_frac=space_frac)
    # --band-block-w: RANDOM-vs-BLOCK MASKING ablation (A1). block_w_band is the leak-safe
    # contiguous width in a band's OWN tokens (masking.py:75, M14 margin 2). At width 1 the
    # cover degenerates to independent uniform per-token selection = RANDOM masking — the control
    # the leakage-free-block claim has never had. None ⇒ unchanged (byte-identical to the lock).
    band_bw = getattr(args, "band_block_w", None)
    if band_bw is not None:
        mask_cfg = replace(mask_cfg, block_w_band=band_bw)
    # --per-band-space / --space-block-w-bands: R19+R20 merged arm. The three TIME masks are
    # already independent per band; SPACE was the lone shared axis, and this makes BOTH axes follow
    # one rule. Widths come from the measured along-shaft r(d) per band (R20). Exact-count snapping
    # is per band, so masked token count is unchanged ⇒ matched-visible. None/False ⇒ locked config.
    if getattr(args, "per_band_space", False):
        mask_cfg = replace(mask_cfg, per_band_space=True)
    sbw = getattr(args, "space_block_w_bands", None)
    if sbw is not None:
        parts = tuple(int(x) for x in sbw.split(","))
        if len(parts) != 3:
            raise ValueError(f"--space-block-w-bands needs SLOW,MID,HGA (3 ints), got {sbw!r}")
        mask_cfg = replace(mask_cfg, block_w_space_bands=parts)
    model = V3ConvergedModel(
        n_parcels=_n_parcels(sessions), mask_cfg=mask_cfg,
        deep_sup=getattr(args, "deep_sup", True),
        parcel_embed=not getattr(args, "no_parcel_embed", False),
        space_rope=not getattr(args, "no_space_rope", False),
        mae=mae, native_fine_hga=native_fine_hga, early_fusion=early_fusion,
        no_fusion=no_fusion, r6=r6, nf_decimate=nf_decimate,
        mae_stream_weight=getattr(args, "mae_stream_weight", "equal"),
        mae_force_norm_pix=getattr(args, "mae_norm_pix", False),
        mae_hga_envelope=getattr(args, "mae_hga_envelope", False),
    )
    optim = build_v3_optim_cfg(
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio,
        adam_beta2=args.adam_beta2,
        stable_steps=getattr(args, "stable_steps", 0),
        decay_shape=getattr(args, "decay_shape", "cosine"),
    )
    module = V14ConvergedV3Module(
        model=model, optim_config=optim, seed=args.seed,
        monitor_every_n_steps=args.monitor_every_n_steps,
    )
    # batch_unit default: the cross-patient arms (r5 early_fusion, v3r5nf no_fusion, r6) ⇒
    # shaft-level batching; the older session-homogeneous path stays the default for
    # arm0/v3/v3fine. --batch-unit overrides. For r6 this IS delta 1 of 4 vs r4.
    batch_unit = args.batch_unit or (
        "shaft" if (early_fusion or no_fusion or r6) else "session"
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
        band_rates=band_rates,
        batch_unit=batch_unit,
        contact_budget=args.contact_budget,
        shaft_alpha=args.shaft_alpha,
    )
    trainer = _build_trainer(args)
    return module, dm, trainer


def _build_trainer(args: argparse.Namespace) -> pl.Trainer:
    """The locked v3 Trainer (E2 knobs). Single-device unless ``--devices>1``."""
    torch.set_float32_matmul_precision("high")  # audit lever, numerics-safe

    callbacks: list[pl.Callback] = [
        _EpochSeedCallback(),
        _StepTimeCallback(),
        SSLHealthMonitorV3(
            every_n_steps=args.monitor_every_n_steps,
            per_stream_enc12=getattr(args, "per_stream_enc12", False),
            grad_noise_every_n_steps=getattr(args, "grad_noise_every_n_steps", 0),
        ),
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
        # never builds a DDPStrategy). Default OFF: static_graph killed r3 at the first
        # backward (project-v3-r3-ddp-static-graph-crash-2026-07-14); opt back in only on
        # a plain-JEPA baseline with --ddp-static-graph.
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
    # r5 (--frontend v3r5) has 2 bands (v3hga, v3lfs); winsor tuple is (hga, lfs). LFS is
    # sub-HGA raw voltage (same content class as the old SLOW/MID) ⇒ the 15 cap; HGA keeps 20.
    p.add_argument("--session-z-winsor-lfs", dest="winsor_lfs", type=float, default=15.0)
    p.add_argument("--num-workers", type=int, default=4)
    # --- locked optimizer/schedule ---
    p.add_argument("--lr", type=float, default=6e-3)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.04)
    # R2 ablation (#19): drop the DKT parcel-identity embed entirely. It is added to the
    # residual at the encoder AND predictor input, so disabling == adding zero; geometry then
    # reaches the model only via L1 index/time RoPE. FROM SCRATCH only -- branching a ckpt that
    # already trained WITH the embed measures a lesion (#30), not whether the arch needs it.
    p.add_argument("--no-parcel-embed", dest="no_parcel_embed", action="store_true",
                   help="R2 ablation: build encoder+predictor without the parcel identity embed")
    # ABLATION A2 (2026-07-28): the control for the sensor-index-as-physics tokenization claim.
    # Zeroes L1RoPE's INDEX frequencies in every L1 block of both towers (time-RoPE untouched),
    # so within-shaft contact ORDER stops reaching the model while the parcel/DKT tag stays.
    # ⚠️ L1RoPE.idx_freq is persistent=False ⇒ NOT in the state_dict ⇒ a resume/cooldown that
    # omits this flag SILENTLY restores base_index=8. Any --resume-ckpt of an A2 run MUST re-pass it.
    p.add_argument("--no-space-rope", dest="no_space_rope", action="store_true",
                   help="A2 ablation: zero the contact-index RoPE axis (time RoPE unchanged)")
    p.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=5000)
    p.add_argument("--min-lr-ratio", dest="min_lr_ratio", type=float, default=1.0)
    # WSD cooldown (branch a stable-phase ckpt): flat until --stable-steps, then
    # decay to min_lr_ratio·peak over the tail by --decay-shape. Defaults reduce
    # to the locked warmup→cosine (stable_steps=0 ⇒ no plateau).
    p.add_argument("--stable-steps", dest="stable_steps", type=int, default=0,
                   help="WSD plateau length; LR held at peak until this step, then decays")
    p.add_argument("--decay-shape", dest="decay_shape", default="cosine",
                   choices=["cosine", "1-sqrt", "linear"],
                   help="LR decay shape after the stable plateau")
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
                        "identical.")
    p.add_argument("--frontend", dest="frontend",
                   choices=("v3", "v3fine", "v3r5", "v3r5nf", "v3r5nffast", "v3r6"),
                   default="v3",
                   help="temporal front-end: 'v3' (default, uniform-32Hz PerBandStem — the "
                        "arm0 recipe, byte-identical), 'v3fine' (FineHgaStem on native-rate "
                        "caches: SLOW 4Hz / MID 16Hz / HGA 128Hz conv-pooled 128→32Hz), "
                        "'v3r5' (EarlyFusionStem, Chang 2-stream: 2 caches v3hga|v3lfs both "
                        "@64 Hz, fused 5ch→256→stride-2→32Hz tokens; single-rate, no band embed, "
                        "accept-the-bleed in_loss=masked), 'v3r5nf' (NoFusionStem: the "
                        "EarlyFusionStem separated into 2 stems + 2 token streams + independent "
                        "masks/heads, MAE-only), 'v3r5nffast' (v3r5nf with the first stem conv "
                        "at stride 2 → net 4× → 16 Hz tokens; physical stride-2 RoPE + block_w=3, "
                        "SAME caches), or 'v3r6' (arm0's r4-MAE 3-band |STFT| frontend VERBATIM — "
                        "same 3 uniform 32 Hz caches, same PerBandStem, same gather — with FOUR "
                        "deltas: shaft-level batching, no norm_pix, no M14 margin gate "
                        "(in_loss=masked), and per-SENSOR band time masks + a predictor band "
                        "embed; MAE-only). 'v3r6' takes the SAME 3 caches as 'v3'/arm0; 'v3fine' "
                        "REQUIRES the fine-HGA cache; 'v3r5'/'v3r5nf'/'v3r5nffast' REQUIRE exactly "
                        "the 2 caches --band-cache-dir v3hga --band-cache-dir v3lfs (HGA first).")
    p.add_argument("--mae-stream-weight", dest="mae_stream_weight",
                   choices=("equal", "pooled"), default="equal",
                   help="v3r5nf MAE loss weighting (Ben 07-22): 'equal' (1:1, DEFAULT) averages "
                        "the two per-stream mean-MSEs so each stream is 50%% of the gradient, "
                        "invariant to HGA's bin granularity (lifts LFS 20%%→50%%); 'pooled' (4:1) "
                        "sums per-channel SE over both streams ⇒ HGA:LFS = 4:1 (matches r5-fused). "
                        "No-op outside the v3r5nf (no-fusion) MAE path.")
    p.add_argument("--mae-norm-pix", dest="mae_norm_pix", action="store_true",
                   help="ISOLATION OFAT (norm_pix attribution): force He-2021 per-token norm_pix "
                        "ON even under r6, decoupled from the r6 contract's `norm_pix=not r6`. "
                        "Default OFF ⇒ byte-identical to r6. Use ONLY to measure the norm_pix "
                        "delta single-handed vs r6 (r6 vs r4-MAE CS gap); NOT for production.")
    p.add_argument("--mae-hga-envelope", dest="mae_hga_envelope", action="store_true",
                   help="OFAT (2026-07-28): HGA's MAE target becomes the MEAN of its 7 |STFT| "
                        "bins; SLOW and MID keep per-bin targets. Measured rationale — the "
                        "y=y_env+y_res EV split on the r6 recon dump gives HGA env .154 vs shape "
                        ".043 (average it), MID .119/.113 (a wash), SLOW .109/.181 (per-bin earns "
                        "it). Input unchanged, no head resized (HGA reuses the pad machinery: "
                        "feat_count 1, column 0 only). r6-only, ⊥ --mae-norm-pix. Default OFF ⇒ "
                        "byte-identical to r6. NOTE: the wandb hga recon-r panels become "
                        "ENVELOPE r, not comparable to per-bin runs.")
    p.add_argument("--band-block-w", dest="band_block_w", type=int, default=None,
                   help="ABLATION A1 (2026-07-28): contiguous mask block width in a band's OWN "
                        "tokens (masking.py block_w_band, unset ⇒ 4 = the M14 leak margin 2). "
                        "Set 1 to make _cover_rank degenerate to independent uniform per-token "
                        "selection = RANDOM masking — the control our leakage-free-block claim has "
                        "never had. r4/r6 path only (sample_masks); r5 uses --temporal-block-w.")
    p.add_argument("--per-band-space", dest="per_band_space", action="store_true",
                   help="ABLATION R19 (2026-07-29): draw the SPATIAL mask INDEPENDENTLY per band "
                        "instead of one tube shared by SLOW/MID/HGA (masking.py per_band_space). "
                        "The three TIME masks are already per-band independent; space was the lone "
                        "exception. Exact-count snapping is per band ⇒ identical masked token count "
                        "⇒ MATCHED-VISIBLE. A contact can then be spatially masked in HGA while "
                        "visible in SLOW/MID, which is the cross-band-bridge test. r6 only.")
    p.add_argument("--space-block-w-bands", dest="space_block_w_bands", type=str, default=None,
                   help="R20: per-band spatial block width as SLOW,MID,HGA (e.g. '6,4,2'). "
                        "REQUIRES --per-band-space. Unset ⇒ block_w_space (4) for all three. "
                        "Derived from the measured along-shaft r(d) per band, replacing the "
                        "hard-coded 4 that — unlike block_w_band — has no leak-margin derivation. "
                        "Width changes ARRANGEMENT only, never count.")
    p.add_argument("--temporal-block-w", dest="temporal_block_w", type=int, default=None,
                   help="temporal mask block width in TOKENS (masking.py: the τ-anchored SSL "
                        "difficulty knob; block half-width must clear the ~83 ms LFS decorrelation "
                        "horizon). r5-fused/r4: the single grid's width, unset ⇒ 5 (156 ms @32 Hz). "
                        "v3r5nf: sets BOTH streams unless --hga-block-w/--lfs-block-w override.")
    p.add_argument("--hga-block-w", dest="hga_block_w", type=int, default=None,
                   help="v3r5nf ONLY: HGA-stream temporal block width in TOKENS. Unset ⇒ "
                        "--temporal-block-w if given, else V3MaskConfig default (3). The near-white "
                        "HGA stream stays hard at a narrow block.")
    p.add_argument("--lfs-block-w", dest="lfs_block_w", type=int, default=None,
                   help="v3r5nf ONLY: LFS-stream temporal block width in TOKENS. Unset ⇒ "
                        "--temporal-block-w if given, else V3MaskConfig default (5) — wider than HGA "
                        "so the slow LFS stream can't trivially in-fill the masked run.")
    p.add_argument("--mask-space-frac", dest="mask_space_frac", type=float, default=None,
                   help="HARD-MASKING OFAT: per-shaft SPATIAL mask fraction (masking.py space_frac; "
                        "each shaft masks round(frac*n_s) of its OWN contacts, keep_alive reserves "
                        "one visible). Default None ⇒ 0.50 (byte-identical to r6, 0.75 total "
                        "space-union-time). 0.80 with time held at 0.50 ≈ 0.90 total, aimed at the "
                        "load-bearing spatial axis (image-MAE-hard space, ASR-standard time).")
    # --- batching unit (shaft-level cross-patient vs session-homogeneous) ---
    p.add_argument("--batch-unit", dest="batch_unit", choices=("session", "shaft"),
                   default=None,
                   help="batch granularity. 'session' = v3 session-homogeneous (one patient/"
                        "step). 'shaft' = cross-patient shaft-level: each step packs shafts from "
                        "the GLOBAL pool (K distinct patients/step, constant compile footprint at "
                        "scale). Default: 'shaft' for --frontend v3r5, else 'session'. Requires "
                        "--contact-budget when shaft.")
    p.add_argument("--contact-budget", dest="contact_budget", type=int, default=None,
                   help="shaft mode: contacts per pack. Pins grid.total to ONE compiled shape "
                        "(overfill-and-trim closes it exactly, no pad). Set to ~match session "
                        "tokens/step (batch_size x mean-session-N).")
    p.add_argument("--shaft-alpha", dest="shaft_alpha", type=float, default=0.5,
                   help="shaft mode: temperature for P(subject) ~ n_shafts^alpha (0 = subject-"
                        "uniform, 1 = shaft-uniform, 0.5 = sqrt-tempered default).")
    # --- trainer/precision (E2) ---
    p.add_argument("--ssl-max-steps", dest="ssl_max_steps", type=int, required=True)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--accelerator", default="gpu")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--monitor-every-n-steps", dest="monitor_every_n_steps",
                   type=int, default=1)
    p.add_argument("--grad-noise-every-n-steps", dest="grad_noise_every_n_steps",
                   type=int, default=0,
                   help="cadence for the grad_noise_scale (McCandlish B_simple critical-batch) "
                        "monitor, on its OWN schedule (~28 ms, AdamW-moment reductions, no extra "
                        "fwd/bwd). 0 = OFF (default). Set e.g. 25 to size batch vs critical batch: "
                        "train_mon_grad_noise_scale is B_simple in SAMPLES — compare to tokens/step "
                        "(≫ ⇒ below critical batch, raising batch buys ~linear step-efficiency).")
    p.add_argument("--per-stream-enc12", dest="per_stream_enc12", action="store_true",
                   help="v3r5nf diagnostic: split the enc12 tap by stream (HGA/LFS) and log each "
                        "stream's OWN rankme + feat_std. OFF by default (Ben 2026-07-22) — its 2 "
                        "extra SVDs/monitor-step are the fast-arm 'comb' (~0.15 s/step). Opt in "
                        "only for a diagnostic run.")
    p.add_argument("--log-every-n-steps", dest="log_every_n_steps", type=int, default=1,
                   help="wandb flush cadence; 1 = per-step resolution (Ben 2026-07-11) "
                        "so update_cos/grad-spike/feat_std are not window-averaged away")
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
                        "config. The resumed trainable-param set must match the ckpt's. "
                        "Default None = fresh run from step 0.")
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
    # v3r5 (fused) AND v3r5nf (no-fusion) both read exactly the 2 caches (v3hga, v3lfs).
    is_two_stream = getattr(args, "frontend", "v3") in ("v3r5", "v3r5nf", "v3r5nffast")
    n_want = 2 if is_two_stream else 3
    if len(args.band_cache_dirs) != n_want:
        order = "v3hga, v3lfs" if is_two_stream else "slow, mid, hga"
        raise ValueError(
            f"--frontend {args.frontend} needs {n_want} --band-cache-dir ({order}), "
            f"got {len(args.band_cache_dirs)}"
        )
    pl.seed_everything(args.seed, workers=True)

    _, _, _, _, band_rates = _frontend_config(args)
    # winsor is per-band in cache order: two-stream (r5/r5nf) = (hga, lfs); arm0/v3fine/v3r6 =
    # (slow, mid, hga).
    winsor = (
        (args.winsor_hga, args.winsor_lfs)
        if is_two_stream
        else (args.winsor_slow, args.winsor_mid, args.winsor_hga)
    )
    sessions = load_v3_sessions(
        sessions=_parse_sessions(args.sessions),
        band_cache_dirs=args.band_cache_dirs,
        span_dir=args.span_dir,
        parcel_fn=make_bt_parcel_fn(args.bt_root),
        lof_report_path=args.lof_report_path,
        winsor=winsor,
        band_rates=band_rates,
    )
    module, dm, trainer = build_v3_training(sessions, args)
    if args.compile:
        # dynamic=True (SHAPE-GENERIC graphs), NOT dynamic=False. Measured on GH200 @15k
        # contacts (2026-07-22): dynamic=False STORMS on the shaft-pack workload — S (shaft
        # count) and max_c vary near-continuously pack-to-pack (far beyond the 13 session
        # shapes the old cache-sizing theory assumed), so every step is a fresh shape ⇒
        # perpetual recompile, ~100 s/step, warmup never ends (the "size the cache bigger"
        # fix does NOT work here; the working set is unbounded). dynamic=None broke
        # build_pack_plan (symbolic-value guards, packing.py:84 recompiled every step).
        # dynamic=True traces 39 shape-generic graphs ONCE during warmup and reuses them
        # across novel shapes (+0 new graphs on 5 unseen shapes) ⇒ 1.65× (472→278 ms/step),
        # no storm. The njt attention (dynamo.disable) and mask sampling (.item() graph-
        # break, masking.py:115) stay eager, so RNG + attention are byte-identical to eager.
        # CAVEAT — NOT bitwise-neutral: compiled-vs-eager loss diverges ~1% (18-23× the
        # 4.4e-4 eager run-to-run nondeterminism floor), a systematic bf16 fusion-reorder
        # shift. The speedup is INSEPARABLE from it: encoder-block-only compile is floor-
        # neutral (1×) but 0× faster; the 1.65× lives in the head/loss/glue fusion that
        # causes the shift. So --compile is an opt-in throughput/precision tradeoff — keep
        # it OFF for runs that must be numerically comparable to an eager lineage (e.g. a
        # board-decisive arm); flip it ON for fully-compiled lineages or when wall-clock
        # gates. Full scope: memory project-r5-eager-compile-speedup-scope-2026-07-22.
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
        module.model = torch.compile(module.model, dynamic=True)  # type: ignore[assignment]
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
