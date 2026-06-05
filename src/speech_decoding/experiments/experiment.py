from __future__ import annotations

import os
import typing as tp
from pathlib import Path

import exca
import lightning.pytorch as pl
import pydantic
import torch

# 2026-05-30 speedup audit (Tier-1): let the CUDA caching allocator grow
# segments instead of fragmenting into fixed blocks. Read once when the
# allocator first initializes, so it must be set before any CUDA op —
# module import in the submitit worker is the earliest reliable point
# (the login-node ``scripts/dcc/dispatch`` env never reaches the GPU job).
# ``setdefault`` so an explicit override on the job still wins. Benign
# allocator hygiene; no effect on CPU/test runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger
from lightning.pytorch.loggers.logger import DummyLogger
from torch.utils.data import DataLoader

from neuraltrain import BaseLoss, BaseMetric, BaseModelConfig, LightningOptimizer
from neuraltrain.utils import BaseExperiment, CsvLoggerConfig, WandbLoggerConfig

from speech_decoding.experiments.collapse_guard import CollapseGuardCallback
from speech_decoding.experiments.data import Data
from speech_decoding.experiments.experiment_logging import ExperimentLogger
from speech_decoding.experiments.module import BrainModule


def _is_global_zero() -> bool:
    """True on the rank-0 worker (or any single-process run).

    Under srun-DDP (``--cluster slurm`` + ``tasks_per_node>1``) the exca task
    body ``run()`` executes once per rank, BEFORE any Lightning Trainer exists,
    so rank must be read from the launcher environment. ``srun`` exports
    ``SLURM_PROCID`` for every task; Lightning's own subprocess launcher exports
    ``RANK``. A run with neither (exca local, or a single srun task) is rank 0.
    Used to gate file-writing side effects (run-log sidecars) that share a fixed
    path across ranks and would otherwise race.
    """
    for var in ("SLURM_PROCID", "RANK"):
        v = os.environ.get(var)
        if v is not None:
            return v == "0"
    return True


class Experiment(BaseExperiment):
    """NeuralTrain/Exca experiment contract for speech decoding runs.

    Inherits from `neuraltrain.utils.BaseExperiment` so `neuraltrain.utils.run_grid`
    can dispatch grids of this experiment as one Slurm array per sweep via
    `infra.job_array()`.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    data: Data
    brain_model_config: BaseModelConfig
    loss: BaseLoss
    optim: LightningOptimizer
    metrics: list[BaseMetric]

    seed: int = 33
    n_epochs: int = 1
    # Optional step budget for SSL/distill phases. When set it caps training at
    # ``max_steps`` optimizer steps and disables the epoch cap (``max_epochs=-1``)
    # so ``estimated_stepping_batches`` equals ``max_steps`` exactly. ``None``
    # (default) keeps the epoch budget. Real SSL runs over multi-hundred-hour
    # corpora are budgeted in steps, not epochs — "1 epoch" over the joint corpus
    # is enormous and uneven across phases. NOTE: no LR scheduler is currently
    # configured in the v14 dispatch optim (constant LR); if a warmup/cosine
    # schedule is added later, ``estimated_stepping_batches`` already feeds its
    # ``total_steps`` via ``configure_optimizers`` so the horizon tracks this cap.
    max_steps: int | None = None
    # Lightning ``gradient_clip_val``. ``None`` (default) → no clipping
    # (back-compat for the tiny test experiments). The v14 §7 recipe locks
    # ``grad_clip = 1.0`` as a non-swept universal; the v14 dispatch sets it.
    gradient_clip_val: float | None = None
    # Lightning ``accumulate_grad_batches``. 1 (default) → no accumulation, bit
    # -for-bit the prior behavior. >1 sums grads over N micro-batches before each
    # optimizer step → effective batch = batch_size * N, at N× the wall-clock per
    # step (it does NOT reduce the per-step horizon — ``max_steps`` still counts
    # optimizer steps, so ``estimated_stepping_batches`` already tracks the right
    # cosine horizon). The effective-batch lever for the 32 GB card when a bigger
    # GPU is unavailable; a larger physical batch on a 48 GB+ card is preferred.
    accumulate_grad_batches: int = 1
    x_name: str | tuple[str, ...] = "input"
    y_name: str = "target"
    target_field: str = "code"
    n_outputs: int | None = None
    test_only: bool = False
    fast_dev_run: bool | int = False
    accelerator: str = "auto"
    devices: int | str = "auto"
    # Lightning ``strategy``. ``None`` (default) → Lightning auto-selects
    # (single-device on 1 GPU; plain DDP under multi-rank srun). The staged
    # B36 SSL phases leave whole submodules out of the active loss by design
    # (P1 trains the front-end only — the pool / inter-parcel encoder /
    # predictor carry ``requires_grad=True`` but get no gradient; the
    # predictor is P2-only). Plain DDP's reducer rejects that on the 2nd
    # iteration ("parameters that were not used in producing the loss"), so a
    # multi-rank run MUST pass ``"ddp_find_unused_parameters_true"``. Set only
    # under DDP (tasks_per_node>1) so the single-GPU path stays non-DDP. The
    # strategy itself is numerically inert (AdamW skips grad=None params, wd=0);
    # the only multi-rank delta is the standard DDP mean-of-means reweighting of
    # the per-element masked loss (see dispatch_v14._resolve_ddp_strategy call
    # site). The string variant keeps the config pydantic-serialisable.
    ddp_strategy: str | None = None
    # Lightning trainer precision. Default ``None`` → fp32 (back-compat).
    # Pass ``"bf16-mixed"`` for activation-half precision on Ada/Hopper
    # GPUs; pass ``"16-mixed"`` for older fp16 mixed (requires GradScaler
    # under the hood). Added 2026-05-29 after the B31 Lite Phase-4 baseline
    # OOM'd repeatedly on RTX 5000 Ada at fp32 — see
    # [[feedback_dcc_partition_default_coganlab_gpu_2026_05_29]].
    precision: str | None = None
    log_every_n_steps: int = 10
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    early_stopping_patience: int | None = None
    checkpoint_monitor: str = "val_loss"
    collapse_guard: bool = True
    # Optimizer-step count of LR warmup, handed to the collapse guard so its
    # soft criteria (RankMe/coverage/loss-blowup) are not believed while the LR
    # is still ramping (representations legitimately look low-rank early). 0 →
    # the guard's check-count grace only. Set by the dispatch from --warmup-steps.
    guard_warmup_min_step: int = 0
    # Validation cadence as a step count. On a ``max_steps``-budgeted SSL
    # phase that ends mid-epoch (1500 steps < one BT-lite epoch), epoch-
    # boundary validation never runs, so the collapse-guard soft panel
    # would never evaluate. Setting this forces validation every N steps.
    # ``None`` → Lightning's default (epoch-boundary only).
    val_check_interval: int | float | None = None
    csv_config: CsvLoggerConfig | None = None
    wandb_config: WandbLoggerConfig | None = None
    # B36 WS-E (E3/E4): cross-phase checkpoint handoff. ``pretrained_ckpt``
    # warm-starts the brain module from a prior phase's snapshot before fit;
    # ``snapshot_ckpt_to`` writes this run's transferable state after test so
    # the next phase can load it. Both default off — inert for single-phase
    # runs. The multi-phase driver (``experiments/v14_phase_pipeline.py``)
    # wires one phase's ``snapshot_ckpt_to`` to the next's ``pretrained_ckpt``.
    pretrained_ckpt: str | None = None
    snapshot_ckpt_to: str | None = None
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    def _input_tensor_name(self) -> str:
        if isinstance(self.x_name, str):
            return self.x_name
        return self.x_name[0]

    def _infer_n_outputs(self, train_loader: DataLoader) -> int:
        if self.n_outputs is not None:
            return self.n_outputs
        triggers = train_loader.dataset.triggers
        if self.target_field not in triggers.columns:
            raise ValueError(f"Missing trigger target column: {self.target_field!r}")
        return int(triggers[self.target_field].nunique())

    def _build_brain_module(self, train_loader: DataLoader) -> BrainModule:
        batch = next(iter(train_loader))
        x = batch.data[self._input_tensor_name()]
        model = self.brain_model_config.build(
            n_in_channels=int(x.shape[1]),
            n_outputs=self._infer_n_outputs(train_loader),
        )
        metrics = {metric.log_name: metric.build() for metric in self.metrics}
        return BrainModule(
            model=model,
            loss=self.loss.build(),
            optim_config=self.optim,
            metrics=metrics,
            x_name=self.x_name,
            y_name=self.y_name,
        )

    def _artifact_root(self) -> Path | None:
        """Per-run root for Lightning side-artifacts (checkpoints, CSV logs).

        Routes them under this config's unique exca ``uid_folder`` so chained
        phases (P1→P2→P3→P4, each a distinct exca UID via ``clone_obj``) and
        grid cells never collide on a shared ``checkpoints/`` or
        ``lightning_logs/version_N`` — each phase/cell owns its own subtree and
        keeps a stable ``best.ckpt``/``last.ckpt`` rather than Lightning's
        ``-v1`` auto-increment. Falls back to ``infra.folder`` (the prior flat
        layout) when no ``uid_folder`` is available, matching
        :meth:`_artifact_dir_and_uid`'s own fallback so that edge is unchanged.

        Note the cross-phase warm-start handoff is unaffected: it uses explicit
        ``snapshot_ckpt_to`` / ``pretrained_ckpt`` paths, not this auto dir.
        """
        if self.infra.folder is None:
            return None
        try:
            uid_folder = self.infra.uid_folder(create=True)
        except RuntimeError:
            uid_folder = None
        return Path(uid_folder) if uid_folder is not None else Path(self.infra.folder)

    def _logger(self) -> Logger | list[Logger]:
        loggers: list[Logger] = []
        if self.csv_config is not None:
            loggers.append(self.csv_config.build())
        if self.wandb_config is not None:
            loggers.append(self.wandb_config.build())
        if loggers:
            return loggers
        root = self._artifact_root()
        if root is not None:
            return CSVLogger(save_dir=str(root / "lightning"))
        return DummyLogger()

    def _callbacks(self) -> list[pl.Callback]:
        callbacks: list[pl.Callback] = [LearningRateMonitor(logging_interval="epoch")]
        root = self._artifact_root()
        if root is not None:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(root / "checkpoints"),
                    filename="best",
                    monitor=self.checkpoint_monitor,
                    save_last=True,
                    save_top_k=1,
                )
            )
        if self.early_stopping_patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor=self.checkpoint_monitor,
                    mode="min",
                    patience=self.early_stopping_patience,
                )
            )
        # #54: active collapse/divergence kill switch. Disarmed under
        # fast_dev_run (a 1-batch smoke would never reach the sustain
        # window and the soft alarms are noisy on a single batch); the
        # non-finite-loss catch is irrelevant there too. Writes its
        # diagnosis JSON next to the checkpoints (``root`` may be None on
        # an infra-less run — the guard still aborts, just skips the file).
        if self.collapse_guard and not self.fast_dev_run:
            callbacks.append(
                CollapseGuardCallback(
                    artifact_root=root,
                    warmup_min_step=self.guard_warmup_min_step,
                )
            )
        return callbacks

    def _trainer(self) -> pl.Trainer:
        kwargs: dict[str, tp.Any] = {
            "max_epochs": self.n_epochs,
            "accelerator": self.accelerator,
            "devices": self.devices,
            "logger": self._logger(),
            "callbacks": self._callbacks(),
            "log_every_n_steps": self.log_every_n_steps,
            "enable_checkpointing": self.infra.folder is not None,
            "fast_dev_run": self.fast_dev_run,
        }
        if self.ddp_strategy is not None:
            kwargs["strategy"] = self.ddp_strategy
        if self.max_steps is not None:
            # Step budget overrides the epoch cap; max_epochs=-1 lets the run go
            # the full max_steps (estimated_stepping_batches == max_steps).
            kwargs["max_steps"] = self.max_steps
            kwargs["max_epochs"] = -1
        if self.val_check_interval is not None:
            # A step-count cadence so the collapse-guard soft panel runs on
            # a max_steps phase that never reaches an epoch boundary.
            #
            # UNIT FIX (#66, 2026-06-05): Lightning counts an INT
            # ``val_check_interval`` in training MICRO-batches, not optimizer
            # steps. Our value is specified in OPTIMIZER steps (the
            # ``--ssl-val-check-interval`` flag + the collapse-guard cadence
            # intent), so under grad-accumulation convert: 1 optimizer step ==
            # ``accumulate_grad_batches`` micro-batches. Without this, accum=16
            # made validation fire 16× too often (~9 opt-steps apart, ~160
            # checks/phase) and consumed ~80% of wall-clock. A FLOAT value
            # (fraction-of-epoch) is Lightning-native and passes through.
            vci = self.val_check_interval
            if isinstance(vci, int):
                if self.accumulate_grad_batches > 1:
                    vci = vci * self.accumulate_grad_batches
                # An int cadence is a GLOBAL step count, not a within-epoch
                # offset. Lightning otherwise enforces ``val_check_interval <=
                # batches-per-epoch`` (because ``check_val_every_n_epoch``
                # defaults to 1) and raises before step 0 when the converted
                # cadence exceeds one epoch — which it does on the locked gate
                # (2400 micro-batches > ~1752/epoch at bs=2). Disabling the
                # epoch gate makes validation fire every ``vci`` batches across
                # epoch boundaries, the intended step-cadence semantics.
                kwargs["check_val_every_n_epoch"] = None
            kwargs["val_check_interval"] = vci
        if self.limit_train_batches is not None:
            kwargs["limit_train_batches"] = self.limit_train_batches
        if self.limit_val_batches is not None:
            kwargs["limit_val_batches"] = self.limit_val_batches
        if self.limit_test_batches is not None:
            kwargs["limit_test_batches"] = self.limit_test_batches
        if self.precision is not None:
            kwargs["precision"] = self.precision
        if self.gradient_clip_val is not None:
            kwargs["gradient_clip_val"] = self.gradient_clip_val
        if self.accumulate_grad_batches != 1:
            kwargs["accumulate_grad_batches"] = self.accumulate_grad_batches
        return pl.Trainer(**kwargs)

    def _artifact_dir_and_uid(self) -> tuple[Path | None, str]:
        if self.infra.folder is None:
            return None, ""
        try:
            uid_folder = self.infra.uid_folder(create=True)
        except RuntimeError:
            uid_folder = None
        if uid_folder is None:
            return Path(self.infra.folder) / "records", ""
        return Path(uid_folder), Path(uid_folder).name

    def _load_pretrained(self, brain_module: pl.LightningModule, ckpt_path: str) -> None:
        """E4: warm-start ``brain_module`` from a prior phase's snapshot.

        Delegates to the module's ``load_transferable_state`` protocol (the
        v14 BrainModules implement it: encoder strict-load + teacher re-sync,
        phase-local keys dropped). A module without the protocol is a hard
        error — the handoff contract is explicit, never a silent no-op.
        """
        loader = getattr(brain_module, "load_transferable_state", None)
        if loader is None:
            raise TypeError(
                f"{type(brain_module).__name__} has no load_transferable_state; "
                "pretrained_ckpt handoff needs the transferable-state protocol."
            )
        # Gate-D fix: `_snapshot` runs INSIDE the exca-cached `run()` body, so a
        # cache-HIT phase returns its stored result without re-writing its
        # snapshot. If the prior phase's `.ckpt` was purged (e.g. a `--work-dir`
        # on /work's 75-day-purge tier while the exca metadata cache persists)
        # the bare torch.load below would raise an opaque FileNotFoundError on a
        # re-run. Fail with an actionable message instead.
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(
                f"pretrained_ckpt {ckpt_path!r} is missing. The prior phase's "
                "snapshot is written inside its exca-cached run(), so a cache-hit "
                "on a purged/cleared work_dir leaves no .ckpt to warm-start from. "
                "Re-run the prior phase with its exca cache cleared, or keep "
                "--work-dir on the same persistence tier as EXCA_CACHE_FOLDER."
            )
        # weights_only=True: the snapshot is a pure dict-of-tensor-dicts
        # (``transferable_state``); load it under the safe unpickler.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        loader(state, strict=True)

    def _snapshot(self, brain_module: pl.LightningModule, ckpt_path: str) -> None:
        """E3: write the trained module's transferable state for the next phase."""
        snap = getattr(brain_module, "transferable_state", None)
        if snap is None:
            raise TypeError(
                f"{type(brain_module).__name__} has no transferable_state; "
                "snapshot_ckpt_to handoff needs the transferable-state protocol."
            )
        path = Path(ckpt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(snap(), str(path))

    def _train_and_test(self) -> dict[str, float | None]:
        pl.seed_everything(self.seed, workers=True)
        # CR01: pl.seed_everything may miss some CUDA RNG init paths in
        # DDP-forked workers; explicit manual_seed_all closes that gap.
        torch.cuda.manual_seed_all(self.seed)
        # CR02: per-worker numpy + random + torch RNG seeded as seed+worker_id.
        loaders = self.data.build(worker_seed=self.seed)
        brain_module = self._build_brain_module(loaders["train"])
        # B36 WS-E (E4): warm-start the encoder (+PMA from P3) from the prior
        # phase BEFORE fit, so the optimizer/scheduler see the loaded weights.
        if self.pretrained_ckpt is not None:
            self._load_pretrained(brain_module, self.pretrained_ckpt)
        trainer = self._trainer()
        if not self.test_only:
            trainer.fit(brain_module, loaders["train"], loaders["val"])
        results = trainer.test(
            brain_module, loaders["test"], ckpt_path=self._test_ckpt_path(trainer),
        )
        # B36 WS-E (E3): snapshot this phase's transferable state for the next.
        # Under DDP the weights are kept in sync across ranks, so only global-
        # zero writes the handoff ckpt — 4 ranks torch.save-ing the same path
        # would corrupt it (and break the next phase's strict load). torch.save
        # has no collective, so a rank-gated write needs no barrier; the next
        # phase is a separate submitit job that starts only after this job's
        # ranks have all exited.
        if self.snapshot_ckpt_to is not None and trainer.is_global_zero:
            self._snapshot(brain_module, self.snapshot_ckpt_to)
        return dict(results[0]) if results else {}

    def _test_ckpt_path(self, trainer: pl.Trainer) -> str | None:
        """Which weights the test pass evaluates: val-best when early-stopping
        ran, else the in-memory (last) weights.

        EarlyStopping does NOT restore best weights, so after it fires the module
        sits ``patience`` epochs past the val optimum — reporting the P4 submission
        metric on those over-fit weights would undercut the early-stop. So the one
        phase that early-stops (the supervised P4 probe) tests its ``best.ckpt``.
        The SSL/distill phases never early-stop: they keep last-epoch weights, and
        the cross-phase ``_snapshot`` handoff snapshots those same last-epoch
        weights (unchanged). Falls back to in-memory if no best checkpoint was
        written (e.g. a step budget that stops before the first validation)."""
        if self.early_stopping_patience is None or self.test_only:
            return None
        cb = trainer.checkpoint_callback
        best = getattr(cb, "best_model_path", "") if cb is not None else ""
        return "best" if best else None

    @infra.apply
    def run(self) -> dict[str, float | None]:
        artifact_dir, exca_uid = self._artifact_dir_and_uid()
        # Under srun-DDP this body runs once per rank. The ExperimentLogger
        # writes fixed-name sidecars into the shared artifact_dir, so only
        # global-zero may write them — concurrent rank writers would race on the
        # same file. Non-zero ranks still run the full train/test so they join
        # every DDP collective; they just skip the run-log.
        if artifact_dir is None or not _is_global_zero():
            return self._train_and_test()
        with ExperimentLogger(
            artifact_dir=artifact_dir,
            stage="neuraltrain",
            run_kind="train",
            seed=str(self.seed),
            exca_uid=exca_uid,
            config_json=self.model_dump_json(exclude={"infra"}),
        ) as run_log:
            output = self._train_and_test()
            primary_name = next(iter(output), "")
            primary_value = output.get(primary_name) if primary_name else None
            run_log.set_metrics(
                output,
                primary_metric_name=primary_name,
                primary_metric_value=(
                    float(primary_value)
                    if isinstance(primary_value, int | float)
                    else None
                ),
            )
            return output
