from __future__ import annotations

import json

import pandas as pd

import neuralset as ns

from speech_decoding.experiments import Data, Experiment


class TinySplitStudy(ns.Step):
    def _run(self):
        rows = []
        splits = ["train"] * 4 + ["val"] * 2 + ["test"] * 2
        for idx, split in enumerate(splits):
            rows.append(
                {
                    "type": "Stimulus",
                    "start": float(idx * 2),
                    "duration": 0.2,
                    "timeline": "run0",
                    "code": idx % 2,
                    "split": split,
                }
            )
        return ns.events.standardize_events(pd.DataFrame(rows))


def tiny_data(batch_size: int = 2) -> Data:
    return Data(
        study=TinySplitStudy(),
        segmenter={
            "extractors": {
                "input": {
                    "name": "Pulse",
                    "event_types": "Stimulus",
                    "frequency": 16.0,
                    "aggregation": "single",
                },
                "target": {
                    "name": "EventField",
                    "event_types": "Stimulus",
                    "event_field": "code",
                },
            },
            "trigger_query": "type == 'Stimulus'",
            "start": 0.0,
            "duration": 1.0,
        },
        batch_size=batch_size,
    )


def test_data_builds_split_loaders() -> None:
    loaders = tiny_data().build()

    assert set(loaders) == {"train", "val", "test"}
    batch = next(iter(loaders["train"]))
    assert batch.data["input"].shape == (2, 1, 16)
    assert batch.data["target"].shape == (2, 1)


def test_experiment_dry_run(tmp_path) -> None:
    run_root = tmp_path / "runs"
    xp = Experiment(
        data=tiny_data(),
        brain_model_config={
            "name": "SimpleConvTimeAgg",
            "hidden": 4,
            "depth": 1,
            "kernel_size": 3,
            "merger_config": None,
        },
        loss={"name": "CrossEntropyLoss"},
        optim={"optimizer": {"name": "Adam", "lr": 1e-3}},
        metrics=[
            {
                "name": "Accuracy",
                "log_name": "acc",
                "kwargs": {"task": "multiclass", "num_classes": 2},
            }
        ],
        n_epochs=1,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        infra={"folder": str(run_root), "cluster": None},
    )

    result = xp.run()

    assert isinstance(result, dict)
    records = list(run_root.rglob("experiment_record.json"))
    assert len(records) == 1


def _tiny_mlp_xp(run_root, n_epochs: int) -> Experiment:
    return Experiment(
        data=tiny_data(),
        brain_model_config={"name": "TinyMLP", "hidden": 8},
        loss={"name": "CrossEntropyLoss"},
        optim={"optimizer": {"name": "Adam", "lr": 1e-2}},
        metrics=[
            {
                "name": "Accuracy",
                "log_name": "acc",
                "kwargs": {"task": "multiclass", "num_classes": 2},
            }
        ],
        n_epochs=n_epochs,
        accelerator="cpu",
        devices=1,
        log_every_n_steps=1,
        infra={"folder": str(run_root), "cluster": None},
    )


def test_substrate_smoke_full_fit_caches(tmp_path) -> None:
    """End-to-end substrate integration: TinyMLP fits via Lightning, exca caches.

    Asserts (a) Experiment.run returns real test metrics, (b) ExperimentLogger
    writes one record per uid, (c) the second `.run()` call hits the exca cache
    instead of re-fitting (no second record dir, identical return value).
    """
    import speech_decoding.models  # noqa: F401  # registers TinyMLP discriminator
    run_root = tmp_path / "runs"

    first = _tiny_mlp_xp(run_root, n_epochs=2).run()
    assert isinstance(first, dict)
    assert "test_loss" in first

    records = sorted(run_root.rglob("experiment_record.json"))
    assert len(records) == 1, [str(r) for r in records]

    record_payload = json.loads(records[0].read_text())
    assert record_payload["status"] == "succeeded"
    assert record_payload["primary_metric_name"] == "test_loss"
    assert isinstance(record_payload["primary_metric_value"], float)
    assert record_payload["exca_uid"]

    uid_dirs_before = sorted(p for p in run_root.rglob("*") if p.is_dir())

    second = _tiny_mlp_xp(run_root, n_epochs=2).run()
    assert second == first  # exca cache hit
    uid_dirs_after = sorted(p for p in run_root.rglob("*") if p.is_dir())
    assert uid_dirs_before == uid_dirs_after


def test_checkpoints_namespaced_per_run(tmp_path) -> None:
    """Two distinct-config runs into one folder must not collide on checkpoints.

    Pre-fix both wrote to ``<folder>/checkpoints/`` with filename ``best`` +
    ``save_last``; Lightning auto-incremented the second to ``last-v1.ckpt`` /
    ``best-v1.ckpt``, so a P1→P2→P3→P4 chain (each phase a distinct exca UID
    sharing one folder) silently shuffled which phase owned which file. The
    per-run ``uid_folder`` routing gives each its own ``checkpoints/last.ckpt``.
    """
    import speech_decoding.models  # noqa: F401  # registers TinyMLP discriminator
    run_root = tmp_path / "runs"

    _tiny_mlp_xp(run_root, n_epochs=1).run()  # seed=33 (default) -> UID A
    # clone_obj rebinds infra so the diff lands in the cached run() body and the
    # second config gets a distinct UID (model_copy would not — see
    # v14_phase_pipeline.configure_phase_handoff).
    _tiny_mlp_xp(run_root, n_epochs=1).infra.clone_obj({"seed": 7}).run()  # UID B

    # No flat shared checkpoints dir, and no Lightning -v auto-increment.
    assert not (run_root / "checkpoints").exists()
    last_ckpts = sorted(run_root.rglob("last.ckpt"))
    assert len(last_ckpts) == 2, [str(p) for p in last_ckpts]
    assert not list(run_root.rglob("last-v*.ckpt"))
    assert not list(run_root.rglob("best-v*.ckpt"))


def _trainer_only_xp(*, n_epochs: int, max_steps: int | None) -> Experiment:
    import speech_decoding.models  # noqa: F401  # registers TinyMLP discriminator

    return Experiment(
        data=tiny_data(),
        brain_model_config={"name": "TinyMLP", "hidden": 8},
        loss={"name": "CrossEntropyLoss"},
        optim={"optimizer": {"name": "Adam", "lr": 1e-2}},
        metrics=[
            {
                "name": "Accuracy",
                "log_name": "acc",
                "kwargs": {"task": "multiclass", "num_classes": 2},
            }
        ],
        n_epochs=n_epochs,
        max_steps=max_steps,
        accelerator="cpu",
        devices=1,
        infra={"folder": None, "cluster": None},
    )


def test_max_steps_overrides_epoch_cap() -> None:
    """#32: a step budget caps optimizer steps and disables the epoch cap, so the
    cosine LR horizon (estimated_stepping_batches) == max_steps."""
    trainer = _trainer_only_xp(n_epochs=100, max_steps=7)._trainer()
    assert trainer.max_steps == 7
    assert trainer.max_epochs == -1


def test_no_max_steps_keeps_epoch_cap() -> None:
    """Default (max_steps=None) keeps the epoch budget; Lightning's max_steps=-1."""
    trainer = _trainer_only_xp(n_epochs=3, max_steps=None)._trainer()
    assert trainer.max_epochs == 3
    assert trainer.max_steps == -1


def test_gradient_clip_val_passed_to_trainer() -> None:
    """#37: gradient_clip_val reaches the Lightning trainer when set; default
    None leaves clipping off (back-compat for the tiny test experiments)."""
    base = _trainer_only_xp(n_epochs=1, max_steps=None)
    assert base._trainer().gradient_clip_val is None  # default: no clipping

    clipped = base.model_copy(update={"gradient_clip_val": 1.0})
    assert clipped._trainer().gradient_clip_val == 1.0


def test_accumulate_grad_batches_passed_to_trainer() -> None:
    """#42: accumulate_grad_batches reaches the trainer when >1; the default 1
    is omitted so the trainer keeps its own default (bit-for-bit prior run)."""
    base = _trainer_only_xp(n_epochs=1, max_steps=None)
    assert base.accumulate_grad_batches == 1  # default: no accumulation
    assert base._trainer().accumulate_grad_batches == 1  # Lightning default

    accum = base.model_copy(update={"accumulate_grad_batches": 8})
    assert accum._trainer().accumulate_grad_batches == 8


def test_profiler_env_var_attaches_lightning_profiler(monkeypatch) -> None:
    """Speedup MISS-1: nothing in the stack measures where the step goes. The
    V14_PROFILER env var (operational, NOT a config field → no exca uid change,
    no cache fork) attaches the matching Lightning profiler. Unset (default)
    leaves the trainer byte-identical — Lightning's no-op PassThroughProfiler."""
    from lightning.pytorch.profilers import PassThroughProfiler, SimpleProfiler

    base = _trainer_only_xp(n_epochs=1, max_steps=None)

    monkeypatch.delenv("V14_PROFILER", raising=False)
    assert isinstance(base._trainer().profiler, PassThroughProfiler)

    monkeypatch.setenv("V14_PROFILER", "simple")
    assert isinstance(base._trainer().profiler, SimpleProfiler)


def test_val_check_interval_converted_optsteps_to_microbatches() -> None:
    """#66: our val_check_interval is in OPTIMIZER steps, but Lightning counts
    an int val_check_interval in training MICRO-batches. Under grad-accum the
    Trainer kwarg must be (opt-steps × accumulate_grad_batches) — else with
    accum=16 it validates ~16× too often (the gate-47722655 cadence bug)."""
    base = _trainer_only_xp(n_epochs=100, max_steps=1500)
    # accum=1: opt-step == micro-batch, value passes through unchanged.
    no_accum = base.model_copy(update={"val_check_interval": 150})
    assert no_accum._trainer().val_check_interval == 150
    # accum=16: 150 opt-steps → 2400 micro-batches (the gate config).
    accum = base.model_copy(
        update={"val_check_interval": 150, "accumulate_grad_batches": 16}
    )
    assert accum._trainer().val_check_interval == 2400


def test_int_val_check_interval_disables_epoch_gate() -> None:
    """#66 M1: an int step-cadence is a GLOBAL step count, so the per-epoch
    gate must be disabled — else Lightning raises ``val_check_interval >
    batches-per-epoch`` before step 0 on the locked gate (2400 micro-batches >
    ~1752/epoch at bs=2). A float cadence keeps the default epoch gate (1)."""
    base = _trainer_only_xp(n_epochs=100, max_steps=1500)
    accum = base.model_copy(
        update={"val_check_interval": 150, "accumulate_grad_batches": 16}
    )
    assert accum._trainer().check_val_every_n_epoch is None
    # No accum: still an int cadence → still disabled.
    no_accum = base.model_copy(update={"val_check_interval": 150})
    assert no_accum._trainer().check_val_every_n_epoch is None


def test_val_check_interval_float_not_converted() -> None:
    """A FLOAT val_check_interval is Lightning-native fraction-of-epoch and
    must pass through unscaled regardless of grad-accum; the per-epoch gate
    stays at its Lightning default (1, validate within every epoch)."""
    base = _trainer_only_xp(n_epochs=3, max_steps=None)
    frac = base.model_copy(
        update={"val_check_interval": 0.5, "accumulate_grad_batches": 16}
    )
    assert frac._trainer().val_check_interval == 0.5
    assert frac._trainer().check_val_every_n_epoch == 1


def test_guard_warmup_min_step_default_zero() -> None:
    """#67: the guard warmup-step gate defaults to 0 (check-count grace only),
    so prior behaviour is unchanged when the dispatch does not set it."""
    base = _trainer_only_xp(n_epochs=1, max_steps=None)
    assert base.guard_warmup_min_step == 0


class _StubCkptCb:
    def __init__(self, best_model_path: str) -> None:
        self.best_model_path = best_model_path


class _StubTrainer:
    def __init__(self, best_model_path: str = "") -> None:
        self.checkpoint_callback = _StubCkptCb(best_model_path)


def test_test_ckpt_path_best_only_for_early_stopping_phase() -> None:
    """#32 audit follow-up: EarlyStopping doesn't restore best weights, so the
    one phase that early-stops (the supervised P4 probe) tests its val-best
    checkpoint; SSL phases (no early-stop) test in-memory last-epoch weights —
    which the cross-phase handoff also snapshots, so that path is unchanged."""
    ssl = _trainer_only_xp(n_epochs=3, max_steps=None)  # early_stopping_patience=None
    # No early-stop -> in-memory, even if a best ckpt happens to exist.
    assert ssl._test_ckpt_path(_StubTrainer("/x/best.ckpt")) is None

    p4 = ssl.model_copy(update={"early_stopping_patience": 10})
    # Early-stop + a written best.ckpt -> evaluate best.
    assert p4._test_ckpt_path(_StubTrainer("/x/best.ckpt")) == "best"
    # Early-stop but no best written (stopped before first validation) -> in-memory.
    assert p4._test_ckpt_path(_StubTrainer("")) is None


def test_periodic_last_ckpt_is_metric_independent_on_step_cadence(tmp_path) -> None:
    """C5: on a step-cadence (max_steps SSL/distill) run, last.ckpt must advance on
    a fixed OPTIMIZER-step cadence regardless of whether val_loss improved, so a
    SLURM-requeue resume never rolls back more than one val window. Lightning gates
    last.ckpt on metric improvement in on_validation_end (model_checkpoint.py:516),
    so a second metric-INDEPENDENT ModelCheckpoint (every_n_train_steps) owns
    last.ckpt while the best.ckpt callback keeps best-by-metric only (save_last off
    -> exactly one owner, no Lightning -v auto-increment). On an epoch-cadence run
    the single best callback owns last.ckpt as before. A THIRD metric-independent
    'ladder' callback (save_top_k=-1, every 500 steps) keeps EVERY checkpoint on
    both cadences for post-hoc probe selection + resume insurance (never-kill
    directive, [[feedback_never_kill_runs_save_every_500_2026_06_11]])."""
    from lightning.pytorch.callbacks import ModelCheckpoint

    base = _tiny_mlp_xp(tmp_path / "runs", n_epochs=1)

    stepped = base.model_copy(update={"val_check_interval": 150})
    ckpts = [c for c in stepped._callbacks() if isinstance(c, ModelCheckpoint)]
    # best + last + ladder.
    assert len(ckpts) == 3
    best = [c for c in ckpts if c.monitor is not None]
    # last (save_top_k=0, save_last) and ladder (save_top_k=-1) are both metric-independent.
    metric_indep = [c for c in ckpts if c.monitor is None]
    last = [c for c in metric_indep if c.save_top_k == 0]
    ladder = [c for c in metric_indep if c.save_top_k == -1]
    assert len(best) == 1 and len(last) == 1 and len(ladder) == 1
    # best.ckpt: top-1 by the monitor; save_last OFF so it never fights for last.ckpt.
    assert best[0].save_top_k == 1 and best[0].save_last is False
    assert best[0].monitor == stepped.checkpoint_monitor
    # last.ckpt: metric-independent, fired purely on the OPTIMIZER-step cadence
    # (val_check_interval is already in optimizer steps == every_n_train_steps unit).
    assert last[0].save_last is True and last[0].save_top_k == 0
    assert last[0]._every_n_train_steps == 150
    # ladder.ckpt: keep EVERY checkpoint on a fixed 500-step cadence (never-kill).
    assert ladder[0].save_top_k == -1 and ladder[0].save_last is False
    assert ladder[0]._every_n_train_steps == 500

    # epoch-cadence (val_check_interval=None): best owns last.ckpt; ladder still present.
    epoch_ckpts = [c for c in base._callbacks() if isinstance(c, ModelCheckpoint)]
    assert len(epoch_ckpts) == 2
    epoch_best = [c for c in epoch_ckpts if c.monitor is not None]
    epoch_ladder = [c for c in epoch_ckpts if c.monitor is None and c.save_top_k == -1]
    assert len(epoch_best) == 1 and len(epoch_ladder) == 1
    assert epoch_best[0].save_last is True and epoch_best[0].save_top_k == 1


def test_lr_log_interval_default_epoch(tmp_path) -> None:
    """Default LearningRateMonitor cadence is per-epoch (unchanged)."""
    from lightning.pytorch.callbacks import LearningRateMonitor

    xp = _tiny_mlp_xp(tmp_path / "runs", n_epochs=1)
    lrm = [c for c in xp._callbacks() if isinstance(c, LearningRateMonitor)]
    assert len(lrm) == 1 and lrm[0].logging_interval == "epoch"


def test_lr_log_interval_step_when_set(tmp_path) -> None:
    """--live sets lr_log_interval='step' so the LR-schedule curve is dense for
    the nano learning-dynamics dashboard.
    reports/nano_dynamics_dashboard_handoff_2026_06_07.md."""
    from lightning.pytorch.callbacks import LearningRateMonitor

    xp = _tiny_mlp_xp(tmp_path / "runs", n_epochs=1).model_copy(
        update={"lr_log_interval": "step"}
    )
    lrm = [c for c in xp._callbacks() if isinstance(c, LearningRateMonitor)]
    assert len(lrm) == 1 and lrm[0].logging_interval == "step"


def test_logger_builds_config_with_save_dir(tmp_path) -> None:
    """neuraltrain's Csv/WandbLoggerConfig.build() REQUIRE save_dir; _logger must
    pass it. The prior no-arg .build() was a latent TypeError on any configured
    logger (e.g. every --live wandb run)."""
    from lightning.pytorch.loggers import CSVLogger
    from neuraltrain.utils import CsvLoggerConfig

    xp = _tiny_mlp_xp(tmp_path / "runs", n_epochs=1).model_copy(
        update={"csv_config": CsvLoggerConfig()}
    )
    loggers = xp._logger()
    loggers = loggers if isinstance(loggers, list) else [loggers]
    assert any(isinstance(lg, CSVLogger) for lg in loggers)
