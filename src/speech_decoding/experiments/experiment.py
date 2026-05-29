from __future__ import annotations

import typing as tp
from pathlib import Path

import exca
import lightning.pytorch as pl
import pydantic
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger
from lightning.pytorch.loggers.logger import DummyLogger
from torch.utils.data import DataLoader

from neuraltrain import BaseLoss, BaseMetric, BaseModelConfig, LightningOptimizer
from neuraltrain.utils import BaseExperiment, CsvLoggerConfig, WandbLoggerConfig

from speech_decoding.experiments.data import Data
from speech_decoding.experiments.experiment_logging import ExperimentLogger
from speech_decoding.experiments.module import BrainModule


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
    x_name: str | tuple[str, ...] = "input"
    y_name: str = "target"
    target_field: str = "code"
    n_outputs: int | None = None
    test_only: bool = False
    fast_dev_run: bool | int = False
    accelerator: str = "auto"
    devices: int | str = "auto"
    log_every_n_steps: int = 10
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    early_stopping_patience: int | None = None
    checkpoint_monitor: str = "val_loss"
    csv_config: CsvLoggerConfig | None = None
    wandb_config: WandbLoggerConfig | None = None
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

    def _logger(self) -> Logger | list[Logger]:
        loggers: list[Logger] = []
        if self.csv_config is not None:
            loggers.append(self.csv_config.build())
        if self.wandb_config is not None:
            loggers.append(self.wandb_config.build())
        if loggers:
            return loggers
        if self.infra.folder is not None:
            return CSVLogger(save_dir=str(Path(self.infra.folder) / "lightning"))
        return DummyLogger()

    def _callbacks(self) -> list[pl.Callback]:
        callbacks: list[pl.Callback] = [LearningRateMonitor(logging_interval="epoch")]
        if self.infra.folder is not None:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(Path(self.infra.folder) / "checkpoints"),
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
        if self.limit_train_batches is not None:
            kwargs["limit_train_batches"] = self.limit_train_batches
        if self.limit_val_batches is not None:
            kwargs["limit_val_batches"] = self.limit_val_batches
        if self.limit_test_batches is not None:
            kwargs["limit_test_batches"] = self.limit_test_batches
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

    def _train_and_test(self) -> dict[str, float | None]:
        pl.seed_everything(self.seed, workers=True)
        # CR01: pl.seed_everything may miss some CUDA RNG init paths in
        # DDP-forked workers; explicit manual_seed_all closes that gap.
        torch.cuda.manual_seed_all(self.seed)
        # CR02: per-worker numpy + random + torch RNG seeded as seed+worker_id.
        loaders = self.data.build(worker_seed=self.seed)
        brain_module = self._build_brain_module(loaders["train"])
        trainer = self._trainer()
        if not self.test_only:
            trainer.fit(brain_module, loaders["train"], loaders["val"])
        results = trainer.test(brain_module, loaders["test"])
        return dict(results[0]) if results else {}

    @infra.apply
    def run(self) -> dict[str, float | None]:
        artifact_dir, exca_uid = self._artifact_dir_and_uid()
        if artifact_dir is None:
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
