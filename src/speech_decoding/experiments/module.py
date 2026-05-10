from __future__ import annotations

import torch
from lightning import pytorch as pl
from torch import nn
from torchmetrics import Metric

from neuraltrain.optimizers import BaseOptimizer


class BrainModule(pl.LightningModule):
    """Minimal Lightning wrapper around a NeuralTrain-built brain model."""

    def __init__(
        self,
        *,
        model: nn.Module,
        loss: nn.Module,
        optim_config: BaseOptimizer,
        metrics: dict[str, Metric],
        x_name: str | tuple[str, ...] = "input",
        y_name: str = "target",
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = loss
        self.optim_config = optim_config
        self.x_name = x_name
        self.y_name = y_name
        self.metrics = nn.ModuleDict(
            {
                f"{split}_{name}": metric.clone()
                for split in ("val", "test")
                for name, metric in metrics.items()
            }
        )

    def forward(self, batch):
        if isinstance(self.x_name, str):
            return self.model(batch.data[self.x_name])
        return self.model(*(batch.data[name] for name in self.x_name))

    def _target(self, batch) -> torch.Tensor:
        target = batch.data[self.y_name]
        if target.ndim > 1 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        return target.long()

    def _run_step(self, batch, step_name: str) -> torch.Tensor:
        target = self._target(batch)
        pred = self.forward(batch)
        loss = self.loss(pred, target)
        self.log(f"{step_name}_loss", loss, on_epoch=True, prog_bar=True)

        for name, metric in self.metrics.items():
            if name.startswith(f"{step_name}_"):
                metric.update(pred, target)
                self.log(name, metric, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._run_step(batch, "train")

    def validation_step(self, batch, batch_idx) -> None:
        self._run_step(batch, "val")

    def test_step(self, batch, batch_idx) -> None:
        self._run_step(batch, "test")

    def configure_optimizers(self):
        try:
            return self.optim_config.build(
                self.parameters(),
                total_steps=self.trainer.estimated_stepping_batches,
            )
        except TypeError:
            return self.optim_config.build(self.parameters())
