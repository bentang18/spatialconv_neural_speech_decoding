"""Stage 1: Synthetic pretraining with masked span prediction.

Trains on unlimited synthetic data for S_total/2 steps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from speech_decoding.pretraining.pretrain_model import PretrainModel
from speech_decoding.pretraining.synthetic_pipeline import SyntheticDataPipeline

logger = logging.getLogger(__name__)


@dataclass
class Stage1Config:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 2500
    batch_size: int = 8
    T: int = 200  # frames at 200Hz (1.0s)
    grad_clip: float = 1.0


class Stage1Trainer:
    def __init__(self, model, pipeline, config, device="cpu"):
        self.model = model.to(device)
        self.pipeline = pipeline
        self.config = config
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.steps)
        self.step_count = 0

    def train_step(self):
        self.model.train()
        batch = self.pipeline.generate_batch(
            batch_size=self.config.batch_size,
            T=self.config.T,
            seed=self.step_count,
        ).to(self.device)

        result = self.model(batch, compute_loss=True)
        loss = result["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step_count += 1

        return {"loss": loss.item(), "step": self.step_count}

    def train(self):
        metrics_history = []
        for step in range(self.config.steps):
            metrics = self.train_step()
            metrics_history.append(metrics)
            if step % 100 == 0:
                logger.info("Stage 1 step %d: loss=%.4f", step, metrics["loss"])
        return metrics_history
