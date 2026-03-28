"""Stage 2: Neural masked span prediction on real trial data.

Trains encoder+predictor on unlabeled response-locked trial epochs.
No phoneme labels used. Source patients only (exclude dev + target).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from speech_decoding.pretraining.pretrain_model import PretrainModel

logger = logging.getLogger(__name__)


@dataclass
class Stage2Config:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 5000
    batch_size: int = 8
    checkpoint_dir: str | None = None
    checkpoint_every: int = 1000
    grad_clip: float = 1.0


class Stage2Trainer:
    """Train PretrainModel on real trial data with masked span prediction."""

    def __init__(self, model: PretrainModel, config: Stage2Config, device: str = "cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.steps)
        self.step_count = 0

    def _filter_patients(self, patient_data, exclude=None):
        if exclude is None:
            return patient_data
        return {k: v for k, v in patient_data.items() if k not in exclude}

    def _sample_batch(self, patient_data):
        """Sample a same-patient batch to avoid mixed-grid stacking."""
        pids = list(patient_data.keys())
        pid = pids[np.random.randint(len(pids))]
        trials = patient_data[pid]
        idx = np.random.randint(len(trials), size=self.config.batch_size)
        return trials[idx].to(self.device)

    def train_step(self, patient_data):
        self.model.train()
        batch = self._sample_batch(patient_data)
        result = self.model(batch, compute_loss=True)
        loss = result["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step_count += 1

        return {"loss": loss.item(), "step": self.step_count}

    def train(self, patient_data, exclude=None):
        filtered = self._filter_patients(patient_data, exclude)
        if not filtered:
            raise ValueError("No patients remaining after exclusion")

        metrics_history = []
        for step in range(self.config.steps):
            metrics = self.train_step(filtered)
            metrics_history.append(metrics)

            if step % 100 == 0:
                logger.info("Step %d: loss=%.4f", step, metrics["loss"])

            if (self.config.checkpoint_dir
                and self.config.checkpoint_every > 0
                and (step + 1) % self.config.checkpoint_every == 0):
                self._save_checkpoint(step)

        return metrics_history

    def _save_checkpoint(self, step):
        path = Path(self.config.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"step": step, "model": self.model.state_dict(),
             "optimizer": self.optimizer.state_dict()},
            path / f"checkpoint_step{step:06d}.pt",
        )
