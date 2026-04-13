"""Semi-supervised Stage 2 training: joint SSL + CE on labeled data.

Combines two objectives in a single training loop:
1. SSL loss (BYOL/VICReg/JEPA/masked) on ALL patients (no labels used)
2. CE classification loss on LABELED target patient (supervised signal)

The backbone is shared, so CE gradients teach phoneme discrimination while
SSL gradients teach augmentation invariance across patients. This is the
principled approach for our setting: too little data for pure SSL to
discover phoneme structure, but labels exist for the target patient.

Inspired by CEBRA (Schneider et al., Nature 2023): joint unsupervised
(time-contrastive) + supervised (behavior-contrastive) objectives for
neural recordings with partial labels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)


@dataclass
class SemiSupervisedConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 5000
    batch_size: int = 8
    grad_clip: float = 1.0
    # CE classification settings
    n_positions: int = 3
    n_classes: int = 9
    alpha: float = 1.0  # weight of CE loss relative to SSL loss
    ce_batch_size: int = 8


class SemiSupervisedStage2Trainer:
    """Joint SSL + supervised training on shared backbone.

    Each step:
    1. Sample batch from all patients → SSL loss (via model.forward())
    2. Sample labeled batch from target → encode → mean-pool → CE loss
    3. Combined loss = ssl_loss + alpha * ce_loss → backprop
    """

    def __init__(
        self,
        model: nn.Module,
        config: SemiSupervisedConfig,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Classification head: operates on mean-pooled encoder features
        gru_hidden = getattr(model, 'gru_hidden', 32)
        gru_out_dim = gru_hidden * 2
        n_out = config.n_positions * config.n_classes
        self.head = nn.Linear(gru_out_dim, n_out).to(device)

        # Single optimizer for model + head
        self.optimizer = AdamW(
            list(model.parameters()) + list(self.head.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.steps)
        self.step_count = 0

    def _sample_ssl_batch(self, patient_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample a same-patient batch for SSL."""
        pids = list(patient_data.keys())
        pid = pids[np.random.randint(len(pids))]
        trials = patient_data[pid]
        idx = np.random.randint(len(trials), size=self.config.batch_size)
        return trials[idx].to(self.device)

    def _sample_labeled_batch(
        self,
        grids: torch.Tensor,
        labels: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a labeled batch from target patient."""
        n = len(grids)
        idx = np.random.randint(n, size=self.config.ce_batch_size)
        batch_grids = grids[idx].to(self.device)
        # Labels: list of lists → (B, n_positions) tensor
        batch_labels = torch.tensor(
            [labels[i] for i in idx], dtype=torch.long, device=self.device,
        )
        return batch_grids, batch_labels

    def train_step(
        self,
        patient_data: dict[str, torch.Tensor],
        labeled_grids: torch.Tensor,
        labeled_labels: list[list[int]],
    ) -> dict[str, float]:
        """One training step: SSL + CE combined."""
        self.model.train()
        self.head.train()

        # 1. SSL loss on all patients
        ssl_batch = self._sample_ssl_batch(patient_data)
        ssl_result = self.model(ssl_batch, compute_loss=True)
        ssl_loss = ssl_result["loss"]

        # 2. CE loss on labeled target
        ce_grids, ce_labels = self._sample_labeled_batch(
            labeled_grids, labeled_labels,
        )
        features = self.model.encode(ce_grids)  # (B, T', 2H)
        pooled = features.mean(dim=1)            # (B, 2H)
        logits = self.head(pooled)               # (B, n_pos * n_cls)
        logits = logits.view(-1, self.config.n_positions, self.config.n_classes)

        # Labels are 1-indexed phoneme IDs → 0-indexed for CE
        targets = (ce_labels - 1).clamp(min=0)  # (B, n_positions)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, self.config.n_classes),
            targets.reshape(-1),
        )

        # 3. Combined loss
        loss = ssl_loss + self.config.alpha * ce_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.head.parameters()),
            self.config.grad_clip,
        )
        self.optimizer.step()
        if hasattr(self.model, 'ema_update'):
            self.model.ema_update()
        self.scheduler.step()
        self.step_count += 1

        metrics = {
            "loss": loss.item(),
            "ssl_loss": ssl_loss.item(),
            "ce_loss": ce_loss.item(),
            "step": self.step_count,
        }
        # Pass through SSL component losses
        for key in ("inv_loss", "var_loss", "cov_loss", "loss_12", "loss_21",
                     "pred_loss", "sigreg_loss"):
            if key in ssl_result:
                val = ssl_result[key]
                metrics[key] = val if isinstance(val, float) else val.item()
        return metrics

    def train(
        self,
        patient_data: dict[str, torch.Tensor],
        labeled_grids: torch.Tensor,
        labeled_labels: list[list[int]],
    ) -> list[dict[str, float]]:
        """Full training loop.

        Args:
            patient_data: {patient_id: (N, H, W, T)} for SSL.
            labeled_grids: (N, H, W, T) labeled target patient grids.
            labeled_labels: list of N label lists (1-indexed phoneme IDs).

        Returns:
            List of per-step metrics dicts.
        """
        metrics_history = []
        for step in range(self.config.steps):
            metrics = self.train_step(patient_data, labeled_grids, labeled_labels)
            metrics_history.append(metrics)

            if step % 100 == 0:
                logger.info(
                    "Step %d: loss=%.4f (ssl=%.4f, ce=%.4f)",
                    step, metrics["loss"], metrics["ssl_loss"], metrics["ce_loss"],
                )

        return metrics_history
