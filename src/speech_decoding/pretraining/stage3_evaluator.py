"""Stage 3: Freeze backbone, train CE head per patient.

Uses grouped-by-token CV. Reports PER + content-collapse diagnostics.
Ref: RD-14 (freeze config), RD-17 (CE primary), RD-78 (collapse diagnostics).
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from speech_decoding.pretraining.pretrain_model import PretrainModel
from speech_decoding.evaluation.grouped_cv import (
    _patient_seed,
    build_token_groups,
    create_grouped_splits,
)
from speech_decoding.evaluation.content_collapse import content_collapse_report

logger = logging.getLogger(__name__)


@dataclass
class Stage3Config:
    lr: float = 1e-3
    epochs: int = 100
    patience: int = 10
    n_folds: int = 5
    n_classes: int = 9
    n_positions: int = 3


class Stage3Evaluator:
    """Fine-tune CE head on frozen pretrained features."""

    def __init__(self, model: PretrainModel, config: Stage3Config, device: str = "cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = device

    def _freeze_backbone(self):
        """Freeze backbone params; keep read-in (Conv2d) trainable.

        Backbone LayerNorm is frozen together with the rest of the backbone
        (RD-14 allows unfreezing it, but the default contract freezes it).
        """
        for name, p in self.model.named_parameters():
            if "readin" not in name:
                p.requires_grad = False

    def _create_head(self) -> nn.Linear:
        """Create 27-way CE head (3 positions × 9 phonemes)."""
        n_out = self.config.n_positions * self.config.n_classes
        gru_hidden = self.model.config["gru_hidden"]
        return nn.Linear(gru_hidden * 2, n_out).to(self.device)

    def _train_fold(self, head, train_grids, train_labels, val_grids, val_labels):
        """Train one fold, return val PER and predictions."""
        self._freeze_backbone()
        head_fresh = self._create_head()

        params = [{"params": head_fresh.parameters(), "lr": self.config.lr}]
        readin_params = [p for n, p in self.model.named_parameters()
                         if "readin" in n and p.requires_grad]
        if readin_params:
            params.append({"params": readin_params, "lr": self.config.lr * 3})

        optimizer = AdamW(params, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)

        best_loss = float("inf")
        best_head_state = None
        patience_ctr = 0

        self.model.eval()
        for epoch in range(self.config.epochs):
            head_fresh.train()
            train_g = train_grids.to(self.device)
            with torch.no_grad():
                features = self.model.encode(train_g)
            pooled = features.mean(dim=1)
            logits = head_fresh(pooled)
            per_pos = logits.view(-1, self.config.n_positions, self.config.n_classes)

            targets = torch.tensor(train_labels, device=self.device, dtype=torch.long)
            loss = sum(
                F.cross_entropy(per_pos[:, p, :], targets[:, p] - 1)
                for p in range(self.config.n_positions)
            ) / self.config.n_positions

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            head_fresh.eval()
            with torch.no_grad():
                val_g = val_grids.to(self.device)
                val_feat = self.model.encode(val_g)
                val_pooled = val_feat.mean(dim=1)
                val_logits = head_fresh(val_pooled)
                val_per_pos = val_logits.view(-1, self.config.n_positions, self.config.n_classes)
                val_tgt = torch.tensor(val_labels, device=self.device, dtype=torch.long)
                val_loss = sum(
                    F.cross_entropy(val_per_pos[:, p, :], val_tgt[:, p] - 1)
                    for p in range(self.config.n_positions)
                ) / self.config.n_positions

            if val_loss < best_loss:
                best_loss = val_loss
                best_head_state = deepcopy(head_fresh.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.config.patience:
                    break

        # Fall back to last state if best_head_state was never set
        if best_head_state is None:
            best_head_state = deepcopy(head_fresh.state_dict())

        head_fresh.load_state_dict(best_head_state)
        head_fresh.eval()
        with torch.no_grad():
            val_g = val_grids.to(self.device)
            val_feat = self.model.encode(val_g)
            val_pooled = val_feat.mean(dim=1)
            val_logits = head_fresh(val_pooled)
            val_per_pos = val_logits.view(-1, self.config.n_positions, self.config.n_classes)
            preds = val_per_pos.argmax(dim=-1).cpu().numpy() + 1
            val_tgt_np = np.array(val_labels)

        total, errors = 0, 0
        for pred_seq, true_seq in zip(preds, val_tgt_np):
            for p, t in zip(pred_seq, true_seq):
                total += 1
                if p != t:
                    errors += 1
        per = errors / total if total > 0 else 1.0
        return per, preds

    def evaluate(self, grids, labels, patient_id):
        """Full grouped-by-token CV evaluation.

        Returns dict with mean_per, fold_pers, content_collapse report.
        """
        groups = build_token_groups(labels)
        seed = _patient_seed(patient_id)
        splits = create_grouped_splits(labels, groups,
                                       n_folds=self.config.n_folds, seed=seed)

        fold_pers = []
        all_preds = []
        all_targets = []
        base_model_state = deepcopy(self.model.state_dict())

        for fold_idx, fold in enumerate(splits):
            # CRITICAL: restore pretrained state before each fold
            self.model.load_state_dict(base_model_state)
            train_idx = fold["train_indices"]
            val_idx = fold["val_indices"]

            train_grids = grids[train_idx]
            train_labels_fold = [labels[i] for i in train_idx]
            val_grids = grids[val_idx]
            val_labels_fold = [labels[i] for i in val_idx]

            head = self._create_head()
            per, preds = self._train_fold(
                head, train_grids, train_labels_fold, val_grids, val_labels_fold
            )
            fold_pers.append(per)
            all_preds.extend(preds.tolist())
            all_targets.extend(val_labels_fold)
            logger.info("  Fold %d PER: %.3f", fold_idx, per)

        all_preds_np = np.array(all_preds)
        preds_per_pos = [all_preds_np[:, i] for i in range(self.config.n_positions)]
        sequences = all_preds_np.tolist()
        collapse = content_collapse_report(preds_per_pos, sequences,
                                           n_classes=self.config.n_classes)

        return {
            "mean_per": float(np.mean(fold_pers)),
            "std_per": float(np.std(fold_pers)),
            "fold_pers": fold_pers,
            "content_collapse": collapse,
        }
