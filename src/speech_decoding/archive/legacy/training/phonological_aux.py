"""Label-derived phonological auxiliary targets and losses."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from speech_decoding.data.phoneme_map import ARTICULATORY_MATRIX


def build_feature_targets(
    labels: list[list[int]],
    n_positions: int = 3,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert 1-indexed phoneme labels into articulatory feature targets."""
    matrix = torch.from_numpy(ARTICULATORY_MATRIX)
    rows = []
    for seq in labels:
        rows.append(matrix[[idx - 1 for idx in seq[:n_positions]]])
    target = torch.stack(rows, dim=0)  # (B, P, 15)
    if device is not None:
        target = target.to(device)
    return target


def articulatory_pos_weight(device: torch.device | str | None = None) -> torch.Tensor:
    """Class-balance weights for binary feature BCE."""
    matrix = torch.from_numpy(ARTICULATORY_MATRIX)
    pos = matrix.sum(dim=0)
    neg = matrix.shape[0] - pos
    weight = neg / pos.clamp_min(1.0)
    if device is not None:
        weight = weight.to(device)
    return weight


def per_position_feature_bce_loss(
    logits: torch.Tensor,
    labels: list[list[int]],
    n_positions: int = 3,
) -> torch.Tensor:
    """Mean-pooled BCE loss for per-position articulatory features."""
    n_features = logits.shape[2] // n_positions
    pooled = logits.mean(dim=1).reshape(logits.shape[0], n_positions, n_features)
    targets = build_feature_targets(labels, n_positions=n_positions, device=logits.device)
    pos_weight = articulatory_pos_weight(device=logits.device)
    return F.binary_cross_entropy_with_logits(
        pooled,
        targets,
        pos_weight=pos_weight.view(1, 1, -1),
    )


def per_position_feature_metrics(
    logits: torch.Tensor,
    labels: list[list[int]],
    n_positions: int = 3,
) -> dict[str, float]:
    """Feature-level diagnostics for the auxiliary head."""
    n_features = logits.shape[2] // n_positions
    pooled = logits.mean(dim=1).reshape(logits.shape[0], n_positions, n_features)
    targets = build_feature_targets(labels, n_positions=n_positions).cpu().numpy()
    pred = (torch.sigmoid(pooled).cpu().numpy() >= 0.5).astype(np.float32)
    feature_acc = float((pred == targets).mean())
    exact_match = float((pred == targets).all(axis=-1).mean())
    return {
        "feature_acc": feature_acc,
        "feature_exact": exact_match,
    }
