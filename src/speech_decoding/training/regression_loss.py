"""Loss helpers for framewise regression objectives."""
from __future__ import annotations

import torch


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over masked frames only.

    Args:
        prediction: (B, T, D)
        target: (B, T, D)
        mask: (B, T) binary or continuous frame weights
    """
    weights = mask.to(prediction.dtype).unsqueeze(-1)
    sqerr = (prediction - target) ** 2
    denom = weights.sum() * prediction.shape[-1]
    if denom.item() <= 0:
        return prediction.new_tensor(0.0)
    return (sqerr * weights).sum() / denom


def segment_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    segment_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error on segment-pooled predictions.

    Args:
        prediction: (B, T, D)
        target: (B, S, D)
        segment_mask: (B, S, T) binary or continuous segment weights
    """
    weights = segment_mask.to(prediction.dtype)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    pred_seg = torch.einsum("btd,bst->bsd", prediction, weights) / denom
    sqerr = (pred_seg - target) ** 2
    return sqerr.mean()
