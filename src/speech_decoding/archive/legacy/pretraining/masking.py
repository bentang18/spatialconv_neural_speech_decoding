"""Temporal span masking for masked prediction pretraining.

Masks 40-60% of frames using 3-6 non-overlapping contiguous spans.
Same mask applied across all spatial positions (RD-87).
Ref: RD-70 (masking ratio), RD-47 (masked temporal spans).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def generate_span_mask(
    T: int,
    mask_ratio: tuple[float, float] = (0.4, 0.6),
    n_spans: tuple[int, int] = (3, 6),
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Generate a boolean mask with contiguous spans.

    Args:
        T: Number of time frames.
        mask_ratio: (min, max) fraction of frames to mask.
        n_spans: (min, max) number of spans.
        rng: Random state for reproducibility.

    Returns:
        Boolean array of shape (T,), True = masked.
    """
    if rng is None:
        rng = np.random.RandomState()

    target_ratio = rng.uniform(*mask_ratio)
    target_masked = max(1, int(round(T * target_ratio)))
    num_spans = rng.randint(n_spans[0], n_spans[1] + 1)
    num_spans = min(num_spans, target_masked)

    mask = np.zeros(T, dtype=bool)

    # Distribute target_masked frames across num_spans spans
    span_lengths = np.ones(num_spans, dtype=int)
    remaining = target_masked - num_spans
    for _ in range(remaining):
        span_lengths[rng.randint(num_spans)] += 1

    # Place spans without overlap using gap distribution
    available = T - target_masked
    if available < 0:
        mask[:] = True
        return mask

    gaps = np.zeros(num_spans + 1, dtype=int)
    for _ in range(available):
        gaps[rng.randint(num_spans + 1)] += 1

    pos = 0
    for i in range(num_spans):
        pos += gaps[i]
        end = min(pos + span_lengths[i], T)
        mask[pos:end] = True
        pos = end

    return mask


class SpanMasker(nn.Module):
    """Applies temporal span masking with a learnable [MASK] token.

    The same temporal mask is applied to all items in the batch and
    all spatial positions (RD-87: prevents spatial correlation shortcut).
    """

    def __init__(self, d: int):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(d) * 0.02)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply mask to input.

        Args:
            x: (B, T, D) input features.
            mask: (T,) boolean mask, True = masked.

        Returns:
            x_masked: (B, T, D) with masked frames replaced by [MASK] token.
            mask: (T,) the mask that was applied.
        """
        x_masked = x.clone()
        x_masked[:, mask] = self.mask_token.unsqueeze(0)
        return x_masked, mask
