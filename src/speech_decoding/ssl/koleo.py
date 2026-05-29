"""KoLeo readout uniformity loss (T2.5).

DINOv3-style uniformity regularizer on the readout embedding. Encourages
embeddings to spread out in a batch by penalizing the negative log of each
sample's distance to its nearest in-batch neighbor:

    L_KoLeo = − (1 / N) Σ_i log( min_{j ≠ i} ||x_i − x_j||₂ + ε )

This loss is **optional** in the v14 recipe — see
``project_v14_three_phase_staged_recipe_2026_05_18.md``. Per blocker EX04
(Gram anchoring): DINOv3 Gram-anchoring is deferred for v14's 13M scale, but
KoLeo's micro-batch-16 readout-uniformity term is cheaper and was retained
as a P1 lever. The loss weight is *not* locked here.
"""

from __future__ import annotations

import torch
from torch import Tensor


def koleo_loss(embeddings: Tensor, *, eps: float = 1e-8) -> Tensor:
    """KoLeo uniformity loss over a batch of embeddings.

    ``embeddings`` shape ``(N, D)``; loss is the negative mean log of each
    sample's nearest-neighbor L2 distance.

    The DINOv3 micro-batch convention is to call this on a sub-batch of
    fixed size (default 16) sampled from the larger training batch; that
    sub-batching belongs in the training loop, not this primitive.
    """
    if embeddings.dim() != 2:
        raise ValueError(
            f"expected (N, D) embeddings, got shape {tuple(embeddings.shape)}"
        )
    n = embeddings.shape[0]
    if n < 2:
        # Loss is undefined for a single point; return 0 so the training
        # loop doesn't have to special-case N=1.
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    # L2-normalize so the loss has a stable scale across the readout's d.
    # DINOv3 normalizes; we follow them.
    x = torch.nn.functional.normalize(embeddings, dim=-1, eps=eps)
    # Pairwise squared distances; mask diagonal with +inf to exclude self-pairs.
    dist2 = torch.cdist(x, x, p=2.0).pow(2)
    diag_mask = torch.eye(n, dtype=torch.bool, device=x.device)
    dist2 = dist2.masked_fill(diag_mask, float("inf"))
    nearest = dist2.min(dim=1).values.clamp(min=eps)
    # Take the log of the L2 distance (not squared) per the DINOv3 formula.
    return -(0.5 * nearest.log()).mean()
