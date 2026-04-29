"""Soft parcel embedding (Stage 3 of the B-1 architecture, Task C2).

Replaces the deprecated within-parcel Perceiver summarizer (`#26`). Per
the 2026-04-16 amendment, the per-electrode parcel embedding is a simple
linear combination `support @ P_emb` where `support[N_e, 15]` is the raw
BNA probability over Tier-1 and `P_emb: (15, d)` is a learnable
Xavier-init lookup. Support is RAW — not row-normalized — so electrodes
with low Tier-1 mass naturally produce small embeddings.
"""

from __future__ import annotations

import torch
from torch import nn

from speech_decoding.v14.config import SoftParcelEmbeddingConfig


class SoftParcelEmbedding(nn.Module):
    """Per-electrode parcel embedding via `support @ P_emb`."""

    def __init__(self, cfg: SoftParcelEmbeddingConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or SoftParcelEmbeddingConfig()
        self.n_parcels = cfg.n_parcels
        self.d_model = cfg.d_model
        self.P_emb = nn.Parameter(torch.empty(cfg.n_parcels, cfg.d_model))
        nn.init.xavier_uniform_(self.P_emb)

    def forward(self, support: torch.Tensor) -> torch.Tensor:
        if support.ndim != 3 or support.shape[-1] != self.n_parcels:
            raise ValueError(
                f"expected (B, N_e, {self.n_parcels}), got {tuple(support.shape)}"
            )
        return support @ self.P_emb
