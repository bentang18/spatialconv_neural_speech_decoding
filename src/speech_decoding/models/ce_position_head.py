"""CE classification head with position-specific temporal attention.

Instead of mean-pooling all GRU frames (which mixes pre-speech silence
with execution from all 3 phonemes), each position gets a learned query
vector that attends to the most relevant temporal frames.

Position 1's query learns to attend to early execution frames,
position 3's to later frames. The classifier weights are shared across
positions — position-specificity comes entirely from the attention.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CEPositionHead(nn.Module):
    """Position-specific temporal attention + shared phoneme classifier.

    Input:  (B, T, 2H) — BiGRU hidden states
    Output: (B, n_positions * n_phonemes) — raw logits for CE loss

    Parameters: n_positions * input_dim (queries) + input_dim * n_phonemes + n_phonemes (classifier)
    Default: 3*128 + 128*9 + 9 = 384 + 1,152 + 9 = 1,545 params
    """

    def __init__(
        self,
        input_dim: int = 128,
        n_positions: int = 3,
        n_phonemes: int = 9,
    ):
        super().__init__()
        self.n_positions = n_positions
        self.n_phonemes = n_phonemes
        self.scale = math.sqrt(input_dim)

        # Learned query per position: (n_positions, input_dim)
        self.queries = nn.Parameter(torch.randn(n_positions, input_dim) * 0.02)

        # Shared classifier: maps D → n_phonemes (same weights for all positions,
        # position-specificity comes from the attention pooling)
        self.classifier = nn.Linear(input_dim, n_phonemes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D) where D = 2*H
        B, T, D = h.shape

        # Dot-product attention: each query attends to temporal frames
        # (B, T, D) @ (D, n_pos) → (B, T, n_pos)
        attn_scores = torch.matmul(h, self.queries.T) / self.scale
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T, n_pos)

        # Position-specific pooling: (B, n_pos, D)
        pooled = torch.einsum("btp,btd->bpd", attn_weights, h)

        # Classify each position: (B, n_pos, n_phon)
        logits = self.classifier(pooled)

        # Flatten to match CE loss format: (B, n_pos * n_phon)
        return logits.reshape(B, self.n_positions * self.n_phonemes)
