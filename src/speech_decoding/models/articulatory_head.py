"""Articulatory decomposition CTC head (E2 full model).

Six parallel heads classify articulatory features. A fixed composition
matrix A maps feature scores to phoneme logits. The model learns what
motor cortex encodes (articulatory gestures), not what linguistics
defines (phoneme categories).

Each sub-head is binary or 3-way — simpler than the full 9-way problem.
/b/ vs /p/ differs ONLY in voicing score — the model must learn voicing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArticulatoryCTCHead(nn.Module):
    """Articulatory decomposition CTC head.

    Input:  (B, T, 2H)
    Output: (B, T, 10) log probabilities (blank + 9 phonemes)
    """

    def __init__(self, input_dim: int = 128, blank_bias: float = 2.0):
        super().__init__()
        # 6 articulatory feature heads
        self.cv_head = nn.Linear(input_dim, 2)       # consonant, vowel
        self.place_head = nn.Linear(input_dim, 3)     # bilabial, labiodental, velar
        self.manner_head = nn.Linear(input_dim, 2)    # stop, fricative
        self.voicing_head = nn.Linear(input_dim, 2)   # voiced, voiceless
        self.height_head = nn.Linear(input_dim, 3)    # low, mid, high
        self.back_head = nn.Linear(input_dim, 3)      # front, central, back

        # CTC blank head (separate)
        self.blank_head = nn.Linear(input_dim, 1)
        # Blank bias: phoneme logits are SUMS of 3-4 sub-head outputs,
        # giving them ~sqrt(3-4)x larger std than blank at Kaiming init.
        # +2.0 for LOPO (enough gradient signal to overcome), lower for
        # per-patient (small data can't escape blank collapse at +2.0).
        nn.init.constant_(self.blank_head.bias, blank_bias)

        # Fixed composition matrix: A[i,j] = 1 iff phoneme i has feature j
        # Phoneme order: AA, EH, IY, UH, B, P, V, G, K (ARPA_PHONEMES)
        # Feature order: C V bil lab vel stp fri vcd vcl low mid hi frt cen bck
        self.register_buffer("A", torch.tensor([
            # AA (/a/): V, low, central
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            # EH (/ae/): V, mid, front
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            # IY (/i/): V, high, front
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            # UH (/u/): V, high, back
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            # B: C, bilabial, stop, voiced
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            # P: C, bilabial, stop, voiceless
            [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            # V: C, labiodental, fricative, voiced
            [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            # G: C, velar, stop, voiced
            [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            # K: C, velar, stop, voiceless
            [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ], dtype=torch.float))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, 2H)
        features = torch.cat([
            self.cv_head(h),
            self.place_head(h),
            self.manner_head(h),
            self.voicing_head(h),
            self.height_head(h),
            self.back_head(h),
        ], dim=-1)  # (B, T, 15)

        phoneme_logits = features @ self.A.T  # (B, T, 9)
        blank_logit = self.blank_head(h)       # (B, T, 1)
        # blank is index 0, phonemes 1-9
        return F.log_softmax(torch.cat([blank_logit, phoneme_logits], dim=-1), dim=-1)
