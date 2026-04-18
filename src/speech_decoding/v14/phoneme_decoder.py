"""D1 minimum AR decoder for the per-phoneme v14 stack (plan P4 / Stage 7).

Mean-pool memory → add BOS-aware `prev_phoneme` embedding → Linear to vocab.
Vocab index 9 in the embedding table is reserved for BOS; `prev_phoneme = -1`
at `phoneme_pos == 0` is remapped to 9 before the embedding lookup.
"""

from __future__ import annotations

import torch
from torch import nn

from speech_decoding.v14.config import D1DecoderConfig


BOS_SENTINEL = -1
_BOS_EMBEDDING_INDEX = 9  # assumes vocab_size == 9; asserted in __init__


class D1Decoder(nn.Module):
    """Mean-pool + prev-embedding + Linear head. ~617 params at d=32."""

    def __init__(self, cfg: D1DecoderConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or D1DecoderConfig()
        if cfg.prev_embedding_size != cfg.vocab_size + 1:
            raise ValueError(
                f"prev_embedding_size {cfg.prev_embedding_size} must equal "
                f"vocab_size + 1 (= {cfg.vocab_size + 1})"
            )
        if cfg.vocab_size != 9:
            # The BOS index assumption is specific to the 9-phoneme vocab.
            raise ValueError(f"D1Decoder assumes vocab_size=9, got {cfg.vocab_size}")
        self.d_model = cfg.d_model
        self.vocab_size = cfg.vocab_size
        self.prev_emb = nn.Embedding(cfg.prev_embedding_size, cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(
        self,
        memory: torch.Tensor,       # (B, S, d)
        prev_phoneme: torch.Tensor, # (B,) long in {-1, 0..8}
    ) -> torch.Tensor:
        """Return per-phoneme logits `(B, vocab_size)`."""

        if memory.ndim != 3 or memory.shape[-1] != self.d_model:
            raise ValueError(
                f"memory shape {tuple(memory.shape)} != (B, S, {self.d_model})"
            )
        if prev_phoneme.shape != (memory.shape[0],):
            raise ValueError(
                f"prev_phoneme shape {tuple(prev_phoneme.shape)} != "
                f"({memory.shape[0]},)"
            )

        mem_pooled = memory.mean(dim=1)  # (B, d)

        idx = torch.where(
            prev_phoneme == BOS_SENTINEL,
            torch.full_like(prev_phoneme, _BOS_EMBEDDING_INDEX),
            prev_phoneme,
        )
        prev_vec = self.prev_emb(idx)  # (B, d)

        q = mem_pooled + prev_vec
        return self.head(q)
