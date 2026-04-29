"""D1 minimum AR decoder for the per-phoneme v14 stack (plan P4 / Stage 7).

Reduces memory to (B, d) per `readout_mode`, adds BOS-aware `prev_phoneme`
embedding, projects to vocab. Vocab index 9 in the embedding table is
reserved for BOS; `prev_phoneme = -1` at `phoneme_pos == 0` is remapped
to 9 before the embedding lookup.

Readout modes:
* ``mean_pool`` — ``memory.mean(dim=1)``. Baseline.
* ``cls``       — ``memory[:, 0]``; assumes the model prepended a learnable
                  CLS token before the backbone.
* ``hierarchical`` — reshape ``(B, n_cells*t_tokens, d) → (B, n_cells, t_tokens, d)``,
                  attention-pool over time per cell (learned ``q_temporal``),
                  then attention-pool over cells (learned ``q_cell``).
* ``hierarchical_atlas`` — same as ``hierarchical`` but the cell-pool query
                  is **atlas-anchored** and provided by the caller as
                  ``cell_query: (B, n_cells, d)`` (derived from pooled_support
                  @ parcel_embedding). Temporal query stays learned.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from speech_decoding.v14.config import D1DecoderConfig


BOS_SENTINEL = -1
_BOS_EMBEDDING_INDEX = 9  # assumes vocab_size == 9; asserted in __init__


class D1Decoder(nn.Module):
    """Readout + prev-embedding + Linear head."""

    def __init__(self, cfg: D1DecoderConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or D1DecoderConfig()
        if cfg.prev_embedding_size != cfg.vocab_size + 1:
            raise ValueError(
                f"prev_embedding_size {cfg.prev_embedding_size} must equal "
                f"vocab_size + 1 (= {cfg.vocab_size + 1})"
            )
        if cfg.vocab_size != 9:
            raise ValueError(f"D1Decoder assumes vocab_size=9, got {cfg.vocab_size}")
        _MODES = ("mean_pool", "cls", "hierarchical", "hierarchical_atlas")
        if cfg.readout_mode not in _MODES:
            raise ValueError(
                f"readout_mode must be one of {_MODES}, got {cfg.readout_mode!r}"
            )
        self.d_model = cfg.d_model
        self.vocab_size = cfg.vocab_size
        self.readout_mode = cfg.readout_mode
        self.n_cells = cfg.n_cells
        self.t_tokens = cfg.t_tokens
        self.prev_emb = nn.Embedding(cfg.prev_embedding_size, cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

        if cfg.readout_mode in ("hierarchical", "hierarchical_atlas"):
            self.q_temporal = nn.Parameter(torch.empty(cfg.d_model))
            nn.init.normal_(self.q_temporal, std=1.0 / math.sqrt(cfg.d_model))
        if cfg.readout_mode == "hierarchical":
            self.q_cell = nn.Parameter(torch.empty(cfg.d_model))
            nn.init.normal_(self.q_cell, std=1.0 / math.sqrt(cfg.d_model))

    def _readout(
        self,
        memory: torch.Tensor,
        cell_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce `(B, S, d)` → `(B, d)` per `readout_mode`.

        ``cell_query`` shape `(B, n_cells, d)` is required for
        ``hierarchical_atlas`` and ignored otherwise.
        """

        if self.readout_mode == "mean_pool":
            return memory.mean(dim=1)
        if self.readout_mode == "cls":
            return memory[:, 0]
        # hierarchical or hierarchical_atlas
        B, S, d = memory.shape
        expected = self.n_cells * self.t_tokens
        if S != expected:
            raise ValueError(
                f"hierarchical readout expects S = n_cells*t_tokens = "
                f"{expected}, got S={S}"
            )
        m = memory.view(B, self.n_cells, self.t_tokens, d)
        scale = 1.0 / math.sqrt(d)
        scores_t = (m * self.q_temporal).sum(dim=-1) * scale   # (B, C, T)
        w_t = F.softmax(scores_t, dim=-1).unsqueeze(-1)         # (B, C, T, 1)
        cell_vec = (w_t * m).sum(dim=2)                         # (B, C, d)

        if self.readout_mode == "hierarchical":
            scores_c = (cell_vec * self.q_cell).sum(dim=-1) * scale  # (B, C)
        else:  # hierarchical_atlas
            if cell_query is None:
                raise ValueError(
                    "readout_mode='hierarchical_atlas' requires cell_query "
                    "(B, n_cells, d) to be passed to decoder.forward()"
                )
            if cell_query.shape != (B, self.n_cells, d):
                raise ValueError(
                    f"cell_query shape {tuple(cell_query.shape)} != "
                    f"({B}, {self.n_cells}, {d})"
                )
            scores_c = (cell_vec * cell_query).sum(dim=-1) * scale  # (B, C)
        w_c = F.softmax(scores_c, dim=-1).unsqueeze(-1)              # (B, C, 1)
        return (w_c * cell_vec).sum(dim=1)                           # (B, d)

    def forward(
        self,
        memory: torch.Tensor,       # (B, S, d)
        prev_phoneme: torch.Tensor, # (B,) long in {-1, 0..8}
        cell_query: torch.Tensor | None = None,  # (B, n_cells, d) for hier_atlas
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

        mem_pooled = self._readout(memory, cell_query)  # (B, d)

        idx = torch.where(
            prev_phoneme == BOS_SENTINEL,
            torch.full_like(prev_phoneme, _BOS_EMBEDDING_INDEX),
            prev_phoneme,
        )
        prev_vec = self.prev_emb(idx)  # (B, d)

        q = mem_pooled + prev_vec
        return self.head(q)
