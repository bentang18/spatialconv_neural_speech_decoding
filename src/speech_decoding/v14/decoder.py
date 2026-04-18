"""3-slot AR phoneme decoder for v14-core (Task C4; `#9` / `#28`).

One AR block: causal self-attention over 3 slot queries + cross-attention
over flattened backbone memory `(N_e × T, d)`, then FFN, then a shared
linear vocab head.

Query construction for slot ``i``:

    q_i = base_query + slot_emb[i] + prev_emb(prev_tokens[:, i])

``prev_tokens`` uses BOS sentinel ``-1`` for slot 0; BOS is a dedicated
learnable parameter (``bos_emb``), *not* a vocab index. Teacher forcing
passes the ground-truth previous tokens; `exhaustive_decode` enumerates
all ``9 ** 3 = 729`` sequences and returns the argmax of the summed
slot log-probs (no greedy, no beam; `#9`).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from speech_decoding.v14.config import BackboneConfig, DecoderConfig


class ARDecoder(nn.Module):
    """One-block causal self-attn + cross-attn decoder with 3 slot queries."""

    def __init__(
        self,
        cfg: DecoderConfig | None = None,
        backbone_cfg: BackboneConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = cfg or DecoderConfig()
        bcfg = backbone_cfg or BackboneConfig()
        if bcfg.d_model != cfg.d_model:
            raise ValueError(
                f"decoder d_model {cfg.d_model} != backbone d_model {bcfg.d_model}"
            )

        self.d_model = cfg.d_model
        self.num_queries = cfg.num_queries
        self.vocab_size = cfg.vocab_size

        self.base_query = nn.Parameter(torch.empty(cfg.d_model))
        self.slot_emb = nn.Parameter(torch.empty(cfg.num_queries, cfg.d_model))
        self.bos_emb = nn.Parameter(torch.empty(cfg.d_model))
        nn.init.normal_(self.base_query, std=0.02)
        nn.init.normal_(self.slot_emb, std=0.02)
        nn.init.normal_(self.bos_emb, std=0.02)
        self.prev_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.norm_self = nn.LayerNorm(cfg.d_model)
        self.self_attn = nn.MultiheadAttention(
            cfg.d_model, bcfg.num_heads, dropout=bcfg.dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(cfg.d_model)
        self.cross_attn = nn.MultiheadAttention(
            cfg.d_model, bcfg.num_heads, dropout=bcfg.dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, bcfg.ffn_hidden),
            nn.GELU(),
            nn.Dropout(bcfg.dropout),
            nn.Linear(bcfg.ffn_hidden, cfg.d_model),
            nn.Dropout(bcfg.dropout),
        )
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def _build_queries(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        if prev_tokens.ndim != 2 or prev_tokens.shape[1] != self.num_queries:
            raise ValueError(
                f"prev_tokens shape must be (B, {self.num_queries}), got "
                f"{tuple(prev_tokens.shape)}"
            )
        B = prev_tokens.shape[0]
        is_bos = prev_tokens == -1
        clamped = prev_tokens.clamp(min=0, max=self.vocab_size - 1)
        prev_e = self.prev_emb(clamped)
        prev_e = torch.where(
            is_bos.unsqueeze(-1),
            self.bos_emb.view(1, 1, -1).to(prev_e.dtype),
            prev_e,
        )
        base = self.base_query.view(1, 1, -1).expand(B, self.num_queries, -1)
        slot = self.slot_emb.view(1, self.num_queries, -1).expand(B, -1, -1)
        return base + slot + prev_e

    def _decode_block(
        self, queries: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        L = queries.shape[1]
        causal = torch.triu(
            torch.ones(L, L, device=queries.device, dtype=torch.bool), diagonal=1
        )
        h = self.norm_self(queries)
        sa, _ = self.self_attn(h, h, h, attn_mask=causal, need_weights=False)
        x = queries + sa
        h = self.norm_cross(x)
        ca, _ = self.cross_attn(h, memory, memory, need_weights=False)
        x = x + ca
        h = self.norm_ffn(x)
        x = x + self.ffn(h)
        return self.head(x)

    def forward_teacher(
        self, memory: torch.Tensor, prev_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-forced forward. Returns logits `(B, num_queries, vocab_size)`."""

        queries = self._build_queries(prev_tokens)
        return self._decode_block(queries, memory)

    def exhaustive_decode(self, memory: torch.Tensor) -> torch.Tensor:
        """Enumerate `vocab_size ** num_queries` sequences per batch item.

        Returns the argmax of the summed slot log-probs as a `(B, num_queries)`
        long tensor.
        """

        if memory.ndim != 3:
            raise ValueError(f"memory must be (B, M, d), got {tuple(memory.shape)}")
        B, _, d = memory.shape
        V = self.vocab_size
        L = self.num_queries
        device = memory.device
        seqs = torch.cartesian_prod(*([torch.arange(V, device=device)] * L))  # (V**L, L)
        n_hyp = seqs.shape[0]

        prev = torch.full((n_hyp, L), -1, dtype=torch.long, device=device)
        prev[:, 1:] = seqs[:, :-1]

        out = torch.empty(B, L, dtype=torch.long, device=device)
        for b in range(B):
            mem_b = memory[b : b + 1].expand(n_hyp, -1, d)
            logits = self.forward_teacher(mem_b, prev)  # (n_hyp, L, V)
            log_probs = F.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(2, seqs.unsqueeze(-1)).squeeze(-1)  # (n_hyp, L)
            best = gathered.sum(dim=1).argmax()
            out[b] = seqs[best]
        return out
