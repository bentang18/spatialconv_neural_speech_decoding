"""V-JEPA-2-style 1-layer attentive probe head for the v2 encoder taps.

A single learnable query (folded) cross-attends over a clip's per-token tap set
(``P·k·S`` tokens), one transformer block, then a linear readout. The
capacity-controlled NONLINEAR analogue of the keep-S ridge: the question is whether a
small nonlinear readout on the M3/M4 surfaces clears the raw_tok ridge floor
(cross-subject ``delta_volume`` 0.666).

Folded single-query attention (exact, fewer params — the locked contract):
  - ``W_q``/``W_k`` fold into per-head query vectors ``phi[h] ∈ R^d``. With ONE query
    each head's score is a fixed linear functional of the token, ``⟨phi^h, x_t⟩``;
    ``phi^h`` free in ``R^d`` is ≥ the original ``W_k``-row-space-constrained
    expressiveness, at ``H·d`` params instead of ``2d²``.
  - ``W_v``/``W_o`` are KEPT. For ``H>1`` the per-head value routing (each head's
    weights ``a_t^h`` pick its own ``W_v^h`` slice, ``W_o`` mixes heads afterward) is a
    sum of ``H`` rank-≤``d/H`` maps that does NOT collapse to one shared matrix. (Only
    ``H=1`` lets ``W_o`` fold into ``W_v``.)
The residual base is a SEPARATE learned query vector ``query ∈ R^d`` — the V-JEPA query
token's residual role; its scoring role is what got absorbed into ``phi``.

Block: ``z = query + attn(x)`` → ``z = z + MLP(LN(z))`` → ``LN`` → mean over queries →
linear. Regularizers (all config, swept offline): attention-weight / MLP / residual
dropout, and token-dropout augmentation (drop a fraction of the input token set per clip
— the set-pool analogue of the SSL mask). The head is permutation-invariant over the
token axis by construction, so the parcel id must ride IN each token (it does: M3 tap is
``pool + latent.parcel_embed``, M4 is the latent output)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["AttentiveProbeHead"]


class AttentiveProbeHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int = 6,
        n_queries: int = 1,
        mlp_ratio: float = 2.0,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        token_dropout: float = 0.0,
        n_out: int = 1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.head_dim = d_model // n_heads
        self.token_dropout = float(token_dropout)

        # phi[q,h,:] — the folded scoring query (W_q∘W_k), scores the FULL token.
        self.phi = nn.Parameter(torch.empty(n_queries, n_heads, d_model))
        nn.init.trunc_normal_(self.phi, std=0.02)
        # the residual base (the query token's residual role).
        self.query = nn.Parameter(torch.zeros(n_queries, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(residual_dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(round(d_model * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(mlp_dropout),
        )
        self.head = nn.Linear(d_model, n_out)

    def _token_mask(self, x: Tensor, key_padding_mask: Tensor | None) -> Tensor | None:
        """Combine the (B,T) valid mask with train-time token dropout; never empty."""
        mask = key_padding_mask
        if self.training and self.token_dropout > 0.0:
            B, T, _ = x.shape
            keep = torch.rand(B, T, device=x.device) >= self.token_dropout
            m = keep if mask is None else (mask & keep)
            empty = m.sum(dim=-1, keepdim=True) == 0       # dropped everything in a row
            fallback = torch.ones_like(m) if mask is None else mask
            mask = torch.where(empty, fallback, m)
        return mask

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """``x`` ``(B,T,d)`` token set; ``key_padding_mask`` ``(B,T)`` bool, True = valid.

        Returns ``(B, n_out)`` logits."""
        B, T, d = x.shape
        scale = self.d_model ** -0.5
        scores = torch.einsum("qhd,btd->bqht", self.phi, x) * scale   # (B,Q,H,T)
        mask = self._token_mask(x, key_padding_mask)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        a = self.attn_drop(scores.softmax(dim=-1))                    # (B,Q,H,T)
        v = self.W_v(x).reshape(B, T, self.n_heads, self.head_dim)    # (B,T,H,c)
        o = torch.einsum("bqht,bthc->bqhc", a, v).reshape(B, self.n_queries, d)
        attn_out = self.W_o(o)                                        # (B,Q,d)
        z = self.query[None] + self.resid_drop(attn_out)             # +query residual
        z = z + self.mlp(self.ln1(z))                                # FFN residual
        z = self.ln2(z).mean(dim=1)                                  # pool queries → (B,d)
        return self.head(z)
