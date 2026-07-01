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

Block (V-JEPA CrossAttentionBlock, one layer): keys/values are PRE-NORMED —
``attn(query, LN_in(x))`` — then ``z = query + attn`` → ``z = z + MLP(LN(z))`` →
``LN`` (final norm, pre-head) → mean over queries → linear. Two scale notes vs textbook
MHA, both consequences of the fold: the score ``⟨phi^h, x_t⟩`` is a FULL-d dot (each head
a free ``R^d`` functional, not a ``d/H`` slice), so the temperature is ``1/√d_model`` (the
variance-matching scale for a d-dim dot), NOT ``1/√d_head``. Values ARE per-head standard. Regularizers (all config, swept offline): attention-weight / MLP / residual
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
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        parcel_dropout: float = 0.0,
        token_dropout: float = 0.0,
        tokens_per_parcel: int = 0,
        n_time_frames: int = 0,
        use_mlp: bool = True,
        n_out: int = 1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.head_dim = d_model // n_heads
        self.parcel_dropout = float(parcel_dropout)
        self.token_dropout = float(token_dropout)
        self.tokens_per_parcel = int(tokens_per_parcel)
        # use_mlp gates the FFN residual block — the ONLY nonlinearity on the pooled
        # summary. Off ⇒ attn-pool → linear head (the matched linear baseline); on ⇒
        # attn-pool → MLP → linear (full nonlinear). Same attn-pool + time-tag either way,
        # so the pair isolates the value of nonlinear readout depth.
        self.use_mlp = bool(use_mlp)
        # Learnable TIME positional tag over the S axis (frame = token_idx % S; the k seeds
        # share a frame's tag, parcels repeat it). The one positional axis not in content
        # (freq folds into d, parcel is tagged, the k seeds differ in content). Off (=0)
        # keeps the pool permutation-invariant; on makes it position-aware like keep-S.
        self.n_time_frames = int(n_time_frames)
        if self.n_time_frames > 0:
            self.time_embed = nn.Embedding(self.n_time_frames, d_model)
            nn.init.trunc_normal_(self.time_embed.weight, std=0.02)

        # phi[q,h,:] — the folded scoring query (W_q∘W_k), scores the FULL token.
        self.phi = nn.Parameter(torch.empty(n_queries, n_heads, d_model))
        nn.init.trunc_normal_(self.phi, std=0.02)
        # the residual base (the query token's residual role).
        self.query = nn.Parameter(torch.zeros(n_queries, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)

        # Pre-norm on keys/values (V-JEPA ``xattn(q, LN(X))``): the frozen encoder taps
        # arrive unnormalized (no terminal LN on the tap), so normalize the token set once
        # and score/value off it. Shared by both heads (linear + nonlinear), so it does not
        # perturb the matched comparison.
        self.ln_in = nn.LayerNorm(d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(residual_dropout)

        self.ln2 = nn.LayerNorm(d_model)
        if self.use_mlp:
            self.ln1 = nn.LayerNorm(d_model)
            hidden = int(round(d_model * mlp_ratio))
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(hidden, d_model),
                nn.Dropout(mlp_dropout),
            )
        self.head = nn.Linear(d_model, n_out)

    def _token_mask(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None,
        token_parcel_ids: Tensor | None = None,
    ) -> Tensor | None:
        """Combine the (B,T) valid mask with train-time dropout; never leaves a row empty.

        PARCEL dropout (primary) drops every token of a randomly chosen parcel together —
        the augmentation that mimics the real cross-subject shift (a parcel absent because
        a new subject lacks coverage there). Two layouts:
          - regular blocks (M3/M4 parcel taps): ``P`` contiguous ``tokens_per_parcel``-blocks,
            so a per-parcel keep mask repeat-interleaves to ``(B,T)``.
          - irregular groups (``token_parcel_ids`` ``(T,)``, M2 electrode tap): electrodes of
            one anatomical parcel form an arbitrary-size group, so the keep mask is gathered
            by each token's group id. Takes precedence over the block path when supplied.
        TOKEN dropout (optional, off by default) drops i.i.d. single tokens — a frame-level
        absence that never happens at test time, kept only as an ablation."""
        mask = key_padding_mask
        if not self.training:
            return mask
        B, T, _ = x.shape
        drop: Tensor | None = None
        tpp = self.tokens_per_parcel
        if self.parcel_dropout > 0.0 and token_parcel_ids is not None:
            uniq, inv = torch.unique(token_parcel_ids.to(x.device), return_inverse=True)
            keep_g = torch.rand(B, uniq.numel(), device=x.device) >= self.parcel_dropout
            drop = keep_g[:, inv]                                  # (B,T) gathered by group id
        elif self.parcel_dropout > 0.0 and tpp > 0 and T % tpp == 0:
            keep_p = torch.rand(B, T // tpp, device=x.device) >= self.parcel_dropout
            drop = keep_p.repeat_interleave(tpp, dim=1)            # (B,T) block mask
        if self.token_dropout > 0.0:
            keep_t = torch.rand(B, T, device=x.device) >= self.token_dropout
            drop = keep_t if drop is None else (drop & keep_t)
        if drop is None:
            return mask
        m = drop if mask is None else (mask & drop)
        empty = m.sum(dim=-1, keepdim=True) == 0                   # dropped a whole row
        fallback = torch.ones_like(m) if mask is None else mask
        return torch.where(empty, fallback, m)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        *,
        token_parcel_ids: Tensor | None = None,
    ) -> Tensor:
        """``x`` ``(B,T,d)`` token set; ``key_padding_mask`` ``(B,T)`` bool, True = valid.

        ``token_parcel_ids`` ``(T,)`` long maps each token to its anatomical parcel for
        irregular-group parcel dropout (M2 electrode tap, where electrodes-per-parcel
        varies); omit for the regular ``tokens_per_parcel``-block path (M3/M4).

        Returns ``(B, n_out)`` logits."""
        B, T, d = x.shape
        x = x.float()   # caches store bf16 (half the host-RAM/IO); compute stays fp32
        if self.n_time_frames > 0:                                    # time positional tag
            pos = torch.arange(T, device=x.device) % self.n_time_frames
            x = x + self.time_embed(pos)[None]                        # (B,T,d), frame=t%S
        xn = self.ln_in(x)                                            # pre-norm K/V: xattn(q,LN(X))
        scale = self.d_model ** -0.5
        scores = torch.einsum("qhd,btd->bqht", self.phi, xn) * scale  # (B,Q,H,T)
        mask = self._token_mask(x, key_padding_mask, token_parcel_ids)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        a = self.attn_drop(scores.softmax(dim=-1))                    # (B,Q,H,T)
        v = self.W_v(xn).reshape(B, T, self.n_heads, self.head_dim)   # (B,T,H,c)
        o = torch.einsum("bqht,bthc->bqhc", a, v).reshape(B, self.n_queries, d)
        attn_out = self.W_o(o)                                        # (B,Q,d)
        z = self.query[None] + self.resid_drop(attn_out)             # +query residual
        if self.use_mlp:
            z = z + self.mlp(self.ln1(z))                            # FFN residual (nonlinear)
        z = self.ln2(z).mean(dim=1)                                  # pool queries → (B,d)
        return self.head(z)
