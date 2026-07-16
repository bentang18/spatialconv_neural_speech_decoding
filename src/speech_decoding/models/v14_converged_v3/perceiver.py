"""v14_converged_v3 r4 — Perceiver-IO secondary head (B4), WRITE-ONLY.

Three-stage Perceiver-IO (contract project-r4-contract-2026-07-15 §6) that reads the
CONTEXT encoder's deep-sup output and predicts, per present (parcel, 4 Hz slot), the
joint 6-vector parcel-state Gaussian (secondary_head.GaussianStateHead):

  fused tokens ─▶ ENCODE cross-attn ─▶ M·n_slots latents ─▶ L_sa self-attn (process)
                                                              │
              (parcel,slot) queries ─▶ DECODE cross-attn ◀───┘ ─▶ Gaussian head

LATENTS are a BAND-LESS, contact-LESS, single-rate ``(n_slots, M)`` grid: ``M`` LEARNED
mode vectors SHARED across all slots (so the parameter count is slot-count-independent —
clip length is a runtime flag), slot identity carried by time-RoPE at lattice pos 8·slot.
This is the shared cross-parcel regime (M10): the one slow thing HGA/MID/SLOW are views
of. Bands enter as input views (stage 1) and re-emerge INSIDE the 6-vector output (§7).

BATCHING: padded per-sample tensors with masks — the perceiver is a small (λ≈0.2) head
with a fixed ``n_slots·M`` latent-query set per clip, so cross-attention is naturally
dense; padding the variable token set to ``K`` with a key mask is the standard Perceiver
form and CPU-testable. The flat(r4)→padded regroup is the model-assembly's job (#36).

WRITE-ONLY: the perceiver output never flows back into the primary stream (no FiLM /
read-back) — that coupling constraint is enforced at assembly. This module just maps the
encoder tokens → the per-(parcel,slot) Gaussian.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from speech_decoding.models.v14_converged_v3.attention import (
    LN_EPS,
    NEG_INF_MASK,
    _MLP,
)
from speech_decoding.models.v14_converged_v3.pe import (
    ParcelIdentityEmbed,
    TimeRoPE,
    init_transformer_weights,
)
from speech_decoding.models.v14_converged_v3.secondary_head import GaussianStateHead
from speech_decoding.models.v14_converged_v3.towers import ENC_D_MODEL, N_LEVELS

# The encoder's deep-sup output width = concat of the 4 affine-LN'd taps {3,6,9,12}
# (towers.V3Tower forward: cat of n_levels·d) — the SAME [.,1024] the JEPA predictor reads.
D_TAP: int = ENC_D_MODEL * N_LEVELS  # 256·4 = 1024
D_PERC: int = 128  # Ben-locked 2026-07-15 (width is not the bottleneck — M is)
N_PERC_HEADS: int = 4  # head_dim 32 — inherits the predictor head geometry (PRED_N_HEADS)
M_LATENTS: int = 12  # Ben-locked (M10 n@90% = 8.7 spatial modes per slot)
L_SA: int = 6  # Ben-locked (Perceiver-IO deep process module; ablate 6→12 if underfit)
FUSION_HIDDEN: int = 256  # §6: Linear(1024→256)·GELU·Linear(256→d_perc)
SLOT_STRIDE: int = 8  # a 4 Hz slot s sits at 32 Hz lattice pos 8·s


class _CrossAttnBlock(nn.Module):
    """Pre-norm Perceiver cross-attention + MLP. Query stream carries the residual;
    the context (key/value) is normed once. Shared-lattice time-RoPE on q AND k (q·k
    depends on the relative slot/frame offset), optional additive key mask."""

    def __init__(self, d_model: int, n_heads: int, *, mlp_ratio: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.to_q = nn.Linear(d_model, d_model, bias=True)
        self.to_kv = nn.Linear(d_model, 2 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.rope = TimeRoPE(self.head_dim)
        self.norm_q = nn.LayerNorm(d_model, eps=LN_EPS)
        self.norm_kv = nn.LayerNorm(d_model, eps=LN_EPS)
        self.norm2 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.mlp = _MLP(d_model, mlp_ratio)

    def _attn(
        self,
        xq: Tensor,
        xkv: Tensor,
        pos_q: Tensor,
        pos_kv: Tensor,
        key_mask: Tensor | None,
    ) -> Tensor:
        B, Lq, d = xq.shape
        Lk = xkv.shape[1]
        q = self.to_q(xq).reshape(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.to_kv(xkv).reshape(B, Lk, 2, self.n_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)  # (B, H, Lk, hd)
        q, k = self.rope(q, k, pos_q, pos_kv)
        q, k = q.to(v.dtype), k.to(v.dtype)  # align for strict/flash SDPA (no-op fp32)
        bias = None
        if key_mask is not None:
            # (B, Lk) bool → additive (B, 1, 1, Lk); pad keys → NEG so they never leak.
            bias = torch.where(
                key_mask[:, None, None, :], 0.0, NEG_INF_MASK
            ).to(v.dtype)
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        ctx = ctx.transpose(1, 2).reshape(B, Lq, d)
        return self.out(ctx)

    def forward(
        self,
        xq: Tensor,
        xkv: Tensor,
        pos_q: Tensor,
        pos_kv: Tensor,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        xq = xq + self._attn(self.norm_q(xq), self.norm_kv(xkv), pos_q, pos_kv, key_mask)
        xq = xq + self.mlp(self.norm2(xq))
        return xq


class _SelfAttnBlock(nn.Module):
    """Pre-norm latent self-attention + MLP (dense, all-to-all over the latents), with
    time-RoPE by slot — the regime's temporal propagation across the n_slots slots."""

    def __init__(self, d_model: int, n_heads: int, *, mlp_ratio: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.rope = TimeRoPE(self.head_dim)
        self.norm1 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.norm2 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.mlp = _MLP(d_model, mlp_ratio)

    def _attn(self, x: Tensor, pos: Tensor) -> Tensor:
        B, L, d = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B,H,L,hd)
        q, k = self.rope(q, k, pos, pos)
        q, k = q.to(v.dtype), k.to(v.dtype)
        ctx = F.scaled_dot_product_attention(q, k, v)  # every latent is real → no mask
        ctx = ctx.transpose(1, 2).reshape(B, L, d)
        return self.out(ctx)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        x = x + self._attn(self.norm1(x), pos)
        x = x + self.mlp(self.norm2(x))
        return x


class PerceiverHead(nn.Module):
    """Three-stage Perceiver-IO secondary head → per-(parcel,slot) Gaussian state.

    ``n_slots`` (= n_time // 8, the clip's 4 Hz slot count) is a FORWARD argument, not a
    constructor param: the M mode vectors are shared across slots (slot identity is RoPE),
    so nothing about the module's parameters depends on clip length.
    """

    def __init__(
        self,
        *,
        n_parcels: int,
        d_tap: int = D_TAP,
        d_perc: int = D_PERC,
        n_heads: int = N_PERC_HEADS,
        m_latents: int = M_LATENTS,
        l_sa: int = L_SA,
    ) -> None:
        super().__init__()
        self.d_perc = d_perc
        self.m_latents = m_latents
        # own fusion (§6): SEPARATE weights from JEPA's enc_to_pred, no LN around it
        # (the taps are already affine-LN'd inside the encoder's norms_block).
        self.fusion = nn.Sequential(
            nn.Linear(d_tap, FUSION_HIDDEN),
            nn.GELU(),
            nn.Linear(FUSION_HIDDEN, d_perc),
        )
        # M learned mode vectors, SHARED across all slots (slot identity = time-RoPE).
        self.latents = nn.Parameter(torch.empty(m_latents, d_perc))
        self.encode = _CrossAttnBlock(d_perc, n_heads)
        self.process = nn.ModuleList(
            _SelfAttnBlock(d_perc, n_heads) for _ in range(l_sa)
        )
        self.decode = _CrossAttnBlock(d_perc, n_heads)
        # decode-query CONTENT = parcel identity (NOT a near-zero additive prior — it IS
        # the query seed, so init at the normal 0.02 so queries are distinguishable).
        self.parcel_emb = ParcelIdentityEmbed(n_parcels, d_perc, init_std=0.02)
        self.head = GaussianStateHead(d_perc)
        self.apply(init_transformer_weights)  # V-JEPA trunc_normal(0.02); skips Embedding
        nn.init.trunc_normal_(self.latents, std=0.02)

    def forward(
        self,
        z_tokens: Tensor,
        token_time: Tensor,
        token_mask: Tensor | None,
        query_parcel: Tensor,
        query_slot: Tensor,
        *,
        n_slots: int,
        noise: Tensor | None = None,
        return_latents: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """z_tokens (B,K,d_tap) padded visible-token deep-sup features; token_time (B,K)
        long 32 Hz-lattice frame index; token_mask (B,K) bool (True=real) or None when
        every token is real (exact masking ⇒ no padding, so the objective passes None and
        the encode cross-attention skips the identically-zero key-bias). Decode queries
        query_parcel (B,Q) long, query_slot (B,Q) long (0..n_slots-1). ``noise`` (B,Q,6)
        optional per-query PD floor (the count-dependent floor keyed on each query's parcel
        electrode count — the objective owns the n_elec map and passes it through to the
        head). Returns (mu (B,Q,6), cov (B,Q,6,6)) — the joint Gaussian per (parcel, slot).
        Presence masking + NLL are the objective's job (B3); this returns a Gaussian at
        every query. ``return_latents`` (monitor cadence only) additionally returns the
        processed latent bank ``lat`` (B, n_slots·M, d_perc) — post-``process``, pre-decode —
        for the representation-health monitor; it is a read-only view, the caller detaches.
        """
        B = z_tokens.shape[0]
        dev = z_tokens.device
        h = self.fusion(z_tokens)  # (B, K, d_perc)

        # latents (B, n_slots·M, d) with slot-major order l = s·M + m; RoPE pos = 8·s.
        lat = self.latents[None].expand(n_slots, self.m_latents, self.d_perc)
        lat = lat.reshape(n_slots * self.m_latents, self.d_perc)
        lat = lat[None].expand(B, n_slots * self.m_latents, self.d_perc)
        slot_of_lat = torch.arange(n_slots, device=dev).repeat_interleave(self.m_latents)
        lat_pos = slot_of_lat * SLOT_STRIDE  # (n_slots·M,) — shared across batch

        # Stage 1 — ENCODE: latents cross-attend the fused tokens (key-masked).
        lat = self.encode(lat, h, lat_pos, token_time, key_mask=token_mask)
        # Stage 2 — PROCESS: L_sa latent self-attention blocks.
        for blk in self.process:
            lat = blk(lat, lat_pos)
        # Stage 3 — DECODE: (parcel,slot) queries read the latents (no key mask).
        q = self.parcel_emb(query_parcel)  # (B, Q, d_perc)
        q_pos = query_slot * SLOT_STRIDE  # (B, Q)
        dec = self.decode(q, lat, q_pos, lat_pos, key_mask=None)  # (B, Q, d_perc)
        mu, cov = self.head(dec, noise=noise)
        if return_latents:
            return mu, cov, lat
        return mu, cov
