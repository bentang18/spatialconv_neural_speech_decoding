"""v14_converged_v3 — L1 / L2 attention blocks (Phase 4b).

Two disjoint mixers, pre-norm transformer blocks (memo project-v14-converged-v3-
sensor-architecture):

  L1Block  within-sensor JOINT spatiotemporal SA. Tokens are gathered into the
           padded (n_shafts, max_c) grid; each shaft attends jointly over its
           (contact, time) pairs, block-diagonal across shafts. Q/K carry the
           L1 rotary (index depth + time), QK-norm before RoPE. Pad slots are
           excluded by a key mask ⇒ the varlen block-diagonal is exact.

  L2Block  cross-sensor FACTORIZED SA at same-t. At each time slice the N
           contacts attend across shafts. No RoPE (no shared metric across
           sensors); the sole cross-sensor coordinate is the learned DKT-parcel
           identity, which is added ONCE at the TOWER INPUT (``towers.py`` builds
           ``x + parcel_embed(parcel_id)``) and rides the residual into every
           block — so identity IS present in the residual and in L2's V, not a
           score-space-only term. L2Block itself injects nothing.

Non-redundancy (memo "non-redundant BY CONSTRUCTION") therefore rests on the
DISJOINT PAIR-SETS, not on a V-is-identity-free property: L1 mixes only
within-shaft (block-diagonal) same-shaft (contact,time) pairs; L2 mixes only
cross-shaft same-t pairs. L1's per-pair score stays RoPE-relative on content;
L2's score sees the identity-laden residual. (Audit note L2: same-shaft same-t
pairs are touched by both mixers with independent weights — a bounded overlap,
not strict disjointness; see M2 on identity being absolute + injected-once.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from speech_decoding.models.v14_converged_v3.geometry import L1Geometry
from speech_decoding.models.v14_converged_v3.pe import L1RoPE

LN_EPS = 1e-6  # v14 convention (v14_encoder.LN_EPS)
NEG_INF_MASK = -1e4  # v14 convention: finite (not -inf) → all-masked rows go uniform


class _QKNorm(nn.Module):
    """Per-head RMSNorm over head_dim on Q or K (Wortsman 2023 / Gemma / Qwen3).

    Bounds pre-softmax logits so bf16 matmuls can't run away. Computed in fp32,
    cast back. Applied BEFORE RoPE (per-head norm is rotation-covariant only up
    to its learned gain).
    """

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(head_dim))
        self.eps = LN_EPS

    def forward(self, x: Tensor) -> Tensor:  # x: (..., head_dim)
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.gain.float()).to(dtype)


class _MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int) -> None:
        super().__init__()
        hidden = d_model * mlp_ratio
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class L1Block(nn.Module):
    """Within-shaft joint spatiotemporal attention block (block-diagonal, RoPE)."""

    def __init__(
        self, d_model: int, n_heads: int, *, mlp_ratio: int = 4, qk_norm: bool = True
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # qkv/out biases ON to match upstream V-JEPA 2 (qkv_bias=True, proj bias;
        # vision_transformer.py factories). QK-norm is our only intentional add.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.rope = L1RoPE(self.head_dim)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = _QKNorm(self.head_dim)
            self.k_norm = _QKNorm(self.head_dim)
        self.norm1 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.norm2 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.mlp = _MLP(d_model, mlp_ratio)

    def _attn(self, x: Tensor, geom: L1Geometry, visible: Tensor | None) -> Tensor:
        # x: (B, N, T, d) → gather to (B, S, C, T, d), attend within each shaft.
        # visible: (B, N) bool, True = keepable key (masked electrodes excluded so
        # they never leak into the visible latents). None → all keepable.
        B, N, T, d = x.shape
        S, C = geom.n_shafts, geom.max_c
        gathered = x[:, geom.gather_idx]  # (B, S, C, T, d)
        seq = C * T
        xg = gathered.reshape(B * S, seq, d)

        qkv = self.qkv(xg).reshape(B * S, seq, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B*S, seq, H, hd)
        q = q.transpose(1, 2)  # (B*S, H, seq, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # index-RoPE coordinate = depth (broadcast over T); time coord = arange(T)
        # (broadcast over C). Token order is contact-major: token = c*T + t.
        idx = geom.depth[:, :, None].expand(S, C, T).reshape(S, seq)  # (S, seq)
        tt = torch.arange(T, device=x.device)[None, None, :].expand(S, C, T).reshape(S, seq)
        idx = idx[None].expand(B, S, seq).reshape(B * S, seq)
        tt = tt[None].expand(B, S, seq).reshape(B * S, seq)
        q, k = self.rope(q, k, idx, tt)

        # key mask: pad slots (invalid contacts) always excluded; masked
        # electrodes excluded when `visible` given. (B, S, C) → (B*S, seq).
        key_ok = geom.valid[None].expand(B, S, C)  # (B, S, C) bool
        if visible is not None:
            key_ok = key_ok & visible[:, geom.gather_idx]  # gather vis into grid
        key_ok = key_ok[:, :, :, None].expand(B, S, C, T).reshape(B * S, seq)
        # Additive mask (NEG, not -inf): a fully-excluded row (a whole-shaft-masked
        # query) softmaxes to uniform → finite garbage we discard, never NaN.
        attn_bias = torch.where(key_ok, 0.0, NEG_INF_MASK).view(B * S, 1, 1, seq)

        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        ctx = ctx.transpose(1, 2).reshape(B * S, seq, d)
        ctx = self.out(ctx).reshape(B, S, C, T, d)

        # Allocate the scatter target in ctx's dtype, NOT x's. Under bf16-mixed
        # autocast x = norm1(x) is fp32 (LayerNorm autocasts to fp32) while
        # ctx = self.out(ctx) is bf16; the advanced-index assignment below does NOT
        # promote dtypes (unlike `+`) and would raise "Index put requires source and
        # destination dtypes match". The residual add in forward() promotes back to
        # the fp32 stream. (fp32 tests never hit this — both were fp32.)
        out = ctx.new_zeros(B, N, T, d)
        out[:, geom.gather_idx[geom.valid]] = ctx[:, geom.valid]
        return out

    def forward(
        self, x: Tensor, geom: L1Geometry, visible: Tensor | None = None
    ) -> Tensor:
        x = x + self._attn(self.norm1(x), geom, visible)
        x = x + self.mlp(self.norm2(x))
        return x


class L2Block(nn.Module):
    """Cross-sensor factorized (same-t) attention block.

    Plain same-t cross-shaft self-attention (no RoPE — no shared metric across
    sensors). The parcel/DKT identity is NOT injected here: it is added ONCE to the
    token stream at the tower input (V-JEPA 2.1 modality-embedding style — the
    learned modality embed is added to encoder AND predictor inputs alongside RoPE,
    `methods.tex:141,308`), so it rides the residual into every block's content.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        mlp_ratio: int = 4,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # qkv/out biases ON to match upstream V-JEPA 2 (qkv_bias=True, proj bias).
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = _QKNorm(self.head_dim)
            self.k_norm = _QKNorm(self.head_dim)
        self.norm1 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.norm2 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.mlp = _MLP(d_model, mlp_ratio)

    def _attn(self, x: Tensor, visible: Tensor | None) -> Tensor:
        # x: (B, N, T, d). Attend over N at each time slice (T folded into batch).
        # visible: (B, N) bool — masked electrodes excluded as keys (same t).
        B, N, T, d = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(B * T, N, d)  # (B*T, N, d)
        qkv = self.qkv(xt).reshape(B * T, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B*T, N, H, hd)
        q = q.transpose(1, 2)  # (B*T, H, N, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_bias = None
        if visible is not None:
            # (B, N) → keys at every time slice: (B*T, 1, 1, N).
            key_ok = visible[:, None, :].expand(B, T, N).reshape(B * T, N)
            attn_bias = torch.where(key_ok, 0.0, NEG_INF_MASK).view(B * T, 1, 1, N)

        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        ctx = ctx.transpose(1, 2).reshape(B * T, N, d)
        ctx = self.out(ctx).reshape(B, T, N, d).permute(0, 2, 1, 3)
        return ctx

    def forward(self, x: Tensor, visible: Tensor | None = None) -> Tensor:
        x = x + self._attn(self.norm1(x), visible)
        x = x + self.mlp(self.norm2(x))
        return x
