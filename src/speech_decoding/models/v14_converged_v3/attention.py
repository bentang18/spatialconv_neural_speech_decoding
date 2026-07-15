"""v14_converged_v3 — L1 / L2 attention blocks (Phase 4b).

Two disjoint mixers, pre-norm transformer blocks (memo project-v14-converged-v3-
sensor-architecture):

  L1Block  within-sensor JOINT spatiotemporal SA. Tokens are gathered into the
           padded (n_shafts, max_c) grid; each shaft attends jointly over its
           (contact, time) pairs, block-diagonal across shafts. Q/K carry the
           L1 rotary (index depth + time). Pad slots are excluded by a key
           mask ⇒ the varlen block-diagonal is exact.

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
from speech_decoding.models.v14_converged_v3.packing import PackPlan
from speech_decoding.models.v14_converged_v3.pe import L1RoPE
from speech_decoding.models.v14_converged_v3.varlen import (
    _reference_block_diag,
    njt_block_diag,
    sdpa_block_diag_packed,
)

LN_EPS = 1e-6  # v14 convention (v14_encoder.LN_EPS)
NEG_INF_MASK = -1e4  # v14 convention: finite (not -inf) → all-masked rows go uniform


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
        self, d_model: int, n_heads: int, *, mlp_ratio: int = 4
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # qkv/out biases ON to match upstream V-JEPA 2 (qkv_bias=True, proj bias;
        # vision_transformer.py factories). No QK-norm: upstream ViT-B omits it,
        # and a per-element head_dim gain rotates inconsistently across each RoPE
        # pair (breaks relative-covariance) while being WD-exempt (1-D) — worse
        # than none. Logit runaway is unlikely at d256/ViT-B scale; if it appears
        # at the 10x LR the fix is a pair-shared (not per-element) RMSNorm.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.rope = L1RoPE(self.head_dim)
        self.norm1 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.norm2 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.mlp = _MLP(d_model, mlp_ratio)

    def _attn(self, x: Tensor, geom: L1Geometry, visible: Tensor | None) -> Tensor:
        # x: (B, N, T, d) → gather to (B, S, C, T, d), attend within each shaft.
        # visible: (B, N, T) bool CELL mask, True = keepable key (masked cells excluded
        # so they never leak into the visible latents). None → all keepable. Dual-axis
        # (Ben 2026-07-12): the key mask is per (contact, frame), not per contact.
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

        # index-RoPE coordinate = depth (broadcast over T); time coord = arange(T)
        # (broadcast over C). Token order is contact-major: token = c*T + t.
        idx = geom.depth[:, :, None].expand(S, C, T).reshape(S, seq)  # (S, seq)
        tt = torch.arange(T, device=x.device)[None, None, :].expand(S, C, T).reshape(S, seq)
        idx = idx[None].expand(B, S, seq).reshape(B * S, seq)
        tt = tt[None].expand(B, S, seq).reshape(B * S, seq)
        q, k = self.rope(q, k, idx, tt)

        # key mask: pad slots (invalid contacts) always excluded; masked CELLS excluded
        # when `visible` given. valid (B,S,C) broadcasts over T; the cell mask is gathered
        # into the (S,C,T) grid and ANDed. → (B*S, seq), token order c*T + t.
        key_ok = geom.valid[None, :, :, None].expand(B, S, C, T)  # (B, S, C, T) bool
        if visible is not None:
            key_ok = key_ok & visible[:, geom.gather_idx, :]  # (B,S,C,T) cell vis in grid
        key_ok = key_ok.reshape(B * S, seq)
        # Additive mask (NEG, not -inf): a fully-excluded row (a whole-shaft-masked
        # query) softmaxes to uniform → finite garbage we discard, never NaN.
        attn_bias = torch.where(key_ok, 0.0, NEG_INF_MASK).view(B * S, 1, 1, seq)

        # Unify dtypes right before SDPA (audit L11). RoPE ran in fp32 (cos/sin are
        # fp32 buffers ⇒ q,k promote) while v stays bf16 under autocast, and the bias
        # is fp32. A strict/flash SDPA build requires q,k,v AND attn_mask to share the
        # query dtype; the math backend tolerates the split but production flash won't.
        # Down-cast to v's dtype (keeps rope precise, matches flash semantics; a no-op
        # under fp32 where every tensor is already fp32).
        q, k = q.to(v.dtype), k.to(v.dtype)
        attn_bias = attn_bias.to(v.dtype)

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

    # ── packed (varlen) path (#24) — the PRODUCTION path (towers run packed) ──
    # ``_attn`` above is the padded ORACLE: it stays the CPU-testable reference the
    # packed path is pinned against (test_attention). The packed path avoids the
    # ``max_c`` zero-pad entirely — it attends over the flat, shaft-grouped tokens
    # the pack plan lays out, block-diagonal via ``cu_seqlens`` (flash on GPU, the
    # block-diagonal SDPA reference on CPU). Same weights, same RoPE, same key set
    # per shaft ⇒ per-contact-identical output at every present contact.
    def _attn_packed(
        self,
        x: Tensor,
        plan: PackPlan,
        *,
        backend: str,
        rope_cs: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        # x: (B, P, T, d) packed, shaft-grouped. Token order = (clip, slot, time),
        # matching plan.cu_seqlens (B*S segments, token units, b-major).
        B, P, T, d = x.shape
        total = B * P * T
        qkv = self.qkv(x.reshape(total, d)).reshape(total, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)  # (total, H, hd)

        # RoPE coord per packed token: index = clinical depth (per slot, over T),
        # time = arange(T) (per slot). Flat (clip, slot, time) order. ``rope_cs`` (the
        # B5 hoist) is the tower-precomputed (cos, sin) table shared across all L1
        # blocks — identical to recomputing here (same idx/t, same fp32 table), so
        # when supplied we skip both the idx/t build and the cos/sin math.
        qh, kh = q.transpose(0, 1), k.transpose(0, 1)  # (H, total, hd) — seq=total
        if rope_cs is None:
            idx = plan.depth[:, :, None].expand(B, P, T).reshape(total)  # (total,)
            # time coord = the REAL kept-frame index per token (plan.time_idx), NOT
            # arange — dual-axis drops frames, so slot k of a shaft is its k-th kept
            # real frame; RoPE must see the true frame separations (Ben's catch).
            tt = plan.time_idx.reshape(total)
            qh, kh = self.rope(qh, kh, idx, tt)
        else:
            qh, kh = self.rope.rotate(qh, kh, rope_cs[0], rope_cs[1])
        q, k = qh.transpose(0, 1), kh.transpose(0, 1)  # (total, H, hd)
        # RoPE promotes q,k to fp32 (cos/sin fp32 buffers); align to v for the SDPA
        # (strict/cuDNN needs q,k,v same dtype). No-op under fp32.
        q, k = q.to(v.dtype), k.to(v.dtype)

        # GPU: ragged flash-varlen over cu_seqlens — no pad, no mask (~3× on attention
        # fwd+bwd; the mask-forced mem-efficient kernel was ~76% of the step). Whole-
        # masked shafts are DROPPED (zero-length segments), not padded. Runs eager
        # (dynamo.disable) — the NestedTensor subclass tangent breaks torch-2.10
        # compiled autograd; SDPA is a graph break, projections/RoPE stay compiled.
        # CPU keeps the scatter-to-padded oracle (the test reference the GPU path is
        # pinned against; njt is numerically identical to bf16, grad-matched).
        if q.is_cuda:
            ctx = njt_block_diag(
                q, k, v, plan.cu_seqlens_drop, plan.max_seqlen
            )  # (total, H, hd)
        else:
            max_c = plan.max_seqlen // T
            ctx = sdpa_block_diag_packed(
                q, k, v, plan.grid_pos, plan.n_shafts, max_c, T
            )  # (total, H, hd)
        ctx = self.out(ctx.reshape(total, d)).reshape(B, P, T, d)
        return ctx

    def forward_packed(
        self,
        x: Tensor,
        plan: PackPlan,
        *,
        backend: str = "auto",
        rope_cs: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        x = x + self._attn_packed(self.norm1(x), plan, backend=backend, rope_cs=rope_cs)
        x = x + self.mlp(self.norm2(x))
        return x

    # ── flat per-band (r4) path — the SAME varlen kernels over a FULLY flat token
    # set (no (B,P,T) rectangle). r4 tokens are ragged per (contact, band) so there
    # is no max_c×T grid to reshape through; the pack plan (pack_r4.R4Grid) already
    # lays them out flat + shaft-contiguous with per-token RoPE coords. This is the
    # rectangular ``_attn_packed`` with the rectangle removed: qkv → RoPE(depth,
    # time_pos) → block-diagonal SDPA over ``cu_seqlens`` → out. ``depth`` is the L1
    # index coord (clinical contact index), ``time_pos`` the shared-32Hz lattice
    # position; both are per-token (M,), not per-slot.
    def _attn_flat(
        self,
        x_flat: Tensor,
        depth: Tensor,
        time_pos: Tensor,
        cu_seqlens: Tensor,
        cu_seqlens_drop: Tensor,
        max_seqlen: int,
        *,
        rope_cs: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        # x_flat: (M, d). depth, time_pos: (M,) long. Token order = shaft-contiguous
        # (cu_seqlens delimits each shaft block).
        m, d = x_flat.shape
        qkv = self.qkv(x_flat).reshape(m, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)  # (M, H, hd)
        qh, kh = q.transpose(0, 1), k.transpose(0, 1)  # (H, M, hd)
        if rope_cs is None:
            qh, kh = self.rope(qh, kh, depth, time_pos)
        else:
            qh, kh = self.rope.rotate(qh, kh, rope_cs[0], rope_cs[1])
        q, k = qh.transpose(0, 1), kh.transpose(0, 1)  # (M, H, hd)
        q, k = q.to(v.dtype), k.to(v.dtype)  # align to v for SDPA (no-op under fp32)
        if q.is_cuda:
            ctx = njt_block_diag(q, k, v, cu_seqlens_drop, max_seqlen)  # (M, H, hd)
        else:
            ctx = _reference_block_diag(q, k, v, cu_seqlens)  # (M, H, hd)
        return self.out(ctx.reshape(m, d))

    def forward_flat(
        self,
        x_flat: Tensor,
        depth: Tensor,
        time_pos: Tensor,
        cu_seqlens: Tensor,
        cu_seqlens_drop: Tensor,
        max_seqlen: int,
        *,
        rope_cs: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        x_flat = x_flat + self._attn_flat(
            self.norm1(x_flat), depth, time_pos, cu_seqlens, cu_seqlens_drop,
            max_seqlen, rope_cs=rope_cs,
        )
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat


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
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # qkv/out biases ON to match upstream V-JEPA 2 (qkv_bias=True, proj bias).
        # No QK-norm (see L1Block).
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.norm1 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.norm2 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.mlp = _MLP(d_model, mlp_ratio)

    def _attn(self, x: Tensor, visible: Tensor | None) -> Tensor:
        # x: (B, N, T, d). Attend over N at each time slice (T folded into batch).
        # visible: (B, N, T) bool CELL mask — masked cells excluded as keys, per real t.
        B, N, T, d = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(B * T, N, d)  # (B*T, N, d)
        qkv = self.qkv(xt).reshape(B * T, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B*T, N, H, hd)
        q = q.transpose(1, 2)  # (B*T, H, N, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_bias = None
        if visible is not None:
            # (B, N, T) → keys per real frame t: (B*T, 1, 1, N).
            key_ok = visible.permute(0, 2, 1).reshape(B * T, N)  # (B*T, N) cell vis at t
            # match query dtype for strict/flash SDPA (audit L11; no-op under fp32).
            attn_bias = torch.where(key_ok, 0.0, NEG_INF_MASK).view(B * T, 1, 1, N).to(q.dtype)

        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        ctx = ctx.transpose(1, 2).reshape(B * T, N, d)
        ctx = self.out(ctx).reshape(B, T, N, d).permute(0, 2, 1, 3)
        return ctx

    def forward(self, x: Tensor, visible: Tensor | None = None) -> Tensor:
        x = x + self._attn(self.norm1(x), visible)
        x = x + self.mlp(self.norm2(x))
        return x

    # ── packed path (#24) ── x carries the selected contacts × kept frames. Two
    # regimes, chosen by whether the plan carries the L2 real-frame regroup:
    #
    #   FULL GRID (teacher/predictor, ``plan.perm_l2 is None``): no time masking ⇒
    #     the kept-frame slot k IS real frame k for every shaft, so cross-sensor
    #     same-t attention = one dense SDPA over the P contacts at each slot t. Masked
    #     contacts are physically absent (permutation-invariant over keys ⇒ matches the
    #     padded oracle).
    #
    #   ONLINE (dual-axis encoder, ``plan.perm_l2`` set): heterogeneous time masking ⇒
    #     slot k is a DIFFERENT real frame per shaft, so "same-t" needs a genuine
    #     regroup. ``perm_l2`` reorders the flat visible-cell tokens into time-major
    #     order; ``cu_l2`` marks each (clip, real-frame) block; the SAME block-diagonal
    #     primitive (njt on GPU / reference on CPU) runs one attention per real frame,
    #     no RoPE. The ≥1-live-sensor guarantee keeps every frame block non-empty. The
    #     result is scattered back to (slot, kept-frame) order. Exact same-t, njt-fast.
    def _attn_packed(self, x: Tensor, plan: PackPlan, *, backend: str) -> Tensor:
        B, P, T, d = x.shape
        if plan.perm_l2 is None:
            xt = x.permute(0, 2, 1, 3).reshape(B * T, P, d)  # (B*T, P, d)
            qkv = self.qkv(xt).reshape(B * T, P, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B*T,H,P,hd)
            ctx = F.scaled_dot_product_attention(q, k, v)
            ctx = ctx.transpose(1, 2).reshape(B * T, P, d)
            return self.out(ctx).reshape(B, T, P, d).permute(0, 2, 1, 3)

        # ── real-frame regroup (online) ──────────────────────────────────────
        tot = B * P * T
        perm = plan.perm_l2  # (B, P*T) flat visible-cell → time-major position
        xf = x.reshape(B, P * T, d)
        xtm = xf.gather(1, perm[:, :, None].expand(B, P * T, d))  # (B, P*T, d) time-major
        qkv = self.qkv(xtm.reshape(tot, d)).reshape(tot, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)  # (tot, H, hd) — no RoPE on L2
        if q.is_cuda:
            ctx = njt_block_diag(q, k, v, plan.cu_l2_drop, plan.max_seqlen_l2)
        else:
            ctx = _reference_block_diag(q, k, v, plan.cu_l2)  # (tot, H, hd)
        ctx = self.out(ctx.reshape(tot, d)).reshape(B, P * T, d)  # time-major
        back = ctx.new_zeros(B, P * T, d).scatter_(
            1, perm[:, :, None].expand(B, P * T, d), ctx
        )  # → (slot, kept-frame) order
        return back.reshape(B, P, T, d)

    def forward_packed(self, x: Tensor, plan: PackPlan, *, backend: str = "auto") -> Tensor:
        x = x + self._attn_packed(self.norm1(x), plan, backend=backend)
        x = x + self.mlp(self.norm2(x))
        return x
