"""v14 Perceiver-IO encoder with parcel-id-tagged latents.

References (load order — later wins on conflict):
    ``memory/project_v14_arch_revision_2026_05_19_v3.md`` (v3 base lock)
    ``memory/project_v14_arch_post_v3_amendment_2026_05_19.md`` (v4 — latents
       keep time axis (K*M, T, d); d=256, ~13M params)
    ``memory/project_v14_imindbench_multistft_pivot_2026_05_22.md`` (5/22 —
       Multi-STFT front-end; Phase-3/4 readout split)
    ``memory/project_v14_dk_first_pass_2026_05_13.md`` (DK one-hot routing)
    ``memory/project_v14_loss_design_amendment_b28_2026_05_27.md`` (B28 —
       cross-attn collapsed to single block at position 0 per Perceiver IO
       canonical; ``cross_attn_positions=[0, 3]`` retained as sister flag for
       ``R-perceiver-original-2-cross-attns``)

Contract
--------
Input:
    electrode_tokens : (B, C_max, T_bins, F_bins)  float32
    support          : (B, C_max, K_parcels)       float32 in [0, 1]
    valid_mask       : (B, C_max)                  bool  — True for real, False for pad

Output (encoder):
    parcel_latents   : (B, K_parcels * M, T_bins, d_model) float32

Output (Phase-4 wrapper):
    logits           : (B, n_classes)              float32

Pipeline (B20 v4-invisible-frontend lock 2026-05-24; FE-01..04;
B28 cross-attn collapse 2026-05-27 PM):
    Conv2d(1, d, kernel=(5, 2), stride=(5, 2)) non-overlap patch stem  [FE-02]
      (kernel_freq 3→5 under FE-RAW-1, F=50 raw |STFT| 2026-06-04;
       fbank F=30 sister keeps kernel_freq=3)
      + per-patch freq embedding (F_p=10 d-vectors, additive)           [FE-03]
      + NO time PE at input (RoPE in ❷)
    → (B, C, F_p, T_p, d)   where F_p = floor(F/5)=10, T_p = floor(T/2)
    → N=6 JOINT (t_p·f_p) token blocks per electrode (RoPE on time-axis [FE-04]
      only, hard cross-electrode mask via batch-dim isolation):
      single multi-head SA over flat (F_p·T_p) tokens per electrode
    → (B, C, F_p, T_p, d) preserved (SA flat over F_p·T_p tokens)
    → broadcast K*M parcel-id-tagged latents to (B, K*M, T_p, d)        [LAT-01]
    → per-time-patch cross-attn @ ``cross_attn_positions`` (default
      ``[0]``, B28 Perceiver-IO canonical) with K/V = (C·F_p) tokens and a
      HARD block-diagonal per-parcel pool: parcel-slot l attends ONLY to its
      own DK parcel's electrodes (one-hot ``support``), off-parcel weight is
      exactly 0 (B36 2026-06-01 replaced the soft λ_anat·log(support+ε) bias)
      (position 0 always runs; depth=0 OK)                              [T1.1, T1.3]
    → factorized latent stack: depth=6 blocks of (t_p-SA RoPE × parcel-SA
      × FFN); additional cross-attns fire pre-block at each interior
      position in ``cross_attn_positions`` (``R-perceiver-original-2-
      cross-attns`` sister uses ``[0, 3]``)                              [T1.9, T1.3]
    → (B, L, T_p, d) full parcel×time field carried to the readout
    → Phase-4 readout (B35 2026-05-31,
      [[project_v14_b35_p4_frozen_pma_mean_linear_2026_05_31]]): the frozen
      P3-PMA collapses the parcel/slot axis → (B, T_p, d), then
      mean-over-time → (B, d) → per-task Linear (V14PmaReadout, default
      readout="pma_mean_linear"). Only the linear trains at P4 (PMA and
      encoder frozen). Reverts B34's transient drop-PMA / per-task-
      attentive-query readout, which isn't trainable at Neuroprobe's
      ≤3500-sample/task budget; the attentive query (readout="attentive")
      and flatten (readout="pma_flatten_linear") survive as deferred
      sisters.
"""

from __future__ import annotations

import math
import typing as tp
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from neuraltrain.models.base import BaseModelConfig

from speech_decoding.studies.braintreebank.anatomy import (
    DEFAULT_SUPPORT_BIAS_EPS,
)


def _rope_freqs(head_dim: int, max_seq_len: int, base: float = 10_000.0) -> Tensor:
    """Pre-compute RoPE cos/sin tables of shape ``(max_seq_len, head_dim)``.

    Layout: pairs ``(d_2k, d_2k+1)`` rotate together. Returned tensor packs
    cos in the first half (along last dim) and sin in the second half so a
    single index read at inference time grabs both.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half).float() / half))
    pos = torch.arange(max_seq_len).float()
    angles = torch.outer(pos, inv_freq)  # (T, half)
    cos = angles.cos().repeat_interleave(2, dim=-1)  # (T, head_dim)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    return torch.stack([cos, sin], dim=0)  # (2, T, head_dim)


# Sentinel used by masked_fill on attention logits. `torch.finfo(bf16).min`
# (~-3.4e38) leaves softmax leakage at masked positions because bf16's
# mantissa precision makes `exp(-3.4e38)` non-zero after rescaling — about
# 4.5e-5 per masked position. -1e4 is large enough to drive `softmax` <1e-6
# at masked positions across bf16/fp16/fp32 without overflow. See blocker
# IE12 in docs/neuroprobe/v14_blockers.md.
NEG_INF_MASK_VALUE: float = -1e4

# Renormalisation floor for the hand-rolled ``pool_weights`` path. After the
# post-softmax ``masked_fill(0)`` zeroes off-parcel weights, the on-parcel row
# is divided by its own sum so it totals exactly 1.0. This corrects the
# residual sum-rounding a bf16 softmax leaves on the *attended* weights — it is
# a no-op in fp32 / under autocast-fp32, where the row already sums to 1. A
# no-coverage parcel's row sums to 0 → ``clamp_min`` keeps the division finite
# (the row stays all-zero, no NaN). On-parcel sums are ≈1.0, so this tiny floor
# never rescales a covered row. ``forward`` pools via SDPA (softmax over the
# attended set only) and needs no renormalisation.
_POOL_RENORM_EPS: float = 1e-6


def _apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    """Apply RoPE to ``x`` of shape ``(..., T, head_dim)`` using cached table."""
    cos, sin = rope[0], rope[1]  # (T, head_dim) each
    T = x.shape[-2]
    cos = cos[:T]
    sin = sin[:T]
    x_paired = x.unflatten(-1, (-1, 2))
    x_rotated = torch.stack(
        [-x_paired[..., 1], x_paired[..., 0]], dim=-1
    ).flatten(-2)
    return x * cos + x_rotated * sin


class _MultiHeadCrossAttention(nn.Module):
    """Latent ← electrode cross-attn with a hard block-diagonal parcel pool.

    B36 (2026-06-01, ``project_v14_b36_perparcel_pool_structured_jepa``):
    the soft additive ``λ_anat·log(support+ε)`` Graphormer bias is replaced
    by a hard one-hot DK assignment mask. Parcel-slot query ``l`` attends
    ONLY to the electrode-freq-patch tokens of its own DK parcel
    (``key_mask[b, l, n] == True``). ``forward`` pools via
    ``F.scaled_dot_product_attention`` with an *additive* ``NEG_INF_MASK_VALUE``
    (-1e4) bias at off-parcel keys (NOT a boolean mask): off-parcel
    ``exp(score - 1e4)`` underflows to **exactly** 0 in every dtype, so their
    weight is exactly 0, and no ``(B, H, L, N)`` score matrix is materialised
    (the masked SDPA routes to the mem-efficient backend under autocast
    bf16-mixed — the same OOM the latent-SA path dodges via SDPA). The finite
    sentinel (vs a boolean mask's true -inf) is load-bearing: a no-coverage
    parcel's row is *fully* blocked, and a -inf row would softmax over an empty
    set → NaN in BOTH forward and backward (the latter poisons q/k/v grads even
    when the forward output is zeroed: 0·NaN = NaN). The -1e4 sentinel makes a
    fully-blocked row a *uniform* finite softmax instead, NaN-free on every
    backend; ``forward`` then zeroes those rows → zero context → the residual
    leaves the latent untouched (and a zero upstream grad × finite jacobian
    gives the slot exactly-zero grad), matching ``_compute_latent_valid``
    downstream. ``pool_weights`` mirrors the same partition with an explicit
    masked softmax (off-parcel masked to ``NEG_INF_MASK_VALUE`` then zeroed,
    on-parcel renormalised to sum 1) for the collapse-monitor / test path,
    which needs the weights SDPA cannot return.

    The cross-attn is strict per-time-patch (the caller batches ``T_p`` into
    the batch dim), so there is no RoPE on the keys — temporal position is
    carried by the joint token blocks and the latent time-SA.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _validate(self, latents: Tensor, electrodes: Tensor, key_mask: Tensor) -> None:
        B, L, _ = latents.shape
        N = electrodes.shape[1]
        if key_mask.shape != (B, L, N):
            raise ValueError(
                f"key_mask shape {tuple(key_mask.shape)} does not match "
                f"(B, L, N) = ({B}, {L}, {N})"
            )
        if key_mask.dtype != torch.bool:
            raise TypeError(
                f"key_mask dtype must be torch.bool; got {key_mask.dtype}"
            )

    def pool_weights(
        self,
        latents: Tensor,         # (B, L, d)
        electrodes: Tensor,      # (B, N, d)  N = C * F_p
        key_mask: Tensor,        # (B, L, N) bool — True = on-parcel & kept
    ) -> Tensor:
        """Post-softmax pooling weights ``(B, H, L, N)``: off-parcel == 0
        exactly, on-parcel rows sum to 1, no-coverage rows all-zero.

        Exposed for the WS-A hard-pool test and collapse monitors; ``forward``
        reproduces the same masking inline (recomputing q/k here keeps this
        path read-only and side-effect free)."""
        self._validate(latents, electrodes, key_mask)
        B, L, _ = latents.shape
        N = electrodes.shape[1]
        q = self.q_proj(latents).reshape(B, L, self.n_heads, self.head_dim)
        k = self.kv_proj(electrodes).reshape(
            B, N, 2, self.n_heads, self.head_dim
        ).unbind(dim=2)[0]
        logits = torch.einsum("blhd,bnhd->bhln", q, k) * self.scale
        blocked = (~key_mask).unsqueeze(1)                       # (B, 1, L, N)
        logits = logits.masked_fill(blocked, NEG_INF_MASK_VALUE)
        attn = logits.softmax(dim=-1).masked_fill(blocked, 0.0)
        return attn / attn.sum(dim=-1, keepdim=True).clamp_min(_POOL_RENORM_EPS)

    def forward(
        self,
        latents: Tensor,         # (B, L, d)
        electrodes: Tensor,      # (B, N, d)  N = C * F_p
        key_mask: Tensor,        # (B, L, N) bool — True = on-parcel & kept
    ) -> Tensor:
        self._validate(latents, electrodes, key_mask)
        B, L, _ = latents.shape
        N = electrodes.shape[1]
        q = self.q_proj(latents).reshape(
            B, L, self.n_heads, self.head_dim
        ).transpose(1, 2)                                        # (B, H, L, hd)
        kv = self.kv_proj(electrodes).reshape(B, N, 2, self.n_heads, self.head_dim)
        k, v = (t.transpose(1, 2) for t in kv.unbind(dim=2))     # (B, H, N, hd) each
        # Additive FLOAT mask (0 = attend, NEG_INF_MASK_VALUE = block), NOT a
        # boolean mask. A boolean mask forces SDPA to use true -inf, and a
        # no-coverage parcel's row is FULLY blocked (K=80 is the cross-cohort
        # union; one BT subject covers only ~20 → ~60 empty parcels per
        # subject), so a -inf row would
        # softmax over an empty set → NaN in forward AND backward on CUDA's
        # mem-efficient/flash kernels. Zeroing the forward output (below) does
        # NOT save the backward: 0·NaN = NaN poisons the q/k/v grads. The finite
        # -1e4 sentinel instead makes a fully-blocked row a *uniform* finite
        # softmax, so forward and backward stay finite on every backend; the
        # no-coverage zeroing below then forces the output (and its grad via a
        # zero upstream grad × finite jacobian) to exactly 0. For covered rows
        # exp(score - 1e4) underflows to exactly 0 in fp32/fp16/bf16, so the
        # off-parcel weight is still exactly 0 — matching pool_weights and the
        # project-wide NEG_INF_MASK_VALUE convention (latent-SA mask + the
        # NEG_INF_MASK_VALUE module-constant rationale). No (B, H, L, N) score
        # matrix is materialised (the
        # mem-efficient backend the masked SDPA routes to), dodging the OOM the
        # hand-rolled softmax hits at C=384 × T~130 on 31 GiB GPUs. SDPA's
        # default scale is 1/sqrt(head_dim) == self.scale.
        attn_bias = torch.zeros_like(
            key_mask, dtype=q.dtype
        ).masked_fill_(~key_mask, NEG_INF_MASK_VALUE).unsqueeze(1)  # (B, 1, L, N)
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)  # (B,H,L,hd)
        ctx = ctx.transpose(1, 2).reshape(B, L, -1)              # (B, L, d)
        # No-coverage slot: the uniform softmax above pools mean(v); zero it so
        # the residual leaves the parcel latent untouched — matching
        # pool_weights' renorm-to-zero and _compute_latent_valid. out has
        # bias=False, so out(0) == 0 keeps the row exactly zero.
        no_coverage = ~key_mask.any(dim=-1)                      # (B, L)
        ctx = ctx.masked_fill(no_coverage.unsqueeze(-1), 0.0)
        return self.out(ctx)


class _PlainMultiHeadSelfAttentionRoPE(nn.Module):
    """SA with RoPE applied symmetrically to Q and K (used inside token blocks
    for the time axis). The freq-SA inside the same block uses the plain
    no-RoPE variant — categorical freq embedding carries identity there.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: Tensor, rope: Tensor, *, key_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # x: (B, T, d), rope: (2, T_max, head_dim) packed cos/sin
        # key_mask: (B, T) bool, True = keepable key (B36 C5 freq-patch
        #   exclusion; None → full attention, byte-identical to the no-mask
        #   path). The latent time-SA caller passes None.
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                     # (B, T, H, head_dim) each
        # Move T into a contiguous-last-but-one slot so `_apply_rope` picks it
        # up (it operates on the second-to-last axis). After RoPE, leave the
        # tensors in (B, H, T, head_dim) — the shape SDPA expects.
        q = _apply_rope(q.transpose(1, 2), rope)        # (B, H, T, head_dim)
        k = _apply_rope(k.transpose(1, 2), rope)        # (B, H, T, head_dim)
        v = v.transpose(1, 2)                           # (B, H, T, head_dim)
        attn_mask = None
        if key_mask is not None:
            # (B, T) key-padding → (B, 1, 1, T) bool broadcast over heads and
            # query positions. True = participate (SDPA bool-mask convention).
            attn_mask = key_mask[:, None, None, :].to(torch.bool)
        # SDPA's default scale is 1/sqrt(head_dim), matching ``self.scale``.
        # Routes to FlashAttention / mem-efficient backends under autocast
        # bf16-mixed, so the (B, H, T, T) attn matrix is never materialized
        # — critical because the hand-rolled softmax was upcast to fp32 by
        # PyTorch autocast and OOM'd at C=384 padded electrodes × T~130 on
        # 31 GiB GPUs (see [[feedback_dcc_partition_default_coganlab_gpu_2026_05_29]]).
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        ctx = ctx.transpose(1, 2)                       # (B, T, H, head_dim)
        return self.out(ctx.reshape(B, T, -1))


class _PlainMultiHeadSelfAttention(nn.Module):
    """Self-attn over latents — no positional encoding (parcel-id embedding owns identity).

    Accepts an optional bidirectional ``attn_mask: (B, L, L) bool`` (True =
    allowed). v14-specific extension beyond canonical PerceiverIO/DETR: our
    latents have variable per-subject validity (no-coverage parcels), so we
    must mask invalid latents both as keys/values AND as queries to keep
    covered latents free of no-coverage contamination — preserving the
    zero-per-subject-params claim.

    **B30 lock 2026-05-28** ([[project_v14_anatomy_gated_symmetric_2026_05_28]]):
    bidirectional masking via ``attn_mask`` replaces the pre-B30 key-only
    ``key_padding_mask`` path. Invalid slots fully bypass — they don't
    attend (queries masked) AND aren't attended to (keys masked). When a
    query row is fully masked, softmax over a row of finite
    ``NEG_INF_MASK_VALUE`` (the project convention is ``-1e4``, not
    ``-inf``) produces uniform ``1/L`` attention rather than NaN; we mask
    the attention output to zero post-softmax so the residual
    ``x + 0 = x`` preserves the original value at invalid positions.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,  # (B, L, L) bool — True = allowed
    ) -> Tensor:
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        attn_logits = torch.einsum("blhd,bshd->bhls", q, k) * self.scale
        if attn_mask is not None:
            if attn_mask.shape != (B, L, L):
                raise ValueError(
                    f"attn_mask shape {tuple(attn_mask.shape)} does not match "
                    f"(B, L, L) = ({B}, {L}, {L})"
                )
            if attn_mask.dtype != torch.bool:
                raise TypeError(
                    f"attn_mask dtype must be torch.bool; got {attn_mask.dtype}"
                )
            invalid = ~attn_mask                                  # (B, L, L)
            attn_logits = attn_logits.masked_fill(
                invalid.unsqueeze(1),                              # (B, 1, L, L)
                NEG_INF_MASK_VALUE,
            )
            attn = attn_logits.softmax(dim=-1)
            # B30 bypass: when a query row is fully masked, softmax over
            # a finite ``NEG_INF_MASK_VALUE`` produces uniform 1/L (not
            # NaN, since the mask value is finite). Zero those rows so
            # the SA output at invalid queries is exactly 0 and the
            # residual leaves them unchanged.
            all_masked = invalid.all(dim=-1, keepdim=True)         # (B, L, 1)
            attn = attn.masked_fill(all_masked.unsqueeze(1), 0.0)
        else:
            attn = attn_logits.softmax(dim=-1)
        ctx = torch.einsum("bhls,bshd->blhd", attn, v)
        return self.out(ctx.reshape(B, L, -1))


def _ffn(d_model: int, mult: int = 4) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_model, d_model * mult),
        nn.GELU(),
        nn.Linear(d_model * mult, d_model),
    )


class _PatchStem(nn.Module):
    """FE-02 (B20 v4 lock 2026-05-24): non-overlapping Conv2d patch stem.

    Treats the (F, T) plane per electrode as a 1-channel "image" and applies
    ``Conv2d(1, d, kernel=(k_f, k_t), stride=(k_f, k_t))`` — non-overlap by
    construction. Replaces the v3 per-cell Linear(1, d) projection. Trunc-
    normal init std=0.02 per recipe §1 "Init policy (uniform)".

    FE-RAW-1 (2026-06-04): the default freq kernel/stride is **5** (was 3) to
    absorb the F=50 raw-|STFT| front end into the same F_p=10 token grid as the
    old F=30 filterbank — ``F_p = (50−5)//5 + 1 = 10``, byte-shape-identical to
    ``(30−3)//3 + 1 = 10``. The time kernel/stride is unchanged at 2. The F=30
    const-Q sister passes ``kernel_freq=3`` explicitly.

    Output shape contract::

        input  : (B, C, F_bins, T_bins)
        output : (B, C, F_p, T_p, d)

    with ``F_p = (F_bins - k_f) // k_f + 1`` and ``T_p = (T_bins - k_t) // k_t + 1``.
    """

    def __init__(
        self,
        d_model: int,
        *,
        kernel_freq: int = 5,
        kernel_time: int = 2,
        stride_freq: int | None = None,
        stride_time: int | None = None,
    ) -> None:
        super().__init__()
        if stride_freq is None:
            stride_freq = kernel_freq
        if stride_time is None:
            stride_time = kernel_time
        self.kernel_freq = kernel_freq
        self.kernel_time = kernel_time
        self.stride_freq = stride_freq
        self.stride_time = stride_time
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(kernel_freq, kernel_time),
            stride=(stride_freq, stride_time),
            padding=0,
            bias=True,
        )
        # Trunc-normal init std=0.02 per recipe §1 (BERT / ViT / Perceiver-IO
        # convention; uniform across the whole model).
        nn.init.trunc_normal_(self.conv.weight, std=0.02)
        nn.init.zeros_(self.conv.bias)

    def n_freq_patches(self, n_freq_bins: int) -> int:
        return (n_freq_bins - self.kernel_freq) // self.stride_freq + 1

    def n_time_patches(self, n_time_bins: int) -> int:
        return (n_time_bins - self.kernel_time) // self.stride_time + 1

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, F, T) → (B*C, 1, F, T) → Conv2d → (B*C, d, F_p, T_p)
        B, C, F, T = x.shape
        x_bc = x.reshape(B * C, 1, F, T)
        out = self.conv(x_bc)                                 # (B*C, d, F_p, T_p)
        F_p, T_p = out.shape[-2], out.shape[-1]
        # → (B, C, F_p, T_p, d)
        return out.permute(0, 2, 3, 1).reshape(B, C, F_p, T_p, -1)


def freq_patch_valid_mask(
    valid_bin_mask: Tensor,
    *,
    kernel_freq: int,
    stride_freq: int,
) -> Tensor:
    """B36 C5 — map a per-freq-BIN validity mask to per-freq-PATCH validity.

    The Conv2d patch stem groups ``kernel_freq`` adjacent freq bins (stride
    ``stride_freq``) into one freq patch. A patch is valid iff EVERY bin it
    covers is valid — conservative: a patch straddling the SWEC k21/k22
    boundary (one valid + two invalid bins) is dropped, which is exactly the
    plan's "SWEC k0–21 → F-patch 0–6" (patch 7 = bins {21,22,23} → dropped).

    Inputs
    ------
    valid_bin_mask
        ``(..., F)`` bool, True = the freq bin carries real signal for this
        corpus. SWEC: bins k0–21 True, k22–29 False.

    Output
    ------
    ``(..., F_p)`` bool where ``F_p = (F − kernel_freq)//stride_freq + 1``.
    """
    if valid_bin_mask.dtype != torch.bool:
        valid_bin_mask = valid_bin_mask.to(torch.bool)
    # unfold the bin axis into (F_p, kernel_freq) windows; a patch is valid
    # iff all bins in its window are valid.
    windows = valid_bin_mask.unfold(-1, kernel_freq, stride_freq)  # (..., F_p, kernel_freq)
    return windows.all(dim=-1)


class _JointTokenBlock(nn.Module):
    """FE-04 (B20 v4 lock 2026-05-24): per-electrode JOINT (t_p·f_p) self-
    attention block. Single multi-head SA over the flat ``F_p · T_p`` token
    sequence per electrode, with RoPE on the **time axis only** (tokens at
    the same ``t_p`` index share the same rotary table; freq identity is
    carried by the additive per-patch freq embedding).

    Pre-norm, GeLU, MLP 4×, heads=8 (per recipe §1). Hard cross-electrode
    mask is structural: each electrode lives in the batch dim of the inner
    SA, so there is no K/V pathway from one electrode to another.

    Replaces the v3 ``_TokenBlock`` (factorized t-SA → f-SA → FFN). At v4's
    small per-electrode token count (F_p · T_p ≈ 40 at P4, 80 at P1 5s),
    factorization saving disappears — AST/Audio-MAE/EAT/SSLAM use joint at
    this scope.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = _PlainMultiHeadSelfAttentionRoPE(d_model, n_heads)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model)

    def forward(
        self,
        tokens: Tensor,
        rope_time: Tensor,
        key_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """tokens: ``(B*C, F_p · T_p, d)`` flat joint sequence.
        rope_time: ``(2, F_p · T_p, head_dim)`` per-token RoPE table where
        position i carries the time-patch index ``i // F_p`` rotation
        (built once per forward by the encoder; freq-patch axis collapses
        into the flat token index).
        key_mask: ``(B*C, F_p · T_p)`` bool, True = keepable key. B36 C5 —
        invalid freq-patch tokens (e.g. SWEC k22–29) are excluded as
        self-attention keys. ``None`` → full attention. Positional (not
        keyword) so the ``checkpoint(...)`` call can pass it through.
        """
        x = tokens + self.attn(self.ln_attn(tokens), rope_time, key_mask=key_mask)
        return x + self.ffn(self.ln_ffn(x))


class _LatentSelfAttnBlock(nn.Module):
    """Factorized latent block: time-SA (RoPE on Q+K) × parcel-SA × FFN, pre-LN.

    T1.9 implementation of the v4 spec "factorized throughout (token blocks
    t×f, latent stack t×parcel)". A single block does:

      1. Time-SA across the T axis (RoPE on Q+K), per parcel slot.
      2. Parcel-SA across the K*M axis, per timestep, with a key-padding
         mask so no-coverage parcels are skipped as keys/values.
      3. FFN.

    Working shape is ``(B*T, L, d)`` (matches the cross-attn stack upstream),
    with an inner reshape to ``(B*L, T, d)`` for the time-SA half. ``B``, ``T``,
    and ``L`` are passed explicitly because they cannot be recovered from a
    flat (B*T, L, d) tensor alone.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln_t = nn.LayerNorm(d_model)
        self.attn_t = _PlainMultiHeadSelfAttentionRoPE(d_model, n_heads)
        self.ln_p = nn.LayerNorm(d_model)
        self.attn_p = _PlainMultiHeadSelfAttention(d_model, n_heads)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model)

    def forward(
        self,
        x_bt: Tensor,                          # (B*T, L, d)
        *,
        B: int,
        T: int,
        L: int,
        latent_valid: Optional[Tensor] = None,  # (B*T, L) bool
        rope_t: Tensor,
    ) -> Tensor:
        d = x_bt.shape[-1]
        # Time-SA: (B*T, L, d) → (B, T, L, d) → (B*L, T, d)
        x_bl = (
            x_bt.reshape(B, T, L, d).transpose(1, 2).reshape(B * L, T, d).contiguous()
        )
        x_bl = x_bl + self.attn_t(self.ln_t(x_bl), rope_t)
        # Parcel-SA: (B*L, T, d) → (B, L, T, d) → (B*T, L, d)
        x_bt = (
            x_bl.reshape(B, L, T, d).transpose(1, 2).reshape(B * T, L, d).contiguous()
        )
        if latent_valid is not None:
            # B30: bidirectional mask — inactive slots neither query nor key.
            # (B*T, L) → (B*T, L, L) via outer-AND.
            parcel_attn_mask = (
                latent_valid.unsqueeze(2) & latent_valid.unsqueeze(1)
            )                                                      # (B*T, L, L)
            x_bt = x_bt + self.attn_p(self.ln_p(x_bt), attn_mask=parcel_attn_mask)
        else:
            x_bt = x_bt + self.attn_p(self.ln_p(x_bt))
        # FFN
        return x_bt + self.ffn(self.ln_ffn(x_bt))


class _CrossAttnBlock(nn.Module):
    """Latent ← electrode hard block-diagonal parcel pool, then FFN, pre-LN."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = _MultiHeadCrossAttention(d_model, n_heads)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model)

    def forward(
        self,
        latents: Tensor,
        electrodes: Tensor,
        key_mask: Tensor,
    ) -> Tensor:
        latents = latents + self.attn(
            self.ln_q(latents), self.ln_kv(electrodes), key_mask,
        )
        latents = latents + self.ffn(self.ln_ffn(latents))
        return latents


def _compute_latent_valid(
    *,
    support: Tensor,
    valid_mask: Optional[Tensor],
    m_sub_slots: int,
) -> Tensor:
    """Return ``(B, L=K*M)`` bool mask: True iff parcel ``p`` has ≥1 covered electrode.

    Used as the DETR memory-padding-mask for the readout (Carion 2020 §3.3):
    a parcel slot with no electrode coverage in this subject is "padded memory"
    and must not be attended to by the task query. All ``m_sub_slots`` of a
    parcel share the same validity. When ``valid_mask`` is None, every
    electrode row of ``support`` is treated as real.
    """
    if valid_mask is not None:
        effective_support = support * valid_mask.unsqueeze(-1).to(support.dtype)
    else:
        effective_support = support
    parcel_covered = effective_support.sum(dim=1) > 0  # (B, K)
    B, K = parcel_covered.shape
    return (
        parcel_covered.unsqueeze(-1)
        .expand(B, K, m_sub_slots)
        .reshape(B, K * m_sub_slots)
    )


def compute_latent_valid_3way(
    *,
    support: Tensor,
    valid_mask: Optional[Tensor],
    m_sub_slots: int,
    parcel_time_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """B36 B8 — 3-way slot/time validity for paradigm-B masked JEPA.

    Splits the ``covered`` slot-time grid into the three roles the masked-JEPA
    step needs:

      * ``visible`` ``(B, L, T_p)`` — ``covered ∧ ¬masked``. Fed to the student
        encoder's latent self-attention keys (visible-only forward).
      * ``target``  ``(B, L, T_p)`` — ``covered ∧ masked``. The masked-prediction
        positions: predictor queries + the L1 loss support.
      * ``teacher`` ``(B, L)``      — ``covered`` (time-invariant). The EMA
        teacher encodes the FULL input, so its valid set ignores the mask.

    ``visible ⊎ target == covered`` (per time patch) and ``visible ∩ target ==
    ∅`` by construction. ``parcel_time_mask`` is ``(B, K, T_p)`` (True = masked,
    only over covered parcels); it is lifted to the slot axis via
    ``repeat_interleave`` over ``M`` (slot ``l = k·M + s``).
    """
    teacher = _compute_latent_valid(
        support=support, valid_mask=valid_mask, m_sub_slots=m_sub_slots,
    )                                                                # (B, L)
    if parcel_time_mask.dim() != 3:
        raise ValueError(
            f"parcel_time_mask must be (B, K, T_p); got "
            f"{tuple(parcel_time_mask.shape)}"
        )
    masked_lt = parcel_time_mask.repeat_interleave(m_sub_slots, dim=1)  # (B, L, T_p)
    covered_lt = teacher.unsqueeze(-1)                                  # (B, L, 1)
    visible = covered_lt & ~masked_lt                                  # (B, L, T_p)
    target = covered_lt & masked_lt                                    # (B, L, T_p)
    return visible, target, teacher


def assert_electrode_alignment_integrity(
    support: Tensor, valid_mask: Optional[Tensor],
) -> None:
    """Always-on electrode-row data-integrity guard (L2, 2026-06-09).

    Cheap and false-positive-free on EVERY forward path — the masked-JEPA path
    zeroes ``electrode_tokens``, never ``support`` / ``valid_mask`` (static
    per-subject), and the Lite electrode set flips ``valid`` False on mapped
    electrodes without touching ``support`` (both handled below). Catches
    corruption / row-desync of the anatomy lane the hard pool routes on:

      * DK ``support`` must be one-hot-or-zero (the hard parcel assignment): a row
        summing > 1 means two parcels claim one electrode.
      * every ``valid`` electrode must map to exactly one parcel: a valid row with
        empty support is a ``support`` / ``valid_mask`` row desync.

    The ``electrode_tokens`` ↔ ``support`` (DP1 permutation) cross-check needs
    per-row identity carried with the front-end tokens (L1) and is tracked
    separately. See reports/bt_alignment/electrode_desync_damage_2026_06_09.md.
    """
    if valid_mask is None:
        return
    if tuple(valid_mask.shape) != tuple(support.shape[:2]):
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} != (B, C) "
            f"{tuple(support.shape[:2])} — electrode-axis row desync"
        )
    sup_per_row = support.sum(dim=-1)
    if bool((sup_per_row > 1.0 + 1e-4).any()):
        raise ValueError(
            "DK support is not one-hot-or-zero (a row assigns >1 parcel) — "
            "corrupted support lane / electrode-row desync"
        )
    if bool((valid_mask.bool() & (sup_per_row < 0.5)).any()):
        raise ValueError(
            "a valid electrode maps to empty support — support/valid_mask row "
            "desync (electrode-row alignment contract violated)"
        )


class V14ParcelPerceiverModel(nn.Module):
    """v14 Perceiver-IO encoder with parcel-id-tagged latents.

    v4-invisible-frontend lock 2026-05-24 (B20 LOCK; FE-01..04 in the
    implementation fix list 2026-05-26): the front-end is a non-overlap
    Conv2d patch stem + per-patch freq embedding + N=6 JOINT (t_p·f_p)
    self-attention token blocks per electrode, with RoPE on the time-axis
    only.
    """

    def __init__(
        self,
        *,
        n_freq_bins: int,
        n_time_bins: int,
        k_parcels: int,
        # v4 defaults (5/19 amendment §3 + B20 5/24 lock) — see config below.
        d_model: int = 256,
        n_heads: int = 8,
        depth_self_attn: int = 6,
        # B29 Item 13 lock 2026-05-27 PM-late: default M=1 (was M=4). Latent
        # count drops 320→80; ``LearnableSubSlotEmbed`` degenerates to a
        # single shared bias when M=1, additive into ``LearnableParcelEmbed``.
        # Sister ``R-m4-slots`` P0 sets ``m_sub_slots=4`` via dispatch to
        # restore the prior 320-slot stack as falsifier.
        m_sub_slots: int = 1,
        # FE-04: N=6 joint token blocks (was N=4 factorized t×f under v3).
        n_token_blocks: int = 6,
        # FE-02 / FE-RAW-1: non-overlap Conv2d patch stem kernel/stride.
        # Default freq kernel/stride = 5 (FE-RAW-1, 2026-06-04) so the F=50 raw
        # |STFT| front end maps to F_p=10 (= (50−5)//5+1), byte-shape-identical
        # to the old F=30 filterbank's F_p=10 at kernel 3. Time kernel/stride
        # stays 2. Strides default to kernel size (non-overlap). Override
        # (e.g. kernel_freq=3) for the F=30 const-Q sister or tiny smoke
        # configs where n_freq_bins < k_f would collapse F_p to zero.
        patch_kernel_freq: int = 5,
        patch_kernel_time: int = 2,
        patch_stride_freq: Optional[int] = None,
        patch_stride_time: Optional[int] = None,
        time_last_input: bool = False,
        # Per blocker IE02: precompute RoPE up to ``max_seq_len`` so the same
        # model instance can ingest clips of different T (Phase-1/2 at 5 s vs
        # Phase-3/4 at 1 s) without rebuilding the buffer. The Phase-4 flat
        # head still constrains T at the downstream head boundary. Default
        # mirrors ``n_time_bins`` so existing callers see no behavior change.
        max_seq_len: Optional[int] = None,
        # B28 cross-attn lock 2026-05-27 PM: latent-stack positions at which
        # to fire a cross-attn block. Default ``[0]`` (Perceiver IO canonical,
        # Jaegle 2021 — single encoder cross-attn followed by N latent SA
        # blocks). The ``R-perceiver-original-2-cross-attns`` sister passes
        # ``[0, 3]`` to restore the prior v4 amendment behavior. Position 0
        # is required (the pre-stack routing block); interior positions
        # ``p > 0`` must satisfy ``p < depth_self_attn``.
        cross_attn_positions: Optional[Sequence[int]] = None,
        # B29 Item 11 + 5/28 PM precedent-audit flip 2026-05-28:
        # ``subtype_embed_enabled`` default ON → OFF after Agent 2 found the
        # primary M3AE precedent net-neutral on iEEG (DIVER-1 §4.1 ablation).
        # Sister roster after the flip:
        #   ``R-subtype-embed-input-only`` P0 (PROMOTED, M3AE-faithful — Geng
        #     2022 §3.1 adds modality embed at input only):
        #     subtype_embed_enabled=True, subtype_embed_reuse_kv=False.
        #   ``R-subtype-embed-on-with-kv-reuse`` P0 (NEW, prior default →
        #     sister): subtype_embed_enabled=True, subtype_embed_reuse_kv=True.
        #   ``R-subtype-embed-3way`` P2-if-budget: subtype_vocab=3 matches
        #     DIVER-1 {sEEG-depth, ECoG-grid, ECoG-strip}.
        #   ``R-no-subtype-embed`` retired — IS the default.
        # 5/28 PM-late B32 first-pass-no-input-aug lock: ref_embed default
        # ON → OFF. With ref-aug already off in the default dispatch path
        # (LogStftView is single static shaft-CAR), the ref_embed lookup
        # always indexes the same row; the additive A1 contribution +
        # cross-attn K/V reuse become no-ops at best, distribution drift
        # at worst. Re-enable as P1 sister `R-ref-aug-3-cell` (paired
        # with RefAugMultiStftView) once first-pass results are in.
        # See: memory/project_v14_b32_first_pass_no_input_aug_2026_05_28.md
        subtype_vocab: int = 2,
        subtype_embed_enabled: bool = False,
        subtype_embed_reuse_kv: bool = True,
        ref_embed_enabled: bool = False,
        ref_embed_reuse_kv: bool = True,
        # 2026-05-30 speedup audit (Tier-2, #119 finally wired): when True,
        # the token-block + latent-block stacks run under
        # ``torch.utils.checkpoint`` so per-block activations are recomputed
        # in backward instead of held. Trades ~25-33% extra student-path
        # compute for the dominant retained activation (the token blocks
        # over BC = B·C rows). Default OFF — no behavior change. The forward
        # gates it on ``self.training and torch.is_grad_enabled()`` so the
        # no_grad EMA-teacher pass is never checkpointed. Numerics-safe: the
        # encoder blocks contain no dropout, so the recomputed forward is
        # bit-identical.
        gradient_checkpointing: bool = False,
        # 2026-06-08 ragged front-end (#91): when True AND a ``valid_mask`` is
        # passed, the per-electrode token blocks run only over the VALID
        # electrodes (gathered out of the ``B·C`` rows) and the masked-out pad
        # rows are scattered back as zeros before the cross-attn pool. The token
        # blocks are per-electrode (no cross-electrode path), so every valid
        # electrode's M2/M4 is BIT-IDENTICAL to the dense path; only the dropped
        # pad rows — which the pool already masks via ``key_mask`` and the P1
        # loss already drops via ``valid_mask`` — change (zeros instead of
        # token-block(masked-stem)). Cuts the dominant token-block FFN
        # activation + FLOPs by the pad fraction (~50% at BT-Lite c_max=256).
        # Default OFF (dense path byte-identical to pre-#91). The pad rows are
        # NOT read downstream: the P1 ``p1_frontend_m2_loss`` must be passed
        # ``valid_mask`` (the joint module gates that on this flag) so the M2
        # loss never reconstructs pad electrodes. See test_v14_ragged_frontend.
        ragged_frontend: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.ragged_frontend = ragged_frontend
        self.n_freq_bins = n_freq_bins
        self.n_time_bins = n_time_bins
        self.k_parcels = k_parcels
        self.m_sub_slots = m_sub_slots
        self.d_model = d_model
        self.time_last_input = time_last_input
        if max_seq_len is None:
            max_seq_len = n_time_bins
        if max_seq_len < n_time_bins:
            raise ValueError(
                f"max_seq_len ({max_seq_len}) must be >= n_time_bins ({n_time_bins})"
            )
        self.max_seq_len = max_seq_len

        # FE-02: non-overlap Conv2d patch stem over the (F, T) plane per
        # electrode. Replaces v3's per-cell Linear(1, d) projection. Output
        # is (B, C, F_p, T_p, d).
        if patch_stride_freq is None:
            patch_stride_freq = patch_kernel_freq
        if patch_stride_time is None:
            patch_stride_time = patch_kernel_time
        self.patch_stem = _PatchStem(
            d_model,
            kernel_freq=patch_kernel_freq,
            kernel_time=patch_kernel_time,
            stride_freq=patch_stride_freq,
            stride_time=patch_stride_time,
        )
        # FE-RAW-1 (2026-06-04): the raw |STFT| front end (kernel_freq=5, the
        # default) requires exactly F=50 bins so the patch stem tiles it
        # leftover-free into F_p=10. A mismatch is a front-end/encoder wiring
        # bug (e.g. an F=30 filterbank fed under the kernel-5 default).
        if patch_kernel_freq == 5 and n_freq_bins != 50:
            raise ValueError(
                f"FE-RAW-1 raw front end (patch_kernel_freq=5) requires "
                f"n_freq_bins=50; got {n_freq_bins}. Pass patch_kernel_freq=3 "
                f"for the F=30 const-Q filterbank sister."
            )
        n_freq_patches = self.patch_stem.n_freq_patches(n_freq_bins)
        max_n_time_patches = self.patch_stem.n_time_patches(max_seq_len)
        if n_freq_patches < 1 or max_n_time_patches < 1:
            raise ValueError(
                f"patch stem produced degenerate output: n_freq_patches="
                f"{n_freq_patches}, max_n_time_patches={max_n_time_patches} "
                f"for n_freq_bins={n_freq_bins}, max_seq_len={max_seq_len}, "
                f"kernel=({patch_kernel_freq},{patch_kernel_time}), stride="
                f"({patch_stride_freq},{patch_stride_time})"
            )
        self.n_freq_patches = n_freq_patches
        self.max_n_time_patches = max_n_time_patches

        # FE-03: per-patch learnable freq embedding (one d-vector per freq
        # patch). Replaces v3's per-bin freq_embed of shape (F, d). Same name
        # so SSL contracts that ref freq_embed by name still resolve, but
        # the first dim is now F_p (post-patch), not F (raw bins).
        self.freq_embed = nn.Parameter(torch.empty(n_freq_patches, d_model))
        nn.init.trunc_normal_(self.freq_embed, std=0.02)

        # LAT-01 (B21 collapse-prevention lock 2026-05-25; recipe §1 "Latent init").
        # The latent-slot tensor (80 at the B29 M=1 default; 320 under the M=4
        # ``R-m4-slots`` sister) is NOT a single free parameter. It is reconstructed
        # at every forward from two embedding tables + a frozen broken-symmetry
        # noise buffer. Identity-anchored init is the symmetry-breaker that lets
        # cross-attn ❺ pool meaningfully — the B36 hard per-parcel pool has no
        # learnable routing bias to break inter-slot ties, so init does.
        #     z[p·M + s] = LearnableParcelEmbed[p] + LearnableSubSlotEmbed[s] + ε
        # where ε ~ N(0, 0.02²) per (p, s, d) is fixed at construction. Replaces
        # the prior `parcel_embedding = nn.Parameter(K, M, d)` single tensor.
        self.learnable_parcel_embed = nn.Parameter(
            torch.empty(k_parcels, d_model)
        )
        nn.init.trunc_normal_(self.learnable_parcel_embed, std=0.02)
        # B29 Item 13: SubSlotEmbed DROPPED at the M=1 default — identity init
        # becomes ``z[p] = LearnableParcelEmbed[p] + ε``. At M=1 a (1, d)
        # sub-slot table would only add a shared-across-parcels bias already
        # spanned by ``learnable_parcel_embed`` (redundant reparameterization).
        # The per-sub-slot table is instantiated ONLY for the ``R-m4-slots``
        # sister (M>1), where distinct sub-slots need a per-slot offset.
        if m_sub_slots > 1:
            self.learnable_subslot_embed = nn.Parameter(
                torch.empty(m_sub_slots, d_model)
            )
            nn.init.trunc_normal_(self.learnable_subslot_embed, std=0.02)
        else:
            self.learnable_subslot_embed = None
        # Frozen broken-symmetry noise: built once at construction, never
        # updated. Persistent so checkpoint round-trip preserves the same ε.
        self.register_buffer(
            "latent_init_noise",
            torch.randn(k_parcels, m_sub_slots, d_model) * 0.02,
            persistent=True,
        )

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        head_dim = d_model // n_heads
        # FE-04: RoPE table for the JOINT token-block self-attention. Each
        # flat token at position i = t_p · F_p + f_p carries the time-patch
        # ``i // F_p`` rotation; the freq-patch axis is left unmodulated and
        # carries identity only through the additive per-patch freq_embed.
        # Pre-tile to (2, F_p · max_T_p, head_dim) so the existing
        # _PlainMultiHeadSelfAttentionRoPE indexer (which slices `[:T]`)
        # Just Works on shorter T_p < max_n_time_patches at forward time.
        base_rope = _rope_freqs(head_dim, max_n_time_patches)  # (2, max_T_p, head_dim)
        flat_pos = torch.arange(n_freq_patches * max_n_time_patches)
        time_patch_idx = flat_pos // n_freq_patches            # i // F_p
        tiled_rope = base_rope[:, time_patch_idx, :]
        self.register_buffer("rope_joint_token", tiled_rope, persistent=False)
        # RoPE table for the latent stack time-SA (per recipe §1, RoPE on
        # latent-stack time axis Q+K). This table is indexed in T_p
        # (post-patch frame count) since the latent stack operates over T_p.
        self.register_buffer("key_rope", base_rope, persistent=False)

        # FE-04: N=6 JOINT (t_p·f_p) self-attention token blocks per electrode.
        # Each block consumes (B*C, F_p·T_p, d) and applies single multi-head
        # SA with the tiled time-axis RoPE above, then FFN. Replaces v3's
        # factorized (t-SA → f-SA → FFN) _TokenBlock.
        self.token_blocks = nn.ModuleList(
            [_JointTokenBlock(d_model, n_heads) for _ in range(n_token_blocks)]
        )
        # B36 B1: terminal LayerNorm of the front-end sub-encoder. The M2 tap
        # (the front-end JEPA target in P1) is returned post-``frontend_ln``,
        # the canonical V-JEPA-2 convention — the EMA teacher mirrors this LN
        # (it lives inside the encoder, so the deepcopy carries it), so the
        # masked-prediction target ``sg(teacher frontend_ln(M2))`` is matched
        # to the student's own normalized front-end output. Mirrors
        # ``encoder_ln`` (the M4 terminal LN) one stage earlier.
        self.frontend_ln = nn.LayerNorm(d_model)

        # B28 cross-attn lock 2026-05-27 PM (supersedes v4 amendment 5/19 §5
        # 2-cross-attn @ {0, 3} default). Perceiver IO (Jaegle 2021,
        # arXiv:2107.14795) is canonically 1 encoder cross-attn + N latent
        # SA blocks; the 2-cross-attn variant cites *original* Perceiver's
        # iterative re-injection (Jaegle 2021a, arXiv:2103.03206), not
        # Perceiver IO. Sister ``R-perceiver-original-2-cross-attns`` passes
        # ``cross_attn_positions=[0, 3]`` to restore the prior default.
        # Position 0 = pre-stack routing (always runs, even at depth=0).
        # Interior positions ``p > 0`` fire pre-block in the latent stack
        # and must satisfy ``p < depth_self_attn``.
        if cross_attn_positions is None:
            cross_attn_positions = [0]
        positions = sorted({int(p) for p in cross_attn_positions})
        for p in positions:
            if p < 0:
                raise ValueError(
                    f"cross_attn_positions entries must be non-negative; got {p}"
                )
        if not positions or positions[0] != 0:
            raise ValueError(
                f"cross_attn_positions must include 0 (the pre-stack routing "
                f"block); got {list(cross_attn_positions)}"
            )
        for p in positions:
            if p > 0 and p >= depth_self_attn:
                raise ValueError(
                    f"cross_attn_positions entry {p} requires depth_self_attn "
                    f"> {p}; got depth_self_attn={depth_self_attn}"
                )
        self._cross_attn_at_block: tuple[int, ...] = tuple(positions)
        self.cross_attns = nn.ModuleList(
            [_CrossAttnBlock(d_model, n_heads) for _ in self._cross_attn_at_block]
        )
        # Legacy single-block alias — older code pokes at `model.cross_attn`
        # to mean "the pre-stack routing block". Production callers should
        # use `model.cross_attns`.
        self.cross_attn = self.cross_attns[0]

        self.latent_blocks = nn.ModuleList(
            [_LatentSelfAttnBlock(d_model, n_heads) for _ in range(depth_self_attn)]
        )
        self.encoder_ln = nn.LayerNorm(d_model)

        # Per-loss-head LayerNorms live on the SSL student bundle
        # (``experiments/v14_joint_module._V14StudentBundle``), NOT on the
        # encoder: ``ln_frame`` always-on; ``ln_mid`` / ``ln_utt`` only for
        # the ``b31_plus_*`` sister variants (B31 dropped LN_mid + LN_utt
        # from the default SSL path). The EMA teacher mirrors that bundle.
        # The encoder formerly carried its own dead ``ln_mid/ln_frame/ln_utt``
        # (B21/B22 era) — removed as never-read scaffold per the 5/29 drift
        # audit.

        # B29 Item 11 lock 2026-05-27 PM-late: subtype + ref-operator
        # per-clip embeddings. Both additive at A1 patch-embed (broadcast
        # over electrodes per clip) and — by default — reused as additive
        # in cross-attn K/V (same broadcast). The reuse defaults are
        # ``True`` to match the B29 lock default; sisters flip them off.
        #
        #   subtype_vocab=2 default: {0=sEEG-depth, 1=ECoG}
        #   subtype_vocab=3 sister:  {0=sEEG-depth, 1=ECoG-grid, 2=ECoG-strip}
        #   ref vocab fixed at 3:    {0=shaftCAR, 1=bipolar, 2=Laplacian}
        if subtype_vocab not in (2, 3):
            raise ValueError(
                f"subtype_vocab must be 2 (binary) or 3 (DIVER-1 three-way); "
                f"got {subtype_vocab}"
            )
        self.subtype_vocab = subtype_vocab
        self.subtype_embed_enabled = bool(subtype_embed_enabled)
        self.subtype_embed_reuse_kv = bool(subtype_embed_reuse_kv)
        self.ref_embed_enabled = bool(ref_embed_enabled)
        self.ref_embed_reuse_kv = bool(ref_embed_reuse_kv)
        if self.subtype_embed_enabled:
            self.subtype_embed = nn.Embedding(subtype_vocab, d_model)
            nn.init.trunc_normal_(self.subtype_embed.weight, std=0.02)
        else:
            self.subtype_embed = None  # type: ignore[assignment]
        if self.ref_embed_enabled:
            # Fixed 3-cell vocab ({shaftCAR, bipolar, Laplacian}) per
            # ``project_v14_ref_aug_input_distribution_lock_2026_05_27`` —
            # raw is skipped per per-corpus definitional ambiguity.
            self.ref_embed = nn.Embedding(3, d_model)
            nn.init.trunc_normal_(self.ref_embed.weight, std=0.02)
        else:
            self.ref_embed = None  # type: ignore[assignment]

    def patch_grid_shape(self, electrode_tokens: Tensor) -> tuple[int, int, int]:
        """B36 B5/B8 — the ``(C, F_p, T_p)`` front-end token grid for a batch.

        Lets the SSL trainer size the P1 token-mask ``(B, C, F_p, T_p)`` and
        the P2 parcel-time mask ``(B, K, T_p)`` without re-deriving the
        patch-stem arithmetic. ``electrode_tokens`` is the raw encoder input
        ``(B, C, T_bins, F_bins)`` (or ``(B, C, F_bins, T_bins)`` under
        ``time_last_input``); returns the post-patch ``(C, F_p, T_p)``.
        """
        if self.time_last_input:
            _B, C, _F, T = electrode_tokens.shape
        else:
            _B, C, T, _F = electrode_tokens.shape
        F_p = self.patch_stem.n_freq_patches(self.n_freq_bins)
        T_p = self.patch_stem.n_time_patches(T)
        return C, F_p, T_p

    # B36 WS-E (phase orchestration). Top-level submodules whose parameters
    # belong to each staging group. The FRONT-END produces the M2 tap (patch
    # stem → per-patch freq embed → joint token blocks → frontend_ln); P1
    # trains it alone. The PARCEL side is everything downstream of M2 — the
    # per-parcel pool (``cross_attns``), the inter-parcel self-attention
    # (``latent_blocks``), the terminal ``encoder_ln``, and the learnable
    # latent-slot tables; P2 trains these (plus the student-only predictor,
    # which lives on the BrainModule) while the front-end rides at LR/10.
    # Per-clip conditioning embeds (``subtype_embed`` / ``ref_embed``,
    # default OFF per B32) inject at the front-end ``cond_emb`` upstream of
    # the token blocks, so they group with the front-end. ``latent_init_noise``
    # and the RoPE tables are buffers (no grad) and never appear here.
    _FRONTEND_PARAM_TOPS: tp.ClassVar[frozenset[str]] = frozenset(
        {"patch_stem", "freq_embed", "token_blocks", "frontend_ln",
         "subtype_embed", "ref_embed"}
    )
    _PARCEL_PARAM_TOPS: tp.ClassVar[frozenset[str]] = frozenset(
        {"cross_attns", "latent_blocks", "encoder_ln",
         "learnable_parcel_embed", "learnable_subslot_embed"}
    )

    def partition_parameters_for_staging(
        self,
    ) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Split encoder params into ``(frontend, parcel)`` for B36 staging.

        ``named_parameters`` dedups the ``self.cross_attn = self.cross_attns[0]``
        alias automatically, so every parameter lands in exactly one bucket.
        Any parameter whose top-level attribute is in neither set raises — a
        new encoder parameter must be consciously assigned to a stage, never
        silently defaulted into the wrong LR group.
        """
        frontend: list[nn.Parameter] = []
        parcel: list[nn.Parameter] = []
        for name, p in self.named_parameters():
            top = name.split(".", 1)[0]
            if top in self._FRONTEND_PARAM_TOPS:
                frontend.append(p)
            elif top in self._PARCEL_PARAM_TOPS:
                parcel.append(p)
            else:
                raise RuntimeError(
                    f"V14ParcelPerceiverModel parameter {name!r} (top-level "
                    f"{top!r}) is unassigned to a B36 staging group; add it to "
                    "_FRONTEND_PARAM_TOPS or _PARCEL_PARAM_TOPS in "
                    "partition_parameters_for_staging."
                )
        return frontend, parcel

    def forward(
        self,
        electrode_tokens: Tensor,   # (B, C, T_bins, F_bins) or (B, C, F_bins, T_bins) if time_last_input
        support: Tensor,            # (B, C, K_parcels)
        valid_mask: Optional[Tensor] = None,  # (B, C) bool
        *,
        # B36 hard pool: the one-hot DK ``support`` IS the parcel assignment,
        # so there is no ``log(support + ε)`` smoothing term. ``eps`` is kept
        # in the signature (vestigial, reserved for the gated ``R-bna-soft``
        # routing sister) but unused on the default hard-pool path.
        eps: float = DEFAULT_SUPPORT_BIAS_EPS,
        return_taps: bool = False,
        # M3 (first-routing latent, pre-LN_mid) is OPT-IN. B31 dropped the
        # ``L_mid_slot`` arm from the default SSL surface (B22 superseded) and
        # the live B36 masked-JEPA path reads only M2 (P1) / M4 (P2), so the
        # M3 reshape would be pure dead compute on every P2 forward. It is
        # computed + returned ONLY for the retained ``R-add-m3-loss`` /
        # ``b31_plus_m3`` P0 sister, which passes ``return_m3=True``.
        return_m3: bool = False,
        # MASK-03 (B03 mask-discipline lock 2026-05-25 PM): per-electrode
        # SHAFT mask (B, C) bool — True = DROP. A dropped electrode is
        # non-attendable by every parcel (folded into the hard-pool
        # ``key_mask`` below). Combines with ``~valid_mask`` (pad mask).
        # P1 leaves this as None (no shaft drop). Pure DROP — no [MASK]
        # token (paradigm B).
        shaft_mask: Optional[Tensor] = None,
        # B29 Item 11 lock 2026-05-27 PM-late: per-clip conditioning ids.
        # When the corresponding embed is enabled and ``subject_subtype`` /
        # ``ref_idx`` is provided, the embedding vector is added at A1
        # (post patch-stem / post freq_embed) and — by default
        # (``*_embed_reuse_kv=True``) — also re-added on the electrode K/V
        # tokens going into the cross-attn block(s). When ``None`` AND the
        # embedding is enabled, the contribution falls back to embedding(0)
        # (i.e. a learned constant per encoder forward) so callers without
        # subtype/ref metadata still produce non-degenerate features.
        subject_subtype: Optional[Tensor] = None,
        ref_idx: Optional[Tensor] = None,
        # B36 B5 (paradigm-B masked JEPA, visible-only student forward). Both
        # default ``None`` → byte-identical to the unmasked forward.
        #   ``token_mask``: (B, C, F_p, T_p) bool, True = masked — the P1
        #     front-end M2 token mask. Masked post-patch tokens are zeroed
        #     before the token blocks so visible tokens never encode them.
        #   ``parcel_time_mask``: (B, K, T_p) bool, True = masked — the P2
        #     parcel-time block mask. Its electrodes are zeroed at the masked
        #     time patches (front-end input drop, derived via the one-hot
        #     ``support``) and the masked parcel-times are excluded from the
        #     latent self-attention keys (visible-only encoder).
        token_mask: Optional[Tensor] = None,
        parcel_time_mask: Optional[Tensor] = None,
        # B36 WS-B efficiency (P1 reads ONLY M2 — the P1 predictor operates on
        # the M2 tap): when True, return
        # ``{"M2": frontend_ln(token-block output)}`` and skip the entire
        # downstream pool / inter-parcel encoder / encoder_ln. M2 is computed
        # upstream of the pool, so the returned tap is byte-identical to the
        # full forward's M2 — the skipped stages produce M3/M4 which P1 never
        # reads and which carry no P1 gradient anyway. Requires return_taps.
        m2_only: bool = False,
        # B36 C5: per-corpus freq-PATCH validity ``(F_p,)`` or ``(B, F_p)`` bool,
        # True = valid. Derived from the per-corpus valid-bin mask via
        # ``freq_patch_valid_mask``. Invalid freq patches (SWEC k22–29 →
        # F-patches 7–9) are excluded as keys in BOTH the per-electrode joint
        # token-block self-attention AND the hard-pool cross-attention. ``None``
        # → every freq patch valid (the BT all-30-bins-valid default), making
        # this path byte-identical to the pre-C5 forward.
        freq_patch_valid: Optional[Tensor] = None,
    ) -> Tensor | dict[str, Tensor]:
        """Returns parcel latents shaped ``(B, K*M, T_p, d)`` where ``T_p``
        is the post-patch frame count from the Conv2d patch stem.

        v4 amendment (5/19) §1 — "THE BIG CORRECTION": latents keep the time
        axis. Cross-attn is strict per-time-patch: latents at t_p attend only
        to electrode tokens at the same t_p (over all C electrodes × F_p freq
        patches). Implemented by batching T_p into the batch dim so the
        existing attention modules work unchanged.

        LAT-05 (B22 collapse-prevention dense-features 2026-05-25): when
        ``return_taps=True``, return a dict of intermediate-stream outputs
        used by the SSL loss heads::

            "M2": (B, C, F_p, T_p, d)   per-electrode-patch tokens, post token
                                        blocks / pre cross-attn-0. Feeds
                                        ``L_pre_frame``.
            "M3": (B, L, T_p, d)        first post-routing latents, post
                                        cross-attn-0 / pre self-attn-0. Feeds
                                        ``L_mid_slot`` (with ``LN_mid``).
            "M4": (B, L, T_p, d)        final encoder output, post encoder_ln
                                        and pre any task-head LayerNorm. Feeds
                                        ``L_post_frame`` (with ``LN_frame``),
                                        ``L_post_utterance`` (with ``LN_utt``),
                                        and ``L_DKoleo``.

        ``LN_mid`` / ``LN_frame`` / ``LN_utt`` are NOT applied here — the
        SSL loss heads apply them, so each head sees an independently-
        normalized stream. Default ``return_taps=False`` preserves the
        single-tensor ``(B, K*M, T_p, d)`` return for downstream callers
        (``V14ParcelPerceiverWithHead``, ``V14ParcelCollapsePMA``).
        """
        # Normalize input to (B, C, F, T) — what _PatchStem wants.
        if self.time_last_input:
            x_in = electrode_tokens                              # (B, C, F, T)
        else:
            x_in = electrode_tokens.transpose(-1, -2)            # default (B, C, T, F) → (B, C, F, T)
        B, C, F, T = x_in.shape
        if F != self.n_freq_bins:
            raise ValueError(
                f"expected {self.n_freq_bins} freq bins, got {F}"
            )
        if T > self.max_seq_len:
            raise ValueError(
                f"got T={T} but max_seq_len={self.max_seq_len}; rebuild "
                f"with a larger max_seq_len"
            )
        if support.shape != (B, C, self.k_parcels):
            raise ValueError(
                f"support shape {tuple(support.shape)} does not match "
                f"(B, C, K) = ({B}, {C}, {self.k_parcels})"
            )
        # L2 always-on electrode-row data-integrity guard (2026-06-09).
        assert_electrode_alignment_integrity(support, valid_mask)

        # FE-02: non-overlap Conv2d patch stem. (B, C, F, T) → (B, C, F_p, T_p, d).
        x = self.patch_stem(x_in)                                   # (B, C, F_p, T_p, d)
        F_p, T_p = x.shape[2], x.shape[3]
        if F_p != self.n_freq_patches:
            raise ValueError(
                f"patch stem produced F_p={F_p} but init-time n_freq_patches="
                f"{self.n_freq_patches} — likely a kernel/stride mismatch"
            )

        # B36 C5: normalize the per-corpus freq-patch validity to (B, F_p) bool.
        # None → all valid (BT default; the whole block below is then a no-op).
        freq_patch_valid_bf: Optional[Tensor] = None
        if freq_patch_valid is not None:
            fpv = freq_patch_valid.to(torch.bool)
            if fpv.shape not in {(F_p,), (B, F_p)}:
                raise ValueError(
                    f"freq_patch_valid shape {tuple(freq_patch_valid.shape)} "
                    f"must be (F_p,) or (B, F_p) = ({F_p},) / ({B}, {F_p})"
                )
            if fpv.dim() == 1:
                fpv = fpv.unsqueeze(0).expand(B, F_p)
            # All-valid is byte-identical to None; only carry a real mask.
            if not bool(fpv.all()):
                freq_patch_valid_bf = fpv

        # FE-03: per-patch freq embedding broadcast over T_p.
        # freq_embed: (F_p, d) → unsqueeze(1) → (F_p, 1, d) → broadcasts to
        # (..., F_p, T_p, d). Trailing-dim alignment is over the last 3 axes.
        x = x + self.freq_embed.unsqueeze(1)

        # B29 Item 11 (A1 additive): per-clip subtype + ref embeddings added
        # at the patch-embed output (before token blocks). Looked up once
        # per clip and broadcast over (C, F_p, T_p). 5/28 audit + B32 flip
        # both default branches OFF: ``subtype_embed_enabled=False`` IS the
        # default; ``ref_embed_enabled=False`` IS the default. Sister
        # ``R-subtype-embed-on-with-kv-reuse`` P0 re-enables subtype;
        # sister ``R-ref-aug-3-cell`` P1 re-enables ref (paired with
        # RefAugMultiStftView). None ids fall back to embed(0) so callers
        # without metadata still produce a non-degenerate forward.
        if self.subtype_embed_enabled or self.ref_embed_enabled:
            cond_emb = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
            if self.subtype_embed_enabled:
                if subject_subtype is None:
                    sub_ids = torch.zeros(B, dtype=torch.long, device=x.device)
                else:
                    # NeuralSet collates per-event TimedArrays into a leading
                    # batch axis. Our extractor emits ``(1, 1)`` (one channel,
                    # one sample) per event — required by NeuralSet's
                    # ``BaseExtractor._missing_default`` invariant
                    # (``tensor.shape[:-1]`` must be non-empty when
                    # frequency != 0) — which collates to ``(B, 1, 1)``. Strip
                    # all trailing singleton axes so the embedding lookup
                    # stays 1-D. ``(B,)`` is already accepted unchanged.
                    # Bind to a non-Optional local so the loop body's
                    # rebinding doesn't break narrowing.
                    sst: Tensor = subject_subtype
                    while sst.dim() > 1 and sst.shape[-1] == 1:
                        sst = sst.squeeze(-1)
                    if sst.shape != (B,):
                        raise ValueError(
                            f"subject_subtype shape {tuple(sst.shape)} "
                            f"does not match (B,) = ({B},)"
                        )
                    if sst.dtype not in (torch.long, torch.int32, torch.int64):
                        raise TypeError(
                            f"subject_subtype dtype must be integer; got "
                            f"{sst.dtype}"
                        )
                    sub_ids = sst.to(torch.long).to(x.device)
                    if (sub_ids < 0).any() or (sub_ids >= self.subtype_vocab).any():
                        raise ValueError(
                            f"subject_subtype ids must be in [0, {self.subtype_vocab}); "
                            f"got min={sub_ids.min().item()}, max={sub_ids.max().item()}"
                        )
                cond_emb = cond_emb + self.subtype_embed(sub_ids)
            if self.ref_embed_enabled:
                if ref_idx is None:
                    ref_ids = torch.zeros(B, dtype=torch.long, device=x.device)
                else:
                    # See subject_subtype above — same trailing-singleton
                    # squeeze contract; covers both (B, 1) and (B, 1, 1).
                    # Bind to a non-Optional local so the loop body's
                    # rebinding doesn't break narrowing.
                    ri: Tensor = ref_idx
                    while ri.dim() > 1 and ri.shape[-1] == 1:
                        ri = ri.squeeze(-1)
                    if ri.shape != (B,):
                        raise ValueError(
                            f"ref_idx shape {tuple(ri.shape)} does not "
                            f"match (B,) = ({B},)"
                        )
                    if ri.dtype not in (torch.long, torch.int32, torch.int64):
                        raise TypeError(
                            f"ref_idx dtype must be integer; got {ri.dtype}"
                        )
                    ref_ids = ri.to(torch.long).to(x.device)
                    if (ref_ids < 0).any() or (ref_ids >= 3).any():
                        raise ValueError(
                            f"ref_idx ids must be in [0, 3); got "
                            f"min={ref_ids.min().item()}, max={ref_ids.max().item()}"
                        )
                cond_emb = cond_emb + self.ref_embed(ref_ids)
            # Broadcast (B, d) → (B, 1, 1, 1, d) → (B, C, F_p, T_p, d).
            x = x + cond_emb.view(B, 1, 1, 1, self.d_model)
        else:
            cond_emb = None

        # B36 B5: visible-only masking. Zero the masked front-end tokens
        # (post-patch, post-freq/cond-embed) BEFORE the token blocks so visible
        # tokens never encode masked content — the leakage-free paradigm-B
        # contract. ``token_mask`` (P1) masks individual (electrode, freq-patch,
        # time-patch) cells; ``parcel_time_mask`` (P2) drops a covered parcel's
        # electrodes at the masked time patches (derived from the one-hot
        # ``support``). Zeroing a token makes the masked input a constant, so a
        # visible token's output is independent of any masked input value.
        token_drop: Optional[Tensor] = None  # (B, C, F_p, T_p) bool
        if token_mask is not None:
            if token_mask.shape != (B, C, F_p, T_p):
                raise ValueError(
                    f"token_mask shape {tuple(token_mask.shape)} does not "
                    f"match (B, C, F_p, T_p) = ({B}, {C}, {F_p}, {T_p})"
                )
            token_drop = token_mask
        if parcel_time_mask is not None:
            if parcel_time_mask.shape != (B, self.k_parcels, T_p):
                raise ValueError(
                    f"parcel_time_mask shape {tuple(parcel_time_mask.shape)} "
                    f"does not match (B, K, T_p) = "
                    f"({B}, {self.k_parcels}, {T_p})"
                )
            # electrode_time_drop[b, c, t] = ∃k: support[b,c,k]>0 ∧ mask[b,k,t].
            onehot = (support > 0).to(x.dtype)                          # (B, C, K)
            etd = (
                torch.einsum("bck,bkt->bct", onehot, parcel_time_mask.to(x.dtype))
                > 0
            )                                                           # (B, C, T_p)
            etd_grid = etd.unsqueeze(2).expand(B, C, F_p, T_p)          # (B,C,F_p,T_p)
            token_drop = etd_grid if token_drop is None else (token_drop | etd_grid)
        if token_drop is not None:
            x = x.masked_fill(token_drop.unsqueeze(-1), 0.0)

        # FE-04: per-electrode JOINT (t_p·f_p) self-attention token blocks.
        # Flatten the per-electrode plane into a single token sequence,
        # ordered (t_p outer, f_p inner) so flat-index i = t_p · F_p + f_p
        # matches the tiled rope_joint_token table.
        BC = B * C
        x_joint = (
            x.permute(0, 1, 3, 2, 4)                                # (B, C, T_p, F_p, d)
             .reshape(BC, T_p * F_p, self.d_model)
        )
        # Slice the precomputed tiled rope to the current flat length so the
        # existing _apply_rope (which does `cos[:T]`) finds matching entries.
        rope_token = self.rope_joint_token[:, : T_p * F_p, :]
        # B36 C5: per-electrode joint token-block key mask. Flat token order is
        # (t_p outer, f_p inner) — token i = t_p·F_p + f_p — so freq validity
        # tiles over T_p and replicates across electrodes (all C share a
        # corpus's freq mask). None (BT) → full attention.
        token_key_mask: Optional[Tensor] = None
        if freq_patch_valid_bf is not None:
            fk = (
                freq_patch_valid_bf.unsqueeze(1)
                .expand(B, T_p, F_p)
                .reshape(B, T_p * F_p)
            )                                                         # (B, T_p·F_p)
            token_key_mask = (
                fk.unsqueeze(1).expand(B, C, T_p * F_p).reshape(BC, T_p * F_p)
            )                                                         # (BC, T_p·F_p)
        # 2026-05-30 speedup audit (#119): activation checkpointing on the
        # block stacks. Gated on training + grad so the no_grad EMA-teacher
        # pass (and eval) run normally — checkpointing only helps the
        # gradient-carrying student backward. No-op when the flag is off.
        use_ckpt = (
            self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        )

        def _run_token_blocks(xj: Tensor, kpm: Optional[Tensor]) -> Tensor:
            for token_block in self.token_blocks:
                if use_ckpt:
                    xj = checkpoint(
                        token_block, xj, rope_token, kpm,
                        use_reentrant=False,
                    )
                else:
                    xj = token_block(xj, rope_token, kpm)
            return xj

        # #91 ragged front-end: the token blocks are per-electrode (the flat
        # ``B·C`` leading dim has no cross-electrode attention — each row is one
        # electrode's (t_p, f_p) plane), so dropping the pad rows leaves every
        # VALID row's output bit-identical while cutting the dominant token-block
        # FFN activation/FLOPs by the pad fraction. Gather the valid rows, run
        # the blocks, scatter back into a zero-filled (B·C, …) buffer. The pad
        # rows become zeros — harmless: the cross-attn pool masks them via
        # ``key_mask`` (support=0 | ~valid_mask) and the P1 M2 loss drops them
        # via ``valid_mask``. Default OFF / no valid_mask → the dense loop runs.
        ragged = self.ragged_frontend and valid_mask is not None
        if ragged:
            assert valid_mask is not None  # narrowed by ``ragged``
            valid_rows = valid_mask.reshape(BC)                      # (B·C,) bool
            valid_idx = valid_rows.nonzero(as_tuple=True)[0]         # (N_valid,)
            if valid_idx.numel() == 0:
                # Whole-batch all-pad (no real electrode anywhere). Unreachable
                # under normal loading, but the gather would feed a 0-row tensor
                # to the token blocks' SDPA and crash; the scatter result IS all
                # zeros, so produce it directly and skip the blocks. The pool's
                # no-coverage path then zeros every parcel (matching dense).
                x_joint = x_joint.new_zeros(BC, T_p * F_p, self.d_model)
            else:
                xj_v = x_joint.index_select(0, valid_idx)            # (N_valid, T_p·F_p, d)
                kpm_v = (
                    token_key_mask.index_select(0, valid_idx)
                    if token_key_mask is not None else None
                )
                xj_v = _run_token_blocks(xj_v, kpm_v)
                x_joint = xj_v.new_zeros(
                    BC, T_p * F_p, self.d_model
                ).index_copy(0, valid_idx, xj_v)
        else:
            x_joint = _run_token_blocks(x_joint, token_key_mask)
        # Reshape back to (B, C, F_p, T_p, d) for cross-attn consumption.
        x = (
            x_joint.reshape(B, C, T_p, F_p, self.d_model)
                   .permute(0, 1, 3, 2, 4)
                   .contiguous()
        )                                                            # (B, C, F_p, T_p, d)

        # B36 WS-B P1 efficiency early-exit: P1 reads ONLY M2 (the P1 predictor
        # operates on the M2 tap), so the hard pool + inter-parcel encoder +
        # encoder_ln below are dead compute in P1 (they build M3/M4, which the
        # P1 loss never reads and which carry no P1 gradient — M2 is taken
        # pre-pool). Skip them. The
        # returned M2 == ``frontend_ln(x)`` is identical to the full-forward
        # tap below; the monitors run their own full-input forward (no
        # ``m2_only``) so RankMe still sees M4.
        if m2_only:
            if not return_taps:
                raise ValueError("m2_only=True requires return_taps=True")
            return {"M2": self.frontend_ln(x)}

        # B36 hard block-diagonal parcel pool (replaces the soft
        # λ_anat·log(support+ε) Graphormer bias). The one-hot DK ``support``
        # (support[b, c, k] ∈ {0, 1}) IS the assignment: parcel-slot l = k·M+s
        # attends ONLY to electrode c iff ``support[b, c, k] > 0``. After
        # FE-02 the cross-attn keys are (C · F_p) tokens per time-patch; all
        # F_p patches of an electrode share its parcel, so the per-electrode
        # attendability replicates over F_p. ``key_mask[b, l, n] == True``
        # means token n is on-parcel for slot l AND not dropped.
        L = self.k_parcels * self.m_sub_slots
        assigned_kc = support > 0                                       # (B, C, K) bool
        attend_lc = (
            assigned_kc.transpose(1, 2)                                 # (B, K, C)
            .unsqueeze(2)
            .expand(B, self.k_parcels, self.m_sub_slots, C)
            .reshape(B, L, C)
        )                                                               # (B, L=K*M, C)

        # MASK-03: drop set = invalid (pad) | shaft (P2 block). A dropped
        # electrode is non-attendable by every parcel; all F_p patches of a
        # dropped electrode share the same DROP.
        drop_electrode: Optional[Tensor] = None
        if valid_mask is not None:
            drop_electrode = ~valid_mask                                # (B, C)
        if shaft_mask is not None:
            if shaft_mask.shape != (B, C):
                raise ValueError(
                    f"shaft_mask shape {tuple(shaft_mask.shape)} does not "
                    f"match (B, C) = ({B}, {C})"
                )
            drop_electrode = (
                shaft_mask if drop_electrode is None else drop_electrode | shaft_mask
            )
        if drop_electrode is not None:
            attend_lc = attend_lc & ~drop_electrode.unsqueeze(1)        # (B, L, C)

        # Expand per-electrode attendability over the F_p freq patches.
        key_mask_lcf = (
            attend_lc.unsqueeze(-1)
                     .expand(B, L, C, F_p)
                     .reshape(B, L, C * F_p)
        )                                                               # (B, L, C·F_p) bool

        # B36 C5: invalid freq patches are non-attendable keys for every parcel
        # slot. The C·F_p key axis is ordered (C outer, F_p inner) — matching
        # ``electrodes_bt`` below — so freq validity replicates across the C
        # electrodes. None (BT) → no-op.
        if freq_patch_valid_bf is not None:
            freq_key_valid = (
                freq_patch_valid_bf.unsqueeze(1)
                .expand(B, C, F_p)
                .reshape(B, C * F_p)
            )                                                           # (B, C·F_p)
            key_mask_lcf = key_mask_lcf & freq_key_valid.unsqueeze(1)   # (B, L, C·F_p)

        # LAT-01: reconstruct the (K, M, d) parcel-embedding tensor at every
        # forward from learnable_parcel_embed + ε (identity-anchored init,
        # B21 lock 2026-05-25). B29 Item 13: at the M=1 default the per-sub-slot
        # table is dropped (z[p] = ParcelEmbed[p] + ε); it is added back only
        # for the R-m4-slots sister (M>1).
        parcel_embedding = (
            self.learnable_parcel_embed.unsqueeze(1)         # (K, 1, d)
            + self.latent_init_noise                          # (K, M, d) frozen
        )                                                     # (K, M, d) via broadcast
        if self.learnable_subslot_embed is not None:
            parcel_embedding = (
                parcel_embedding
                + self.learnable_subslot_embed.unsqueeze(0)  # (1, M, d)
            )
        latents_t = (
            parcel_embedding.reshape(L, self.d_model)
            .unsqueeze(0)                                               # (1, L, d)
            .unsqueeze(2)                                               # (1, L, 1, d)
            .expand(B, L, T_p, self.d_model)
        )

        # Batch T_p into the batch dim so the attention modules see a flat
        # (B*T_p, ?, d) tensor each. Cross-attn keys at time-patch t_p are
        # the C · F_p electrode-freq-patch tokens at that t_p.
        latents_bt = latents_t.transpose(1, 2).reshape(B * T_p, L, self.d_model)
        electrodes_bt = (
            x.permute(0, 3, 1, 2, 4)                                    # (B, T_p, C, F_p, d)
             .reshape(B * T_p, C * F_p, self.d_model)
        )

        # B29 Item 11 (cross-attn K/V reuse): when ``subtype_embed_reuse_kv``
        # or ``ref_embed_reuse_kv`` are True (defaults), re-add the per-clip
        # conditioning embeddings to the electrode tokens going into the
        # cross-attn block(s). M3AE (Geng 2022 §3.1) adds at input only;
        # sister ``R-subtype-embed-input-only`` flips ``subtype_embed_reuse_kv``
        # off to falsify M3AE-vs-v14-default on iEEG.
        if cond_emb is not None and (
            self.subtype_embed_reuse_kv or self.ref_embed_reuse_kv
        ):
            kv_extra = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
            if self.subtype_embed_enabled and self.subtype_embed_reuse_kv:
                if subject_subtype is None:
                    sub_ids_kv = torch.zeros(B, dtype=torch.long, device=x.device)
                else:
                    # Strip trailing singletons (NeuralSet collates
                    # extractor's (1, 1) into (B, 1, 1)); see the A1
                    # additive branch for the same contract.
                    sst_kv: Tensor = subject_subtype
                    while sst_kv.dim() > 1 and sst_kv.shape[-1] == 1:
                        sst_kv = sst_kv.squeeze(-1)
                    sub_ids_kv = sst_kv.to(torch.long).to(x.device)
                kv_extra = kv_extra + self.subtype_embed(sub_ids_kv)
            if self.ref_embed_enabled and self.ref_embed_reuse_kv:
                if ref_idx is None:
                    ref_ids_kv = torch.zeros(B, dtype=torch.long, device=x.device)
                else:
                    ri_kv: Tensor = ref_idx
                    while ri_kv.dim() > 1 and ri_kv.shape[-1] == 1:
                        ri_kv = ri_kv.squeeze(-1)
                    ref_ids_kv = ri_kv.to(torch.long).to(x.device)
                kv_extra = kv_extra + self.ref_embed(ref_ids_kv)
            # Broadcast (B, d) → (B, T_p, C·F_p, d) → (B·T_p, C·F_p, d).
            kv_extra_bt = (
                kv_extra.view(B, 1, 1, self.d_model)
                        .expand(B, T_p, C * F_p, self.d_model)
                        .reshape(B * T_p, C * F_p, self.d_model)
            )
            electrodes_bt = electrodes_bt + kv_extra_bt

        # key_mask is time-invariant (anatomy + drop set don't depend on t);
        # replicate per time-patch to match the batched (B·T_p) attention.
        key_mask_bt = (
            key_mask_lcf.unsqueeze(1)
                        .expand(B, T_p, L, C * F_p)
                        .reshape(B * T_p, L, C * F_p)
        )

        # B29 Item 12 (5/27 PM-late) + B30 lock (2026-05-28): latent-SA
        # bidirectional ``attn_mask`` is always support-derived (parcels
        # with ≥1 covered electrode), per-subject + time-invariant. The
        # pre-B29 ``parcels_supervised[subject]`` override is gone; the
        # pre-B30 key-only ``key_padding_mask`` path is gone.
        latent_valid = _compute_latent_valid(
            support=support,
            valid_mask=valid_mask,
            m_sub_slots=self.m_sub_slots,
        )                                                            # (B, L) covered
        if parcel_time_mask is not None:
            # B36 B8 (3-way ``latent_valid``): the latent-SA keys are the
            # VISIBLE set = covered & ~masked, per time patch (visible-only
            # encoder). ``parcel_time_mask`` (B, K, T_p) → slot axis via
            # repeat_interleave over M (slot l = k·M+s) → exclude masked
            # parcel-times as both queries and keys.
            ptm_l = parcel_time_mask.repeat_interleave(self.m_sub_slots, dim=1)  # (B, L, T_p)
            visible_blt = latent_valid.unsqueeze(-1) & ~ptm_l                    # (B, L, T_p)
            latent_valid_bt = visible_blt.permute(0, 2, 1).reshape(B * T_p, L)
        else:
            latent_valid_bt = latent_valid.unsqueeze(1).expand(B, T_p, L).reshape(B * T_p, L)

        # B28 lock 2026-05-27 PM: cross-attn fires at position 0 ONLY by
        # default (was {0, 3} under v4 amendment 5/19 §5). Position 0
        # (pre-stack routing) ALWAYS runs, even at depth=0. Interior
        # positions are empty unless ``cross_attn_positions`` adds them
        # (sister ``R-perceiver-original-2-cross-attns`` = [0, 3]); the rest
        # of the stack is the latent block (t-SA × parcel-SA × FFN).
        latents_bt = self.cross_attns[0](
            latents_bt, electrodes_bt, key_mask_bt,
        )
        # LAT-05 tap M3: first-routing post cross-attn-0 / pre self-attn-0.
        # Captured BEFORE LN_mid so the loss head owns the normalization.
        # Opt-in (``return_m3``) — dead on the live B36 path (see signature).
        m3_bt = latents_bt if (return_taps and return_m3) else None

        interior_cross_attn = {
            pos: blk
            for pos, blk in zip(self._cross_attn_at_block[1:], list(self.cross_attns)[1:])
        }
        for i, block in enumerate(self.latent_blocks):
            interior = interior_cross_attn.get(i)
            if interior is not None:
                latents_bt = interior(
                    latents_bt, electrodes_bt, key_mask_bt,
                )
            if use_ckpt:
                latents_bt = checkpoint(
                    block,
                    latents_bt,
                    B=B, T=T_p, L=L,
                    latent_valid=latent_valid_bt,
                    rope_t=self.key_rope,
                    use_reentrant=False,
                )
            else:
                latents_bt = block(
                    latents_bt,
                    B=B, T=T_p, L=L,
                    latent_valid=latent_valid_bt,
                    rope_t=self.key_rope,
                )

        latents_bt = self.encoder_ln(latents_bt)
        # Unflatten T_p back out → (B, L, T_p, d).
        out = latents_bt.reshape(B, T_p, L, self.d_model).transpose(1, 2).contiguous()
        if not return_taps:
            return out
        taps = {
            # M2: per-electrode-patch state post token blocks, returned
            #     post-``frontend_ln`` (B36 B1 — the V-JEPA-2-canonical
            #     terminal-LN target convention; teacher mirrors this LN).
            #     (B, C, F_p, T_p, d).
            "M2": self.frontend_ln(x),
            # M4: final encoder output, post encoder_ln (no task-head LN).
            "M4": out,
        }
        # M3 (first post-routing latent, pre LN_mid) only on the opt-in
        # ``R-add-m3-loss`` sister path — see ``return_m3`` in the signature.
        if return_m3:
            assert m3_bt is not None  # return_taps=True ⇒ captured above.
            # LAT-05: re-shape M3 to (B, L, T_p, d) for symmetry with M4.
            taps["M3"] = m3_bt.reshape(B, T_p, L, self.d_model).transpose(1, 2).contiguous()
        return taps


def _sinusoidal_1d(positions: Tensor, dim: int) -> Tensor:
    """Fixed sin/cos positional embedding for one integer axis.

    ``positions`` ``(B, N)`` (long/float) → ``(B, N, dim)``. Standard
    Transformer sinusoid; zero params.
    """
    pos = positions.to(torch.float32).unsqueeze(-1)            # (B, N, 1)
    half = (dim + 1) // 2
    idx = torch.arange(half, device=positions.device, dtype=torch.float32)
    inv_freq = torch.exp(-math.log(10000.0) * (2.0 * idx / dim))  # (half,)
    ang = pos * inv_freq.view(1, 1, -1)                        # (B, N, half)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, N, 2*half)
    return emb[..., :dim]


def factored_sinusoidal_pos_emb(axis_ids: Sequence[Tensor], dim: int) -> Tensor:
    """B36 §5 — fixed factored sinusoidal positional embedding for the JEPA
    predictor's (id/parcel + position) tagging.

    Each axis in ``axis_ids`` (a list of ``(B, N)`` integer tensors — e.g.
    ``[parcel_id, time]`` for the P2 parcel-time grid, ``[electrode, freq_patch,
    time]`` for the P1 front-end grid) gets a contiguous ``dim // n_axes`` slice
    of the embedding (the last axis absorbs the remainder). Zero params
    (V-JEPA convention: the predictor's only learnable piece is the mask token;
    positions are fixed sinusoids). Returns ``(B, N, dim)``.
    """
    if not axis_ids:
        raise ValueError("axis_ids must be non-empty")
    n_axes = len(axis_ids)
    base = dim // n_axes
    sizes = [base] * n_axes
    sizes[-1] += dim - base * n_axes                  # remainder → last axis
    parts = [_sinusoidal_1d(ids, sz) for ids, sz in zip(axis_ids, sizes)]
    return torch.cat(parts, dim=-1)                   # (B, N, dim)


class JepaPredictor(nn.Module):
    """B36 §5 paradigm-B masked-JEPA predictor (replaces ``Predictor2Block``).

    A separate narrow transformer that predicts masked-cell features from the
    **visible** encoder tokens (V-JEPA / Brain-JEPA paradigm B). Default =
    **3 blocks @ d=128, 4 heads, MLP 4×, terminal ``Linear → d_model``, NO
    per-head LN** (~0.6M params), discarded after SSL. Depth is a config knob
    (D2 center 3; sweep {2, 3, 4}). ``R-p1-predictor-large`` = 16@512.

    **Positional scheme — RoPE-on-time-only** (Ben 2026-06-04,
    [[project_v14_predictor_design_rope_lock_2026_06_04]]). RoPE assumes an
    ordered/metric axis; our parcel axis is *unordered anatomy*, so RoPE is
    applied to the TIME axis only (on both context and query tokens, inside
    the block self-attention — the encoder-consistent ``_JointTokenBlock`` path)
    and the non-time identity (freq-patch for P1, parcel-id for P2) is carried
    by a **learned additive ``id_embed``** on the query. Context carries its
    identity through content (the encoded token), so it gets RoPE-time only.
    This fixes the pre-fix ``context_pos=None`` time-blindness and mirrors the
    encoder (RoPE time + learned ``freq_embed`` / ``ParcelEmbed``). Sisters:
    ``R-pred-2d-rope-freq`` (P1 freq also rotary), ``R-pred-additive-sincos``
    (the pre-fix fixed-sinusoid scheme, a must-beat falsifier — its builder
    ``factored_sinusoidal_pos_emb`` is retained).

    The predictor consumes:

    * ``context`` ``(B, N_ctx, d_model)`` — visible encoder tokens (P1: visible
      front-end M2 tokens; P2: visible parcel M4 tokens), projected to ``hidden``.
    * ``context_time_ids`` / ``query_time_ids`` ``(N_ctx,)`` / ``(N_qry,)`` long
      — the time-patch index of every context / query token (shared across the
      batch; the P1/P2 grids are batch-identical). Drive the per-token RoPE.
    * ``query_id`` ``(N_qry,)`` long — the non-time identity index of every
      masked target slot (freq-patch ∈ [0, F_p) for P1, parcel-slot ∈ [0, L)
      for P2). Looked up in ``id_embed`` and added to the single learnable
      ``mask_token`` → "learnable mask tokens tagged by (RoPE-time + learned
      identity)" (B2).

    Context and queries are concatenated into one sequence; ``depth``
    encoder-consistent pre-norm ``_JointTokenBlock`` blocks (RoPE-time SA +
    GELU MLP 4×) attend over both (bidirectional, key-padding-masked); the
    query slice is read out and projected back to ``d_model``. With a
    ``query_valid`` mask the predictions are gathered to ``(n_masked, d_model)``
    — exactly the masked positions the L1 loss scores (B6). The raw projection
    is returned with NO LN (only the EMA-teacher target is normed, by the
    encoder's own terminal LN — V-JEPA-2 §2.1).
    """

    def __init__(
        self,
        d_model: int,
        *,
        n_identity: int,
        hidden: int = 128,
        n_heads: int = 4,
        depth: int = 3,
        max_time_patches: int = 64,
    ) -> None:
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError(f"hidden={hidden} not divisible by n_heads={n_heads}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1; got {depth}")
        if n_identity < 1:
            raise ValueError(f"n_identity must be >= 1; got {n_identity}")
        self.d_model = d_model
        self.hidden = hidden
        self.depth = depth
        self.n_identity = n_identity
        self.input_proj = nn.Linear(d_model, hidden)
        # Single learnable mask token (V-JEPA): every masked query starts from
        # this vector, then gets its identity via ``id_embed`` and its time via
        # RoPE inside the blocks.
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        # Learned additive embedding for the UNORDERED non-time identity axis
        # (freq-patch for P1, parcel-slot for P2). Replaces the pre-fix fixed
        # ``factored_sinusoidal_pos_emb`` tag — the parcel axis has no metric,
        # so a learned table (not a sinusoid / not RoPE) is the right prior.
        self.id_embed = nn.Embedding(n_identity, hidden)
        # Encoder-consistent RoPE-time blocks (pre-norm, RoPE on Q+K time axis,
        # GELU MLP 4×, no per-head LN). Reuses the exact ``_JointTokenBlock``
        # the front-end token blocks use, so the autocast/dtype behaviour is
        # identical to the proven encoder path.
        self.blocks = nn.ModuleList(
            [_JointTokenBlock(hidden, n_heads) for _ in range(depth)]
        )
        self.output_proj = nn.Linear(hidden, d_model)
        # RoPE table over the time axis, precomputed to ``max_time_patches`` and
        # gathered per-token by time index at forward time (supports any T_p ≤
        # max without recompute, incl. P4's shorter grid). Non-persistent: a
        # pure function of (head_dim, max_time_patches).
        head_dim = hidden // n_heads
        self.register_buffer(
            "_rope_base", _rope_freqs(head_dim, max_time_patches), persistent=False,
        )
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.id_embed.weight, std=0.02)

    def forward(
        self,
        context: Tensor,                            # (B, N_ctx, d_model)
        *,
        context_time_ids: Tensor,                   # (N_ctx,) long
        query_time_ids: Tensor,                     # (N_qry,) long
        query_id: Tensor,                           # (N_qry,) long
        context_key_padding_mask: Optional[Tensor] = None,  # (B, N_ctx) True=ignore
        query_valid: Optional[Tensor] = None,       # (B, N_qry) True=real masked slot
    ) -> Tensor:
        """Predict masked-slot features from the visible ``context``.

        Returns ``(n_masked, d_model)`` (gathered at ``query_valid``) when
        ``query_valid`` is given, else ``(B, N_qry, d_model)``. Padded query
        slots (``~query_valid``) and padded context cells are excluded as
        attention keys so they never leak into a real prediction; if every
        token in a row is padded the row is un-padded (its outputs are gathered
        away anyway) so the SDPA softmax never sees an all-masked key set (NaN).
        """
        B, n_ctx, _ = context.shape
        n_qry = query_time_ids.shape[0]
        h_ctx = self.input_proj(context)                        # (B, N_ctx, h)
        q = self.mask_token.view(1, 1, -1) + self.id_embed(query_id).unsqueeze(0)
        q = q.expand(B, n_qry, -1)                              # (B, N_qry, h)
        seq = torch.cat([h_ctx, q], dim=1)                     # (B, N_ctx+N_qry, h)

        # Per-token RoPE-time table for the concatenated sequence: gather the
        # precomputed base table at each token's time index (context then
        # query). ``_apply_rope`` (inside the blocks) reads it positionally.
        time_ids = torch.cat([context_time_ids, query_time_ids])  # (N_total,)
        rope = self._rope_base.index_select(1, time_ids)        # (2, N_total, head_dim)

        # Key-keep mask (SDPA bool convention inside _JointTokenBlock: True =
        # participate). Visible context cells + real masked query slots are
        # keepable keys; padded context + padded query slots are not.
        ctx_pad = (
            context_key_padding_mask
            if context_key_padding_mask is not None
            else torch.zeros(B, n_ctx, dtype=torch.bool, device=seq.device)
        )
        qry_pad = (
            ~query_valid
            if query_valid is not None
            else torch.zeros(B, n_qry, dtype=torch.bool, device=seq.device)
        )
        keep = ~torch.cat([ctx_pad, qry_pad], dim=1)            # (B, N_total) True=keep
        # All-padded row → softmax over an empty key set → NaN. Un-pad such rows
        # (their query outputs are gathered away by query_valid, so inert).
        none_keep = ~keep.any(dim=1)
        if bool(none_keep.any()):
            keep = keep.clone()
            keep[none_keep] = True

        for block in self.blocks:
            seq = block(seq, rope, keep)
        out = self.output_proj(seq[:, n_ctx:, :])               # (B, N_qry, d_model)
        if query_valid is not None:
            return out[query_valid]                             # (n_masked, d_model)
        return out


class V14ParcelCollapsePMA(nn.Module):
    """Parcel collapse via PMA k=1 (Set Transformer): one learnable query
    attends to all ``K*M`` parcel latents per timestep, producing one ``d``
    vector per ``t``. Input ``(B, L, T, d)`` → output ``(B, T, d)``.

    **Training contract** (B31 lock 2026-05-28,
    [[project_v14_b31_vjepa2_canonical_loss_2026_05_28]]; P4 per B35
    2026-05-31, [[project_v14_b35_p4_frozen_pma_mean_linear_2026_05_31]]):

    * **P1+P2 (joint SSL)** — PMA is NOT in the loss path. The B31 default
      drops ``L_post_utterance`` from the SSL aggregator; PMA receives no
      gradient. The ``R-add-utterance-loss`` sister constructs an
      unfrozen PMA in this phase to falsify the drop.
    * **P3 (Whisper distillation)** — PMA is **unfrozen** and trained by
      the cross-modal distillation gradient (Antonello/Shimizu precedent;
      B33 project-up).
    * **P4 (Neuroprobe probe)** — PMA is **kept, FROZEN** (loaded from the
      P3 checkpoint): it collapses the parcel/slot axis → ``(B, T_p, d)``,
      then :class:`V14PmaReadout` applies mean-over-time → per-task Linear
      (B35; reverts B34's drop-PMA-add-attentive-query change). The forward
      below already produces ``(B, T_p, d)`` keyed by ``latent_valid`` — P4
      reuses that exact contract; there is no second collapse path.

    Default is ``freeze=False`` so the natural P3 construction picks up an
    unfrozen PMA without ceremony; the P4 readout re-freezes it
    (:class:`V14PmaReadout` calls ``requires_grad_(False)``). Init is
    random. Also instantiated by the ``R-add-utterance-loss`` SSL sister.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.query = nn.Parameter(
            torch.randn(1, 1, d_model) * (1.0 / math.sqrt(d_model))
        )
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(
        self,
        latents: Tensor,                              # (B, L, T, d)
        latent_valid: Optional[Tensor] = None,        # (B, L) bool
    ) -> Tensor:
        B, L, T, d = latents.shape
        # Batch T into batch dim: (B*T, L, d).
        kv_input = latents.transpose(1, 2).reshape(B * T, L, d)
        q = self.q_proj(self.ln_q(self.query)).reshape(
            1, 1, self.n_heads, self.head_dim
        ).expand(B * T, 1, self.n_heads, self.head_dim)
        kv = self.kv_proj(self.ln_kv(kv_input)).reshape(
            B * T, L, 2, self.n_heads, self.head_dim
        )
        k, v = kv.unbind(dim=2)
        attn_logits = torch.einsum("blhd,bshd->bhls", q, k) * self.scale
        if latent_valid is not None:
            invalid_bt = (
                (~latent_valid).unsqueeze(1).expand(B, T, L).reshape(B * T, L)
            )
            attn_logits = attn_logits.masked_fill(
                invalid_bt.unsqueeze(1).unsqueeze(1),                  # (B*T, 1, 1, L)
                NEG_INF_MASK_VALUE,
            )
        attn = attn_logits.softmax(dim=-1)
        ctx = torch.einsum("bhls,bshd->blhd", attn, v).reshape(B * T, 1, d)
        ctx = self.out_proj(ctx)                                       # (B*T, 1, d)
        return ctx.reshape(B, T, d)


class V14Phase4FlatHead(nn.Module):
    """Generic flatten-time + linear head. Input ``(B, T, d)`` → ``(B,
    n_classes)``.

    **Not the Phase-4 default** (B35 lock 2026-05-31,
    [[project_v14_b35_p4_frozen_pma_mean_linear_2026_05_31]]: the default
    means over time, not flattens). Reusable primitive: the
    ``readout="pma_flatten_linear"`` build (the ``R-p4-flatten`` sister)
    flattens the frozen-PMA-collapsed ``(B, T_p, d)`` via this head's
    ``Linear(T_p·d, n_classes)`` logic, wired through :class:`V14PmaReadout`.
    """

    def __init__(self, n_time_bins: int, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.n_time_bins = n_time_bins
        self.d_model = d_model
        self.classifier = nn.Linear(n_time_bins * d_model, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, d)
        B = x.shape[0]
        return self.classifier(x.reshape(B, -1))


class V14PerTaskAttentivePooler(nn.Module):
    """Phase-4 per-task attentive readout — **no longer the default**
    (B35 lock 2026-05-31,
    [[project_v14_b35_p4_frozen_pma_mean_linear_2026_05_31]] reverted B34's
    default to frozen-PMA → mean → linear). Kept as the
    ``readout="attentive"`` / ``R-p4-attentive`` sister — revisit only with
    a much larger per-task train budget (792k trainable params is
    ~152–453× over Neuroprobe's ≤3500-sample/task data). A fresh trainable
    attentive probe over the FULL parcel×time encoder output
    ``(B, L, T, d)``.

    Mechanism — lean V-JEPA §4.3 attentive probe (Set Transformer PMA
    k=1, Lee 2019 §3.2; class-attention, CaiT 2021 §3)::

        (B, L, T, d) → flatten to (B, L*T, d) tokens
          → ONE cross-attention layer, single learnable query, keys/values
            = the (L*T) tokens, anatomically-inactive parcel slots masked
            out via ``latent_valid``
          → residual add to query → LN → 2-layer MLP
          → Linear(d, n_classes)

    Deliberately the LEAN single-query / single-cross-attn-layer form, NOT
    the V-JEPA-2 §5 4-block probe: the Neuroprobe gate is subject-leakage-
    bound on the CSubject prong, and the attentive-probe capacity caution
    (P4 precedent audit Q5: V-JEPA §4.3 +17/+16.1 over avg-pool;
    "Attention, Please!" 2025) says keep the per-task probe small and
    always report the :class:`V14MeanPoolLinearHead` leakage-control
    baseline alongside.

    Always trainable — this IS the per-task probe (the P4 frozen-encoder
    protocol freezes the backbone, never the probe).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_classes: int,
        *,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.query = nn.Parameter(
            torch.randn(1, 1, d_model) * (1.0 / math.sqrt(d_model))
        )
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln_post = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.classifier = nn.Linear(d_model, n_classes)
        for lin in (self.q_proj, self.kv_proj, self.out_proj):
            nn.init.trunc_normal_(lin.weight, std=0.02)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        latents: Tensor,                              # (B, L, T, d)
        latent_valid: Optional[Tensor] = None,        # (B, L) bool
    ) -> Tensor:
        B, L, T, d = latents.shape
        tokens = latents.reshape(B, L * T, d)
        q = self.q_proj(self.ln_q(self.query)).reshape(
            1, 1, self.n_heads, self.head_dim
        ).expand(B, 1, self.n_heads, self.head_dim)
        kv = self.kv_proj(self.ln_kv(tokens)).reshape(
            B, L * T, 2, self.n_heads, self.head_dim
        )
        k, v = kv.unbind(dim=2)
        attn_logits = torch.einsum("bqhd,bshd->bhqs", q, k) * self.scale   # (B,H,1,L*T)
        if latent_valid is not None:
            invalid = (
                (~latent_valid).unsqueeze(-1).expand(B, L, T).reshape(B, L * T)
            )                                                              # (B, L*T)
            attn_logits = attn_logits.masked_fill(
                invalid.unsqueeze(1).unsqueeze(1),                         # (B,1,1,L*T)
                NEG_INF_MASK_VALUE,
            )
        attn = attn_logits.softmax(dim=-1)
        ctx = torch.einsum("bhqs,bshd->bqhd", attn, v).reshape(B, 1, d)
        ctx = self.out_proj(ctx)                                          # (B,1,d)
        h = self.query.expand(B, 1, d) + ctx                             # residual to query
        h = h + self.mlp(self.ln_post(h))                                # MLP block
        return self.classifier(h.squeeze(1))                            # (B, n_classes)


class V14MeanPoolLinearHead(nn.Module):
    """``R-p4-meanpool-no-pma`` sister readout (B35, 2026-05-31): masked
    mean-pool the encoder output ``(B, L, T, d)`` over anatomically-valid
    ``(parcel, time)`` cells → ``(B, d)`` → ``Linear(d, n_classes)``.

    **Means over parcel×time directly, SKIPPING the PMA** — distinct from
    the B35 default :class:`V14PmaReadout` (``pma_mean_linear``), which
    means over time *after* the frozen PMA collapses parcels. Leakage /
    ablation reference: does the PMA's learned parcel-pooling beat a dumb
    parcel mean? (MAE §4.3; PopT Fig. 17 frozen-aggregator+linear sanity.)
    """

    def __init__(self, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.classifier = nn.Linear(d_model, n_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        latents: Tensor,                              # (B, L, T, d)
        latent_valid: Optional[Tensor] = None,        # (B, L) bool
    ) -> Tensor:
        B, L, T, d = latents.shape
        if latent_valid is None:
            pooled = latents.reshape(B, L * T, d).mean(dim=1)
        else:
            mask = latent_valid.to(latents.dtype).reshape(B, L, 1, 1)
            num = (latents * mask).sum(dim=(1, 2))                        # (B, d)
            den = (
                latent_valid.to(latents.dtype).sum(dim=1) * T
            ).clamp(min=1.0).unsqueeze(-1)                               # (B, 1)
            pooled = num / den
        return self.classifier(pooled)


class V14PmaReadout(nn.Module):
    """B35 Phase-4 default readout (2026-05-31,
    [[project_v14_b35_p4_frozen_pma_mean_linear_2026_05_31]]): the
    **frozen** P3-PMA collapses the parcel/slot axis, then a temporal op
    over ``T_p``, then a per-task ``Linear``::

        latents (B, L, T_p, d)
          → frozen PMA (parcel-collapse, key-masked by latent_valid) → (B, T_p, d)
          → temporal {mean | flatten | timeattn}                     → (B, d) or (B, T_p·d)
          → Linear(., n_classes)                                     → (B, n_classes)

    ``temporal="mean"`` is the B35 default (``pma_mean_linear``): mean over
    ``T_p`` → ``(B, d)`` → ``Linear(d, n_classes)``. ``"flatten"``
    (``pma_flatten_linear`` / ``R-p4-flatten``) restores B31's
    ``Linear(T_p·d, n_classes)`` flat head. ``"timeattn"``
    (``pma_timeattn_linear`` / ``R-p4-time-attn-pool``) convex-pools over
    ``T_p`` with one learned score vector (~257 params) before the linear.

    The PMA is the P3-trained query, loaded from the P3 checkpoint into
    ``readout.pma.*`` and **frozen here** (``requires_grad_(False)``); the
    linear (and, for ``timeattn``, the score vector) are the ONLY trainable
    P4 params — the readout the Neuroprobe ≤3500-sample/task budget can
    actually fit (B35 rationale). The encoder is frozen by the trainer.
    """

    def __init__(
        self,
        pma: "V14ParcelCollapsePMA",
        temporal: str,
        d_model: int,
        n_classes: int,
        n_time_patches: int,
    ) -> None:
        super().__init__()
        if temporal not in ("mean", "flatten", "timeattn"):
            raise ValueError(f"unknown temporal={temporal!r}")
        self.pma = pma
        self.pma.requires_grad_(False)                  # frozen P3-PMA at P4
        self.temporal = temporal
        if temporal == "flatten":
            self.classifier = nn.Linear(n_time_patches * d_model, n_classes)
        else:
            self.classifier = nn.Linear(d_model, n_classes)
        if temporal == "timeattn":
            self.time_score = nn.Linear(d_model, 1)     # ~257p, convex pool over T_p
            nn.init.trunc_normal_(self.time_score.weight, std=0.02)
            nn.init.zeros_(self.time_score.bias)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        latents: Tensor,                              # (B, L, T, d)
        latent_valid: Optional[Tensor] = None,        # (B, L) bool
    ) -> Tensor:
        collapsed = self.pma(latents, latent_valid=latent_valid)   # (B, T, d), PMA frozen
        if self.temporal == "mean":
            pooled = collapsed.mean(dim=1)                         # (B, d)
        elif self.temporal == "flatten":
            pooled = collapsed.reshape(collapsed.shape[0], -1)     # (B, T·d)
        else:  # timeattn
            scores = self.time_score(collapsed).squeeze(-1)        # (B, T)
            weights = scores.softmax(dim=1).unsqueeze(-1)          # (B, T, 1)
            pooled = (collapsed * weights).sum(dim=1)              # (B, d)
        return self.classifier(pooled)                            # (B, n_classes)


# Phase-3 distillation pool + projection live on the TEACHER side, not here.
# Per the B05/B06 lock (2026-05-25 PM) the v14 student is an identity
# passthrough at its 8 Hz native rate; the Whisper-L8 teacher is pooled
# 50 → 8 Hz by ``extractors.whisper_teacher_pool.triangular_pool_50_to_8_hz``
# and projected 1280 → 256 by ``models.whisper_adapter.WhisperAdapter``.
# The former student-side ``V14Phase3TimePoolTriangular`` +
# ``V14Phase3DistillHead`` (5/22-era student-side pool @ 10 Hz / 50 buckets)
# were removed as stale scaffold; they predated the teacher-side relocation.


class V14ParcelPerceiverWithHead(nn.Module):
    """Encoder + Phase-4 per-task readout (B35 lock 2026-05-31,
    [[project_v14_b35_p4_frozen_pma_mean_linear_2026_05_31]]).

    Phase-4 downstream pipeline::

        encoder(...)   → (B, L, T, d)
        readout(...)   → (B, n_classes)    per-task probe

    ``readout`` is a :class:`V14PmaReadout` for the default + flatten +
    time-attn options (``readout="pma_mean_linear"`` default,
    ``"pma_flatten_linear"``, ``"pma_timeattn_linear"``): a **frozen**
    P3-PMA collapses parcels → ``(B, T_p, d)``, then a temporal op →
    Linear. The B34 sisters stay selectable:
    :class:`V14PerTaskAttentivePooler` (``"attentive"``) over the full
    parcel×time field, and :class:`V14MeanPoolLinearHead` (``"meanpool"``,
    the ``R-p4-meanpool-no-pma`` reference). The frozen-encoder protocol
    freezes ``encoder``; the only trainable P4 params are the readout's
    linear (and the attentive sister's query/MLP).
    """

    def __init__(
        self,
        encoder: V14ParcelPerceiverModel,
        readout: nn.Module,
        eps: float = DEFAULT_SUPPORT_BIAS_EPS,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.readout = readout
        self.eps = eps

    def forward(
        self,
        electrode_tokens: Tensor,
        support: Tensor,
        valid_mask: Optional[Tensor] = None,
        *,
        eps: Optional[float] = None,
        # Forward B29 conditioning through to the inner encoder so Phase-4
        # downstream + dispatch's ``cfg.build()`` path can exercise it (not
        # just unit-test mocks).
        shaft_mask: Optional[Tensor] = None,
        subject_subtype: Optional[Tensor] = None,
        ref_idx: Optional[Tensor] = None,
    ) -> Tensor:
        eps_used = self.eps if eps is None else eps
        latents = self.encoder(
            electrode_tokens,
            support,
            valid_mask,
            eps=eps_used,
            shaft_mask=shaft_mask,
            subject_subtype=subject_subtype,
            ref_idx=ref_idx,
        )  # (B, L, T, d)
        latent_valid = _compute_latent_valid(
            support=support,
            valid_mask=valid_mask,
            m_sub_slots=self.encoder.m_sub_slots,
        )
        return self.readout(latents, latent_valid=latent_valid)                      # (B, n_classes)


class V14ParcelPerceiver(BaseModelConfig):
    """NeuralTrain config for the v14 first-pass encoder + classifier head.

    See ``memory/project_v14_encoder_design_2026_05_13.md`` for the locked
    defaults. Stage-1 sweeps: ``eps`` is anatomy-prior strength, swept at
    training-config level (not on the model config). ``m_sub_slots``,
    ``d_model``, ``depth_self_attn`` are first-class sweep axes here.
    """

    n_freq_bins: int
    n_time_bins: int
    k_parcels: int
    # v4 amendment 5/19 §3 + B20 5/24 lock + B29 Item 13 (M=1 default):
    # d=256, heads=8 (32 dim/head). ~14.235M params at K=80, M=1, depth=6,
    # T=8 (T_p), F=12 (F_p), N=6 (post-B28 cross-attn collapse + post-B29
    # M=1 + B29 Item 11 subtype/ref embeds when enabled).
    d_model: int = 256
    n_heads: int = 8
    depth_self_attn: int = 6
    # B29 Item 13 lock 2026-05-27 PM-late: default M=1. Sister
    # ``R-m4-slots`` P0 sets this to 4 to restore the prior 320-slot stack.
    m_sub_slots: int = 1
    # FE-04: N=6 joint token blocks (was N=4 factorized t×f under v3).
    n_token_blocks: int = 6
    # FE-02 / FE-RAW-1: non-overlap Conv2d patch stem kernel/stride. Default
    # freq kernel/stride = 5 (FE-RAW-1, 2026-06-04) for the F=50 raw |STFT|
    # front end → F_p=10. The F=30 const-Q filterbank sister sets this to 3.
    patch_kernel_freq: int = 5
    patch_kernel_time: int = 2
    patch_stride_freq: int | None = None
    patch_stride_time: int | None = None
    eps: float = DEFAULT_SUPPORT_BIAS_EPS
    time_last_input: bool = False
    # Per blocker IE02: RoPE table size. None ⇒ same as n_time_bins (legacy
    # behavior). Set explicitly when the same model instance ingests clips
    # of variable T (e.g. 5 s pretrain + 1 s downstream).
    max_seq_len: int | None = None
    # B28 cross-attn lock 2026-05-27 PM: default ``[0]`` (Perceiver IO
    # canonical). The ``R-perceiver-original-2-cross-attns`` sister sets
    # this to ``[0, 3]`` via the dispatch flag.
    cross_attn_positions: list[int] | None = None
    # B29 Item 11 + 5/28 PM precedent-audit flip 2026-05-28: subtype default
    # ON → OFF (Agent 2 found M3AE precedent net-neutral on iEEG via DIVER-1
    # §4.1). Ref defaults unchanged. Sisters flip via dispatch:
    #   R-subtype-embed-input-only      → subtype_embed_enabled=True,
    #                                     subtype_embed_reuse_kv=False (M3AE)
    #   R-subtype-embed-on-with-kv-reuse → subtype_embed_enabled=True,
    #                                     subtype_embed_reuse_kv=True
    #   R-subtype-embed-3way             → subtype_vocab=3 (DIVER-1)
    #   R-ref-aug-3-cell                 → ref_embed_enabled=True paired
    #                                      with RefAugMultiStftView (B32).
    subtype_vocab: int = 2
    subtype_embed_enabled: bool = False
    subtype_embed_reuse_kv: bool = True
    ref_embed_enabled: bool = False
    ref_embed_reuse_kv: bool = True

    # B35 P4 readout selector (2026-05-31,
    # [[project_v14_b35_p4_frozen_pma_mean_linear_2026_05_31]]; reverts
    # B34). Default "pma_mean_linear" = V14PmaReadout: the FROZEN P3-PMA
    # collapses parcels → (B, T_p, d), then mean-over-time → Linear; only
    # the linear trains at P4. "pma_flatten_linear" (R-p4-flatten) restores
    # B31's flatten→Linear; "pma_timeattn_linear" (R-p4-time-attn-pool)
    # convex-pools over T_p. "attentive" = V14PerTaskAttentivePooler
    # (R-p4-attentive sister, 792k trainable — not fittable at Neuroprobe's
    # ≤3500-sample/task budget, hence demoted). "meanpool" =
    # V14MeanPoolLinearHead (R-p4-meanpool-no-pma, means over parcel×time
    # directly, skipping the PMA). The P3 PMA weights load into
    # readout.pma.* at P4 (trainer-level); the encoder-freeze for the real
    # probe is also a trainer concern (load checkpoint → freeze encoder).
    readout: tp.Literal[
        "pma_mean_linear", "pma_flatten_linear", "pma_timeattn_linear",
        "attentive", "meanpool",
    ] = "pma_mean_linear"

    # 2026-05-30 speedup audit (Tier-2, #119): activation checkpointing on
    # the encoder block stacks. Default OFF (no behavior change). True
    # recomputes per-block activations in backward to cut peak memory —
    # insurance for memory-bound configs / larger batches. Numerics-safe
    # (no dropout in the blocks). Threaded into the encoder; the no_grad
    # teacher pass is never checkpointed (see the forward's training/grad
    # gate).
    gradient_checkpointing: bool = False

    # 2026-06-08 ragged front-end (#91): skip pad electrodes in the per-electrode
    # token blocks (gather valid rows → run blocks → scatter pad rows as zeros).
    # Default OFF (dense path byte-identical). Pairs with the P1 loss's
    # ``valid_mask`` exclusion (gated on this flag in the joint module) so the M2
    # loss never reconstructs pad electrodes. Cuts the token-block FFN activation
    # + FLOPs by the pad fraction (~50% at BT-Lite c_max=256 ⇒ unblocks raw bs=8
    # and ~halves front-end step time on padded batches).
    ragged_frontend: bool = False

    # SSL-pretrain dispatch flags persisted on the model config so they
    # ride along with the run-record YAML. The encoder ``build`` does
    # not branch on them — they are metadata for sister-cell rollouts
    # and for an SSL trainer that consumes them later. Typed via
    # :class:`Literal` so reconstruction from a persisted config rejects
    # typos at deserialization rather than waiting for the (absent) SSL
    # consumer to silently no-op them.
    dkoleo_mode: tp.Literal[
        "off", "intra_clip_slots", "batch_cls_unit", "vicreg_slot_variance",
    ] = "off"
    phase_mode: tp.Literal["joint_b29", "split_p1_p2"] = "joint_b29"

    def build(
        self,
        n_classes: int | None = None,
        *,
        n_in_channels: int | None = None,
        n_outputs: int | None = None,
    ) -> nn.Module:
        """Build a v14 encoder + classifier head.

        Accepts both the standalone ``n_classes=`` form and the NeuralTrain
        ``Experiment._build_brain_module`` convention ``build(n_in_channels=...,
        n_outputs=...)``. ``n_in_channels`` is informational only — v14 handles
        variable C via the per-batch ``valid_mask``.
        """
        if n_classes is None:
            n_classes = n_outputs
        if n_classes is None:
            raise ValueError("V14ParcelPerceiver.build needs n_classes or n_outputs")

        encoder = V14ParcelPerceiverModel(
            n_freq_bins=self.n_freq_bins,
            n_time_bins=self.n_time_bins,
            k_parcels=self.k_parcels,
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth_self_attn=self.depth_self_attn,
            m_sub_slots=self.m_sub_slots,
            n_token_blocks=self.n_token_blocks,
            patch_kernel_freq=self.patch_kernel_freq,
            patch_kernel_time=self.patch_kernel_time,
            patch_stride_freq=self.patch_stride_freq,
            patch_stride_time=self.patch_stride_time,
            time_last_input=self.time_last_input,
            max_seq_len=self.max_seq_len,
            cross_attn_positions=self.cross_attn_positions,
            subtype_vocab=self.subtype_vocab,
            subtype_embed_enabled=self.subtype_embed_enabled,
            subtype_embed_reuse_kv=self.subtype_embed_reuse_kv,
            ref_embed_enabled=self.ref_embed_enabled,
            ref_embed_reuse_kv=self.ref_embed_reuse_kv,
            gradient_checkpointing=self.gradient_checkpointing,
            ragged_frontend=self.ragged_frontend,
        )
        # B35: P4 readout. The "pma_*" options collapse parcels with a
        # FROZEN P3-PMA, then mean/flatten/timeattn over T_p → Linear (only
        # the linear trains; the P3 PMA weights load into readout.pma.* at
        # the trainer level). "attentive"/"meanpool" consume the full
        # (B, L, T, d) field directly (B34 sisters).
        readout: nn.Module
        if self.readout in (
            "pma_mean_linear", "pma_flatten_linear", "pma_timeattn_linear",
        ):
            pma = V14ParcelCollapsePMA(
                d_model=self.d_model,
                n_heads=self.n_heads,
            )
            temporal = {
                "pma_mean_linear": "mean",
                "pma_flatten_linear": "flatten",
                "pma_timeattn_linear": "timeattn",
            }[self.readout]
            n_time_patches = encoder.patch_stem.n_time_patches(self.n_time_bins)
            readout = V14PmaReadout(
                pma=pma,
                temporal=temporal,
                d_model=self.d_model,
                n_classes=n_classes,
                n_time_patches=n_time_patches,
            )
        elif self.readout == "attentive":
            readout = V14PerTaskAttentivePooler(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_classes=n_classes,
            )
        elif self.readout == "meanpool":
            readout = V14MeanPoolLinearHead(
                d_model=self.d_model,
                n_classes=n_classes,
            )
        else:
            raise ValueError(f"unknown readout={self.readout!r}")
        return V14ParcelPerceiverWithHead(encoder, readout, eps=self.eps)
