"""v14 Perceiver-IO encoder with parcel-id-tagged latents.

Reference: ``memory/project_v14_encoder_design_2026_05_13.md`` (this commit),
``memory/project_v14_parcel_token_readout_2026_04_26.md`` (canonical arch
spec), ``memory/project_v14_dk_first_pass_2026_05_13.md`` (data-side pivot).

Contract
--------
Input:
    electrode_tokens : (B, C_max, T_bins, F_bins)  float32
    support          : (B, C_max, K_parcels)       float32 in [0, 1]
    valid_mask       : (B, C_max)                  bool  — True for real, False for pad

Output (encoder):
    parcel_latents   : (B, K_parcels * M, d_model) float32

Output (classifier head):
    logits           : (B, n_classes)              float32

Pipeline:
    Linear(F_bins → d) → per-electrode temporal self-attn (RoPE on T_bins)
    → flatten to (B, C*T_bins, d)
    → cross-attn from K*M parcel-id-tagged latents, with log(support + eps) bias
    → 6× latent self-attn
    → optional DETR-style task readout (one learnable query per task)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

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


class _MultiHeadCrossAttentionWithBias(nn.Module):
    """Latent ← electrode cross-attn with additive QK bias broadcast over heads.

    Optionally applies one-sided RoPE to the K vectors based on their time-bin
    index. Q (latents) has no time axis, so it stays unrotated — this is
    unusual versus canonical bidirectional Q+K RoPE but principled here:
    absolute K position is what we need, and it lets us honor the
    "RoPE on T_bins" spec commitment without introducing a separate
    per-electrode temporal self-attn block.
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

    def forward(
        self,
        latents: Tensor,         # (B, L, d)
        electrodes: Tensor,      # (B, N, d)  N = C * T_bins
        bias: Tensor,            # (B, L, N) — log(support+eps) + valid_mask
        *,
        key_rope: Optional[Tensor] = None,  # (2, T_bins, head_dim) cos/sin
        t_bins: Optional[int] = None,
    ) -> Tensor:
        B, L, _ = latents.shape
        N = electrodes.shape[1]
        q = self.q_proj(latents).reshape(B, L, self.n_heads, self.head_dim)
        kv = self.kv_proj(electrodes).reshape(B, N, 2, self.n_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        if key_rope is not None:
            if t_bins is None or N % t_bins != 0:
                raise ValueError(
                    f"key_rope requires t_bins that evenly divides N={N}, got t_bins={t_bins}"
                )
            C = N // t_bins
            # (B, N, H, head_dim) → (B, C, T, H, head_dim) → (B, C, H, T, head_dim)
            k_rot = k.reshape(B, C, t_bins, self.n_heads, self.head_dim).transpose(2, 3)
            k_rot = _apply_rope(k_rot, key_rope)
            # back to (B, N, H, head_dim)
            k = k_rot.transpose(2, 3).reshape(B, N, self.n_heads, self.head_dim)
        logits = torch.einsum("blhd,bnhd->bhln", q, k) * self.scale
        logits = logits + bias.unsqueeze(1)  # broadcast over heads
        attn = logits.softmax(dim=-1)
        ctx = torch.einsum("bhln,bnhd->blhd", attn, v)
        return self.out(ctx.reshape(B, L, -1))


class _PlainMultiHeadSelfAttention(nn.Module):
    """Self-attn over latents — no positional encoding (parcel-id embedding owns identity).

    Accepts an optional ``key_padding_mask: (B, L) bool`` (True = valid, False
    = padded). v14-specific extension beyond canonical PerceiverIO/DETR: our
    latents have variable per-subject validity (no-coverage parcels), so we
    must mask invalid latents as keys/values to keep covered latents free of
    no-coverage contamination — preserving the zero-per-subject-params claim.
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
        key_padding_mask: Optional[Tensor] = None,  # (B, L) bool — True = valid
    ) -> Tensor:
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        attn_logits = torch.einsum("blhd,bshd->bhls", q, k) * self.scale
        if key_padding_mask is not None:
            invalid = ~key_padding_mask                       # (B, L)
            attn_logits = attn_logits.masked_fill(
                invalid.unsqueeze(1).unsqueeze(1),             # (B, 1, 1, L)
                torch.finfo(attn_logits.dtype).min,
            )
        attn = attn_logits.softmax(dim=-1)
        ctx = torch.einsum("bhls,bshd->blhd", attn, v)
        return self.out(ctx.reshape(B, L, -1))


def _ffn(d_model: int, mult: int = 4) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_model, d_model * mult),
        nn.GELU(),
        nn.Linear(d_model * mult, d_model),
    )


class _LatentSelfAttnBlock(nn.Module):
    """Latent self-attn + FFN, pre-LN.

    Forwards ``latent_valid: (B, L) bool`` as a key-padding mask to the inner
    attention (see :class:`_PlainMultiHeadSelfAttention`). Invalid positions
    still produce query outputs, but no covered position attends to them as
    keys/values.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _PlainMultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model)

    def forward(
        self,
        x: Tensor,
        latent_valid: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=latent_valid)
        x = x + self.ffn(self.ln2(x))
        return x


class _CrossAttnBlock(nn.Module):
    """Latent ← electrode cross-attn with anatomy bias, then FFN, pre-LN."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = _MultiHeadCrossAttentionWithBias(d_model, n_heads)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model)

    def forward(
        self,
        latents: Tensor,
        electrodes: Tensor,
        bias: Tensor,
        *,
        key_rope: Optional[Tensor] = None,
        t_bins: Optional[int] = None,
    ) -> Tensor:
        latents = latents + self.attn(
            self.ln_q(latents), self.ln_kv(electrodes), bias,
            key_rope=key_rope, t_bins=t_bins,
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


class V14ParcelPerceiverModel(nn.Module):
    """v14 Perceiver-IO encoder with parcel-id-tagged latents."""

    def __init__(
        self,
        *,
        n_freq_bins: int,
        n_time_bins: int,
        k_parcels: int,
        d_model: int = 128,
        n_heads: int = 4,
        depth_self_attn: int = 6,
        m_sub_slots: int = 4,
        time_last_input: bool = False,
    ) -> None:
        super().__init__()
        self.n_freq_bins = n_freq_bins
        self.n_time_bins = n_time_bins
        self.k_parcels = k_parcels
        self.m_sub_slots = m_sub_slots
        self.d_model = d_model
        self.time_last_input = time_last_input

        self.input_proj = nn.Linear(n_freq_bins, d_model)
        self.parcel_embedding = nn.Parameter(
            torch.randn(k_parcels, m_sub_slots, d_model) * (1.0 / math.sqrt(d_model))
        )
        self.cross_attn = _CrossAttnBlock(d_model, n_heads)
        # Time positional encoding for cross-attn keys (one-sided RoPE on K).
        # Q (latents) has no time, so it is unrotated. The spec commitment
        # "RoPE on per-electrode T_bins axis" is honored here without a
        # separate temporal self-attn block.
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        head_dim = d_model // n_heads
        self.register_buffer(
            "key_rope", _rope_freqs(head_dim, n_time_bins), persistent=False
        )
        self.latent_blocks = nn.ModuleList(
            [_LatentSelfAttnBlock(d_model, n_heads) for _ in range(depth_self_attn)]
        )
        self.encoder_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        electrode_tokens: Tensor,   # (B, C, T_bins, F_bins) or (B, C, F_bins, T_bins) if time_last_input
        support: Tensor,            # (B, C, K_parcels)
        valid_mask: Optional[Tensor] = None,  # (B, C) bool
        *,
        eps: float = DEFAULT_SUPPORT_BIAS_EPS,
    ) -> Tensor:
        if self.time_last_input:
            electrode_tokens = electrode_tokens.transpose(-1, -2)
        B, C, T, F = electrode_tokens.shape
        if F != self.n_freq_bins:
            raise ValueError(
                f"expected {self.n_freq_bins} freq bins, got {F}"
            )
        if T != self.n_time_bins:
            raise ValueError(
                f"expected {self.n_time_bins} time bins, got {T}"
            )
        if support.shape != (B, C, self.k_parcels):
            raise ValueError(
                f"support shape {tuple(support.shape)} does not match "
                f"(B, C, K) = ({B}, {C}, {self.k_parcels})"
            )

        x = self.input_proj(electrode_tokens)              # (B, C, T, d)
        electrodes = x.reshape(B, C * T, self.d_model)     # (B, N=C*T, d)

        bias = torch.log(support + eps)                    # (B, C, K)
        bias_per_latent = bias.unsqueeze(-1).expand(B, C, self.k_parcels, self.m_sub_slots)
        bias_per_latent = bias_per_latent.reshape(B, C, self.k_parcels * self.m_sub_slots)
        bias_full = bias_per_latent.unsqueeze(2).expand(B, C, T, self.k_parcels * self.m_sub_slots)
        bias_full = bias_full.reshape(B, C * T, self.k_parcels * self.m_sub_slots)
        bias_full = bias_full.transpose(1, 2)              # (B, L, N)

        if valid_mask is not None:
            invalid = ~valid_mask                          # (B, C)
            mask_per_token = invalid.unsqueeze(-1).expand(B, C, T).reshape(B, C * T)
            bias_full = bias_full.masked_fill(
                mask_per_token.unsqueeze(1), torch.finfo(bias_full.dtype).min
            )

        L = self.k_parcels * self.m_sub_slots
        latents = self.parcel_embedding.reshape(1, L, self.d_model).expand(B, L, self.d_model)

        latents = self.cross_attn(
            latents, electrodes, bias_full,
            key_rope=self.key_rope, t_bins=T,
        )

        latent_valid = _compute_latent_valid(
            support=support,
            valid_mask=valid_mask,
            m_sub_slots=self.m_sub_slots,
        )
        for block in self.latent_blocks:
            latents = block(latents, latent_valid=latent_valid)
        return self.encoder_ln(latents)


class V14ClassifierHead(nn.Module):
    """DETR-style readout: one full decoder layer (cross-attn + FFN) on a
    single learnable task query, then a linear classifier.

    Layout matches canonical DETR (Carion 2020 §3.3) decoder layer minus the
    trivial query self-attn (we have N=1 query so it would attend only to
    itself):

        q_state ← task_query  (expanded to B)
        q_state ← q_state + cross_attn(LN(q_state), latents, latents, mask)
        q_state ← q_state + ffn(LN(q_state))
        logits  ← classifier(q_state)

    The ``latent_valid`` kwarg masks no-coverage parcel slots in the cross-attn
    keys (memory-padding mask). The residual + FFN brings the head to a
    canonical DETR decoder layer; earlier the FFN was missing — unagreed
    implementation choice flagged 2026-05-13 and corrected here.
    """

    def __init__(self, d_model: int, n_classes: int, n_heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * (1.0 / math.sqrt(d_model)))
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(
        self,
        latents: Tensor,
        latent_valid: Optional[Tensor] = None,  # (B, L) bool
    ) -> Tensor:
        B, L, d = latents.shape
        q_state = self.query.expand(B, 1, d)                          # (B, 1, d)

        q = self.q_proj(self.ln_q(q_state)).reshape(
            B, 1, self.n_heads, self.head_dim
        )
        kv = self.kv_proj(self.ln_kv(latents)).reshape(
            B, L, 2, self.n_heads, self.head_dim
        )
        k, v = kv.unbind(dim=2)
        attn_logits = torch.einsum("blhd,bshd->bhls", q, k) * self.scale
        if latent_valid is not None:
            invalid = ~latent_valid                                    # (B, L)
            attn_logits = attn_logits.masked_fill(
                invalid.unsqueeze(1).unsqueeze(1),                     # (B, 1, 1, L)
                torch.finfo(attn_logits.dtype).min,
            )
        attn = attn_logits.softmax(dim=-1)
        ctx = torch.einsum("bhls,bshd->blhd", attn, v).reshape(B, 1, d)
        ctx = self.out_proj(ctx)

        h = q_state + ctx                                              # cross-attn residual
        h = h + self.ffn(self.ln_ffn(h))                               # FFN residual
        pooled = h.squeeze(1)                                          # (B, d)
        return self.classifier(pooled)


class V14ParcelPerceiverWithHead(nn.Module):
    """Encoder + classifier head bundled for single-task training."""

    def __init__(
        self,
        encoder: V14ParcelPerceiverModel,
        head: V14ClassifierHead,
        eps: float = DEFAULT_SUPPORT_BIAS_EPS,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.eps = eps

    def forward(
        self,
        electrode_tokens: Tensor,
        support: Tensor,
        valid_mask: Optional[Tensor] = None,
        *,
        eps: Optional[float] = None,
    ) -> Tensor:
        eps_used = self.eps if eps is None else eps
        latents = self.encoder(electrode_tokens, support, valid_mask, eps=eps_used)
        latent_valid = _compute_latent_valid(
            support=support,
            valid_mask=valid_mask,
            m_sub_slots=self.encoder.m_sub_slots,
        )
        return self.head(latents, latent_valid=latent_valid)


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
    d_model: int = 128
    n_heads: int = 4
    depth_self_attn: int = 6
    m_sub_slots: int = 4
    eps: float = DEFAULT_SUPPORT_BIAS_EPS
    time_last_input: bool = False

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
            time_last_input=self.time_last_input,
        )
        head = V14ClassifierHead(
            d_model=self.d_model, n_classes=n_classes, n_heads=self.n_heads
        )
        return V14ParcelPerceiverWithHead(encoder, head, eps=self.eps)
