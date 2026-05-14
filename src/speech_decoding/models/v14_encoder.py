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


class _RoPEMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE applied to Q and K."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer(
            "rope", _rope_freqs(self.head_dim, max_seq_len), persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B', T, d)
        Bp, T, _ = x.shape
        qkv = self.qkv(x).reshape(Bp, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B', T, H, head_dim)
        q = _apply_rope(q.transpose(1, 2), self.rope).transpose(1, 2)
        k = _apply_rope(k.transpose(1, 2), self.rope).transpose(1, 2)
        attn = torch.einsum("bthd,bshd->bhts", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        ctx = torch.einsum("bhts,bshd->bthd", attn, v)
        return self.out(ctx.reshape(Bp, T, -1))


class _MultiHeadCrossAttentionWithBias(nn.Module):
    """Latent ← electrode cross-attn with additive QK bias broadcast over heads."""

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
    ) -> Tensor:
        B, L, _ = latents.shape
        N = electrodes.shape[1]
        q = self.q_proj(latents).reshape(B, L, self.n_heads, self.head_dim)
        kv = self.kv_proj(electrodes).reshape(B, N, 2, self.n_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        logits = torch.einsum("blhd,bnhd->bhln", q, k) * self.scale
        logits = logits + bias.unsqueeze(1)  # broadcast over heads
        attn = logits.softmax(dim=-1)
        ctx = torch.einsum("bhln,bnhd->blhd", attn, v)
        return self.out(ctx.reshape(B, L, -1))


class _PlainMultiHeadSelfAttention(nn.Module):
    """Self-attn over latents — no positional encoding (parcel-id embedding owns identity)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        attn = torch.einsum("blhd,bshd->bhls", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        ctx = torch.einsum("bhls,bshd->blhd", attn, v)
        return self.out(ctx.reshape(B, L, -1))


def _ffn(d_model: int, mult: int = 4) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_model, d_model * mult),
        nn.GELU(),
        nn.Linear(d_model * mult, d_model),
    )


class _TemporalEncoderBlock(nn.Module):
    """Per-electrode temporal self-attn with RoPE + FFN, pre-LN."""

    def __init__(self, d_model: int, n_heads: int, max_t_bins: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _RoPEMultiHeadSelfAttention(d_model, n_heads, max_t_bins)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class _LatentSelfAttnBlock(nn.Module):
    """Latent self-attn + FFN, pre-LN."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _PlainMultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
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

    def forward(self, latents: Tensor, electrodes: Tensor, bias: Tensor) -> Tensor:
        latents = latents + self.attn(self.ln_q(latents), self.ln_kv(electrodes), bias)
        latents = latents + self.ffn(self.ln_ffn(latents))
        return latents


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
        depth_temporal: int = 1,
        m_sub_slots: int = 4,
    ) -> None:
        super().__init__()
        self.n_freq_bins = n_freq_bins
        self.n_time_bins = n_time_bins
        self.k_parcels = k_parcels
        self.m_sub_slots = m_sub_slots
        self.d_model = d_model

        self.input_proj = nn.Linear(n_freq_bins, d_model)
        self.temporal_blocks = nn.ModuleList(
            [
                _TemporalEncoderBlock(d_model, n_heads, n_time_bins)
                for _ in range(depth_temporal)
            ]
        )
        self.parcel_embedding = nn.Parameter(
            torch.randn(k_parcels, m_sub_slots, d_model) * (1.0 / math.sqrt(d_model))
        )
        self.cross_attn = _CrossAttnBlock(d_model, n_heads)
        self.latent_blocks = nn.ModuleList(
            [_LatentSelfAttnBlock(d_model, n_heads) for _ in range(depth_self_attn)]
        )
        self.encoder_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        electrode_tokens: Tensor,   # (B, C, T_bins, F_bins)
        support: Tensor,            # (B, C, K_parcels)
        valid_mask: Optional[Tensor] = None,  # (B, C) bool
        *,
        eps: float = DEFAULT_SUPPORT_BIAS_EPS,
    ) -> Tensor:
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
        x = x.reshape(B * C, T, self.d_model)
        for block in self.temporal_blocks:
            x = block(x)
        x = x.reshape(B, C, T, self.d_model)
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

        latents = self.cross_attn(latents, electrodes, bias_full)
        for block in self.latent_blocks:
            latents = block(latents)
        return self.encoder_ln(latents)


class V14ClassifierHead(nn.Module):
    """DETR-style readout: one learnable query → cross-attn over latents → linear."""

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
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, latents: Tensor) -> Tensor:
        B, L, d = latents.shape
        q = self.q_proj(self.ln_q(self.query.expand(B, 1, d))).reshape(
            B, 1, self.n_heads, self.head_dim
        )
        kv = self.kv_proj(self.ln_kv(latents)).reshape(
            B, L, 2, self.n_heads, self.head_dim
        )
        k, v = kv.unbind(dim=2)
        attn = (torch.einsum("blhd,bshd->bhls", q, k) * self.scale).softmax(dim=-1)
        ctx = torch.einsum("bhls,bshd->blhd", attn, v).reshape(B, 1, d)
        pooled = self.out_proj(ctx).squeeze(1)
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
        return self.head(latents)


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
    depth_temporal: int = 1
    m_sub_slots: int = 4
    eps: float = DEFAULT_SUPPORT_BIAS_EPS

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
            depth_temporal=self.depth_temporal,
            m_sub_slots=self.m_sub_slots,
        )
        head = V14ClassifierHead(
            d_model=self.d_model, n_classes=n_classes, n_heads=self.n_heads
        )
        return V14ParcelPerceiverWithHead(encoder, head, eps=self.eps)
