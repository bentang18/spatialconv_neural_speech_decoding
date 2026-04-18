"""Combined spatiotemporal attention backbone for v14-core (Task C3).

Amended `#27`: replaces the factored (space/time) backbone with one
combined-attention pass per block over the flat sequence of `(N_e × T)`
tokens. ``B = 3`` blocks, 4 heads × ``head_dim = 16``, FFN width ``4d``,
pre-norm, dropout 0.1. RoPE is applied on the **temporal axis only**
(rotation depends on `t_a − t_b`, not on the electrode axis). No SC/FC
bias in the baseline (deferred to ablation A4 per the amended `#8`).

Masking (per the amendment):
- Keys: `electrode_active_mask` broadcast across time; inactive keys are
  masked out of attention.
- Queries: inactive rows are zeroed in the block output.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from speech_decoding.v14.config import BackboneConfig


_ROPE_BASE = 10000.0
_ROPE_MAX_T = 64  # headroom beyond T=28 (#6 token rate)


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate even/odd feature pairs by per-token angles.

    ``x`` is ``(B, S, H, D_h)``; ``cos``/``sin`` are ``(S, D_h // 2)``.
    """

    cos_ = cos.view(1, cos.shape[0], 1, cos.shape[1])
    sin_ = sin.view(1, sin.shape[0], 1, sin.shape[1])
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rot_even = x_even * cos_ - x_odd * sin_
    rot_odd = x_even * sin_ + x_odd * cos_
    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


class CombinedAttentionBlock(nn.Module):
    """One pre-norm combined-attention + FFN block."""

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.dropout_p = cfg.dropout

        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_hidden, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        active_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, S, d = x.shape
        active = active_mask.to(x.dtype).unsqueeze(-1)  # (B, S, 1)

        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(B, S, self.num_heads, self.head_dim)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        q = q.transpose(1, 2)  # (B, H, S, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Key mask: True = attend to this key. (B, 1, 1, S) broadcasts over (H, S_query).
        key_mask = active_mask.view(B, 1, 1, S)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=key_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(B, S, d)
        attn = self.o_proj(attn)
        attn = self.resid_dropout(attn)
        attn = attn * active
        x = x + attn

        h = self.norm2(x)
        h = self.ffn(h)
        h = h * active
        x = x + h

        return x * active  # invariant: inactive rows exit this block as zero


class Backbone(nn.Module):
    """3-block combined-attention stack over `(N_e × T)` tokens."""

    def __init__(self, cfg: BackboneConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or BackboneConfig()
        if cfg.num_heads * cfg.head_dim != cfg.d_model:
            raise ValueError(
                f"num_heads * head_dim ({cfg.num_heads} * {cfg.head_dim}) != "
                f"d_model ({cfg.d_model})"
            )
        if cfg.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {cfg.head_dim}")
        self.d_model = cfg.d_model
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.num_blocks = cfg.num_blocks

        self.blocks = nn.ModuleList(
            [CombinedAttentionBlock(cfg) for _ in range(cfg.num_blocks)]
        )

        half = cfg.head_dim // 2
        freqs = 1.0 / (_ROPE_BASE ** (torch.arange(half, dtype=torch.float32) / half))
        t_arange = torch.arange(_ROPE_MAX_T, dtype=torch.float32)
        angles = t_arange.unsqueeze(1) * freqs.unsqueeze(0)
        self.register_buffer("_rope_cos", torch.cos(angles), persistent=False)
        self.register_buffer("_rope_sin", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        active_mask: torch.Tensor,
        time_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the combined-attention stack.

        ``x``: ``(B, S, d)`` with ``S = N_e * T``.
        ``active_mask``: ``(B, S)`` bool.
        ``time_ids``: ``(S,)`` long, each token's time index in ``[0, T)``.
        """

        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"expected (B, S, {self.d_model}), got {tuple(x.shape)}"
            )
        if int(time_ids.max().item()) >= _ROPE_MAX_T:
            raise ValueError(
                f"time_id {int(time_ids.max().item())} >= {_ROPE_MAX_T} RoPE horizon"
            )
        cos = self._rope_cos[time_ids]
        sin = self._rope_sin[time_ids]
        x = x * active_mask.to(x.dtype).unsqueeze(-1)
        for block in self.blocks:
            x = block(x, active_mask, cos, sin)
        return x
