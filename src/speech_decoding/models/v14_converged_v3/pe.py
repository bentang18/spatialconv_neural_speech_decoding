"""v14_converged_v3 — positional encoding (Phase 3).

Memo: project-v14-converged-v3-sensor-architecture.

L1RoPE — within-sensor JOINT spatiotemporal rotary. head_dim is split EQUALLY
across two axes (contact-index, time), each with the standard RoPE freq schedule
(base 10000; VideoRoPE low-freq-on-time is an ablation, not the default). RoPE
rotates Q/K by ABSOLUTE position but the q·k score depends only on the RELATIVE
offset (Δindex, Δtime) — so a global per-shaft depth flip/offset is a no-op
(depth-flip is retired to a QC flag). Applied within a shaft (the block-diagonal
L1 attention supplies per-shaft grouping); the raw clinical contact index (with
drop-gaps) is the spatial coordinate.

ParcelIdentityEmbed — L2 cross-sensor identity: a LEARNED embedding indexed by
the DKT/DK hard parcel tag (the only cross-subject-meaningful "who am I"), added
to tokens. NO geometric RoPE across sensors — there is no shared spatial metric
(RAS dropped, MNI banned); cross-sensor coupling is functional, learned.

Pair convention matches v14_encoder._rope_freqs: dims (2j, 2j+1) rotate together
(repeat_interleave(2)); rotate_half = [-x_odd, x_even].
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def init_transformer_weights(m: nn.Module, init_std: float = 0.02) -> None:
    """V-JEPA 2 module init (``vision_transformer.py:130-141``): Linear weights
    ``trunc_normal_(std)`` + zero bias; LayerNorm weight 1 / bias 0. Applied via
    ``module.apply(...)``; Embeddings are skipped (they self-init in ``__init__``)."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=init_std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(m.weight, 1.0)


def _rotate_half(x: Tensor) -> Tensor:
    x_paired = x.unflatten(-1, (-1, 2))  # (..., hd/2, 2)
    return torch.stack([-x_paired[..., 1], x_paired[..., 0]], dim=-1).flatten(-2)


class L1RoPE(nn.Module):
    """Mixed 2-axis (contact-index, time) rotary PE, equal head_dim split."""

    def __init__(
        self,
        head_dim: int,
        *,
        base_index: float = 10_000.0,
        base_time: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"L1RoPE needs head_dim % 4 == 0 (equal even split across 2 axes), "
                f"got {head_dim}"
            )
        self.head_dim = head_dim
        pairs = head_dim // 4  # rotation pairs per axis
        idx_freq = 1.0 / (base_index ** (torch.arange(pairs).float() / pairs))
        t_freq = 1.0 / (base_time ** (torch.arange(pairs).float() / pairs))
        self.register_buffer("idx_freq", idx_freq, persistent=False)
        self.register_buffer("t_freq", t_freq, persistent=False)

    def _cos_sin(self, idx: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        # idx, t: (..., seq) → cos/sin: (..., seq, head_dim)
        ang_idx = idx[..., None].float() * self.idx_freq  # (..., seq, pairs)
        ang_t = t[..., None].float() * self.t_freq
        ang = torch.cat([ang_idx, ang_t], dim=-1)  # (..., seq, head_dim/2)
        cos = ang.cos().repeat_interleave(2, dim=-1)  # (..., seq, head_dim)
        sin = ang.sin().repeat_interleave(2, dim=-1)
        return cos, sin

    def forward(
        self, q: Tensor, k: Tensor, idx: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """q, k: (..., H, seq, head_dim); idx, t: (..., seq) (no head axis)."""
        cos, sin = self._cos_sin(idx, t)  # (..., seq, head_dim)
        cos = cos.unsqueeze(-3)  # (..., 1, seq, head_dim) → broadcast over heads
        sin = sin.unsqueeze(-3)
        q_out = q * cos + _rotate_half(q) * sin
        k_out = k * cos + _rotate_half(k) * sin
        return q_out, k_out


class ParcelIdentityEmbed(nn.Module):
    """Learned per-parcel identity embedding indexed by the DKT/DK hard tag."""

    def __init__(self, n_parcels: int, d_model: int, *, init_std: float = 0.02) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_parcels, d_model)
        nn.init.trunc_normal_(self.embed.weight, std=init_std)
        # NO weight-decay exemption: this 2-D table IS decayed, matching upstream
        # V-JEPA 2 exactly (`app/vjepa_2_1/utils.py` shape-only rule decays every
        # ≥2-D param, including its own 3-D modality embed) and our established
        # `optim_param_groups.is_no_decay` (ndim<=1). Tagging `_no_weight_decay`
        # here would be a silent divergence — and the splitter ignores it anyway.

    def forward(self, parcel_id: Tensor) -> Tensor:
        """parcel_id: (..., seq) long → (..., seq, d_model)."""
        return self.embed(parcel_id)
