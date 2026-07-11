"""v14_converged_v3 — positional encoding (Phase 3).

Memo: project-v14-converged-v3-sensor-architecture.

L1RoPE — within-sensor JOINT spatiotemporal rotary. head_dim is split EQUALLY
across two axes (contact-index, time), each with the standard RoPE freq schedule.
RoPE rotates Q/K by ABSOLUTE position but the q·k score depends only on the
RELATIVE offset (Δindex, Δtime) — so a global per-shaft depth flip/offset is a
no-op (depth-flip is retired to a QC flag). Applied within a shaft (the
block-diagonal L1 attention supplies per-shaft grouping); the raw clinical contact
index (with drop-gaps) is the spatial coordinate.

BASE (audit L12/#8, 2026-07-10): base is NOT the LLM/video-inherited 10000 — that
is calibrated for context ~1e3-1e5, and against our ranges (index ~20 contacts,
time 128 slots) it leaves most rotary pairs turning <<1 radian end-to-end, i.e.
position-DEAD. A pair j turns R/base^(j/pairs) radians across a range R. Choosing
base so the LOWEST-frequency pair (j=pairs-1) turns ~half a cycle across R (≈R rad,
so ~π-3 rad — full ladder utilized, no wrap-around ambiguity) gives base ≈ R/(2π)
scaled up by the ladder: base_index=8 (R≈20 ⇒ every pair 2.9–20 rad, vs ~3/16 live
at 10000) and base_time=64 (R=128 ⇒ every pair 2.6–128 rad, vs ~9/16 live). Clean
powers of two near the derived optima; the sweep arms are {4,32}=λ_max≈R and
{16,128}=λ_max≈2R. NOTE we deliberately do NOT normalize index to a reference span
(upstream ``interpolate_rope``): our contact index maps to PHYSICAL depth at fixed
inter-contact spacing, so a shorter shaft SHOULD span less phase, and drop-gaps are
real physical gaps — normalizing would erase geometry that raw index encodes, and
because index is already physical, cross-subject phase is consistent WITHOUT it.

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
        base_index: float = 8.0,
        base_time: float = 64.0,
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

    def cos_sin(self, idx: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """The (..., seq, head_dim) fp32 cos/sin rotary table for coords (idx, t).

        B5 hoist (#46): every L1RoPE instance is identical (same head_dim, same
        bases), so within one tower forward the table is IDENTICAL across all L1
        blocks and across every head. Building it once here and feeding
        :meth:`rotate` in each block — instead of each block recomputing the
        ``pow``/``cos``/``sin``/``repeat_interleave`` over ~B·P·T tokens — is
        bit-identical (same fp32 math) and drops (n_L1−1) redundant table builds
        per tower. Kept fp32: the rope math runs in fp32 and the result is
        down-cast AFTER (attention.py); a bf16 table would change the rounding."""
        ang_idx = idx[..., None].float() * self.idx_freq  # (..., seq, pairs)
        ang_t = t[..., None].float() * self.t_freq
        ang = torch.cat([ang_idx, ang_t], dim=-1)  # (..., seq, head_dim/2)
        cos = ang.cos().repeat_interleave(2, dim=-1)  # (..., seq, head_dim)
        sin = ang.sin().repeat_interleave(2, dim=-1)
        return cos, sin

    @staticmethod
    def rotate(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        """Apply a precomputed (from :meth:`cos_sin`) rotary table to q, k.

        q, k: (..., H, seq, head_dim); cos, sin: (..., seq, head_dim) — the head
        axis is broadcast in via ``unsqueeze(-3)``."""
        cos = cos.unsqueeze(-3)  # (..., 1, seq, head_dim) → broadcast over heads
        sin = sin.unsqueeze(-3)
        q_out = q * cos + _rotate_half(q) * sin
        k_out = k * cos + _rotate_half(k) * sin
        return q_out, k_out

    def forward(
        self, q: Tensor, k: Tensor, idx: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """q, k: (..., H, seq, head_dim); idx, t: (..., seq) (no head axis)."""
        cos, sin = self.cos_sin(idx, t)  # (..., seq, head_dim)
        return self.rotate(q, k, cos, sin)


class ParcelIdentityEmbed(nn.Module):
    """Learned per-parcel identity embedding indexed by the DKT/DK hard tag."""

    def __init__(self, n_parcels: int, d_model: int, *, init_std: float = 1e-6) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_parcels, d_model)
        nn.init.trunc_normal_(self.embed.weight, std=init_std)
        # NEAR-ZERO init = upstream parity (audit finding #1, memo's strongest
        # change-candidate). Every additive-to-residual embed in V-JEPA 2.1 inits
        # ~0 so the model GROWS into it (modality embed std 1e-6 at
        # `app/vjepa_2_1/models/vision_transformer.py:187-188`; mask_token zero).
        # This table is added to BOTH encoder+predictor input and rides every block
        # (towers.py:97) — at the old 0.02 it was ~20% of the token signal at init,
        # a strong per-parcel prior from step 0 that biases toward memorizing parcel
        # identity early, exactly the failure mode in the high-repeat 30h regime.
        # 0.02 is retained as the A/B arm (pass init_std=0.02).
        #
        # NO weight-decay exemption: this 2-D table IS decayed, matching upstream
        # V-JEPA 2 exactly (`app/vjepa_2_1/utils.py` shape-only rule decays every
        # ≥2-D param, including its own 3-D modality embed) and our established
        # `optim_param_groups.is_no_decay` (ndim<=1). Tagging `_no_weight_decay`
        # here would be a silent divergence — and the splitter ignores it anyway.

    def forward(self, parcel_id: Tensor) -> Tensor:
        """parcel_id: (..., seq) long → (..., seq, d_model)."""
        return self.embed(parcel_id)
