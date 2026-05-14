"""Cross-attn one-sided RoPE tests.

After dropping the per-electrode temporal self-attn block (the implementation
artifact that was never explicitly agreed), the spec commitment "RoPE on
per-electrode T_bins axis" is honored by applying one-sided RoPE to the K
vectors inside the parcel cross-attention. K has a time index (each
electrode-time-bin token); Q (latents) has no time, so it stays unrotated.

This is unusual versus canonical bidirectional RoPE (Su 2021) but principled:
- absolute K position is what we need (Q has none, so "relative" is degenerate);
- it keeps temporal positional information without inventing a separate
  per-electrode self-attn block.

Pins:
1. The encoder distinguishes time-bin order (permuting a single electrode's
   T_bins changes the output).
2. ``V14ParcelPerceiverModel`` has no ``temporal_blocks`` attribute and the
   config has no ``depth_temporal`` field.
3. End-to-end shape/dtype/finite invariants still hold.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import (
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
)


def _kwargs(**overrides) -> dict:
    base = {
        "n_freq_bins": 4,
        "n_time_bins": 5,
        "k_parcels": 6,
        "d_model": 32,
        "n_heads": 4,
        "depth_self_attn": 1,
        "m_sub_slots": 2,
    }
    base.update(overrides)
    return base


def test_encoder_distinguishes_time_bin_order_via_one_sided_rope() -> None:
    """Permuting an electrode's time-bins must change the encoder output.
    Without temporal PE the cross-attn would treat T_bins as exchangeable —
    a feature loss the spec ("RoPE on T_bins") explicitly forbids.
    """
    torch.manual_seed(0)
    model = V14ParcelPerceiverModel(**_kwargs())
    model.eval()

    B, C, T, F = 1, 2, 5, 4
    electrodes = torch.randn(B, C, T, F)
    support = torch.zeros(B, C, 6)
    support[0, 0, 0] = 1.0
    support[0, 1, 1] = 1.0

    with torch.no_grad():
        out_ref = model(electrodes, support)
        permuted = electrodes.clone()
        permuted[0, 0] = electrodes[0, 0].flip(dims=[0])  # reverse T for electrode 0
        out_perm = model(permuted, support)

    assert not torch.allclose(out_ref, out_perm, atol=1e-5), (
        "encoder must distinguish time-bin order via one-sided RoPE on K"
    )


def test_encoder_has_no_temporal_block_attribute() -> None:
    """The dedicated per-electrode temporal self-attn block was an
    implementation artifact (never agreed). Pin its removal."""
    model = V14ParcelPerceiverModel(**_kwargs())
    assert not hasattr(model, "temporal_blocks"), (
        "temporal_blocks must be removed; cross-attn carries time via RoPE-on-K"
    )


def test_config_has_no_depth_temporal_field() -> None:
    """``V14ParcelPerceiver`` config no longer exposes ``depth_temporal``."""
    cfg = V14ParcelPerceiver(n_freq_bins=4, n_time_bins=5, k_parcels=6)
    assert not hasattr(cfg, "depth_temporal"), (
        "V14ParcelPerceiver must not carry a depth_temporal field"
    )


def test_encoder_forward_shape_dtype_finite_after_block_removal() -> None:
    """End-to-end shape/dtype/finite invariants survive the refactor."""
    model = V14ParcelPerceiverModel(**_kwargs())
    B, C = 2, 7
    electrodes = torch.randn(B, C, 5, 4)
    support = torch.zeros(B, C, 6)
    support[..., 0] = 1.0

    out = model(electrodes, support)
    assert out.shape == (B, 6 * 2, 32)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
