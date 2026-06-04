"""Cross-attn temporal-distinction + interim shape tests.

Lineage note: the v3 spec carried "RoPE on per-electrode T_bins axis" via
one-sided RoPE on cross-attn K vectors (because K had a time axis when
electrode tokens were flattened to ``(B, C*T, d)``). The v4 amendment
(5/19) factorises cross-attn per-timestep — K is now ``(B, C, d)`` per t,
no time index, no RoPE on K. The latent-stack time-SA (T1.9) carries
temporal position instead, via RoPE on the latent time axis.

These tests survive the refactor by checking weaker but still-load-bearing
invariants: the encoder must distinguish time-bin order (each t's
cross-attn sees different electrode values), there is no per-electrode
temporal self-attn block, and the output shape/finiteness contract holds.
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
        "patch_kernel_freq": 3,  # FE-RAW-1: kernel-3 path for the tiny F=4 config
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
    """End-to-end shape/dtype/finite invariants survive the v4 latent-time-axis
    refactor (T1.1) + B20 FE-02 patch stem: output is ``(B, K*M, T_p, d)``
    where ``T_p = (T - k_t) // k_t + 1`` is the post-patch frame count."""
    model = V14ParcelPerceiverModel(**_kwargs())
    B, C, T, F = 2, 7, 5, 4
    electrodes = torch.randn(B, C, T, F)
    support = torch.zeros(B, C, 6)
    support[..., 0] = 1.0

    out = model(electrodes, support)
    # T=5 with kernel_time=2 stride=2: T_p = (5 - 2) // 2 + 1 = 2
    T_p = model.patch_stem.n_time_patches(T)
    assert out.shape == (B, 6 * 2, T_p, 32)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
