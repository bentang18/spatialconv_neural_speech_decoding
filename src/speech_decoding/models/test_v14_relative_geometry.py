"""Tests for RelativeGeometry — the Run-B native-RAS relative positional bias.

Maps a centroid-relative offset Δ (mm) to a per-head additive attention-logit
bias via σ-normalized Fourier features + a tiny MLP. Invariants: shape, zero-init
(bias == 0 at init ⇒ pool starts as pure content attention), translation-
invariance-by-construction (same Δ ⇒ same bias), and direction-sensitivity (the
features distinguish Δ from −Δ, which the reconstruction query needs).
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v2 import RelativeGeometry


def test_geometry_output_shape():
    geom = RelativeGeometry(n_heads=6, sigma_mm=12.0, n_freqs=5)
    delta = torch.randn(2, 7, 11, 3)          # (B, Q, K, 3)
    out = geom(delta)
    assert out.shape == (2, 7, 11, 6)         # (..., n_heads)


def test_geometry_feature_dim():
    geom = RelativeGeometry(n_heads=4, n_freqs=5)
    feats = geom.features(torch.randn(3, 3))
    assert feats.shape == (3, 3 * (2 * 5 + 1))  # per axis: raw + sin,cos×n_freqs


def test_geometry_zero_init_bias_is_zero():
    """Last layer zero-init ⇒ every bias is exactly 0 at init (safe warm start)."""
    geom = RelativeGeometry(n_heads=6, n_freqs=4)
    delta = torch.randn(50, 3) * 10.0
    out = geom(delta)
    assert torch.allclose(out, torch.zeros_like(out))


def test_geometry_same_offset_same_bias():
    """Δ is relative ⇒ translation-invariant. Two electrode/seed pairs with the
    same offset must get an identical bias (train the MLP a step to leave init)."""
    geom = RelativeGeometry(n_heads=3, n_freqs=5)
    for p in geom.mlp[-1].parameters():       # nudge off zero-init
        torch.nn.init.normal_(p, std=0.1)
    d = torch.tensor([[3.0, -4.0, 5.0]])
    b1 = geom(d)
    b2 = geom(d + 0.0)                          # identical Δ
    assert torch.allclose(b1, b2)


def test_geometry_direction_sensitive_features():
    """features(Δ) ≠ features(−Δ) (sin is odd) — the recon query can distinguish
    two electrodes equidistant from the centroid but in opposite directions."""
    geom = RelativeGeometry(n_heads=3, n_freqs=5)
    d = torch.tensor([[2.0, -1.0, 3.0]])
    assert not torch.allclose(geom.features(d), geom.features(-d))


def test_geometry_trained_bias_direction_sensitive():
    """After leaving zero-init, the per-head bias itself differs for ±Δ."""
    geom = RelativeGeometry(n_heads=4, n_freqs=5)
    for p in geom.mlp[-1].parameters():
        torch.nn.init.normal_(p, std=0.2)
    d = torch.tensor([[2.0, -1.0, 3.0]])
    assert not torch.allclose(geom(d), geom(-d))


def test_geometry_omega_is_buffer_moves_with_module():
    geom = RelativeGeometry(n_heads=2, n_freqs=3)
    assert "omega" in dict(geom.named_buffers())
    assert geom.omega.shape == (3,)
    # ω=2π/λ with geometrically increasing wavelengths ⇒ strictly DECREASING ω
    # (index 0 = shortest wavelength/highest freq, resolves neighbours; index -1 =
    # longest wavelength/lowest freq, the smooth cross-parcel ramp).
    assert (geom.omega[1:] < geom.omega[:-1]).all()


def test_geometry_dtype_preserved():
    geom = RelativeGeometry(n_heads=3, n_freqs=4)
    for p in geom.mlp[-1].parameters():
        torch.nn.init.normal_(p, std=0.1)
    out = geom(torch.randn(5, 3, dtype=torch.float32))
    assert out.dtype == torch.float32
