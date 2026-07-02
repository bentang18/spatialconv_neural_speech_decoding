"""Tests for ElectrodeReconHead — the Run-B MNI-free reconstruction decode.

A dropped electrode's teacher frontend feature is reconstructed from its parcel's
k pool seeds by a single positional-query cross-attention. The query is the
electrode's centroid-relative offset (Fourier features), so two electrodes at
different positions decode DIFFERENTLY from the same seeds — the pressure that
forces the pool to keep a spatially-resolved field. Invariants: shape, offset-
sensitivity, per-row independence, gradient flow.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v2 import (
    ElectrodeReconHead,
    RelativeGeometry,
)


def _head(d=32, n_heads=4, n_freqs=5):
    geom = RelativeGeometry(n_heads=n_heads, n_freqs=n_freqs)
    head = ElectrodeReconHead(d, n_heads, geom.feat_dim)
    return head, geom


def test_recon_output_shape():
    head, geom = _head()
    N, k, d = 7, 2, 32
    seeds = torch.randn(N, k, d)
    feats = geom.features(torch.randn(N, 3))
    out = head(seeds, feats)
    assert out.shape == (N, d)


def test_recon_offset_changes_prediction():
    """Same seeds, different centroid-relative offset ⇒ different reconstruction
    (the query addresses WHICH electrode)."""
    head, geom = _head()
    N, k, d = 5, 2, 32
    seeds = torch.randn(N, k, d)
    out_a = head(seeds, geom.features(torch.zeros(N, 3)))
    out_b = head(seeds, geom.features(torch.full((N, 3), 4.0)))
    assert not torch.allclose(out_a, out_b)


def test_recon_same_input_same_output():
    head, geom = _head()
    seeds = torch.randn(3, 2, 32)
    feats = geom.features(torch.randn(3, 3))
    assert torch.equal(head(seeds, feats), head(seeds, feats))


def test_recon_row_independent():
    """Row i's prediction is invariant to other rows' seeds/offsets (batched-
    independent decode — one electrode never leaks into another's recon)."""
    head, geom = _head()
    seeds = torch.randn(4, 2, 32)
    off = torch.randn(4, 3)
    full = head(seeds, geom.features(off))
    single = head(seeds[1:2], geom.features(off[1:2]))
    assert torch.allclose(full[1:2], single, atol=1e-6)


def test_recon_gradient_flows():
    head, geom = _head()
    seeds = torch.randn(6, 2, 32, requires_grad=True)
    off = torch.randn(6, 3)
    out = head(seeds, geom.features(off))
    out.sum().backward()
    assert seeds.grad is not None and torch.isfinite(seeds.grad).all()
    assert head.q_pos.weight.grad is not None      # positional query supervised
