"""P2.3 tests for the converged-v2 set-pool (block-diagonal PMA, k=2)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models import v14_converged_v2 as v2
from speech_decoding.models.v14_converged_v2 import (
    SetPoolV2,
    active_parcels,
    cell_operator_index,
    n_operators,
)

N_PARCELS = 62  # DKT-sized universal table (placeholder; exact count at wiring)


def _setup(clip_s=1.0, b=2, tie_lfs=True, d=32, n_heads=4, k=2):
    """4 electrodes in 2 parcels (labels 5, 9), tied-default operators."""
    bands = v2.bands_for_clip_len(clip_s)
    S = sum(bd.n_tokens for bd in bands)
    parcel_of_electrode = torch.tensor([5, 5, 9, 9])
    parcel_labels, membership = active_parcels(parcel_of_electrode)
    cell_patch = cell_operator_index(bands, tie_lfs=tie_lfs)
    pool = SetPoolV2(d, n_heads, k=k, n_parcels=N_PARCELS, n_op=n_operators(tie_lfs))
    torch.manual_seed(0)
    x = torch.randn(b, 4, S, d)
    return pool, x, membership, parcel_labels, cell_patch, bands, S


def test_operator_index_tied_vs_untied():
    tied = cell_operator_index(v2.BANDS_V2, tie_lfs=True)
    untied = cell_operator_index(v2.BANDS_V2, tie_lfs=False)
    # tied = band_id (LFS 0, HGA 1) ⇒ 2 operators
    assert set(tied.tolist()) == {0, 1}
    assert n_operators(True) == 2
    # untied = freq_patch_id (LFS 0/1/2, HGA 3) ⇒ 4 operators
    assert set(untied.tolist()) == {0, 1, 2, 3}
    assert n_operators(False) == 4


def test_active_parcels():
    poe = torch.tensor([5, 5, 9, 9, 5])
    labels, membership = active_parcels(poe)
    assert labels.tolist() == [5, 9]
    assert membership.tolist() == [[True, True, False, False, True],
                                   [False, False, True, True, False]]


@pytest.mark.parametrize("clip_s, total", [(1.0, 22), (5.0, 110)])
def test_pool_output_shape(clip_s, total):
    pool, x, membership, labels, cell_patch, bands, S = _setup(clip_s)
    out = pool(x, membership, labels, cell_patch)
    assert out.shape == (2, 2, 2, total, 32)  # (B, P, k, S, d)
    assert S == total


def test_block_diagonal_cross_parcel_isolation():
    """Seed of parcel p is invariant to electrodes in OTHER parcels (block-diag)."""
    pool, x, membership, labels, cell_patch, bands, S = _setup()
    pool.eval()
    with torch.no_grad():
        out1 = pool(x, membership, labels, cell_patch)
        x2 = x.clone()
        x2[:, 2:4] += 7.0                       # corrupt parcel-1 electrodes (2,3)
        out2 = pool(x2, membership, labels, cell_patch)
    # parcel 0 (index 0) seeds unchanged; parcel 1 (index 1) changes.
    assert torch.allclose(out1[:, 0], out2[:, 0], atol=1e-6)
    assert not torch.allclose(out1[:, 1], out2[:, 1])


def test_intra_parcel_permutation_invariance():
    """Pooling a parcel's electrode SET is permutation-invariant."""
    pool, x, membership, labels, cell_patch, bands, S = _setup()
    pool.eval()
    perm = torch.tensor([1, 0, 2, 3])           # swap the two parcel-0 electrodes
    with torch.no_grad():
        out1 = pool(x, membership, labels, cell_patch)
        out2 = pool(x[:, perm], membership[:, perm], labels, cell_patch)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_freq_specific_operators():
    """Zeroing the HGA value-operator zeroes ONLY the HGA cells' pooled output."""
    pool, x, membership, labels, cell_patch, bands, S = _setup()  # tied n_op=2
    lfs_band, hga_band = bands
    n_lfs = lfs_band.n_tokens
    # tied: operator = band_id ⇒ HGA cells use operator 1.
    with torch.no_grad():
        pool.W_V[1].zero_()
        out = pool(x, membership, labels, cell_patch)
    # HGA cells (S index ≥ n_lfs) pool zero values ⇒ out (bias-free) exactly 0.
    assert torch.allclose(out[:, :, :, n_lfs:], torch.zeros_like(out[:, :, :, n_lfs:]))
    assert not torch.allclose(out[:, :, :, :n_lfs], torch.zeros_like(out[:, :, :, :n_lfs]))


def test_untied_pool_runs_with_4_operators():
    pool, x, membership, labels, cell_patch, bands, S = _setup(tie_lfs=False)
    assert pool.W_K.shape[0] == 4
    out = pool(x, membership, labels, cell_patch)
    assert out.shape[-2] == S


def test_pool_rejects_wrong_cell_patch_length():
    pool, x, membership, labels, cell_patch, bands, S = _setup()
    with pytest.raises(ValueError, match="cell_patch must be"):
        pool(x, membership, labels, cell_patch[:-1])


def test_pool_gradient_flows():
    pool, x, membership, labels, cell_patch, bands, S = _setup()
    x = x.requires_grad_(True)
    out = pool(x, membership, labels, cell_patch)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert pool.embed_pq.grad is not None  # query embed supervised
