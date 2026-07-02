"""P2.3 tests for the converged-v2 set-pool (block-diagonal PMA, k=2)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models import v14_converged_v2 as v2
from speech_decoding.models.v14_converged_v2 import (
    NEG_INF_MASK_VALUE,
    SetPoolV2,
    active_parcels,
    cell_operator_index,
    n_operators,
)

N_PARCELS = 62  # DKT-sized universal table (placeholder; exact count at wiring)


def _setup(clip_s=1.0, b=2, op_mode="band", d=32, n_heads=4, k=2):
    """4 electrodes in 2 parcels (labels 5, 9), band-tied operators by default."""
    bands = v2.bands_for_clip_len(clip_s)
    S = sum(bd.n_tokens for bd in bands)
    parcel_of_electrode = torch.tensor([5, 5, 9, 9])
    parcel_labels, membership = active_parcels(parcel_of_electrode)
    cell_patch = cell_operator_index(bands, op_mode=op_mode)
    pool = SetPoolV2(d, n_heads, k=k, n_parcels=N_PARCELS, n_op=n_operators(op_mode))
    torch.manual_seed(0)
    x = torch.randn(b, 4, S, d)
    return pool, x, membership, parcel_labels, cell_patch, bands, S


def test_operator_index_shared_band_patch():
    shared = cell_operator_index(v2.BANDS_V2, op_mode="shared")
    band = cell_operator_index(v2.BANDS_V2, op_mode="band")
    patch = cell_operator_index(v2.BANDS_V2, op_mode="patch")
    # shared = one operator for every cell ⇒ 1 operator
    assert set(shared.tolist()) == {0}
    assert n_operators("shared") == 1
    # band = band_id (LFS 0, HGA 1) ⇒ 2 operators
    assert set(band.tolist()) == {0, 1}
    assert n_operators("band") == 2
    # patch = freq_patch_id (LFS 0/1/2, HGA 3) ⇒ 4 operators
    assert set(patch.tolist()) == {0, 1, 2, 3}
    assert n_operators("patch") == 4


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
    pool, x, membership, labels, cell_patch, bands, S = _setup(op_mode="patch")
    assert pool.W_K.shape[0] == 4
    out = pool(x, membership, labels, cell_patch)
    assert out.shape[-2] == S


def test_shared_pool_runs_with_1_operator():
    pool, x, membership, labels, cell_patch, bands, S = _setup(op_mode="shared")
    assert pool.W_K.shape[0] == 1
    assert set(cell_patch.tolist()) == {0}
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


# ---- Run-B pool bias: block (drop / heterogeneous M2 mask) + geom_bias ------
def test_pool_block_and_geom_none_is_identical():
    """block=None, geom_bias=None ⇒ byte-identical to the base pool."""
    pool, x, membership, labels, cell_patch, bands, S = _setup()
    base = pool(x, membership, labels, cell_patch)
    same = pool(x, membership, labels, cell_patch, block=None, geom_bias=None)
    assert torch.equal(base, same)


def test_pool_block_is_localized_and_nan_free():
    """Blocking parcel-9's electrodes at (clip0, cell0) leaves every OTHER
    (clip, cell) bit-identical to base, and the fully-blocked seed rows stay
    finite (the -1e4 sentinel keeps softmax NaN-free)."""
    pool, x, membership, labels, cell_patch, bands, S = _setup()
    B, C = x.shape[0], x.shape[1]
    base = pool(x, membership, labels, cell_patch)
    block = torch.zeros(B, S, C, dtype=torch.bool)
    block[0, 0, 2:4] = True                       # parcel-9 electrodes (idx 2,3) at cell 0
    out = pool(x, membership, labels, cell_patch, block=block)
    assert torch.isfinite(out).all()
    # out shape (B,P,k,S,d); only clip0/cell0 may differ. Everything else equal.
    mask = torch.ones_like(out, dtype=torch.bool)
    mask[0, :, :, 0, :] = False                   # clip0, cell0, all parcels/seeds
    assert torch.equal(out[mask], base[mask])
    assert not torch.equal(out[0, 1, :, 0, :], base[0, 1, :, 0, :])  # parcel-9 seed changed


def test_pool_block_all_cells_equals_geom_neg_inf():
    """Blocking electrode 3 at EVERY (clip,cell) must equal a geom_bias of -1e4
    on electrode-3's column for all seeds/heads — the two additive paths compose
    identically into the SDPA mask."""
    pool, x, membership, labels, cell_patch, bands, S = _setup()
    B, C = x.shape[0], x.shape[1]
    H, Pk = pool.n_heads, membership.shape[0] * pool.k
    block = torch.zeros(B, S, C, dtype=torch.bool)
    block[:, :, 3] = True
    via_block = pool(x, membership, labels, cell_patch, block=block)
    geom = torch.zeros(H, Pk, C)
    geom[:, :, 3] = NEG_INF_MASK_VALUE
    via_geom = pool(x, membership, labels, cell_patch, geom_bias=geom)
    torch.testing.assert_close(via_block, via_geom)


def test_pool_geom_bias_shifts_attention():
    """A nonzero geom_bias changes the pooled output (it enters the logits)."""
    pool, x, membership, labels, cell_patch, bands, S = _setup()
    C = x.shape[1]
    H, Pk = pool.n_heads, membership.shape[0] * pool.k
    base = pool(x, membership, labels, cell_patch)
    geom = torch.zeros(H, Pk, C)
    geom[:, :, 0] = 5.0                            # strongly favour electrode 0
    out = pool(x, membership, labels, cell_patch, geom_bias=geom)
    assert not torch.equal(out, base)
    assert torch.isfinite(out).all()


def test_pool_block_bad_shape_raises():
    pool, x, membership, labels, cell_patch, bands, S = _setup()
    with pytest.raises(ValueError, match="block must be"):
        pool(x, membership, labels, cell_patch, block=torch.zeros(3, 3, dtype=torch.bool))
