"""P2.6 tests for converged-v2 parcel-uniform M2 masks + parcel tube (vectorized).

Locked spec (frontend memo, 2026-06-26): M2 is PARCEL-UNIFORM with a constant
masked count per parcel ``n_mask = n_mask_hga + T_lfs`` — this constancy is what
makes every downstream shape static. HGA = wav2vec2 fill-not-trim spans (exact
``round(frac_hga·T_hga)`` cells, span-granular, last span tail-trimmed); LFS =
freq-tube (exactly one of the 3 log groups). All sampling is vectorized over
``B·P`` parcels (no python loop — the ~455 ms/step mask bubble is forbidden)."""

from __future__ import annotations

import torch

from speech_decoding.models import v14_converged_v2 as v2
from speech_decoding.models.v14_converged_v2 import (
    n_mask_hga,
    sample_m2_masks_v2,
    sample_parcel_tube_v2,
)


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _counts(clip_s):
    bands = v2.bands_for_clip_len(clip_s)
    lfs, hga = bands
    return bands, n_mask_hga(hga), hga.n_time_patches, lfs.n_time_patches, lfs.n_freq_patches


def test_n_mask_hga_frac_half():
    # frac_hga = 0.50 ⇒ 8 @1s (T_hga 16), 40 @5s (T_hga 80).
    _, hga1 = v2.bands_for_clip_len(1.0)
    _, hga5 = v2.bands_for_clip_len(5.0)
    assert n_mask_hga(hga1) == 8
    assert n_mask_hga(hga5) == 40


def test_m2_mask_shape_and_constant_total():
    bands, nmh, T_hga, T_lfs, _ = _counts(1.0)
    B, P = 4, 5
    m2 = sample_m2_masks_v2(B, P, bands, _gen())
    S = sum(b.n_tokens for b in bands)
    assert m2.shape == (B, P, S)
    assert m2.dtype == torch.bool
    total = nmh + T_lfs
    # EVERY parcel masks exactly n_mask = n_mask_hga + T_lfs — the static invariant.
    assert torch.equal(m2.sum(-1), torch.full((B, P), total))


def test_m2_lfs_exactly_one_group():
    """LFS part holds exactly one full log-group block (T_lfs cells, contiguous)."""
    bands, _, _, T_lfs, n_groups = _counts(5.0)
    B, P = 3, 4
    m2 = sample_m2_masks_v2(B, P, bands, _gen())
    n_lfs = n_groups * T_lfs
    lfs_part = m2[..., :n_lfs].reshape(B, P, n_groups, T_lfs)
    per_group = lfs_part.any(-1)                       # (B,P,n_groups) which groups touched
    assert torch.equal(per_group.sum(-1), torch.ones(B, P, dtype=torch.long))  # exactly 1
    # the touched group is fully masked (all T_lfs time cells)
    assert torch.equal(lfs_part.sum(-1).max(-1).values, torch.full((B, P), T_lfs))


def test_m2_hga_exact_count_and_span_granular():
    """HGA part = exactly n_mask_hga cells; span-granular ⇒ ≤1 isolated singleton
    per parcel (only the tail-trimmed last span can leave a stray single)."""
    bands, nmh, _T_hga, T_lfs, n_groups = _counts(5.0)
    B, P = 2, 6
    m2 = sample_m2_masks_v2(B, P, bands, _gen())
    n_lfs = n_groups * T_lfs
    hga = m2[..., n_lfs:]                              # (B,P,T_hga)
    assert torch.equal(hga.sum(-1), torch.full((B, P), nmh))
    # isolated singleton = masked cell whose both time-neighbours are unmasked.
    pad = torch.zeros(B, P, 1, dtype=torch.bool)
    left = torch.cat([pad, hga[..., :-1]], -1)
    right = torch.cat([hga[..., 1:], pad], -1)
    isolated = (hga & ~left & ~right).sum(-1)
    assert (isolated <= 1).all()


def test_m2_deterministic_under_seed():
    bands, *_ = _counts(1.0)
    a = sample_m2_masks_v2(3, 4, bands, _gen(7))
    b = sample_m2_masks_v2(3, 4, bands, _gen(7))
    c = sample_m2_masks_v2(3, 4, bands, _gen(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_m2_parcels_vary_within_clip():
    """Parcel-uniform count, but the masked PATTERN differs across parcels (each
    parcel samples independently) — not a broadcast of one mask."""
    bands, *_ = _counts(5.0)
    m2 = sample_m2_masks_v2(2, 8, bands, _gen())
    # at least two parcels in clip 0 differ
    rows = m2[0]
    assert not (rows == rows[0]).all(-1).all()


def test_parcel_tube_constant_count():
    g = _gen()
    B, P = 5, 16
    for ratio, expect in [(0.35, round(0.35 * 16)), (0.20, round(0.20 * 16))]:
        tube = sample_parcel_tube_v2(B, P, ratio, g)
        assert tube.shape == (B, P)
        n = max(1, min(expect, P - 1))
        assert torch.equal(tube.sum(-1), torch.full((B,), n))
        assert (tube.sum(-1) >= 1).all() and (tube.sum(-1) <= P - 1).all()


def test_parcel_tube_clamps_extremes():
    # ratio 0 ⇒ still ≥1 tubed; ratio 1 ⇒ still ≥1 untubed.
    lo = sample_parcel_tube_v2(4, 10, 0.0, _gen())
    hi = sample_parcel_tube_v2(4, 10, 1.0, _gen())
    assert torch.equal(lo.sum(-1), torch.full((4,), 1))
    assert torch.equal(hi.sum(-1), torch.full((4,), 9))


def test_parcel_tube_single_parcel_inert():
    tube = sample_parcel_tube_v2(3, 1, 0.5, _gen())
    assert tube.shape == (3, 1)
    assert not tube.any()                              # 1-parcel session ⇒ no tube


def test_parcel_tube_varies_across_clips():
    tube = sample_parcel_tube_v2(8, 12, 0.35, _gen())
    assert not (tube == tube[0]).all(-1).all()         # clips tube different subsets
