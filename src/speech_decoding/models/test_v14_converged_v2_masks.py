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
    electrode_drop_count,
    n_mask_hga,
    sample_electrode_drop_v2,
    sample_m2_masks_hetero_v2,
    sample_m2_masks_v2,
    sample_parcel_tube_v2,
)


def _membership(sizes: list[int]) -> torch.Tensor:
    """Partition membership (P,C) bool from per-parcel electrode counts."""
    poe = torch.cat([torch.full((n,), p) for p, n in enumerate(sizes)])
    return poe[None, :] == torch.arange(len(sizes))[:, None]  # (P,C)


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


# ---- Run-B heterogeneous per-electrode M2 masks ---------------------------
def test_m2_hetero_shape_and_constant_count():
    bands, nmh, _T_hga, T_lfs, _ = _counts(1.0)
    B, C = 3, 7
    m2 = sample_m2_masks_hetero_v2(B, C, bands, _gen())
    S = sum(b.n_tokens for b in bands)
    assert m2.shape == (B, C, S)
    assert m2.dtype == torch.bool
    # every electrode masks exactly n_mask = n_mask_hga + T_lfs (static invariant).
    assert torch.equal(m2.sum(-1), torch.full((B, C), nmh + T_lfs))


def test_m2_hetero_electrodes_differ_within_clip():
    """Heterogeneous: electrodes in the same clip mask DIFFERENT cells (each row
    sampled independently) — the property the parcel-uniform mask lacks."""
    bands, *_ = _counts(5.0)
    m2 = sample_m2_masks_hetero_v2(2, 12, bands, _gen())
    rows = m2[0]
    assert not (rows == rows[0]).all(-1).all()


def test_m2_hetero_matches_core_count_and_deterministic():
    bands, nmh, _, T_lfs, n_groups = _counts(5.0)
    a = sample_m2_masks_hetero_v2(3, 5, bands, _gen(7))
    b = sample_m2_masks_hetero_v2(3, 5, bands, _gen(7))
    assert torch.equal(a, b)                                # deterministic under seed
    # LFS part still exactly one full group per electrode.
    n_lfs = n_groups * T_lfs
    lfs_part = a[..., :n_lfs].reshape(3, 5, n_groups, T_lfs)
    assert torch.equal(lfs_part.any(-1).sum(-1), torch.ones(3, 5, dtype=torch.long))


# ---- Run-B whole-electrode drop -------------------------------------------
def test_electrode_drop_count_floor_and_exempt():
    # min_keep=3: parcels of size 1/2/3 exempt (drop 0); size 6 @ρ=0.3 → round(1.8)=2;
    # size 10 @ρ=0.3 → 3; but survivor floor caps at n_elec-min_keep.
    mem = _membership([1, 2, 3, 6, 10])
    nd = electrode_drop_count(mem, drop_frac=0.3, min_keep=3)
    assert nd.tolist() == [0, 0, 0, 2, 3]
    # kept = n_elec - n_drop ≥ min_keep for every parcel that has ≥min_keep.
    n_elec = mem.sum(1)
    assert ((n_elec - nd)[n_elec > 3] >= 3).all()


def test_electrode_drop_count_floor_caps_high_frac():
    # ρ=0.9 on size 5, min_keep=3 → round(4.5)=4 but floor caps at 5-3=2.
    mem = _membership([5])
    assert electrode_drop_count(mem, drop_frac=0.9, min_keep=3).tolist() == [2]


def test_electrode_drop_shape_and_counts_per_parcel():
    mem = _membership([1, 4, 8, 12])
    B = 16
    drop = sample_electrode_drop_v2(B, mem, drop_frac=0.25, min_keep=3, generator=_gen())
    assert drop.shape == (B, sum([1, 4, 8, 12]))
    assert drop.dtype == torch.bool
    nd = electrode_drop_count(mem, 0.25, 3)                 # (P,)
    # every clip drops EXACTLY nd(p) of parcel p's electrodes (constant count).
    per_parcel = (drop[:, None, :] & mem[None]).sum(-1)     # (B,P)
    assert torch.equal(per_parcel, nd[None].expand(B, -1))


def test_electrode_drop_only_members_and_min_keep_survivors():
    mem = _membership([2, 7, 9])
    drop = sample_electrode_drop_v2(24, mem, drop_frac=0.4, min_keep=3, generator=_gen())
    # dropped electrodes are always members of some parcel.
    assert (drop <= mem.any(0)[None]).all()
    # every parcel keeps ≥ min_keep visible electrodes on every clip.
    kept = (~drop[:, None, :] & mem[None]).sum(-1)          # (B,P) visible per parcel
    assert (kept >= torch.tensor([2, 3, 3])[None]).all()   # size-2 parcel exempt (keeps 2)


def test_electrode_drop_count_constant_across_clips_varies_which():
    mem = _membership([10, 10])
    drop = sample_electrode_drop_v2(8, mem, drop_frac=0.3, min_keep=3, generator=_gen())
    per_parcel = (drop[:, None, :] & mem[None]).sum(-1)     # (B,P)
    assert (per_parcel == per_parcel[0]).all()             # count clip-invariant (static shape)
    assert not (drop == drop[0]).all()                     # but WHICH electrodes vary


def test_electrode_drop_deterministic_with_generator():
    mem = _membership([6, 8])
    a = sample_electrode_drop_v2(4, mem, 0.3, 3, generator=_gen(7))
    b = sample_electrode_drop_v2(4, mem, 0.3, 3, generator=_gen(7))
    assert torch.equal(a, b)
