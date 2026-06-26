"""P2.6 tests for the converged-v2 static-shape header + fail-loud invariants."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models import v14_converged_v2 as v2
from speech_decoding.models.v14_converged_v2 import (
    active_parcels,
    compute_static_shapes_v2,
    sample_m2_masks_v2,
    sample_parcel_tube_v2,
)

K = 2


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _session(clip_s=5.0, B=4, tube_ratio=0.35):
    """4 electrodes → 2 parcels (n_p = [2,2]); real sampled masks."""
    bands = v2.bands_for_clip_len(clip_s)
    parcel_of_electrode = torch.tensor([5, 5, 9, 9])
    _labels, membership = active_parcels(parcel_of_electrode)
    P = membership.shape[0]
    m2 = sample_m2_masks_v2(B, P, bands, _gen(1))
    tube = sample_parcel_tube_v2(B, P, tube_ratio, _gen(2))
    return bands, membership, m2, tube


def test_static_header_fields():
    bands, membership, m2, tube = _session(clip_s=5.0, B=4)
    sh = compute_static_shapes_v2(m2, tube, membership, bands, k=K)
    lfs, hga = bands
    S = sum(b.n_tokens for b in bands)
    # LFS tube masks T_lfs = n_time_patches (=10 @5s, one of 3 groups), NOT n_tokens.
    n_mask = v2.n_mask_hga(hga) + lfs.n_time_patches    # 40 + 10 = 50 @5s
    assert sh.b == 4 and sh.c == 4 and sh.P == 2
    assert sh.n_p == (2, 2)
    assert sh.k == 2 and sh.S == S == 110
    assert sh.n_mask == n_mask == 50                   # 45% of 110
    assert sh.s_vis == S - n_mask == 60
    assert sh.n_tube == max(1, min(round(0.35 * 2), 1)) == 1  # P=2 ⇒ clamp to 1
    assert sh.P_vis == 1


def test_derived_gather_lengths():
    bands, membership, m2, tube = _session(clip_s=5.0, B=3)
    sh = compute_static_shapes_v2(m2, tube, membership, bands, k=K)
    assert sh.latent_student == sh.P_vis * sh.k * sh.s_vis
    assert sh.latent_teacher == sh.P * sh.k * sh.S
    assert sh.m2_q == sh.b * sh.c * sh.n_mask
    assert sh.m4_q == sh.b * (sh.n_tube * sh.k * sh.S + sh.P_vis * sh.k * sh.n_mask)


def test_header_1s_counts():
    bands, membership, m2, tube = _session(clip_s=1.0, B=2)
    sh = compute_static_shapes_v2(m2, tube, membership, bands, k=K)
    # 1 s: S=22, n_mask = 8 (HGA) + 2 (LFS T_lfs) = 10, s_vis = 12.
    assert sh.S == 22 and sh.n_mask == 10 and sh.s_vis == 12


def test_fail_loud_m2_count_nonuniform():
    bands, membership, m2, tube = _session()
    m2 = m2.clone()
    m2[0, 0, :] = False          # zero one parcel's mask → count breaks
    m2[0, 0, 0] = True
    with pytest.raises(ValueError, match="M2 count not uniform"):
        compute_static_shapes_v2(m2, tube, membership, bands, k=K)


def test_fail_loud_tube_count_nonuniform():
    bands, membership, m2, tube = _session()
    tube = tube.clone()
    tube[0] = ~tube[0]           # flip clip 0's tube → count differs from others
    # (P=2, n_tube=1 ⇒ flip makes clip0 have 1 too; force a real mismatch instead)
    tube[0, :] = True            # clip 0 now tubes both parcels (count 2 ≠ 1)
    with pytest.raises(ValueError, match="tube count not uniform"):
        compute_static_shapes_v2(m2, tube, membership, bands, k=K)


def test_fail_loud_membership_not_partition():
    bands, membership, m2, tube = _session()
    membership = membership.clone()
    membership[0, 0] = True
    membership[1, 0] = True      # electrode 0 now in both parcels
    with pytest.raises(ValueError, match="not a partition"):
        compute_static_shapes_v2(m2, tube, membership, bands, k=K)


def test_fail_loud_wrong_S():
    bands, membership, m2, tube = _session()
    with pytest.raises(ValueError, match="!= bands S"):
        compute_static_shapes_v2(m2[..., :-1], tube, membership, bands, k=K)


def test_fail_loud_non_bool():
    bands, membership, m2, tube = _session()
    with pytest.raises(ValueError, match="must be bool"):
        compute_static_shapes_v2(m2.float(), tube, membership, bands, k=K)


def test_larger_session_ragged_parcels():
    """n_p ragged (3,1,2); Σ = C; header reflects it."""
    bands = v2.bands_for_clip_len(5.0)
    poe = torch.tensor([5, 5, 5, 9, 20, 20])     # parcels 5(×3), 9(×1), 20(×2)
    _labels, membership = active_parcels(poe)
    P = membership.shape[0]
    m2 = sample_m2_masks_v2(3, P, bands, _gen(1))
    tube = sample_parcel_tube_v2(3, P, 0.35, _gen(2))
    sh = compute_static_shapes_v2(m2, tube, membership, bands, k=K)
    assert sh.c == 6 and sh.P == 3
    assert sh.n_p == (3, 1, 2) and sum(sh.n_p) == sh.c
    assert sh.n_tube == max(1, min(round(0.35 * 3), 2))   # round(1.05)=1
    assert sh.P_vis == sh.P - sh.n_tube
