"""Tests for the B36 paradigm-B masked-JEPA samplers (WS-B B3/B4).

6/03 masking lock (``reports/b36_masking_handoff_2026_06_03.md``): the M2
default is structured 1D bands (:func:`sample_token_band_mask` /
``sample_m2_mask(mask_type="bands")``) and the M4 default is the parcel tube
(:func:`sample_parcel_tube_mask` / ``sample_m4_mask(mask_type="tube")``). The
per-cell Bernoulli (:func:`sample_token_mask`, R-m2-random) and the contiguous
time-block (:func:`sample_parcel_time_block_mask`, R-time-block) are retained
as sisters; their tests live in the "sister" sections below.

Convention under test: a ``True`` entry means MASKED (a prediction target /
dropped from the visible context); visible = ``~mask``.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.ssl.mask import (
    sample_m2_mask,
    sample_m4_mask,
    sample_parcel_time_block_mask,
    sample_parcel_tube_mask,
    sample_token_band_mask,
    sample_token_mask,
    validate_m4_coupling,
)


def _onehot_support(assignments: list[list[int]], k_parcels: int) -> torch.Tensor:
    """assignments[b][c] = parcel index of electrode c in sample b."""
    B = len(assignments)
    C = len(assignments[0])
    sup = torch.zeros(B, C, k_parcels)
    for b, row in enumerate(assignments):
        for c, k in enumerate(row):
            sup[b, c, k] = 1.0
    return sup


def _cover_n_parcels(n_covered: int, *, k_parcels: int, B: int = 4) -> torch.Tensor:
    """Diagonal support covering the first ``n_covered`` of ``k_parcels``."""
    return _onehot_support([list(range(n_covered))] * B, k_parcels=k_parcels)


# ══════════════════════════ B3: M2 structured bands (DEFAULT) ════════════════


def _structure_components(mask_bc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose a single ``(F_p, T_p)`` band mask into its fully-masked rows
    (freq-bands) and fully-masked columns (time-bands)."""
    row_full = mask_bc.all(dim=1)  # (F_p,) rows masked across all time → freq-band
    col_full = mask_bc.all(dim=0)  # (T_p,) cols masked across all freq → time-band
    return row_full, col_full


def test_m2_band_mask_runs_at_fe_raw1_f_p_10_grid() -> None:
    """FE-RAW-1: the M2 mask operates on the POST-PATCH token grid
    ``(B, C, F_p, T_p)``, which the raw F=50 stem (freq kernel 5) floors to
    F_p=10 — identical to the demoted F=30 stem (kernel 3). The sampler is
    F-agnostic: it sees only F_p, so swapping the front end changes nothing.

    Verified end-to-end: the raw stem's actual F_p drives the mask shape, and
    the SWEC raw valid-bin prefix (37 valid bins → F_p prefix) confines the
    freq-bands without error."""
    from speech_decoding.models.v14_encoder import _PatchStem

    raw_stem = _PatchStem(d_model=8, kernel_freq=5, kernel_time=2)
    F_p = raw_stem.n_freq_patches(50)
    assert F_p == 10  # raw F=50 → F_p=10 (== F=30 const-Q stem)

    g = torch.Generator().manual_seed(0)
    B, C, T_p = 4, 6, 20
    mask = sample_m2_mask((B, C, F_p, T_p), held_out_ratio=0.50, generator=g)
    assert tuple(mask.shape) == (B, C, F_p, T_p)
    assert mask.dtype == torch.bool

    # SWEC: 37 of 50 raw bins valid → the valid F_p prefix is the patches whose
    # 5-bin window lies fully inside the 37-valid prefix → floor(37/5)=7.
    valid_fp = torch.zeros(F_p, dtype=torch.bool)
    valid_fp[:7] = True
    masked = sample_m2_mask(
        (B, C, F_p, T_p), held_out_ratio=0.50, generator=g, freq_patch_valid=valid_fp,
    )
    # No freq-band may touch an invalid patch row.
    assert not masked[..., 7:, :].all(dim=-1).any(), (
        "M2 freq-bands must stay inside the SWEC-valid F_p prefix"
    )


def test_band_mask_held_out_fraction_is_exact_union() -> None:
    """Non-overlapping placement → the union held-out fraction is the exact
    ``1-(1-a)(1-b)``. On a 10×20 grid at r=0.50 (a=b=0.293) the per-axis counts
    are 6 time-cols / 3 freq-rows → union exactly 0.51, every electrode."""
    g = torch.Generator().manual_seed(0)
    mask = sample_token_band_mask((4, 8, 10, 20), held_out_ratio=0.50, generator=g)
    assert mask.dtype == torch.bool
    assert tuple(mask.shape) == (4, 8, 10, 20)
    per_elec = mask.float().mean(dim=(2, 3))  # (B, C)
    assert torch.allclose(per_elec, torch.full_like(per_elec, 0.51), atol=1e-6)


def test_band_mask_is_union_of_rows_and_cols() -> None:
    """Every band mask reconstructs exactly from its fully-masked rows
    (freq-bands) ∪ fully-masked columns (time-bands) — the A-JEPA TF union
    structure, not an arbitrary per-cell pattern."""
    g = torch.Generator().manual_seed(1)
    mask = sample_token_band_mask((3, 4, 10, 20), held_out_ratio=0.50, generator=g)
    for b in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            m = mask[b, c]
            row_full, col_full = _structure_components(m)
            recon = row_full.unsqueeze(-1) | col_full.unsqueeze(0)
            assert torch.equal(recon, m), f"({b},{c}) not a row∪col union"


def test_band_mask_time_bands_meet_min_width() -> None:
    """Time-bands (fully-masked columns) come in contiguous runs each ≥ the
    floor (2 patches ≥ 250 ms at 8 Hz)."""
    g = torch.Generator().manual_seed(2)
    mask = sample_token_band_mask(
        (2, 3, 10, 24), held_out_ratio=0.50, generator=g, time_band_floor=2,
    )
    for b in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            _row_full, col_full = _structure_components(mask[b, c])
            idx = torch.nonzero(col_full, as_tuple=False).flatten().tolist()
            if not idx:
                continue
            run = 1
            for prev, cur in zip(idx, idx[1:]):
                if cur == prev + 1:
                    run += 1
                else:
                    assert run >= 2, f"time-band run {run} < floor 2"
                    run = 1
            assert run >= 2, f"final time-band run {run} < floor 2"


def test_band_mask_freq_bands_confined_to_valid_prefix() -> None:
    """With a corpus-valid freq prefix and the time axis suppressed
    (``time_freq_split=(0.0, b)``), the mask is pure freq-bands and never
    touches invalid rows ``[n_valid, F_p)``."""
    g = torch.Generator().manual_seed(3)
    F_p, n_valid = 10, 6
    valid = torch.zeros(F_p, dtype=torch.bool)
    valid[:n_valid] = True
    mask = sample_token_band_mask(
        (4, 4, F_p, 20), generator=g,
        time_freq_split=(0.0, 0.5), freq_patch_valid=valid,
    )
    # No masked cell in any invalid row (no time-band to spill there).
    assert not bool(mask[:, :, n_valid:, :].any())
    # Some valid row IS masked (the freq-band landed inside the prefix).
    assert bool(mask[:, :, :n_valid, :].any())


def test_band_mask_freq_patch_valid_shape_and_prefix_guards() -> None:
    """The band sampler confines via a shared ``(F_p,)`` prefix count, so it
    must REJECT (not silently over-count) shapes its single-layout design
    can't honor: a per-clip ``(B, F_p)`` mask (WS-H per-element extension), a
    wrong-length 1-D mask, and a non-prefix (gappy) valid set."""
    g = torch.Generator().manual_seed(0)
    shape = (2, 3, 10, 12)
    # (B, F_p) per-clip → deferred WS-H path, not silently mis-masked.
    with pytest.raises(NotImplementedError, match="per-clip"):
        sample_token_band_mask(
            shape, generator=g, freq_patch_valid=torch.ones(2, 10, dtype=torch.bool),
        )
    # Wrong-length 1-D.
    with pytest.raises(ValueError, match=r"\(F_p,\)"):
        sample_token_band_mask(
            shape, generator=g, freq_patch_valid=torch.ones(8, dtype=torch.bool),
        )
    # Non-prefix (valid rows not [0, n_valid)) → n_valid-as-count would mask
    # invalid low rows; must raise.
    gappy = torch.tensor([1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
    with pytest.raises(ValueError, match="prefix"):
        sample_token_band_mask(shape, generator=g, freq_patch_valid=gappy)


def test_band_mask_asymmetric_time_freq_split() -> None:
    """R-m2-time-weighted-split sister: an explicit (a, b) split realizes its
    two axes independently. A column is fully masked iff it is a time-band
    (all freq rows hidden); a row iff it is a freq-band — so the per-axis band
    counts are recoverable from the union. a=0.4 → round(0.4·20)=8, floor 2 →
    4 bands × 2 = 8 time cols; b=0.1 → round(0.1·10)=1, floor 1 → 1 freq row."""
    g = torch.Generator().manual_seed(1)
    F_p, T_p = 10, 20
    mask = sample_token_band_mask(
        (3, 4, F_p, T_p), generator=g, time_freq_split=(0.4, 0.1),
    )
    time_band_cols = mask.all(dim=2).sum(dim=-1)  # (B, C) fully-masked cols
    freq_band_rows = mask.all(dim=3).sum(dim=-1)  # (B, C) fully-masked rows
    assert torch.all(time_band_cols == 8)
    assert torch.all(freq_band_rows == 1)


def test_band_mask_same_seed_identical_and_extremes() -> None:
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    shape = (2, 6, 10, 12)
    assert torch.equal(
        sample_token_band_mask(shape, generator=g1),
        sample_token_band_mask(shape, generator=g2),
    )
    g = torch.Generator().manual_seed(0)
    assert not bool(sample_token_band_mask(shape, held_out_ratio=0.0, generator=g).any())


def test_band_mask_ratio_bounds() -> None:
    g = torch.Generator().manual_seed(0)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            sample_token_band_mask((2, 2, 4, 4), held_out_ratio=bad, generator=g)


# ─────────────────────── M2 random sister (R-m2-random) ──────────────────────


def test_token_mask_empirical_fraction_half() -> None:
    """Per-(electrode, freq-patch, time-patch) Bernoulli at 0.5 → empirical
    masked fraction → 0.5 ± 0.02 over many draws."""
    g = torch.Generator().manual_seed(0)
    shape = (4, 8, 10, 20)  # (B, C, F_p, T_p)
    total = 0
    masked = 0
    for _ in range(1000):
        m = sample_token_mask(shape, mask_ratio=0.5, generator=g)
        masked += int(m.sum())
        total += m.numel()
    frac = masked / total
    assert abs(frac - 0.5) < 0.02, f"empirical masked fraction {frac} != 0.5"


def test_token_mask_same_seed_identical() -> None:
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    shape = (2, 6, 5, 12)
    m1 = sample_token_mask(shape, mask_ratio=0.5, generator=g1)
    m2 = sample_token_mask(shape, mask_ratio=0.5, generator=g2)
    assert torch.equal(m1, m2)


def test_token_mask_visible_and_masked_partition() -> None:
    g = torch.Generator().manual_seed(1)
    shape = (3, 4, 5, 6)
    m = sample_token_mask(shape, mask_ratio=0.4, generator=g)
    visible = ~m
    assert not bool((visible & m).any())
    assert bool((visible | m).all())
    assert m.dtype == torch.bool
    assert tuple(m.shape) == shape


def test_token_mask_ratio_bounds() -> None:
    g = torch.Generator().manual_seed(0)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            sample_token_mask((2, 2, 2, 2), mask_ratio=bad, generator=g)


def test_token_mask_extremes() -> None:
    g = torch.Generator().manual_seed(0)
    shape = (2, 3, 4, 5)
    assert not bool(sample_token_mask(shape, mask_ratio=0.0, generator=g).any())
    assert bool(sample_token_mask(shape, mask_ratio=1.0, generator=g).all())


# ══════════════════════════ B4: M4 parcel tube (DEFAULT) ═════════════════════


def test_tube_mask_whole_parcel_masked() -> None:
    """A masked parcel is masked across ALL time-patches (a tube); an unmasked
    covered parcel is masked at NO time-patch (0/1 per parcel, never partial)."""
    g = torch.Generator().manual_seed(3)
    support = _cover_n_parcels(20, k_parcels=24, B=4)
    p_mask, _ = sample_parcel_tube_mask(
        support, n_time_patches=12, mask_ratio=0.20, generator=g,
    )
    per_parcel = p_mask.float().mean(dim=-1)  # (B, K) fraction of time masked
    assert torch.all((per_parcel == 0.0) | (per_parcel == 1.0)), (
        "a parcel is fully tubed or untouched — never partially masked"
    )


def test_tube_mask_count_is_clamped_fraction() -> None:
    """#masked parcels == clamp(round(0.20·N), 1, N − n_min_visible). N=20 →
    round(4.0)=4; N=14 → round(2.8)=3; N=5 → round(1.0)=1. Small-N clamp
    boundary (where ``upper = N − n_min_visible`` binds at the n_min_visible
    floor): N=4 → upper=1 caps at exactly one maskable parcel; N=3 → upper=0,
    the floor fully suppresses the mask (0 masked, all 3 kept visible)."""
    for n_covered, expected in [(20, 4), (14, 3), (5, 1), (4, 1), (3, 0)]:
        g = torch.Generator().manual_seed(0)
        support = _cover_n_parcels(n_covered, k_parcels=40, B=4)
        p_mask, _ = sample_parcel_tube_mask(
            support, n_time_patches=8, mask_ratio=0.20, generator=g,
        )
        n_masked_parcels = (p_mask.any(dim=-1)).sum(dim=-1)  # (B,)
        assert torch.all(n_masked_parcels == expected), (
            f"N={n_covered}: masked {n_masked_parcels.tolist()} != {expected}"
        )


def test_tube_mask_only_covered_parcels() -> None:
    g = torch.Generator().manual_seed(2)
    support = _cover_n_parcels(10, k_parcels=24, B=4)
    p_mask, _ = sample_parcel_tube_mask(
        support, n_time_patches=10, mask_ratio=0.20, generator=g,
    )
    covered = (support.sum(dim=1) > 0)  # (B, K)
    uncovered_masked = p_mask[~covered.unsqueeze(-1).expand_as(p_mask)]
    assert not bool(uncovered_masked.any())


def test_tube_mask_respects_n_min_visible() -> None:
    """After masking, ≥ n_min_visible covered parcels remain visible."""
    g = torch.Generator().manual_seed(5)
    n_covered, n_min = 14, 3
    support = _cover_n_parcels(n_covered, k_parcels=24, B=4)
    p_mask, _ = sample_parcel_tube_mask(
        support, n_time_patches=8, mask_ratio=0.20,
        n_min_visible=n_min, generator=g,
    )
    masked_parcels = p_mask.any(dim=-1).sum(dim=-1)  # (B,)
    visible_covered = n_covered - masked_parcels
    assert torch.all(visible_covered >= n_min)


def test_tube_mask_no_mask_when_n_at_or_below_min_visible() -> None:
    """N ≤ n_min_visible can't mask anything (would drop below the visible
    floor) → 0 masked parcels."""
    g = torch.Generator().manual_seed(1)
    support = _cover_n_parcels(3, k_parcels=10, B=4)
    p_mask, drop = sample_parcel_tube_mask(
        support, n_time_patches=6, mask_ratio=0.20, n_min_visible=3, generator=g,
    )
    assert not bool(p_mask.any())
    assert not bool(drop.any())


def test_tube_drop_is_electrode_level_and_maps_to_masked_parcel() -> None:
    """For a tube, an electrode is dropped at ALL t iff its parcel is masked —
    an electrode-level drop. Every dropped (c, t) maps to a masked parcel."""
    g = torch.Generator().manual_seed(11)
    assignments = [list(range(8)), list(range(8))]  # 8 covered parcels, B=2
    support = _onehot_support(assignments, k_parcels=12)
    p_mask, drop = sample_parcel_tube_mask(
        support, n_time_patches=9, mask_ratio=0.40, generator=g,
    )
    B, C, T_p = drop.shape
    onehot = (support > 0)
    for b in range(B):
        for c in range(C):
            k = int(onehot[b, c].nonzero(as_tuple=False).flatten()[0])
            # electrode-level: dropped at all-t or none, mirroring its parcel
            torch.testing.assert_close(
                drop[b, c].to(torch.float32), p_mask[b, k].to(torch.float32),
            )


def test_tube_mask_shapes_dtypes_and_seed() -> None:
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    support = _cover_n_parcels(6, k_parcels=8, B=1)
    p1, d1 = sample_parcel_tube_mask(support, n_time_patches=10, generator=g1)
    p2, d2 = sample_parcel_tube_mask(support, n_time_patches=10, generator=g2)
    assert tuple(p1.shape) == (1, 8, 10)
    assert tuple(d1.shape) == (1, 6, 10)
    assert p1.dtype == torch.bool and d1.dtype == torch.bool
    assert torch.equal(p1, p2) and torch.equal(d1, d2)


# ────────────────── M4 time-block sister (R-time-block) ──────────────────────


def test_parcel_time_block_mask_contiguous_per_parcel() -> None:
    """For every masked parcel, the masked time indices form a single
    contiguous block."""
    g = torch.Generator().manual_seed(3)
    support = _onehot_support([[0, 0, 1, 2, 3], [0, 1, 1, 2, 3]], k_parcels=6)
    p_mask, _ = sample_parcel_time_block_mask(
        support, n_time_patches=20, mask_ratio=0.3, generator=g
    )
    B, K, T_p = p_mask.shape
    for b in range(B):
        for k in range(K):
            row = p_mask[b, k]
            if not bool(row.any()):
                continue
            idx = torch.nonzero(row, as_tuple=False).flatten()
            assert int(idx.max() - idx.min() + 1) == idx.numel(), (
                f"parcel ({b},{k}) masked indices not contiguous: {idx.tolist()}"
            )


def test_parcel_time_block_mask_fraction_of_covered() -> None:
    g = torch.Generator().manual_seed(5)
    support = _onehot_support([[0, 1, 2, 3, 4, 5, 6, 7]] * 4, k_parcels=10)
    p_mask, _ = sample_parcel_time_block_mask(
        support, n_time_patches=20, mask_ratio=0.3, generator=g
    )
    covered = (support.sum(dim=1) > 0)  # (B, K)
    n_covered_cells = int(covered.sum()) * p_mask.shape[-1]
    frac = int(p_mask.sum()) / n_covered_cells
    assert abs(frac - 0.3) < 0.02, f"covered-cell masked fraction {frac} != 0.3"


def test_parcel_time_block_mask_only_covered_parcels() -> None:
    g = torch.Generator().manual_seed(2)
    support = _onehot_support([[0, 1, 2, 3]], k_parcels=10)
    p_mask, _ = sample_parcel_time_block_mask(
        support, n_time_patches=16, mask_ratio=0.4, generator=g
    )
    covered = (support.sum(dim=1) > 0)  # (B, K)
    uncovered_masked = p_mask[~covered.unsqueeze(-1).expand_as(p_mask)]
    assert not bool(uncovered_masked.any())


def test_parcel_time_block_drop_maps_to_masked_parcel() -> None:
    g = torch.Generator().manual_seed(11)
    assignments = [[0, 0, 1, 2, 3, 3, 4], [0, 1, 2, 2, 3, 4, 4]]
    support = _onehot_support(assignments, k_parcels=8)
    p_mask, drop = sample_parcel_time_block_mask(
        support, n_time_patches=18, mask_ratio=0.3, generator=g
    )
    B, C, T_p = drop.shape
    onehot = (support > 0)
    for b in range(B):
        for c in range(C):
            k = int(onehot[b, c].nonzero(as_tuple=False).flatten()[0])
            torch.testing.assert_close(
                drop[b, c].to(torch.float32), p_mask[b, k].to(torch.float32)
            )


# ════════════════════════ Dispatchers + coupling guard ═══════════════════════


def test_m2_dispatch_bands_default_and_random_sister() -> None:
    g1 = torch.Generator().manual_seed(4)
    g2 = torch.Generator().manual_seed(4)
    shape = (2, 4, 10, 20)
    # default == bands
    assert torch.equal(
        sample_m2_mask(shape, generator=g1),
        sample_token_band_mask(shape, generator=g2),
    )
    # random sister routes to the per-cell Bernoulli
    g3 = torch.Generator().manual_seed(9)
    g4 = torch.Generator().manual_seed(9)
    assert torch.equal(
        sample_m2_mask(shape, mask_type="random", held_out_ratio=0.5, generator=g3),
        sample_token_mask(shape, mask_ratio=0.5, generator=g4),
    )


def test_m2_dispatch_unknown_type_raises() -> None:
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="m2 mask_type"):
        sample_m2_mask((1, 1, 4, 4), mask_type="bogus", generator=g)


def test_m4_dispatch_tube_default_and_time_block_sister() -> None:
    support = _cover_n_parcels(10, k_parcels=16, B=2)
    g1 = torch.Generator().manual_seed(6)
    g2 = torch.Generator().manual_seed(6)
    pt_a, dr_a = sample_m4_mask(support, n_time_patches=8, generator=g1)
    pt_b, dr_b = sample_parcel_tube_mask(support, n_time_patches=8, generator=g2)
    assert torch.equal(pt_a, pt_b) and torch.equal(dr_a, dr_b)

    g3 = torch.Generator().manual_seed(6)
    g4 = torch.Generator().manual_seed(6)
    pt_c, _ = sample_m4_mask(
        support, n_time_patches=8, mask_type="time_block", mask_ratio=0.3, generator=g3,
    )
    pt_d, _ = sample_parcel_time_block_mask(
        support, n_time_patches=8, mask_ratio=0.3, generator=g4,
    )
    assert torch.equal(pt_c, pt_d)

    # n_min_visible is threaded through the dispatcher — pick a value that
    # CHANGES the count vs the default so the check is not vacuous: N=10,
    # n_min_visible=9 → upper=N−9=1, target=round(0.2·10)=2 → mask_count=1,
    # whereas the default 3 → upper=7 → 2. So masked==1 proves the kwarg
    # reached the tube sampler (a dropped kwarg would give 2).
    g5 = torch.Generator().manual_seed(6)
    g6 = torch.Generator().manual_seed(6)
    pt_e, _ = sample_m4_mask(
        support, n_time_patches=8, n_min_visible=9, generator=g5,
    )
    pt_f, _ = sample_parcel_tube_mask(
        support, n_time_patches=8, n_min_visible=9, generator=g6,
    )
    assert torch.equal(pt_e, pt_f)
    assert torch.all(pt_e.any(dim=-1).sum(dim=-1) == 1)  # differs from default (2)


def test_m4_dispatch_mix_not_implemented_and_unknown_raises() -> None:
    support = _cover_n_parcels(6, k_parcels=8, B=1)
    g = torch.Generator().manual_seed(0)
    with pytest.raises(NotImplementedError, match="mix"):
        sample_m4_mask(support, n_time_patches=6, mask_type="mix", generator=g)
    with pytest.raises(ValueError, match="m4 mask_type"):
        sample_m4_mask(support, n_time_patches=6, mask_type="bogus", generator=g)


def test_coupling_guard_valid_pairs_pass() -> None:
    validate_m4_coupling("tube", "cross_time")
    validate_m4_coupling("time_block", "co_temporal")


def test_coupling_guard_h1_leak_and_overhard_rejected() -> None:
    # time_block + cross_time = the H1 leak (masked-at-t, visible-at-t±1).
    with pytest.raises(ValueError, match="H1 leak"):
        validate_m4_coupling("time_block", "cross_time")
    # tube + co_temporal = over-hard (nothing co-temporal to attend).
    with pytest.raises(ValueError):
        validate_m4_coupling("tube", "co_temporal")
