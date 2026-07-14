"""v14_converged_v3 Phase 5 — dual-axis (space × time) block masking (TDD).

Masking-axis inversion (Ben 2026-07-12): mask on BOTH axes independently per sensor
— SPACE (whole-shaft stochastic tier + intra width-4 depth-blocks, snap to exactly
D=round(space_frac·N)) and TIME (width-7 time-blocks per shaft, snap to exactly
T_mask=round(time_frac·T) per shaft). Compose as a per-sensor outer product ⇒ static
shapes (constant N−D visible contacts, constant T_kept visible frames per shaft).
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import (
    V3MaskConfig,
    assert_mask_feasible,
    sample_masks,
)
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _session(shaft_sizes: list[int]):
    labels, parcels = [], []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            parcels.append(s)
    sc = build_sidecar(labels, parcel_id=torch.tensor(parcels, dtype=torch.long))
    return sc, build_l1_geometry(sc)


def _runs(row: torch.Tensor) -> list[int]:
    out, cur = [], 0
    for v in row.tolist():
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def _per_shaft_rate(sc, contact_mask: torch.Tensor) -> torch.Tensor:
    r, s = contact_mask.shape[0], int(sc.n_shafts)
    out = torch.zeros(r, s)
    for si in range(s):
        idx = (sc.shaft_id == si).nonzero(as_tuple=True)[0]
        out[:, si] = contact_mask[:, idx].float().mean(dim=1)
    return out


# ── SPACE axis: exact count, whole tier, blockiness ──────────────────────────
def test_space_exact_constant_count() -> None:
    sc, geom = _session([12, 10, 8, 6])  # N=36
    n = 36
    d = round(V3MaskConfig().space_frac * n)  # 18
    m = sample_masks(geom, n, n_time=96, n_rows=5, generator=_gen())
    assert m.contact_mask.shape == (5, n) and m.contact_mask.dtype == torch.bool
    assert (m.contact_mask.sum(1) == d).all()


def test_space_frac_controls_count() -> None:
    sc, geom = _session([20])
    m = sample_masks(geom, 20, n_time=96, n_rows=3, generator=_gen(), cfg=V3MaskConfig(space_frac=0.4))
    assert (m.contact_mask.sum(1) == 8).all()


def test_space_is_blocky_not_iid() -> None:
    # contiguous depth-blocks ⇒ far fewer runs than an iid coin-flip at the same rate.
    sc, geom = _session([30])
    cfg = V3MaskConfig(space_frac=0.5, whole_shaft_frac=0.0, block_w_space=4)
    m = sample_masks(geom, 30, n_time=96, n_rows=8, generator=_gen(1), cfg=cfg)
    mean_runs = sum(len(_runs(m.contact_mask[r])) for r in range(8)) / 8
    assert mean_runs <= 4.5, f"mean runs/row {mean_runs} — not blocky"
    assert max(max(_runs(m.contact_mask[r]), default=0) for r in range(8)) >= 4


def test_whole_shaft_tier_fully_masks_shafts() -> None:
    # whole_shaft_frac high over few shafts ⇒ some shafts masked 100%.
    sc, geom = _session([5, 5, 5, 5])  # N=20, D=10
    cfg = V3MaskConfig(space_frac=0.5, whole_shaft_frac=0.5)
    m = sample_masks(geom, 20, n_time=96, n_rows=64, generator=_gen(3), cfg=cfg)
    rate = _per_shaft_rate(sc, m.contact_mask)
    n_full = (rate >= 0.999).sum(1)
    assert n_full.float().mean() >= 1.0, f"expected some whole shafts, got {n_full.tolist()}"
    # whole_contact ⊆ contact_mask
    assert (m.whole_contact & ~m.contact_mask).sum() == 0


def test_whole_shaft_count_is_stochastic() -> None:
    # The number of fully-masked shafts VARIES across rows (not pinned) and averages
    # near whole_shaft_frac·S (plus intra saturation).
    sc, geom = _session([8, 8, 8, 8, 8, 8])  # S=6, N=48, D=24
    cfg = V3MaskConfig(space_frac=0.5, whole_shaft_frac=0.25)  # E[K]=1.5
    m = sample_masks(geom, 48, n_time=96, n_rows=400, generator=_gen(4), cfg=cfg)
    # whole_contact tells us the whole tier directly (not intra saturation)
    whole_shafts = torch.zeros(400, 6)
    for si in range(6):
        idx = (sc.shaft_id == si).nonzero(as_tuple=True)[0]
        whole_shafts[:, si] = m.whole_contact[:, idx].all(1).float()
    k_per_row = whole_shafts.sum(1)
    assert k_per_row.unique().numel() >= 3, f"whole count not stochastic: {k_per_row.unique().tolist()}"
    assert 0.5 < k_per_row.mean().item() < 2.5, f"E[K] off: {k_per_row.mean().item():.2f}"


def test_no_keep_alive_shafts_may_saturate() -> None:
    # With the ≥1-visible rule REMOVED, a non-whole shaft is ALLOWED to reach 100%
    # via the intra tier — it just becomes another full drop. So we do NOT assert a
    # pinned whole-shaft count; we only require the global count stays exactly D.
    sc, geom = _session([6, 16, 9, 9, 4, 10, 8, 3])  # varied incl. tiny shafts
    n = sum([6, 16, 9, 9, 4, 10, 8, 3])
    cfg = V3MaskConfig(space_frac=0.5, whole_shaft_frac=0.15)
    m = sample_masks(geom, n, n_time=96, n_rows=500, generator=_gen(0), cfg=cfg)
    assert (m.contact_mask.sum(1) == round(0.5 * n)).all()
    # at least one row somewhere saturates a shaft beyond the whole tier (legal now)
    rate = _per_shaft_rate(sc, m.contact_mask)
    assert (rate >= 0.999).any(), "expected some fully-masked shafts across 500 rows"


# ── TIME axis: exact per-shaft count, blockiness, heterogeneity ──────────────
def test_time_exact_per_shaft_count() -> None:
    sc, geom = _session([12, 10, 8])  # S=3
    t = 96
    tmask = round(V3MaskConfig().time_frac * t)  # 48
    m = sample_masks(geom, 30, n_time=t, n_rows=5, generator=_gen(2))
    assert m.frame_mask.shape == (5, 3, t) and m.frame_mask.dtype == torch.bool
    assert (m.frame_mask.sum(-1) == tmask).all(), "every (row, shaft) must mask exactly T_mask frames"


def test_time_frac_controls_count() -> None:
    sc, geom = _session([12])
    m = sample_masks(geom, 12, n_time=100, n_rows=4, generator=_gen(), cfg=V3MaskConfig(time_frac=0.3))
    assert (m.frame_mask.sum(-1) == 30).all()


def test_time_is_blocky() -> None:
    sc, geom = _session([12])
    m = sample_masks(geom, 12, n_time=96, n_rows=8, generator=_gen(5), cfg=V3MaskConfig(block_w_time=7))
    # 48 masked frames of 96 in width-7 blocks ⇒ few runs (≲ ~8), longest run ≥ 7
    mean_runs = sum(len(_runs(m.frame_mask[r, 0])) for r in range(8)) / 8
    assert mean_runs <= 9.0, f"time mean runs {mean_runs} — not blocky"
    assert max(max(_runs(m.frame_mask[r, 0]), default=0) for r in range(8)) >= 7


def test_blank_frames_are_rare_and_match_the_arithmetic() -> None:
    # THE GUARDIAN IS GONE (M14, 2026-07-15). It used to force ≥1 live shaft to keep
    # every t; M14 measured that this SATURATED the realized block run at ~6 no matter
    # how wide a block was requested, which defeats the whole purpose of the width (it
    # must bury a cell deeper than the STFT window overlap). See masking.py's docstring.
    #
    # The new contract, and this test IS the contract:
    #   1. blank frames (no live shaft keeps t) are ALLOWED — proving the guardian is
    #      really gone, so this is a REGRESSION GUARD: re-adding one fails here.
    #   2. they are RARE, and rare at the rate the arithmetic predicts: a shaft masks t
    #      with prob time_frac, blocks correlate frames WITHIN a shaft but the shafts are
    #      drawn INDEPENDENTLY, so P(blank) ≈ time_frac^V over V live shafts.
    #   3. the STATIC invariant is untouched: still EXACTLY T_mask masked frames per
    #      (row, shaft). This is the one that would break the compiled buffer, and it is
    #      why deleting the guardian is safe — it only ever reordered the cover-rank sort,
    #      which happens BEFORE the `time_rank < t_mask` snap.
    sc, geom = _session([12, 10, 10, 8, 8, 6, 6, 4])  # N=64, S=8
    n, rows, t = 64, 800, 96
    m = sample_masks(geom, n, n_time=t, n_rows=rows, generator=_gen(9))

    # (3) the static invariant — the only one that can break the compiled shapes
    assert (m.frame_mask.sum(-1) == 48).all()

    shaft_of = geom.shaft_of_contact
    vis = ~m.contact_mask  # (R, N)
    live = torch.zeros(rows, int(sc.n_shafts), dtype=torch.bool)
    for si in range(int(sc.n_shafts)):
        idx = (shaft_of == si).nonzero(as_tuple=True)[0]
        live[:, si] = vis[:, idx].any(1)
    covers = (~m.frame_mask) & live[:, :, None]  # (R, S, T) live AND keeping t
    blank = covers.sum(1) == 0  # (R, T)
    blank_frac = blank.float().mean().item()

    # (1) the guardian is gone
    assert blank_frac > 0.0, "no blank frame in 76800 draws — is the guardian back?"
    # (2) rare, and at the predicted rate. V varies per row, so predict per row and pool.
    v_live = live.sum(1).clamp(min=1).float()  # (R,)
    predicted = (V3MaskConfig().time_frac ** v_live).mean().item()
    assert blank_frac < 0.02, f"blank frames not rare: {blank_frac:.4f}"
    assert 0.5 * predicted < blank_frac < 2.0 * predicted, (
        f"blank rate {blank_frac:.5f} does not match time_frac^V = {predicted:.5f} — "
        "the sampler and the arithmetic disagree, so one of them is wrong"
    )


def test_time_is_heterogeneous_across_shafts() -> None:
    # Different shafts mask different frames (independent draws) ⇒ at most t slices are
    # NOT uniformly all-masked/all-visible across shafts. This is what keeps L2 pressured.
    sc, geom = _session([10, 10, 10, 10])  # S=4
    m = sample_masks(geom, 40, n_time=96, n_rows=1, generator=_gen(11))
    fm = m.frame_mask[0]  # (S, T)
    frac_masked_per_t = fm.float().mean(0)  # (T,)
    # some time slices are partially masked across shafts (not all 0 or all 1)
    mixed = ((frac_masked_per_t > 0.1) & (frac_masked_per_t < 0.9)).float().mean()
    assert mixed > 0.5, f"time masking too homogeneous across shafts: {mixed:.2f}"


# ── OUTER-PRODUCT composition / static invariants ────────────────────────────
def test_outer_product_visible_fraction_matches_product() -> None:
    # total visible cells = (N−D)·T_kept ; total masked = 1 − space_keep·time_keep.
    sc, geom = _session([16, 12, 10, 8])  # N=46
    n, t = 46, 96
    cfg = V3MaskConfig(space_frac=0.5, time_frac=0.5)
    m = sample_masks(geom, n, n_time=t, n_rows=32, generator=_gen(7), cfg=cfg)
    d = round(0.5 * n)
    tmask = round(0.5 * t)
    tkept = t - tmask
    # build the cell mask: cell masked iff contact masked OR frame masked for its shaft
    shaft_of = geom.shaft_of_contact  # (N,)
    cell_masked = m.contact_mask[:, :, None] | m.frame_mask[:, shaft_of, :]  # (R, N, T)
    vis = (~cell_masked).sum(dim=(1, 2))  # (R,) visible cells per row
    expected_vis = (n - d) * tkept
    assert (vis == expected_vis).all(), f"visible cells {vis.unique().tolist()} != {expected_vis}"


def test_static_counts_are_constant_across_seeds() -> None:
    sc, geom = _session([14, 12, 10, 8, 6])  # N=50
    n, t = 50, 96
    cfg = V3MaskConfig()  # no cfg passed to sample_masks below ⇒ assert the DEFAULT's invariant
    d = round(cfg.space_frac * n)
    tmask = round(cfg.time_frac * t)
    for seed in range(20):
        m = sample_masks(geom, n, n_time=t, n_rows=8, generator=_gen(seed))
        assert (m.contact_mask.sum(1) == d).all()
        assert (m.frame_mask.sum(-1) == tmask).all()


# ── determinism / independence / feasibility ─────────────────────────────────
def test_deterministic_in_generator_seed() -> None:
    sc, geom = _session([12, 8])
    a = sample_masks(geom, 20, n_time=96, n_rows=4, generator=_gen(7))
    b = sample_masks(geom, 20, n_time=96, n_rows=4, generator=_gen(7))
    c = sample_masks(geom, 20, n_time=96, n_rows=4, generator=_gen(8))
    assert torch.equal(a.contact_mask, b.contact_mask) and torch.equal(a.frame_mask, b.frame_mask)
    assert not torch.equal(a.contact_mask, c.contact_mask)


def test_rows_are_independent() -> None:
    # multi-shaft (realistic); a real montage always has S≫1 anyway. (The old V=1
    # degeneracy note belonged to the time guardian, deleted 2026-07-15 — the sampler
    # no longer has any V-dependent constraint.)
    sc, geom = _session([8, 6, 6])
    m = sample_masks(geom, 20, n_time=96, n_rows=8, generator=_gen(5))
    assert not all(torch.equal(m.contact_mask[0], m.contact_mask[r]) for r in range(1, 8))
    assert not all(torch.equal(m.frame_mask[0], m.frame_mask[r]) for r in range(1, 8))


def test_only_valid_contacts_masked() -> None:
    sc, geom = _session([7, 5])  # N=12
    m = sample_masks(geom, 12, n_time=96, n_rows=4, generator=_gen())
    assert m.contact_mask.shape == (4, 12)
    assert m.contact_mask.sum(1).unique().tolist() == [round(V3MaskConfig().space_frac * 12)]


def test_feasible_passes_for_realistic_seeg() -> None:
    sc, geom = _session([16, 12, 10, 9, 8, 6])  # N=61
    assert_mask_feasible(geom)


def test_feasible_flags_whole_infeasible() -> None:
    # one dominant grid block larger than D ⇒ no shaft fits under D ⇒ whole tier infeasible.
    sc, geom = _session([40, 4, 4])  # N=48, D=24, largest=40>24
    cfg = V3MaskConfig(space_frac=0.5, whole_shaft_frac=0.15)
    with pytest.raises(ValueError, match="whole-shaft infeasible"):
        assert_mask_feasible(geom, cfg)
