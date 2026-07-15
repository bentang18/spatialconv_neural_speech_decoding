"""v14_converged_v3 Phase 5 — unified space × time block masking (TDD).

SPACE is PER-SHAFT balanced (each shaft masks round(space_frac·n_s) of its own contacts via
depth-blocks; keep-alive automatic). TIME is GLOBAL across shafts, ONE unified rule: each band is
masked INDEPENDENTLY, in contiguous width-block_w_band (=4) blocks of its OWN tokens, snapped to
~*_mask_frac. HGA 32 Hz / MID 16 Hz / SLOW 4 Hz — symmetric, no blackout (empty latent windows
emerge where all three masks overlap).
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import (
    MID_STRIDE,
    SLOW_STRIDE,
    V3MaskConfig,
    assert_mask_feasible,
    assert_time_feasible,
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


def _expected_space(cfg: V3MaskConfig, sizes: list[int]) -> int:
    cs = torch.tensor(sizes)
    d_s = torch.round(cfg.space_frac * cs.float()).long()
    if cfg.keep_alive:
        d_s = torch.minimum(d_s, (cs - 1).clamp(min=0))
    return int(d_s.sum())


def _expected_counts(cfg: V3MaskConfig, t: int) -> dict[str, int]:
    t_mid = t // MID_STRIDE
    n_slots = t // SLOW_STRIDE
    return {
        "hga": round(cfg.hga_mask_frac * t),
        "mid": round(cfg.mid_mask_frac * t_mid),
        "slow": round(cfg.slow_mask_frac * n_slots),
        "n_slots": n_slots,
    }


# ── SPACE axis: per-shaft balanced count, blockiness ─────────────────────────
def test_space_exact_constant_count() -> None:
    sizes = [12, 10, 8, 6]
    sc, geom = _session(sizes)  # N=36
    n = 36
    d = _expected_space(V3MaskConfig(), sizes)  # Σ round(0.5·n_s) = 18
    m = sample_masks(geom, n, n_time=128, n_rows=5, generator=_gen())
    assert m.contact_mask.shape == (5, n) and m.contact_mask.dtype == torch.bool
    assert (m.contact_mask.sum(1) == d).all()


def test_space_frac_controls_count() -> None:
    sizes = [20]
    sc, geom = _session(sizes)
    cfg = V3MaskConfig(space_frac=0.4)
    m = sample_masks(geom, 20, n_time=128, n_rows=3, generator=_gen(), cfg=cfg)
    assert (m.contact_mask.sum(1) == _expected_space(cfg, sizes)).all()  # 8


def test_space_is_per_shaft_balanced() -> None:
    # THE FIX (Ben 2026-07-15): every shaft masks ~space_frac of its OWN contacts. No shaft is
    # left fully unmasked while another is hammered — the old global top-D had std 0.18, min 0.00.
    sizes = [12, 10, 14, 8, 10, 12, 6, 9, 7, 11]
    sc, geom = _session(sizes)
    n = sum(sizes)
    m = sample_masks(geom, n, n_time=128, n_rows=200, generator=_gen(1))
    rate = _per_shaft_rate(sc, m.contact_mask)  # (R, S)
    # each shaft is deterministically round(0.5·n_s)/n_s masked — exact, zero variance across rows.
    for si, sz in enumerate(sizes):
        exp = min(round(0.5 * sz), sz - 1) / sz
        assert torch.allclose(rate[:, si], torch.full((200,), exp), atol=1e-6), (
            f"shaft {si} (size {sz}) rate {rate[:, si].unique().tolist()} != {exp}"
        )
    assert rate.min() > 0.0, "some shaft fully unmasked — imbalance not fixed"


def test_space_is_blocky_not_iid() -> None:
    sc, geom = _session([30])
    cfg = V3MaskConfig(space_frac=0.5, whole_shaft_frac=0.0, block_w_space=4)
    m = sample_masks(geom, 30, n_time=128, n_rows=8, generator=_gen(1), cfg=cfg)
    mean_runs = sum(len(_runs(m.contact_mask[r])) for r in range(8)) / 8
    assert mean_runs <= 4.5, f"mean runs/row {mean_runs} — not blocky"
    assert max(max(_runs(m.contact_mask[r]), default=0) for r in range(8)) >= 4


def test_whole_shaft_tier_fully_masks_shafts() -> None:
    sc, geom = _session([5, 5, 5, 5])  # N=20
    cfg = V3MaskConfig(space_frac=0.5, whole_shaft_frac=0.5, keep_alive=False)
    m = sample_masks(geom, 20, n_time=128, n_rows=64, generator=_gen(3), cfg=cfg)
    rate = _per_shaft_rate(sc, m.contact_mask)
    n_full = (rate >= 0.999).sum(1)  # k_max caps whole count at 1 for this montage (D=8)
    assert (n_full >= 1).float().mean() > 0.5, f"expected whole shafts in most rows, got {n_full.tolist()}"
    assert (m.whole_contact & ~m.contact_mask).sum() == 0


def test_whole_shaft_count_is_stochastic() -> None:
    sc, geom = _session([8, 8, 8, 8, 8, 8])  # S=6, N=48
    cfg = V3MaskConfig(space_frac=0.5, whole_shaft_frac=0.25, keep_alive=False)  # E[K]=1.5
    m = sample_masks(geom, 48, n_time=128, n_rows=400, generator=_gen(4), cfg=cfg)
    whole_shafts = torch.zeros(400, 6)
    for si in range(6):
        idx = (sc.shaft_id == si).nonzero(as_tuple=True)[0]
        whole_shafts[:, si] = m.whole_contact[:, idx].all(1).float()
    k_per_row = whole_shafts.sum(1)
    assert k_per_row.unique().numel() >= 3, f"whole count not stochastic: {k_per_row.unique().tolist()}"
    assert 0.5 < k_per_row.mean().item() < 2.5, f"E[K] off: {k_per_row.mean().item():.2f}"


# ── TIME axis: GLOBAL across shafts, per-band budgets ────────────────────────
def test_time_masks_are_global_no_shaft_axis() -> None:
    sc, geom = _session([12, 10, 8])
    t = 128
    m = sample_masks(geom, 30, n_time=t, n_rows=5, generator=_gen(2))
    assert m.hga_mask.shape == (5, t)
    assert m.mid_mask.shape == (5, t // MID_STRIDE)
    assert m.slow_mask.shape == (5, t // SLOW_STRIDE)
    assert all(x.dtype == torch.bool for x in (m.hga_mask, m.mid_mask, m.slow_mask))


def test_time_exact_per_band_counts() -> None:
    sc, geom = _session([12, 10, 8])
    t = 128
    cfg = V3MaskConfig()
    exp = _expected_counts(cfg, t)
    m = sample_masks(geom, 30, n_time=t, n_rows=16, generator=_gen(2), cfg=cfg)
    assert (m.hga_mask.sum(-1) == exp["hga"]).all(), "HGA count not constant"
    assert (m.mid_mask.sum(-1) == exp["mid"]).all(), "MID count = round(mid_frac·T/2) not constant"
    assert (m.slow_mask.sum(-1) == exp["slow"]).all(), "SLOW count = round(slow_frac·T/8) not constant"


def test_all_three_bands_near_50pct() -> None:
    # ONE rule, three grids: HGA/MID/SLOW each masked ~50% of their OWN tokens (Ben 2026-07-15:
    # "each band its own independent temporal masking ... slow can be masked well ... 50% symmetry").
    sc, geom = _session([12, 10])
    t = 128
    m = sample_masks(geom, 22, n_time=t, n_rows=32, generator=_gen(3))
    assert (m.hga_mask.float().mean() - 0.50).abs() < 1e-6
    assert (m.mid_mask.float().mean() - 0.50).abs() < 1e-6
    assert (m.slow_mask.float().mean() - 0.50).abs() < 1e-6


def test_each_band_is_blocky_width4() -> None:
    # Leak-safe: each band masked in contiguous blocks of ≥block_w_band (=4) of its OWN tokens.
    sc, geom = _session([12])
    t = 128
    m = sample_masks(geom, 12, n_time=t, n_rows=16, generator=_gen(7))
    for name, band in (("hga", m.hga_mask), ("mid", m.mid_mask), ("slow", m.slow_mask)):
        longest = max(max(_runs(band[r]), default=0) for r in range(16))
        assert longest >= 4, f"{name}: longest run {longest} < 4 ⇒ not leak-safe width-4 blocky"


def test_bands_are_masked_independently() -> None:
    # No coupling: a SLOW-masked slot does NOT force its HGA/MID tokens masked (that was blackout).
    # Across many rows there exist slots where SLOW is masked but some HGA in that slot is visible.
    sc, geom = _session([10])
    t = 128
    m = sample_masks(geom, 10, n_time=t, n_rows=64, generator=_gen(6))
    hpp = SLOW_STRIDE
    found_slow_masked_hga_visible = False
    for r in range(64):
        for k in m.slow_mask[r].nonzero(as_tuple=True)[0].tolist():
            if not m.hga_mask[r, k * hpp:(k + 1) * hpp].all():
                found_slow_masked_hga_visible = True
    assert found_slow_masked_hga_visible, "bands appear coupled — SLOW-masked slots always empty HGA"


def test_slow_frac_controls_slow_count() -> None:
    sc, geom = _session([12])
    t = 128
    cfg = V3MaskConfig(slow_mask_frac=0.25)  # 0.25·16 = 4 SLOW slots
    m = sample_masks(geom, 12, n_time=t, n_rows=8, generator=_gen(), cfg=cfg)
    assert (m.slow_mask.sum(-1) == 4).all()
    exp = _expected_counts(cfg, t)
    assert (m.hga_mask.sum(-1) == exp["hga"]).all()  # HGA unaffected by SLOW frac (independent)


# ── OUTER-PRODUCT composition / static invariants ────────────────────────────
def test_outer_product_visible_hga_cells_matches_product() -> None:
    sizes = [16, 12, 10, 8]
    sc, geom = _session(sizes)  # N=46
    n, t = 46, 128
    cfg = V3MaskConfig()
    m = sample_masks(geom, n, n_time=t, n_rows=32, generator=_gen(7), cfg=cfg)
    d = _expected_space(cfg, sizes)
    exp = _expected_counts(cfg, t)
    t_kept = t - exp["hga"]
    cell_masked = m.contact_mask[:, :, None] | m.hga_mask[:, None, :]  # (R, N, T_hga)
    vis = (~cell_masked).sum(dim=(1, 2))
    assert (vis == (n - d) * t_kept).all(), f"HGA visible {vis.unique().tolist()} != {(n-d)*t_kept}"


def test_static_counts_are_constant_across_seeds() -> None:
    sizes = [14, 12, 10, 8, 6]
    sc, geom = _session(sizes)  # N=50
    n, t = 50, 128
    cfg = V3MaskConfig()
    d = _expected_space(cfg, sizes)
    exp = _expected_counts(cfg, t)
    for seed in range(20):
        m = sample_masks(geom, n, n_time=t, n_rows=8, generator=_gen(seed))
        assert (m.contact_mask.sum(1) == d).all()
        assert (m.hga_mask.sum(-1) == exp["hga"]).all()
        assert (m.mid_mask.sum(-1) == exp["mid"]).all()
        assert (m.slow_mask.sum(-1) == exp["slow"]).all()


# ── determinism / independence / feasibility ─────────────────────────────────
def test_deterministic_in_generator_seed() -> None:
    sc, geom = _session([12, 8])
    a = sample_masks(geom, 20, n_time=128, n_rows=4, generator=_gen(7))
    b = sample_masks(geom, 20, n_time=128, n_rows=4, generator=_gen(7))
    c = sample_masks(geom, 20, n_time=128, n_rows=4, generator=_gen(8))
    assert torch.equal(a.contact_mask, b.contact_mask) and torch.equal(a.hga_mask, b.hga_mask)
    assert torch.equal(a.mid_mask, b.mid_mask) and torch.equal(a.slow_mask, b.slow_mask)
    assert not (torch.equal(a.contact_mask, c.contact_mask) and torch.equal(a.hga_mask, c.hga_mask))


def test_rows_are_independent() -> None:
    sc, geom = _session([8, 6, 6])
    m = sample_masks(geom, 20, n_time=128, n_rows=8, generator=_gen(5))
    assert not all(torch.equal(m.contact_mask[0], m.contact_mask[r]) for r in range(1, 8))
    assert not all(torch.equal(m.hga_mask[0], m.hga_mask[r]) for r in range(1, 8))
    assert not all(torch.equal(m.mid_mask[0], m.mid_mask[r]) for r in range(1, 8))


def test_only_valid_contacts_masked() -> None:
    sizes = [7, 5]
    sc, geom = _session(sizes)  # N=12
    m = sample_masks(geom, 12, n_time=128, n_rows=4, generator=_gen())
    assert m.contact_mask.shape == (4, 12)
    assert m.contact_mask.sum(1).unique().tolist() == [_expected_space(V3MaskConfig(), sizes)]


def test_feasible_passes_for_realistic_seeg() -> None:
    sc, geom = _session([16, 12, 10, 9, 8, 6])  # N=61
    assert_mask_feasible(geom)
    assert_time_feasible(128)


def test_feasible_flags_whole_infeasible() -> None:
    sc, geom = _session([40, 4, 4])  # N=48, D≈24, largest=40>24
    cfg = V3MaskConfig(space_frac=0.5, whole_shaft_frac=0.15, keep_alive=False)
    with pytest.raises(ValueError, match="whole-shaft infeasible"):
        assert_mask_feasible(geom, cfg)


def test_time_feasible_flags_bad_clip_and_short_grid() -> None:
    with pytest.raises(ValueError, match="multiple of SLOW_STRIDE"):
        assert_time_feasible(100)
    with pytest.raises(ValueError, match="clip too short"):
        # n_time=16 ⇒ SLOW grid only 2 slots < block_w_band=4 ⇒ no leak-safe SLOW block fits.
        assert_time_feasible(16)


def test_time_feasible_passes_3s_clip() -> None:
    assert_time_feasible(96)  # 3 s @ 32 Hz: n_slots=12, t_mid=48, all bands leak-safe


# ── KEEP-ALIVE floor (Design B, L1-only predictor: no shaft may be fully masked) ──
def _dead_shaft_rows(sc, geom, contact_mask: torch.Tensor) -> torch.Tensor:
    r, n = contact_mask.shape
    s = int(sc.n_shafts)
    soc = geom.shaft_of_contact
    vps = geom.valid.sum(1)
    masked_per = torch.zeros(r, s, dtype=torch.long)
    masked_per.scatter_add_(1, soc[None].expand(r, n), contact_mask.long())
    return (masked_per == vps[None]).sum(1)


def test_keepalive_no_dead_shaft_default() -> None:
    sc, geom = _session([16, 12, 8, 6, 4, 4, 3, 2])  # N=55, has size-2/3/4
    n = 55
    for seed in range(6):
        m = sample_masks(geom, n, n_time=128, n_rows=500, generator=_gen(seed))
        dead = _dead_shaft_rows(sc, geom, m.contact_mask)
        assert dead.sum().item() == 0, f"seed {seed}: {int(dead.sum())} dead-shaft events"


def test_keepalive_off_allows_whole_drops() -> None:
    # keep_alive=False + whole tier ⇒ whole shafts fully masked (dead by design).
    sc, geom = _session([4, 4, 3, 3, 2, 4, 3, 2])  # tiny-stress, N=25
    n = 25
    cfg = V3MaskConfig(keep_alive=False, whole_shaft_frac=0.4)
    m = sample_masks(geom, n, n_time=128, n_rows=500, generator=_gen(0), cfg=cfg)
    assert _dead_shaft_rows(sc, geom, m.contact_mask).sum().item() > 0


def test_keepalive_protects_every_shaft() -> None:
    sc, geom = _session([8, 6, 5, 4])  # N=23
    n = 23
    m = sample_masks(geom, n, n_time=128, n_rows=200, generator=_gen(1))
    vis = ~m.contact_mask
    soc = geom.shaft_of_contact
    for si in range(int(sc.n_shafts)):
        idx = (soc == si).nonzero(as_tuple=True)[0]
        assert vis[:, idx].any(1).all(), f"shaft {si} fully masked in some row"


def test_keepalive_clamps_per_shaft_to_n_minus_one() -> None:
    # size-2 shafts at space_frac 0.5 ⇒ round(1)=1 ≤ n_s−1=1 (already fits); size-2 keeps 1 visible.
    sc, geom = _session([2, 2, 2, 2, 2])  # N=10, S=5
    n = 10
    m = sample_masks(geom, n, n_time=128, n_rows=50, generator=_gen(0))
    assert (m.contact_mask.sum(1) == 5).all(), "each size-2 shaft masks exactly 1 ⇒ Σ=5"
    assert _dead_shaft_rows(sc, geom, m.contact_mask).sum().item() == 0


def test_keepalive_feasible_flags_all_size1() -> None:
    sc, geom = _session([1, 1, 1, 1])  # N=4, every shaft size 1 ⇒ nothing maskable under keep-alive
    with pytest.raises(ValueError, match="not in"):
        assert_mask_feasible(geom)
