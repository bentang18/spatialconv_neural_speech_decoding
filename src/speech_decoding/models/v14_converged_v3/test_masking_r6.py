"""r6 masking — ``sample_masks_r6``.

r4's 3-band leak-safe STRUCTURE (one per-shaft-balanced SPACE mask shared across bands +
three width-``block_w_band`` band TIME masks on the SLOW/MID/HGA native grids) crossed with
PER-SENSOR INDEPENDENCE (each band's time mask is ``(R,N,T_b)``, drawn per contact, not the
r4 global ``(R,T_b)``) — Ben 2026-07-23: the encoder is L1-within-shaft only, so per-sensor
masking is free extra diversity. The M14 margin gate is GONE downstream (``in_loss == masked``);
this sampler only lays the width-4 blocks. Invariants named + asserted + printed.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import (
    V3MaskConfig,
    V3MasksR6,
    assert_mask_feasible,
    sample_masks_r6,
)
from speech_decoding.models.v14_converged_v3.pack_r4 import build_r4_grid, token_flags_r6
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


def test_shape_and_four_fields() -> None:
    # T=32 ⇒ HGA 32 (32 Hz), MID 16 (16 Hz), SLOW 4 (4 Hz); SLOW grid 4 == block_w_band (leak-safe).
    sc, geom = _session([3, 3, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=5, generator=_gen())
    assert isinstance(m, V3MasksR6)
    assert m.contact_mask.shape == (5, n)
    assert m.hga_mask.shape == (5, n, 32)
    assert m.mid_mask.shape == (5, n, 16)
    assert m.slow_mask.shape == (5, n, 4)
    print("[check] OK V3MasksR6: shared contact (R,N) + per-sensor SLOW/MID/HGA (R,N,T_b)")


def test_each_band_temporal_count_is_exact_per_sensor() -> None:
    # per-band PER-SENSOR count == round(frac · band-grid length) EXACTLY. This is what keeps the
    # visible-token count a per-session constant under per-sensor masking: every contact hides the
    # same number of frames ⇒ per-shaft total = n_contacts · cnt ⇒ m_vis/cu_seqlens unchanged.
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    for frac in (0.25, 0.5, 0.75):
        cfg = V3MaskConfig(
            hga_mask_frac=frac, mid_mask_frac=frac, slow_mask_frac=frac
        )
        m = sample_masks_r6(geom, n, n_time=32, n_rows=16, generator=_gen(1), cfg=cfg)
        for name, tm, length in (
            ("hga", m.hga_mask, 32), ("mid", m.mid_mask, 16), ("slow", m.slow_mask, 4)
        ):
            want = round(frac * length)
            counts = tm.sum(-1)  # (R, N) per contact
            assert torch.all(counts == want), (name, frac, counts.unique().tolist())
    print("[check] OK per-band per-sensor count == round(frac·length) exactly, all 3 bands")


def test_space_balanced_shared_and_keep_alive() -> None:
    # SPACE is ONE mask shared across bands (r4 outer-product structure), per-shaft balanced.
    sc, geom = _session([4, 4, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=64, generator=_gen(3))
    csum = m.contact_mask.long().sum(1)  # (R,) total masked per row
    assert torch.all(csum == 6), csum.unique().tolist()  # 3 shafts × round(0.5·4)=2
    assert not m.contact_mask.all(1).any()  # keep-alive: never a fully-masked row
    print("[check] OK shared per-shaft-balanced space (6/12), keep-alive holds")


def test_masked_counts_constant_across_rows_static_shapes() -> None:
    sc, geom = _session([3, 5, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=8, generator=_gen(2))
    per_row = m.contact_mask.long().sum(1)
    assert torch.all(per_row == per_row[0])
    for tm in (m.hga_mask, m.mid_mask, m.slow_mask):
        per_sensor = tm.long().sum(-1)  # (R, N)
        assert torch.all(per_sensor == per_sensor[0, 0])
    print("[check] OK contact + per-band counts row-constant ⇒ static compiled shapes")


def test_bands_are_independent() -> None:
    # r4 structure: the three band masks are drawn INDEPENDENTLY on their own grids. Compare the
    # bands on their shared coarse (SLOW) lattice — a SLOW-masked slot need not be HGA/MID-masked.
    sc, geom = _session([4, 4, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=8, generator=_gen(5))
    hga_on_slow = m.hga_mask[..., ::8]  # (R,S,4) HGA sampled at SLOW lattice positions
    mid_on_slow = m.mid_mask[..., ::4]  # (R,S,4) MID sampled at SLOW lattice positions
    assert not torch.equal(hga_on_slow, m.slow_mask)
    assert not torch.equal(mid_on_slow, m.slow_mask)
    assert not torch.equal(hga_on_slow, mid_on_slow)
    print("[check] OK SLOW/MID/HGA masks independent (differ at shared lattice positions)")


def test_temporal_is_per_sensor_independent_within_a_shaft() -> None:
    # THE r6 invariant: unlike r4's GLOBAL band masks, each band mask is per-SENSOR — and the test
    # that separates per-sensor from the old per-shaft draw is that two contacts of the SAME shaft
    # get different masks. Compare within shaft 1 (contacts 3..7) only.
    sc, geom = _session([3, 5, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=4, generator=_gen(2))
    same_shaft = [i for i in range(n) if int(geom.shaft_of_contact[i]) == 1]
    assert len(same_shaft) >= 2
    for tm in (m.hga_mask, m.mid_mask, m.slow_mask):
        any_differ = any(
            not torch.equal(tm[r, i], tm[r, j])
            for r in range(tm.shape[0])
            for a, i in enumerate(same_shaft)
            for j in same_shaft[a + 1 :]
        )
        assert any_differ, "contacts of one shaft share a band mask — not per-sensor independent"
    print("[check] OK each band mask is (R,N,T_b) per-sensor independent WITHIN a shaft")


def test_block_width_controls_masked_run_length() -> None:
    # block_w_band lays contiguous width-blocks: wider ⇒ longer contiguous masked runs at the SAME
    # frac (the M14 leak-safe knob). Assert the WIDTH drives it on the HGA grid (narrow vs wide).
    sc, geom = _session([6, 6])
    n = int(geom.valid.sum())
    t = 32

    def mean_max_run(tm: torch.Tensor) -> float:
        runs = []
        for row in tm.reshape(-1, t):
            best = cur = 0
            for v in row.tolist():
                cur = cur + 1 if v else 0
                best = max(best, cur)
            runs.append(best)
        return sum(runs) / len(runs)

    narrow = sample_masks_r6(
        geom, n, n_time=t, n_rows=64, generator=_gen(11),
        cfg=V3MaskConfig(block_w_band=1),
    )
    wide = sample_masks_r6(
        geom, n, n_time=t, n_rows=64, generator=_gen(11),
        cfg=V3MaskConfig(block_w_band=8),
    )
    r_narrow, r_wide = mean_max_run(narrow.hga_mask), mean_max_run(wide.hga_mask)
    assert r_wide > r_narrow + 1.0, (r_narrow, r_wide)
    print(f"[check] OK block_w_band drives run length: w=1 {r_narrow:.2f} < w=8 {r_wide:.2f}")


def test_band_block_w_1_is_uniform_random_masking() -> None:
    """A1's premise: at ``block_w_band=1`` the cover degenerates to an independent uniform draw.

    "Shorter runs" (the test above) does NOT prove randomness on its own, so assert the two
    properties that actually define it:
      1. MARGINAL uniformity — every grid position is masked at the same rate.
      2. NO CONTIGUITY — P(i+1 masked | i masked) ≈ the without-replacement rate 15/31 ≈ .484,
         whereas the leak-safe width-4 block drives that conditional far above it by construction.
    Without this, the random-masking ablation would not be testing what it claims to test.
    """
    sc, geom = _session([6, 6])
    n = int(geom.valid.sum())
    t, rows = 32, 512

    def hga(width: int) -> torch.Tensor:
        m = sample_masks_r6(
            geom, n, n_time=t, n_rows=rows, generator=_gen(7),
            cfg=V3MaskConfig(block_w_band=width),
        )
        return m.hga_mask.reshape(-1, t).float()

    w1, w4 = hga(1), hga(4)

    rate = w1.mean(0)  # (T,) per-position marginal
    dev = float((rate - 0.5).abs().max())
    assert dev < 0.08, (dev, rate.tolist())

    def adjacency(m: torch.Tensor) -> float:
        both = (m[:, :-1] * m[:, 1:]).sum()
        return float(both / m[:, :-1].sum())

    a1, a4 = adjacency(w1), adjacency(w4)
    assert a1 < 0.55, a1
    assert a4 > 0.65, a4
    print(
        f"[check] OK block_w_band=1 is UNIFORM RANDOM: max |rate-0.5| {dev:.4f} < .08; "
        f"P(i+1|i) w=1 {a1:.3f} (chance 15/31={15 / 31:.3f}) vs w=4 {a4:.3f}"
    )


def test_ntime_not_multiple_of_slow_stride_raises() -> None:
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    for bad in (20, 12):  # not divisible by SLOW_STRIDE=8
        try:
            sample_masks_r6(geom, n, n_time=bad, n_rows=2, generator=_gen())
        except ValueError:
            continue
        raise AssertionError(f"n_time={bad} should raise (not a multiple of SLOW_STRIDE)")
    print("[check] OK n_time not divisible by SLOW_STRIDE=8 raises")


def test_deterministic_in_generator_seed() -> None:
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    a = sample_masks_r6(geom, n, n_time=32, n_rows=4, generator=_gen(7))
    b = sample_masks_r6(geom, n, n_time=32, n_rows=4, generator=_gen(7))
    c = sample_masks_r6(geom, n, n_time=32, n_rows=4, generator=_gen(8))
    same = (
        torch.equal(a.contact_mask, b.contact_mask)
        and torch.equal(a.hga_mask, b.hga_mask)
        and torch.equal(a.mid_mask, b.mid_mask)
        and torch.equal(a.slow_mask, b.slow_mask)
    )
    diff = not torch.equal(a.hga_mask, c.hga_mask)
    assert same and diff
    print(f"[check] OK same seed → identical ({same}); different seed → differs ({diff})")


# ── space_frac == 0.0: the no-spatial-masking arm ────────────────────────────
# Ben 2026-07-28: ms70 → keeper → ms25 was monotone in the probe as spatial masking fell,
# so the endpoint (no spatial masking at all, time held at the ASR-convention 0.50) has to
# be reachable. It was not: assert_mask_feasible rejected Σd_s == 0. These pin down that the
# arm is WELL-DEFINED, not merely permitted — the loss must still have targets.


def test_space_frac_zero_is_feasible_and_masks_no_contacts() -> None:
    sc, geom = _session([3, 3, 4])
    cfg = V3MaskConfig(space_frac=0.0)
    assert_mask_feasible(geom, cfg)  # must not raise
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=5, generator=_gen(), cfg=cfg)
    assert not m.contact_mask.any(), "space_frac=0 must mask zero contacts"
    # the time masks are untouched — each band still hides round(0.50 * T_b) per sensor
    for name, band in (("slow", m.slow_mask), ("mid", m.mid_mask), ("hga", m.hga_mask)):
        assert band.any(), f"{name} time mask empty — the arm would have no loss targets"
    print(f"[check] space_frac=0 OK: contacts masked=0, time masks non-empty (N={n})")


def test_space_frac_zero_still_leaves_loss_targets() -> None:
    """The guard that matters: masked == time mask alone, and it is NOT empty."""
    sc, geom = _session([3, 3, 4])
    n = int(geom.valid.sum())
    grid = build_r4_grid(geom, n_time=32)
    m = sample_masks_r6(
        geom, n, n_time=32, n_rows=4, generator=_gen(), cfg=V3MaskConfig(space_frac=0.0)
    )
    masked, in_loss = token_flags_r6(grid, m)
    assert masked.any(), "no masked tokens ⇒ _masked_mean_l1 would divide by zero"
    assert bool((in_loss == masked).all()), "r6 contract: in_loss == masked"
    frac = float(masked.float().mean())
    # space ∪ time collapses to time alone; the three bands sit at 0.50 each.
    assert 0.3 < frac < 0.7, f"masked fraction {frac:.3f} implausible for time-only masking"
    print(f"[check] space_frac=0 loss targets OK: masked fraction={frac:.3f}, in_loss==masked")


def test_degenerate_montage_still_raises_when_masking_was_requested() -> None:
    """The guard keeps its real job: space_frac > 0 that silently masks nothing is a bug."""
    sc, geom = _session([1, 1, 1])  # every shaft size 1 ⇒ keep_alive forces d_s = 0
    try:
        assert_mask_feasible(geom, V3MaskConfig(space_frac=0.5))
    except ValueError as e:
        assert "not in (0, N=" in str(e)
        print(f"[check] degenerate montage still raises: {e}")
        return
    raise AssertionError("space_frac=0.5 on all-size-1 shafts must still raise")
