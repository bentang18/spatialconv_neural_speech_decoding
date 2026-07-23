"""r6 masking — ``sample_masks_r6``.

r4's 3-band leak-safe STRUCTURE (one per-shaft-balanced SPACE mask shared across bands +
three width-``block_w_band`` band TIME masks on the SLOW/MID/HGA native grids) crossed with
r5's PER-SHAFT INDEPENDENCE (each band's time mask is ``(R,S,T_b)``, drawn per shaft, not the
r4 global ``(R,T_b)``). The M14 margin-2 in_loss gate is retained DOWNSTREAM (objective) — this
sampler only lays the width-4 blocks; it does not gate. Invariants named + asserted + printed.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import (
    V3MaskConfig,
    V3MasksR6,
    sample_masks_r6,
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


def test_shape_and_four_fields() -> None:
    # T=32 ⇒ HGA 32 (32 Hz), MID 16 (16 Hz), SLOW 4 (4 Hz); SLOW grid 4 == block_w_band (leak-safe).
    sc, geom = _session([3, 3, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=5, generator=_gen())
    assert isinstance(m, V3MasksR6)
    assert m.contact_mask.shape == (5, n)
    assert m.hga_mask.shape == (5, geom.n_shafts, 32)
    assert m.mid_mask.shape == (5, geom.n_shafts, 16)
    assert m.slow_mask.shape == (5, geom.n_shafts, 4)
    print("[check] OK V3MasksR6: shared contact (R,N) + per-shaft SLOW/MID/HGA (R,S,T_b)")


def test_each_band_temporal_count_is_exact_per_shaft() -> None:
    # per-band per-shaft count == round(frac · band-grid length) EXACTLY (static compiled shapes).
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
            counts = tm.sum(-1)  # (R, S) per shaft
            assert torch.all(counts == want), (name, frac, counts.unique().tolist())
    print("[check] OK per-band per-shaft count == round(frac·length) exactly, all 3 bands")


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
        per_shaft = tm.long().sum(-1)  # (R, S)
        assert torch.all(per_shaft == per_shaft[0, 0])
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


def test_temporal_is_per_shaft_independent() -> None:
    # THE r5-borrowed invariant: unlike r4's GLOBAL band masks, each band mask is per-shaft.
    sc, geom = _session([3, 5, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=4, generator=_gen(2))
    for tm in (m.hga_mask, m.mid_mask, m.slow_mask):
        any_differ = any(
            not torch.equal(tm[r, i], tm[r, j])
            for r in range(tm.shape[0])
            for i in range(tm.shape[1])
            for j in range(i + 1, tm.shape[1])
        )
        assert any_differ, "shafts share a band mask — not per-shaft independent"
    print("[check] OK each band mask is (R,S,T_b) per-shaft independent")


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
