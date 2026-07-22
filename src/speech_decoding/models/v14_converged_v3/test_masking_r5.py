"""r5 (Chang 2-stream) single-band masking — ``sample_masks_r5``.

r5 has ONE early-fused 32 Hz band, so masking is: the SAME per-shaft balanced SPACE tier as
``sample_masks`` + a PER-SHAFT temporal mask (``temporal_mask_frac``) on the single grid (no
SLOW/MID/HGA trio) — each shaft hides its own timepoints (shaft = independent axiom).
Invariants named + asserted + printed.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import (
    V3MaskConfig,
    V3MasksR5,
    sample_masks_r5,
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


def test_shape_and_type() -> None:
    sc, geom = _session([3, 3, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r5(geom, n, n_time=16, n_rows=5, generator=_gen())
    assert isinstance(m, V3MasksR5)
    assert m.contact_mask.shape == (5, n)
    assert m.temporal_mask.shape == (5, geom.n_shafts, 16)  # (R, S, T) — per-shaft time axis
    print(f"[check] OK V3MasksR5 contact {tuple(m.contact_mask.shape)} temporal {tuple(m.temporal_mask.shape)}")


def test_temporal_exact_count_and_frac_controls_it() -> None:
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    t = 20
    for frac in (0.25, 0.5, 0.75):
        cfg = V3MaskConfig(temporal_mask_frac=frac)
        m = sample_masks_r5(geom, n, n_time=t, n_rows=16, generator=_gen(1), cfg=cfg)
        want = round(frac * t)
        counts = m.temporal_mask.sum(-1)  # (R, S) — count over the T axis, per shaft
        assert torch.all(counts == want), (frac, counts.unique().tolist())
    print(f"[check] OK temporal count == round(frac·T) exactly per (row, shaft) for frac in {{0.25,0.5,0.75}}")


def test_temporal_is_per_shaft_independent() -> None:
    # (R, S, T): each shaft has its OWN temporal mask — shafts within a row hide DIFFERENT
    # timepoints (the independence axiom). Count per shaft is identical (shape-neutral), but
    # the masked SET differs across shafts, else this collapses back to a shared mask.
    sc, geom = _session([3, 5, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r5(geom, n, n_time=24, n_rows=4, generator=_gen(2))
    assert m.temporal_mask.dim() == 3 and m.temporal_mask.shape == (4, geom.n_shafts, 24)
    # some pair of shafts in some row must differ — a shared mask would make all rows equal.
    tm = m.temporal_mask
    any_differ = any(
        not torch.equal(tm[r, i], tm[r, j])
        for r in range(tm.shape[0]) for i in range(tm.shape[1]) for j in range(i + 1, tm.shape[1])
    )
    assert any_differ, "shafts share a temporal mask — not independent"
    print("[check] OK temporal mask is (R, S, T) per-shaft — shafts hide different timepoints")


def test_temporal_is_blocky_via_temporal_block_w() -> None:
    # Contiguous width-``temporal_block_w`` blocks (r5's OWN knob, not block_w_band) ⇒ mean
    # masked-run length grows with the width; an iid Bernoulli mask would be ~1-2.
    sc, geom = _session([6, 6])
    n = int(geom.valid.sum())
    means = {}
    for w in (4, 6):
        cfg = V3MaskConfig(temporal_mask_frac=0.5, temporal_block_w=w)
        m = sample_masks_r5(geom, n, n_time=64, n_rows=16, generator=_gen(7), cfg=cfg)
        runs = [r for row in m.temporal_mask.reshape(-1, 64) for r in _runs(row)]  # per (row,shaft)
        means[w] = sum(runs) / len(runs)
    assert means[6] > means[4] > 1.5, means  # wider blocks ⇒ longer runs
    print(f"[check] OK temporal_block_w drives blockiness: mean run w4={means[4]:.2f} < w6={means[6]:.2f}")


def test_default_temporal_block_w_is_five() -> None:
    # r5 default block width = 5 tokens = 156 ms floor @ 32 Hz (~190 ms mean run after overlap;
    # Ben 2026-07-22, τ-anchored to LFS 83 ms first-zero, deliberately shorter than speech 200 ms).
    assert V3MaskConfig().temporal_block_w == 5
    print("[check] OK default temporal_block_w == 5 (156 ms floor, ~190 ms mean run @ 32 Hz)")


def test_space_balanced_and_keep_alive() -> None:
    # SPACE tier == sample_masks: per-shaft ~space_frac, ≥1 visible per shaft (keep-alive).
    sc, geom = _session([4, 4, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r5(geom, n, n_time=16, n_rows=64, generator=_gen(3))
    # per-shaft: each shaft masks round(0.5·4)=2 of its 4 contacts ⇒ 2 visible ≥ 1.
    csum = m.contact_mask.long().sum(1)  # (R,) total masked per row
    assert torch.all(csum == 6)  # 3 shafts × 2 masked
    assert not m.contact_mask.all(1).any()  # never a fully-masked row (keep-alive)
    print(f"[check] OK space balanced (6/12 masked/row), keep-alive holds")


def test_deterministic_in_generator_seed() -> None:
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    a = sample_masks_r5(geom, n, n_time=16, n_rows=4, generator=_gen(7))
    b = sample_masks_r5(geom, n, n_time=16, n_rows=4, generator=_gen(7))
    c = sample_masks_r5(geom, n, n_time=16, n_rows=4, generator=_gen(8))
    same = torch.equal(a.temporal_mask, b.temporal_mask) and torch.equal(a.contact_mask, b.contact_mask)
    diff = not torch.equal(a.temporal_mask, c.temporal_mask)
    assert same and diff
    print(f"[check] OK same seed → identical ({same}); different seed → differs ({diff})")
