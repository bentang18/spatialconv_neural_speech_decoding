"""v14_converged_v3 Phase 5 — electrode-unit time-tube masking (TDD).

Memo project-v14-converged-v3-sensor-architecture, two-tier per-SHAFT scheme
(Ben 2026-07-09): (1) WHOLE-SENSOR tier — ``whole_shaft_frac`` of shafts masked
100% (the patient-invariant / L2 augmentation); (2) WITHIN-SENSOR tier — every
surviving shaft masked at its OWN ratio ``r ~ Uniform[r_lo, r_hi]`` via contiguous
depth-blocks, width ``Uniform{block_w_lo..hi}``, floor 4 = along-shaft HGA autocorr.

Locked contract: ``mask_frac 0.60 ⇒ M = round(0.60·N)`` held out, CONSTANT count
(⇒ static shapes / compile-once-per-session); per-shaft ratios average ~M and one
reconciliation argsort holds the count EXACTLY while never trimming whole shafts.
FULLY VECTORIZED (cover-rank + per-shaft top-k, no python loop). Vectorized over R
independent rows (one per clip in the batch).
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import (
    V3MaskConfig,
    sample_contact_mask,
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


def _runs(mask_row: torch.Tensor) -> list[int]:
    """Lengths of maximal True runs in a 1-D bool row."""
    out, cur = [], 0
    for v in mask_row.tolist():
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def _per_shaft_rate(sc, mask: torch.Tensor) -> torch.Tensor:
    """(R, S) fraction of each shaft's contacts masked."""
    R, S = mask.shape[0], int(sc.n_shafts)
    out = torch.zeros(R, S)
    for s in range(S):
        idx = (sc.shaft_id == s).nonzero(as_tuple=True)[0]
        out[:, s] = mask[:, idx].float().mean(dim=1)
    return out


def test_exact_constant_masked_count() -> None:
    sc, geom = _session([12, 10, 8, 6])  # N = 36
    n = 36
    m = round(0.60 * n)  # 22
    mask = sample_contact_mask(geom, n, n_rows=5, generator=_gen())
    assert mask.shape == (5, n)
    assert mask.dtype == torch.bool
    assert (mask.sum(dim=1) == m).all()


def test_mask_frac_config_controls_count() -> None:
    sc, geom = _session([20])
    mask = sample_contact_mask(
        geom, 20, n_rows=3, generator=_gen(), cfg=V3MaskConfig(mask_frac=0.5)
    )
    assert (mask.sum(dim=1) == 10).all()


def test_masking_is_blocky_not_iid_scatter() -> None:
    # Contiguous blocks ⇒ FAR fewer runs than an iid coin flip at the same rate.
    # Fixed per-shaft ratio at the target (r=mask_frac, no whole tier) isolates the
    # block mechanism: natural total = M, no reconciliation padding.
    sc, geom = _session([30])
    cfg = V3MaskConfig(
        mask_frac=0.6, block_w_lo=4, block_w_hi=8, whole_shaft_frac=0.0, r_lo=0.6, r_hi=0.6
    )
    mask = sample_contact_mask(geom, 30, n_rows=8, generator=_gen(1), cfg=cfg)
    mean_runs = sum(len(_runs(mask[r])) for r in range(8)) / 8
    assert mean_runs <= 4.0, f"mean runs/row {mean_runs} — not blocky enough"
    assert max(max(_runs(mask[r]), default=0) for r in range(8)) >= 4


def test_wider_blocks_give_longer_runs() -> None:
    sc, geom = _session([40])
    narrow = V3MaskConfig(
        mask_frac=0.5, block_w_lo=4, block_w_hi=4, whole_shaft_frac=0.0, r_lo=0.5, r_hi=0.5
    )
    wide = V3MaskConfig(
        mask_frac=0.5, block_w_lo=10, block_w_hi=10, whole_shaft_frac=0.0, r_lo=0.5, r_hi=0.5
    )
    mn = sample_contact_mask(geom, 40, n_rows=8, generator=_gen(2), cfg=narrow)
    mw = sample_contact_mask(geom, 40, n_rows=8, generator=_gen(2), cfg=wide)
    max_narrow = sum(max(_runs(mn[r]), default=0) for r in range(8)) / 8
    max_wide = sum(max(_runs(mw[r]), default=0) for r in range(8)) / 8
    assert max_wide > max_narrow, f"wide {max_wide} !> narrow {max_narrow}"


def test_whole_shaft_tier_masks_entire_shafts() -> None:
    # whole_shaft_frac=0.5 over 4 shafts ⇒ exactly 2 shafts masked 100% (never
    # trimmed), and those shafts are FULLY masked.
    sc, geom = _session([5, 5, 5, 5])  # N=20, M=12
    cfg = V3MaskConfig(mask_frac=0.6, whole_shaft_frac=0.5)
    mask = sample_contact_mask(geom, 20, n_rows=16, generator=_gen(3), cfg=cfg)
    rate = _per_shaft_rate(sc, mask)  # (16, 4)
    n_full = (rate >= 0.999).sum(dim=1)  # fully-masked shafts per row
    assert (n_full >= 2).all(), f"expected ≥2 whole shafts, got {n_full.tolist()}"


def test_whole_shaft_count_matches_frac() -> None:
    # round(whole_shaft_frac·S) shafts are masked 100%; the rest are NOT full.
    sc, geom = _session([8, 8, 8, 8, 8, 8])  # S=6, N=48
    cfg = V3MaskConfig(mask_frac=0.6, whole_shaft_frac=0.34)  # round(0.34·6)=2
    mask = sample_contact_mask(geom, 48, n_rows=200, generator=_gen(4), cfg=cfg)
    rate = _per_shaft_rate(sc, mask)  # (200, 6)
    n_full = (rate >= 0.999).float().mean(dim=0).sum()  # avg fully-masked shafts/row
    assert abs(n_full.item() - 2.0) < 0.25, f"avg whole shafts {n_full:.2f} != ~2"


def test_within_shaft_ratio_spreads_over_range() -> None:
    # The per-shaft ratio r~Uniform[r_lo,r_hi] gives a BOUNDED spread — no shaft
    # masked "too much" (the pooled-budget clumping is gone). r_mean = mask_frac and
    # no whole tier ⇒ minimal reconciliation, so realised rates track the draw.
    sc, geom = _session([16] * 8)  # 8 equal shafts, N=128
    cfg = V3MaskConfig(
        mask_frac=0.5, whole_shaft_frac=0.0, block_w_lo=4, block_w_hi=8, r_lo=0.3, r_hi=0.7
    )
    mask = sample_contact_mask(geom, 128, n_rows=2000, generator=_gen(6), cfg=cfg)
    rate = _per_shaft_rate(sc, mask).reshape(-1)  # all (row, shaft) rates
    assert abs(rate.mean().item() - 0.5) < 0.03
    # bounded: essentially nothing above r_hi (no more pathological ~100% shafts)
    assert (rate > 0.85).float().mean().item() < 0.02
    # genuinely varying (not collapsed to a single per-shaft rate)
    assert rate.std().item() > 0.06


def test_fixed_within_ratio_masks_every_shaft_at_that_rate() -> None:
    # r_lo=r_hi=0.5, no whole tier ⇒ every shaft masked ~50%, uniformly.
    sc, geom = _session([12] * 5)  # N=60
    cfg = V3MaskConfig(
        mask_frac=0.5, whole_shaft_frac=0.0, block_w_lo=4, block_w_hi=8, r_lo=0.5, r_hi=0.5
    )
    mask = sample_contact_mask(geom, 60, n_rows=1000, generator=_gen(7), cfg=cfg)
    rate = _per_shaft_rate(sc, mask)  # (1000, 5)
    assert (rate.mean(dim=0) - 0.5).abs().max().item() < 0.03


def test_only_valid_contacts_masked_and_time_tube_expand() -> None:
    sc, geom = _session([7, 5])  # N=12
    mask = sample_contact_mask(geom, 12, n_rows=4, generator=_gen())
    assert mask.shape == (4, 12)
    T = 128
    tube = mask[:, :, None].expand(4, 12, T)
    assert tube.shape == (4, 12, T)
    assert (tube.any(dim=2) == mask).all()  # a masked contact ⇒ its whole tube


def test_deterministic_in_generator_seed() -> None:
    sc, geom = _session([12, 8])
    a = sample_contact_mask(geom, 20, n_rows=4, generator=_gen(7))
    b = sample_contact_mask(geom, 20, n_rows=4, generator=_gen(7))
    c = sample_contact_mask(geom, 20, n_rows=4, generator=_gen(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_rows_are_independent() -> None:
    sc, geom = _session([20])
    mask = sample_contact_mask(geom, 20, n_rows=8, generator=_gen(5))
    assert not all(torch.equal(mask[0], mask[r]) for r in range(1, 8))


def test_shallow_edge_coverage_is_uniform() -> None:
    # Contiguous-block masking under-covers shaft boundaries: contact 0 is coverable
    # ONLY by a block starting at 0. Negative starts + random tiebreak fix it so the
    # shallowest contact's marginal mask rate ≈ interior ≈ the per-shaft ratio.
    sc, geom = _session([12, 12, 12, 12, 12])  # 5 equal shafts, len 12
    n = 60
    cfg = V3MaskConfig(
        mask_frac=0.6, block_w_lo=4, block_w_hi=8, whole_shaft_frac=0.0, r_lo=0.6, r_hi=0.6
    )
    mask = sample_contact_mask(geom, n, n_rows=3000, generator=_gen(0), cfg=cfg).float()
    shallow_rates, interior_rates = [], []
    for s in range(5):
        idx = (sc.shaft_id == s).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(sc.depth[idx])]
        rate = mask[:, idx].mean(dim=0)  # (12,) per depth-rank
        shallow_rates.append(rate[0].item())
        interior_rates.append(rate[3:9].mean().item())
    shallow = sum(shallow_rates) / 5
    interior = sum(interior_rates) / 5
    assert abs(shallow - 0.60) < 0.08, f"shallow rate {shallow:.3f} far from 0.60"
    assert abs(shallow - interior) < 0.08, f"shallow {shallow:.3f} vs interior {interior:.3f}"
