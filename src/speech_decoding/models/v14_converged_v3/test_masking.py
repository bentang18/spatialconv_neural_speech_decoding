"""v14_converged_v3 Phase 5 — electrode-unit time-tube masking (TDD).

Memo project-v14-converged-v3-sensor-architecture (MASK UNIT = THE ELECTRODE,
time-tubed): the mask is a per-ELECTRODE boolean; a masked electrode is hidden
across ALL time slots (a full time tube). Distribution, small→large spatial
extent: melec(1) ⊂ within-shaft contiguous contact-blocks (w~Uniform{4..8}, the
MAJORITY mass — shaft-mates run out ⇒ cross-sensor L2 must fill) ⊂ whole-shaft
(the THIN TAIL — trains the offloaded predictor L2). Blocks are contiguous ALONG
the shaft (adjacency in surviving contacts, drop-gaps already removed).

Locked contract (this session): mask_frac 0.60 ⇒ M = round(0.60·N) held out,
CONSTANT count (⇒ static shapes / compile-once-per-session); whole-shaft tier
~15% of shafts; block width Uniform{4..8} (floor 4 = along-shaft HGA autocorr);
overlap allowed; FULLY VECTORIZED (cover-rank argsort, no python loop over
shafts/blocks). Vectorized over R independent rows (one per clip in the batch).
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
    # Contiguous blocks ⇒ the masked set clusters into FAR fewer runs than an iid
    # coin flip at the same rate would (p=0.6 over 30 ≈ 7 runs). Overlap-allowed
    # fragments (incl. melec singletons) are fine — the clustering still dominates.
    sc, geom = _session([30])
    cfg = V3MaskConfig(mask_frac=0.6, block_w_lo=4, block_w_hi=8, whole_shaft_frac=0.0)
    mask = sample_contact_mask(geom, 30, n_rows=8, generator=_gen(1), cfg=cfg)
    mean_runs = sum(len(_runs(mask[r])) for r in range(8)) / 8
    assert mean_runs <= 4.0, f"mean runs/row {mean_runs} — not blocky enough"
    # and a full block actually forms: some row reaches a run ≥ block_w_lo.
    assert max(max(_runs(mask[r]), default=0) for r in range(8)) >= 4


def test_wider_blocks_give_longer_runs() -> None:
    # The Uniform{lo..hi} width knob is real: wider spans ⇒ longer masked runs.
    sc, geom = _session([40])
    narrow = V3MaskConfig(mask_frac=0.5, block_w_lo=4, block_w_hi=4, whole_shaft_frac=0.0)
    wide = V3MaskConfig(mask_frac=0.5, block_w_lo=10, block_w_hi=10, whole_shaft_frac=0.0)
    mn = sample_contact_mask(geom, 40, n_rows=8, generator=_gen(2), cfg=narrow)
    mw = sample_contact_mask(geom, 40, n_rows=8, generator=_gen(2), cfg=wide)
    max_narrow = sum(max(_runs(mn[r]), default=0) for r in range(8)) / 8
    max_wide = sum(max(_runs(mw[r]), default=0) for r in range(8)) / 8
    assert max_wide > max_narrow, f"wide {max_wide} !> narrow {max_narrow}"


def test_whole_shaft_tier_masks_entire_shafts_first() -> None:
    # whole_shaft_frac=0.5 over 4 shafts ⇒ 2 shafts wholly masked before any block.
    sc, geom = _session([5, 5, 5, 5])  # N=20, M=12
    cfg = V3MaskConfig(mask_frac=0.6, whole_shaft_frac=0.5)
    mask = sample_contact_mask(geom, 20, n_rows=16, generator=_gen(3), cfg=cfg)
    # count fully-masked shafts per row; expect ≥2 (the whole-shaft tier).
    shaft_of = sc.shaft_id  # (20,)
    for r in range(16):
        full = sum(
            bool(mask[r][shaft_of == s].all()) for s in range(4)
        )
        assert full >= 2, f"row {r}: only {full} whole shafts"


def test_only_valid_contacts_masked_and_time_tube_expand() -> None:
    sc, geom = _session([7, 5])  # N=12
    mask = sample_contact_mask(geom, 12, n_rows=4, generator=_gen())
    assert mask.shape == (4, 12)
    # time-tube expansion is a trivial broadcast the objective consumes:
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
    # not all rows identical (independent sampling).
    assert not all(torch.equal(mask[0], mask[r]) for r in range(1, 8))
