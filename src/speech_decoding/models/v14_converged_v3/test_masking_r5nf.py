"""v3r5nf (no-fusion) masking — ``sample_masks_r5nf``.

Two INDEPENDENT r5-style masks, one per separated stream (HGA, LFS): the SAME per-shaft
balanced SPACE + per-shaft temporal TIME tier as ``sample_masks_r5``, drawn SEPARATELY ⇒
HGA(e,t) can be masked while LFS(e,t) is visible. Invariants named + asserted + printed.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import (
    V3MaskConfig,
    V3MasksR5NF,
    sample_masks_r5nf,
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
    sc, geom = _session([3, 3, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r5nf(geom, n, n_time=16, n_rows=5, generator=_gen())
    assert isinstance(m, V3MasksR5NF)
    assert m.hga_contact_mask.shape == (5, n)
    assert m.lfs_contact_mask.shape == (5, n)
    assert m.hga_temporal_mask.shape == (5, geom.n_shafts, 16)
    assert m.lfs_temporal_mask.shape == (5, geom.n_shafts, 16)
    print("[check] OK V3MasksR5NF 4 fields with r5-shaped tensors")


def test_each_stream_temporal_count_is_exact_and_frac_controls_it() -> None:
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    t = 20
    for frac in (0.25, 0.5, 0.75):
        cfg = V3MaskConfig(temporal_mask_frac=frac)
        m = sample_masks_r5nf(geom, n, n_time=t, n_rows=16, generator=_gen(1), cfg=cfg)
        want = round(frac * t)
        for name, tm in (("hga", m.hga_temporal_mask), ("lfs", m.lfs_temporal_mask)):
            counts = tm.sum(-1)  # (R, S) per shaft
            assert torch.all(counts == want), (name, frac, counts.unique().tolist())
    print("[check] OK per-stream temporal count == round(frac·T) exactly, both streams")


def test_each_stream_space_balanced_and_keep_alive() -> None:
    sc, geom = _session([4, 4, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r5nf(geom, n, n_time=16, n_rows=64, generator=_gen(3))
    for name, cm in (("hga", m.hga_contact_mask), ("lfs", m.lfs_contact_mask)):
        csum = cm.long().sum(1)  # (R,) total masked per row
        assert torch.all(csum == 6), (name, csum.unique().tolist())  # 3 shafts × round(0.5·4)=2
        assert not cm.all(1).any(), name  # keep-alive: never a fully-masked row
    print("[check] OK per-stream space balanced (6/12), keep-alive holds")


def test_masked_counts_constant_across_rows_static_shapes() -> None:
    # per-shaft balanced ⇒ each stream's masked count is a per-session CONSTANT (static shapes).
    sc, geom = _session([3, 5, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r5nf(geom, n, n_time=24, n_rows=8, generator=_gen(2))
    for cm in (m.hga_contact_mask, m.lfs_contact_mask):
        per_row = cm.long().sum(1)
        assert torch.all(per_row == per_row[0])
    for tm in (m.hga_temporal_mask, m.lfs_temporal_mask):
        per_shaft = tm.long().sum(-1)  # (R, S)
        assert torch.all(per_shaft == per_shaft[0, 0])
    print("[check] OK masked counts row-constant per stream ⇒ static compiled shapes")


def test_hga_and_lfs_masks_are_independent() -> None:
    # THE no-fusion invariant: the two streams are sampled INDEPENDENTLY (sequential draws),
    # so their masked sets differ — HGA(e,t) can be masked while LFS(e,t) is visible.
    sc, geom = _session([4, 4, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r5nf(geom, n, n_time=24, n_rows=8, generator=_gen(5))
    contact_differ = not torch.equal(m.hga_contact_mask, m.lfs_contact_mask)
    temporal_differ = not torch.equal(m.hga_temporal_mask, m.lfs_temporal_mask)
    assert contact_differ and temporal_differ, (contact_differ, temporal_differ)
    print(f"[check] OK HGA vs LFS masks independent: contact_differ={contact_differ} "
          f"temporal_differ={temporal_differ}")


def test_temporal_is_per_shaft_independent() -> None:
    sc, geom = _session([3, 5, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r5nf(geom, n, n_time=24, n_rows=4, generator=_gen(2))
    for tm in (m.hga_temporal_mask, m.lfs_temporal_mask):
        any_differ = any(
            not torch.equal(tm[r, i], tm[r, j])
            for r in range(tm.shape[0]) for i in range(tm.shape[1]) for j in range(i + 1, tm.shape[1])
        )
        assert any_differ, "shafts share a temporal mask — not independent"
    print("[check] OK each stream's temporal mask is (R,S,T) per-shaft independent")


def test_deterministic_in_generator_seed() -> None:
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    a = sample_masks_r5nf(geom, n, n_time=16, n_rows=4, generator=_gen(7))
    b = sample_masks_r5nf(geom, n, n_time=16, n_rows=4, generator=_gen(7))
    c = sample_masks_r5nf(geom, n, n_time=16, n_rows=4, generator=_gen(8))
    same = (
        torch.equal(a.hga_temporal_mask, b.hga_temporal_mask)
        and torch.equal(a.lfs_temporal_mask, b.lfs_temporal_mask)
        and torch.equal(a.hga_contact_mask, b.hga_contact_mask)
        and torch.equal(a.lfs_contact_mask, b.lfs_contact_mask)
    )
    diff = not torch.equal(a.hga_temporal_mask, c.hga_temporal_mask)
    assert same and diff
    print(f"[check] OK same seed → identical ({same}); different seed → differs ({diff})")
