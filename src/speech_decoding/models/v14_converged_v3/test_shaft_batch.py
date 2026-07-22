"""v14_converged_v3 — shaft-level batching data-layer tests (pool, temperature, collate).

Covers the three rule-INDEPENDENT pieces (identical for bucketing or packing):
  * pool enumeration groups shafts by subject across that subject's sessions;
  * the temperature sampler realises P(subject) ∝ n_shafts^α at the α=0/0.5/1 checkpoints;
  * the super-montage collate unions cross-patient shafts into a B=1 ``V3Batch`` whose flat grid
    total is EXACTLY ΣN·k_full (the fixed-shape mechanic), with the packing pad hitting an exact
    target — and two patients' identically named ("LA") shafts stay DISTINCT blocks.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.pack_r4 import band_token_counts, build_r4_grid
from speech_decoding.models.v14_converged_v3.shaft_batch import (
    PAD_PARCEL_ID,
    ShaftClipSample,
    TemperatureShaftSampler,
    build_shaft_pool,
    collate_shaft_pack,
)
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

T = 16  # SLOW 2 + MID 8 + HGA 16 = k_full 26
_F = (7, 6, 7)


def _stub_session(session_key, shaft_sizes):
    """A minimal V3SessionSpec stand-in: only ``.setup.geom`` and ``.session_key`` are read."""
    labels, parcels = [], []
    for sh, nc in enumerate(shaft_sizes):
        for c in range(1, nc + 1):
            labels.append(f"S{sh}x{c}")
            parcels.append(0)
    geom = build_l1_geometry(build_sidecar(labels, parcel_id=torch.tensor(parcels)))
    return SimpleNamespace(setup=SimpleNamespace(geom=geom), session_key=session_key)


def _clip(depth, parcels, key, shaft_id):
    d = torch.tensor(depth)
    n = len(depth)
    bands = tuple(torch.randn(n, _F[b], T) for b in range(3))
    return ShaftClipSample(bands=bands, depth=d, parcel_id=torch.tensor(parcels),
                           session_key=key, shaft_id=shaft_id)


# ── pool ────────────────────────────────────────────────────────────────────────
def test_pool_groups_shafts_by_subject_across_sessions() -> None:
    # subject A: one session, 2 shafts. subject B: TWO sessions, 4 + 4 = 8 shafts.
    sessions = [
        _stub_session(("A", 0), [3, 5]),
        _stub_session(("B", 0), [6, 7, 8, 9]),
        _stub_session(("B", 1), [10, 11, 12, 13]),
    ]
    pool = build_shaft_pool(sessions)
    assert len(pool.refs) == 2 + 4 + 4
    assert pool.subjects == ("A", "B")
    assert pool.subject_n_shafts.tolist() == [2, 8]  # B's shafts pooled across its 2 sessions
    # every ref carries the right valid-contact count and subject.
    a_refs = [pool.refs[i] for i in pool.subject_ref_idxs[0]]
    assert sorted(r.n_contacts for r in a_refs) == [3, 5]
    assert all(r.subject == "A" for r in a_refs)
    print(f"[check] pool: {len(pool.refs)} shafts, subjects {pool.subjects}, "
          f"n_shafts {pool.subject_n_shafts.tolist()} OK")


def test_pool_skips_fully_dropped_shafts() -> None:
    # a geom with a 0-valid shaft: build one then blank a shaft's valid row.
    s = _stub_session(("A", 0), [3, 4])
    g = s.setup.geom
    g.valid[1] = False  # shaft 1 fully dropped
    pool = build_shaft_pool([s])
    assert len(pool.refs) == 1 and pool.refs[0].n_contacts == 3
    print("[check] pool: fully-dropped shaft skipped OK")


# ── temperature ─────────────────────────────────────────────────────────────────
def _subject_freq(alpha, n=40000, seed=0):
    # A: 2 shafts, B: 8 shafts (10 total). Draw n shafts, return P(subject=A).
    sessions = [_stub_session(("A", 0), [3, 5]),
                _stub_session(("B", 0), [6, 7, 8, 9]),
                _stub_session(("B", 1), [10, 11, 12, 13])]
    pool = build_shaft_pool(sessions)
    smp = TemperatureShaftSampler(pool, alpha=alpha)
    g = torch.Generator().manual_seed(seed)
    a = sum(1 for _ in range(n) if smp.draw(g).subject == "A")
    return a / n


def test_temperature_endpoints_and_middle() -> None:
    # α=0 ⇒ subject-uniform ⇒ P(A)=1/2; α=1 ⇒ shaft-uniform ⇒ P(A)=2/10; α=0.5 ⇒ √-tempered.
    p0, p1, phalf = _subject_freq(0.0), _subject_freq(1.0), _subject_freq(0.5)
    exp_half = (2 ** 0.5) / (2 ** 0.5 + 8 ** 0.5)  # 0.333…
    print(f"[check] temperature P(A): α0={p0:.3f}(→0.500) α1={p1:.3f}(→0.200) "
          f"α0.5={phalf:.3f}(→{exp_half:.3f})")
    assert abs(p0 - 0.5) < 0.02
    assert abs(p1 - 0.2) < 0.02
    assert abs(phalf - exp_half) < 0.02


# ── collate (super-montage) ───────────────────────────────────────────────────────
def test_collate_unions_cross_patient_shafts_into_b1_super_montage() -> None:
    # two clips that would COLLIDE if not relabelled: both originate from a shaft named "LA"
    # (depths reused). The collate must keep them as two DISTINCT blocks.
    c0 = _clip([1, 2, 4], [5, 5, 5], ("PatA", 0), shaft_id=0)          # 3 contacts, parcel 5
    c1 = _clip([1, 2, 3, 5, 6], [12] * 5, ("PatB", 0), shaft_id=0)     # 5 contacts, parcel 12
    batch = collate_shaft_pack([c0, c1])

    assert batch.geom.n_shafts == 2
    assert batch.geom.valid.sum(dim=1).tolist() == [3, 5]              # partition preserved
    assert batch.parcel_id.tolist() == [5, 5, 5, 12, 12, 12, 12, 12]   # concat order
    assert batch.bands[0].shape == (1, 8, _F[0], T)                    # B=1 super-montage
    # depths (index-RoPE coords) survive the relabel, per block.
    assert batch.geom.depth[0, :3].tolist() == [1, 2, 4]
    assert batch.geom.depth[1, :5].tolist() == [1, 2, 3, 5, 6]
    # the flat grid total is EXACTLY ΣN·k_full — the fixed-shape mechanic.
    k_full = sum(band_token_counts(T))
    grid = build_r4_grid(batch.geom, n_time=T)
    assert grid.total == 8 * k_full
    assert not hasattr(batch, "stat_mean") and not hasattr(batch, "stat_std")  # secondary PURGED
    print(f"[check] collate: 2 cross-patient shafts → B=1 super-montage, "
          f"grid.total={grid.total}=8×{k_full}, blocks distinct OK")


def test_collate_pad_to_total_fixes_the_shape_exactly() -> None:
    # packing tail-slack pad: ΣN=8 padded to 10 ⇒ grid.total is EXACTLY the target shape.
    c0 = _clip([1, 2, 4], [5, 5, 5], ("PatA", 0), 0)
    c1 = _clip([1, 2, 3, 5, 6], [12] * 5, ("PatB", 0), 0)
    batch = collate_shaft_pack([c0, c1], pad_to_total=10)
    k_full = sum(band_token_counts(T))
    grid = build_r4_grid(batch.geom, n_time=T)
    assert grid.total == 10 * k_full                       # exact fixed shape
    assert batch.geom.n_shafts == 3                        # one filler shaft appended
    assert batch.geom.valid.sum(dim=1).tolist() == [3, 5, 2]
    assert batch.parcel_id[-2:].tolist() == [PAD_PARCEL_ID, PAD_PARCEL_ID]
    assert batch.session_key == ("shaft_pack", 10)         # shape key for plan-cache
    print(f"[check] collate pad_to_total=10: grid.total={grid.total}=10×{k_full}, "
          f"1 filler shaft (2 contacts) OK")


def test_pack_draw_overfills_and_trims_to_exact_budget() -> None:
    # overfill-and-trim closes the budget EXACTLY (drop, no pad): Σ n_keep == budget, and
    # ONLY the last shaft is trimmed (all earlier shafts keep their full contact count).
    sessions = [_stub_session(("A", 0), [6, 7, 8]), _stub_session(("B", 0), [9, 10, 11])]
    pool = build_shaft_pool(sessions)
    smp = TemperatureShaftSampler(pool, alpha=0.5)
    g = torch.Generator().manual_seed(3)
    pack = smp.draw_pack_to_budget(g, contact_budget=30)
    tot = sum(n_keep for _, n_keep in pack)
    assert tot == 30 and len(pack) >= 1                       # exact close, no pad
    assert all(nk == r.n_contacts for r, nk in pack[:-1])     # earlier shafts kept whole
    assert 1 <= pack[-1][1] <= pack[-1][0].n_contacts         # only the last is trimmed
    print(f"[check] pack draw: {len(pack)} shafts, Σn_keep={tot}==30 (last trimmed) OK")
