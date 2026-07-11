"""v14_converged_v3 — packed (varlen) L1/L2 == padded oracle (#24) TDD.

The packed path is the production path; the padded ``_attn`` in ``attention.py`` is
the CPU-testable ORACLE. These tests pin the numerical contract: for every contact
present in BOTH runs, ``forward_packed`` (reference backend) reproduces the padded
``forward`` exactly. flash validates against the same oracle on GPU at F2.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.attention import L1Block, L2Block
from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.packing import build_pack_plan
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

D, H, T = 32, 4, 5


def _sc(labels, parcels):
    return build_sidecar(labels, parcel_id=torch.tensor(parcels, dtype=torch.long))


def _two_shaft():
    sc = _sc(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    return sc, build_l1_geometry(sc)


def _gather(x_full, plan):
    B, _, t, d = x_full.shape
    idx = plan.order[:, :, None, None].expand(-1, -1, t, d)
    return x_full.gather(1, idx)  # (B, P, T, D)


def _scatter(x_packed, plan, N):
    B, P, t, d = x_packed.shape
    full = x_packed.new_zeros(B, N, t, d)
    idx = plan.order[:, :, None, None].expand(-1, -1, t, d)
    full.scatter_(1, idx, x_packed)
    return full


def _assert_visible_match(full_packed, padded, visible):
    vis = visible[:, :, None, None].expand_as(full_packed)
    assert torch.allclose(full_packed[vis], padded[vis], atol=1e-5)


# ── L1 ───────────────────────────────────────────────────────────────────────


def test_l1_packed_unmasked_matches_padded_all_contacts() -> None:
    sc, geom = _two_shaft()
    N = 5
    blk = L1Block(D, H).eval()
    x = torch.randn(2, N, T, D)
    padded = blk(x, geom)  # (B, N, T, D)
    plan = build_pack_plan(geom, n_time=T, batch=2, n_selected=N, visible=None)
    out = blk.forward_packed(_gather(x, plan), plan, backend="reference")
    full = _scatter(out, plan, N)
    assert torch.allclose(full, padded, atol=1e-5)  # every contact present in both


def test_l1_packed_masked_matches_padded_on_visible() -> None:
    # Partial masks per shaft (each keeps ≥1). M_vis = 3, constant per clip.
    sc, geom = _two_shaft()
    N = 5
    blk = L1Block(D, H).eval()
    x = torch.randn(2, N, T, D)
    visible = torch.tensor([
        [True, False, True, False, True],   # LA:0,2  LB:4
        [False, True, True, True, False],   # LA:1,2  LB:3
    ])
    padded = blk(x, geom, visible)
    plan = build_pack_plan(geom, n_time=T, batch=2, n_selected=3, visible=visible)
    out = blk.forward_packed(_gather(x, plan), plan, backend="reference")
    full = _scatter(out, plan, N)
    _assert_visible_match(full, padded, visible)


def test_l1_packed_whole_shaft_masked_matches_visible_shaft() -> None:
    # Whole shaft B masked (contacts 3,4 gone) → LB is a 0-length block; LA's
    # visible rows must still match the padded run.
    sc, geom = _two_shaft()
    N = 5
    blk = L1Block(D, H).eval()
    x = torch.randn(1, N, T, D)
    visible = torch.tensor([[True, True, True, False, False]])
    padded = blk(x, geom, visible)
    plan = build_pack_plan(geom, n_time=T, batch=1, n_selected=3, visible=visible)
    out = blk.forward_packed(_gather(x, plan), plan, backend="reference")
    full = _scatter(out, plan, N)
    _assert_visible_match(full, padded, visible)


def test_l1_packed_precomputed_rope_cs_matches_recompute() -> None:
    # B5 hoist: feeding forward_packed a precomputed cos/sin table must be
    # BIT-IDENTICAL to letting the block recompute it (rope_cs=None). Proves the
    # tower can build the table once and share it across every L1 block.
    sc, geom = _two_shaft()
    N = 5
    blk = L1Block(D, H).eval()
    x = torch.randn(2, N, T, D)
    plan = build_pack_plan(geom, n_time=T, batch=2, n_selected=N, visible=None)
    xg = _gather(x, plan)
    B, P = xg.shape[0], xg.shape[1]
    total = B * P * T
    idx = plan.depth[:, :, None].expand(B, P, T).reshape(total)
    tt = torch.arange(T)[None, None, :].expand(B, P, T).reshape(total)
    cos, sin = blk.rope.cos_sin(idx, tt)

    recompute = blk.forward_packed(xg, plan, backend="reference")
    hoisted = blk.forward_packed(xg, plan, backend="reference", rope_cs=(cos, sin))
    assert torch.equal(recompute, hoisted)


# ── L2 ───────────────────────────────────────────────────────────────────────


def test_l2_packed_unmasked_matches_padded() -> None:
    _, geom = _two_shaft()
    N = 5
    blk = L2Block(D, H).eval()
    x = torch.randn(2, N, T, D)
    padded = blk(x)  # visible=None
    plan = build_pack_plan(geom, n_time=T, batch=2, n_selected=N, visible=None)
    out = blk.forward_packed(_gather(x, plan))
    full = _scatter(out, plan, N)
    assert torch.allclose(full, padded, atol=1e-5)


def test_l2_packed_masked_matches_padded_on_visible() -> None:
    # Padded L2 with visible key-mask vs packed L2 over only-visible contacts.
    _, geom = _two_shaft()
    N = 5
    blk = L2Block(D, H).eval()
    x = torch.randn(2, N, T, D)
    visible = torch.tensor([
        [True, False, True, False, True],
        [False, True, True, True, False],
    ])
    padded = blk(x, visible)
    plan = build_pack_plan(geom, n_time=T, batch=2, n_selected=3, visible=visible)
    out = blk.forward_packed(_gather(x, plan))
    full = _scatter(out, plan, N)
    _assert_visible_match(full, padded, visible)
