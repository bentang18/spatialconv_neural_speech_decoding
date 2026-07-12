"""v14_converged_v3 — packed (varlen) L1/L2 == padded oracle (#24, dual-axis) TDD.

The packed path is the production path; the padded ``_attn`` in ``attention.py`` is
the CPU-testable ORACLE. These tests pin the numerical contract: for every VISIBLE
CELL present in BOTH runs, ``forward_packed`` (reference backend) reproduces the padded
``forward`` exactly. flash validates against the same oracle on GPU at F2.

Dual-axis (Ben 2026-07-12): masking drops CELLS on two axes — SPACE (whole contacts,
per-contact ``visible`` (B,N)) and TIME (frames per shaft, ``frame_keep`` (B,S,T),
homogeneous within a shaft). The composed cell visibility is
``cell_vis = visible[:,:,None] & frame_keep[:, shaft_of, :]`` (B,N,T). The L1 block
attends within a shaft over its ``C_kept × T_kept`` rectangle; the L2 block attends
cross-sensor at the SAME real frame, regrouping the heterogeneously time-masked cells.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.attention import L1Block, L2Block
from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.packing import (
    build_pack_plan,
    gather_cells,
    scatter_cells,
)
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

D, H, T = 32, 4, 5


def _sc(labels, parcels):
    return build_sidecar(labels, parcel_id=torch.tensor(parcels, dtype=torch.long))


def _two_shaft():
    sc = _sc(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    return sc, build_l1_geometry(sc)


def _cell_vis(visible, frame_keep, geom):
    # (B,N) spatial ∧ (B,S,T) per-shaft temporal → (B,N,T) cell visibility.
    return visible[:, :, None] & frame_keep[:, geom.shaft_of_contact, :]


def _packed_full(blk, x, plan, N):
    xg = gather_cells(x, plan.order, plan.time_idx)  # (B, P, T_kept, d)
    out = blk.forward_packed(xg, plan, backend="reference")
    return scatter_cells(out, plan.order, plan.time_idx, N, T)  # (B, N, T, d)


def _assert_cell_match(full_packed, padded, cell_vis):
    v = cell_vis[:, :, :, None].expand_as(full_packed)  # (B,N,T,d) from (B,N,T)
    assert torch.allclose(full_packed[v], padded[v], atol=1e-5)


# ── L1 ───────────────────────────────────────────────────────────────────────


def test_l1_packed_unmasked_matches_padded_all_contacts() -> None:
    _, geom = _two_shaft()
    N = 5
    blk = L1Block(D, H).eval()
    x = torch.randn(2, N, T, D)
    padded = blk(x, geom)  # visible=None → all cells
    plan = build_pack_plan(geom, n_time=T, batch=2, n_selected=N, visible=None)
    full = _packed_full(blk, x, plan, N)
    assert torch.allclose(full, padded, atol=1e-5)  # every cell present in both


def test_l1_packed_masked_matches_padded_on_visible() -> None:
    # Dual-axis: partial SPACE masks per shaft (M_vis=3) + per-shaft TIME masks
    # (drop 1 frame/shaft → T_kept=4). L1 must match on every visible cell.
    _, geom = _two_shaft()
    N = 5
    blk = L1Block(D, H).eval()
    x = torch.randn(2, N, T, D)
    visible = torch.tensor([
        [True, False, True, False, True],   # LA:0,2  LB:4
        [False, True, True, True, False],   # LA:1,2  LB:3
    ])
    # per-shaft kept frames (True=kept); each shaft keeps exactly 4 → T_kept=4.
    frame_keep = torch.tensor([
        [[False, True, True, True, True], [True, True, True, True, False]],
        [[True, True, False, True, True], [True, False, True, True, True]],
    ])  # (B=2, S=2, T=5)
    cell_vis = _cell_vis(visible, frame_keep, geom)
    padded = blk(x, geom, cell_vis)
    plan = build_pack_plan(
        geom, n_time=T, batch=2, n_selected=3, visible=visible, frame_keep=frame_keep
    )
    full = _packed_full(blk, x, plan, N)
    _assert_cell_match(full, padded, cell_vis)


def test_l1_packed_whole_shaft_masked_matches_visible_shaft() -> None:
    # Whole shaft B masked (contacts 3,4 gone) → LB is a 0-length block; LA's visible
    # cells must still match. Time masking active on the surviving shaft.
    _, geom = _two_shaft()
    N = 5
    blk = L1Block(D, H).eval()
    x = torch.randn(1, N, T, D)
    visible = torch.tensor([[True, True, True, False, False]])
    # every shaft (even spatially-dead B) drops exactly 1 frame ⇒ T_kept=4 uniform,
    # mirroring production's exact-T_mask-per-shaft invariant (t_kept is a constant).
    frame_keep = torch.tensor([[[True, False, True, True, True],
                               [True, True, False, True, True]]])
    cell_vis = _cell_vis(visible, frame_keep, geom)
    padded = blk(x, geom, cell_vis)
    plan = build_pack_plan(
        geom, n_time=T, batch=1, n_selected=3, visible=visible, frame_keep=frame_keep
    )
    full = _packed_full(blk, x, plan, N)
    _assert_cell_match(full, padded, cell_vis)


def test_l1_packed_precomputed_rope_cs_matches_recompute() -> None:
    # B5 hoist: feeding forward_packed a precomputed cos/sin table must be
    # BIT-IDENTICAL to letting the block recompute it (rope_cs=None). Uses the plan's
    # REAL frame indices (plan.time_idx) as the RoPE time coordinate, not arange.
    _, geom = _two_shaft()
    N = 5
    blk = L1Block(D, H).eval()
    x = torch.randn(2, N, T, D)
    plan = build_pack_plan(geom, n_time=T, batch=2, n_selected=N, visible=None)
    xg = gather_cells(x, plan.order, plan.time_idx)
    B, P = xg.shape[0], xg.shape[1]
    total = B * P * plan.t_kept
    idx = plan.depth[:, :, None].expand(B, P, plan.t_kept).reshape(total)
    tt = plan.time_idx.reshape(total)
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
    full = _packed_full(blk, x, plan, N)
    assert torch.allclose(full, padded, atol=1e-5)


def test_l2_packed_space_only_matches_padded_on_visible() -> None:
    # SPACE-only masking (frame_keep=None ⇒ perm_l2 None ⇒ dense full-slot L2). Padded
    # L2 with a contact-uniform cell mask vs packed L2 over only-visible contacts.
    _, geom = _two_shaft()
    N = 5
    blk = L2Block(D, H).eval()
    x = torch.randn(2, N, T, D)
    visible = torch.tensor([
        [True, False, True, False, True],
        [False, True, True, True, False],
    ])
    cell_vis = visible[:, :, None].expand(2, N, T)  # contact-uniform over T
    padded = blk(x, cell_vis)
    plan = build_pack_plan(geom, n_time=T, batch=2, n_selected=3, visible=visible)
    full = _packed_full(blk, x, plan, N)
    _assert_cell_match(full, padded, cell_vis)


def test_l2_packed_time_masked_regroup_matches_padded_on_visible() -> None:
    # TIME masking makes slot-index ≠ real-frame per shaft ⇒ the L2 regroup path
    # (perm_l2 set). Keep ALL contacts, drop a different frame per shaft so at frame 1
    # only shaft B is visible and at frame 3 only shaft A — the heterogeneity that
    # exercises the real-frame regroup. Every real frame keeps ≥1 shaft (the guarantee).
    _, geom = _two_shaft()
    N = 5
    blk = L2Block(D, H).eval()
    x = torch.randn(2, N, T, D)
    visible = torch.ones(2, N, dtype=torch.bool)
    frame_keep = torch.tensor([
        [[True, False, True, True, True], [True, True, True, False, True]],
        [[True, True, True, False, True], [True, False, True, True, True]],
    ])  # (B,S,T) each shaft keeps 4 → T_kept=4
    cell_vis = _cell_vis(visible, frame_keep, geom)
    padded = blk(x, cell_vis)
    plan = build_pack_plan(
        geom, n_time=T, batch=2, n_selected=N, visible=visible, frame_keep=frame_keep
    )
    assert plan.perm_l2 is not None  # regroup engaged
    full = _packed_full(blk, x, plan, N)
    _assert_cell_match(full, padded, cell_vis)
