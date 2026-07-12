"""v14_converged_v3 — varlen pack plan for the ragged L1/L2 attention (#24, dual-axis).

The L1 block attends jointly over (contact, time) WITHIN a shaft, block-diagonal by
shaft. The L2 block attends cross-sensor at the SAME real frame t. The padded
``(n_shafts, max_c)`` gather (``geometry.py``) runs on any backend but pays the max_c
zero-pad. This module builds the RAGGED alternative: the selected VISIBLE CELLS of
every clip laid END TO END, with ``cu_seqlens`` marking each block — the flat layout
``flash_attn_varlen_func`` / ``nested_tensor_from_jagged`` consumes, and that the SDPA
block-diagonal reference reproduces exactly (dual backend, ``attention``).

DUAL-AXIS (masking-axis inversion, Ben 2026-07-12). Masking now drops CELLS on two
axes — SPACE (whole contacts) and TIME (frames per shaft) — so a surviving shaft is a
``C_kept × T_kept`` rectangle (homogeneous-intra ⇒ every kept contact of a shaft shares
the same kept-frame set). The plan therefore compacts BOTH axes:

  * ``order`` (B, P)         — the P = N−D spatially-visible contacts, shaft-grouped.
  * ``time_idx`` (B, P, T_kept) — the REAL kept-frame indices per packed contact (its
    shaft's kept frames). Used as the L1 RoPE time coordinate (true frame geometry,
    NOT arange) and to gather/scatter the compacted cell buffer.

Two axes ⇒ two block structures over the SAME visible-cell token set:

  * L1 (within-shaft): tokens grouped by shaft → block = ``C_kept·T_kept`` tokens.
    ``cu_seqlens`` / ``cu_seqlens_drop`` as before but on ``T_kept``.
  * L2 (cross-sensor same-t): tokens REGROUPED by real frame t → block = the contacts
    visible at t. Heterogeneous time masking means the visible-contact set differs per
    t, so L2 needs a genuine regroup: ``perm_l2`` reorders the flat visible-cell tokens
    into time-major order and ``cu_l2`` marks each (clip, frame) block. The ≥1-live-
    sensor guarantee (masking.py) keeps every frame block non-empty. Only the ONLINE
    encoder (compacted, heterogeneous) needs this; the teacher/predictor run the full
    grid (``frame_keep=None`` ⇒ ``T_kept=T``, slot-index == real-t, no regroup).

HARD CONSTRAINT (Ben's RoPE catch): the plan carries ``depth`` = the RAW clinical
contact index and ``time_idx`` = the RAW frame index (drop/mask gaps preserved), NEVER
a condensed 0..P / 0..T_kept — only the true separations matter for RoPE.

Compile-safety: selections are FIXED-shape sorts (``n_selected`` = P and ``T_kept`` are
per-session constants — never a data-dependent ``.nonzero()``); ``cu_*`` are
constant-length tensors whose VALUES vary per step. The plan recompiles never.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from speech_decoding.models.v14_converged_v3.geometry import L1Geometry


@dataclass(frozen=True)
class PackPlan:
    """Ragged per-clip layout for one visible-cell selection (online or full-grid)."""

    order: Tensor  # (B, P) long — selected contacts per clip, shaft-grouped, into N
    depth: Tensor  # (B, P) long — clinical depth per selected contact (gaps kept)
    time_idx: Tensor  # (B, P, T_kept) long — REAL kept-frame index per (slot, kept-frame).
    # Full grid ⇒ arange(T) broadcast; online ⇒ the contact's shaft's kept frames.
    cu_seqlens: Tensor  # (B*S+1,) int32 — per (clip, shaft) L1 block bound, TOKEN units
    cu_seqlens_drop: Tensor  # (n_real+1,) int64 — cu_seqlens minus zero-length (whole-
    # masked) shafts, for the GPU njt L1 path. One CPU sync, done once per plan.
    max_seqlen: int  # max L1 block length in tokens (static bound = max_c * T_kept)
    n_selected: int  # P — selected contacts per clip (constant per session)
    t_kept: int  # T_kept — kept frames per shaft (constant per session)
    n_tokens: int  # B * P * T_kept — total packed visible-cell tokens (constant)
    grid_pos: Tensor  # (B, P) long — flat (shaft*max_c + slot_in_shaft) grid cell per
    # packed slot, for the SDPA-per-block L1 oracle scatter (varlen.sdpa_block_diag_packed).
    n_shafts: int  # S — shaft count (grid row count); grid width = max_c = max_seqlen // T_kept
    # ── L2 real-frame regroup (ONLINE encoder only; None on the full grid) ──────
    perm_l2: Tensor | None  # (B, P*T_kept) long — flat visible-cell → time-major position.
    cu_l2: Tensor | None  # (B*T+1,) int32 — per (clip, real-frame) L2 block bound, tokens.
    cu_l2_drop: Tensor | None  # (n_real+1,) int64 — cu_l2 minus empty frames, for njt L2.
    max_seqlen_l2: int  # max L2 block length (static bound = N contacts at one frame)


def _drop_offsets(cu: Tensor) -> Tensor:
    """Compact a cu_seqlens (drop zero-length segments) → int64 for njt. ONE CPU sync,
    paid once per plan so the towers reuse it sync-free across blocks."""
    off = cu.to(torch.int64)
    keep = off[1:] != off[:-1]
    return torch.cat([off[:1], off[1:][keep]])


def build_pack_plan(
    geom: L1Geometry,
    *,
    n_time: int,
    batch: int,
    n_selected: int,
    visible: Tensor | None = None,
    frame_keep: Tensor | None = None,
) -> PackPlan:
    """Build the varlen pack plan for ``batch`` clips.

    ``visible`` (B, N) bool selects the ONLINE encoder's per-clip spatially-visible
    contacts; ``None`` selects every valid contact (teacher/predictor, batch-shared).
    ``n_selected`` = P is the per-clip selected count (a per-session CONSTANT).

    ``frame_keep`` (B, S, T) bool = the per-shaft kept-frame set (True = kept). When
    given (online encoder), the plan compacts the time axis to ``T_kept`` and builds
    the L2 real-frame regroup. ``None`` (teacher/predictor) ⇒ full ``T``, no regroup.
    """
    device = geom.gather_idx.device
    S = geom.n_shafts
    T = n_time
    B = batch
    P = n_selected

    canon = geom.gather_idx[geom.valid]  # (N,) long — contacts in (shaft, slot) order
    depth_canon = geom.depth[geom.valid]  # (N,) long — clinical depth, gaps preserved
    N = int(canon.shape[0])
    shaft_of_contact = geom.shaft_of_contact  # (N,) long — shaft id per contact-in-N

    if visible is None:
        vis_canon = torch.ones((B, N), dtype=torch.bool, device=device)
        vis_full = torch.ones((B, N), dtype=torch.bool, device=device)
    else:
        vis_canon = visible[:, canon]  # (B, N) bool in canonical order
        vis_full = visible  # (B, N) bool in contact order (per-shaft count)

    # FIXED-shape contact selection (compile-safe, no .nonzero()).
    ar = torch.arange(N, device=device)
    key = torch.where(vis_canon, ar[None, :].expand(B, N), N)
    pos = key.sort(dim=1).values[:, :P]  # (B, P) canonical positions of the P selected
    order = canon[pos]  # (B, P) long — selected contacts in N, shaft-grouped
    depth = depth_canon[pos]  # (B, P) long — clinical depth per selected slot

    grid_flat = torch.arange(S * geom.max_c, device=device)
    canon_grid = grid_flat[geom.valid.reshape(-1)]  # (N,) grid cell per canon position
    grid_pos = canon_grid[pos]  # (B, P) long — grid cell per packed slot

    # ── time compaction: kept frame indices per packed contact ────────────────
    if frame_keep is None:
        t_kept = T
        time_idx = torch.arange(T, device=device)[None, None, :].expand(B, P, T)
    else:
        # T_kept is a per-session constant (= T − T_mask); every shaft keeps the same
        # count by construction (masking snaps each shaft to exactly T_mask masked).
        t_kept = int(frame_keep[0].sum(-1).max().item())
        ar_t = torch.arange(T, device=device)
        keyt = torch.where(frame_keep, ar_t[None, None, :].expand(B, S, T), T)
        tidx_shaft = keyt.sort(dim=2).values[:, :, :t_kept]  # (B, S, T_kept) real kept frames
        shaft_of_order = shaft_of_contact[order]  # (B, P)
        time_idx = tidx_shaft.gather(
            1, shaft_of_order[:, :, None].expand(B, P, t_kept)
        )  # (B, P, T_kept) real kept frames per packed contact

    # ── L1 blocks (per shaft): seg = v_per * T_kept tokens ────────────────────
    onehot = F.one_hot(shaft_of_contact, S).to(torch.float32)  # (N, S)
    v_per = (vis_full.to(torch.float32) @ onehot).round().to(torch.long)  # (B, S)
    seg_tok = (v_per * t_kept).reshape(B * S)  # (B*S,) tokens per (clip, shaft)
    cu = torch.zeros(B * S + 1, dtype=torch.int32, device=device)
    cu[1:] = seg_tok.cumsum(0).to(torch.int32)
    cu_drop = _drop_offsets(cu)

    # ── L2 blocks (per real frame): regroup, ONLINE only ──────────────────────
    if frame_keep is None:
        perm_l2 = None
        cu_l2 = None
        cu_l2_drop = None
        max_seqlen_l2 = N
    else:
        # each visible-cell token (slot p, kept-frame k) has real frame time_idx[b,p,k].
        # Sort tokens within a clip by real frame → time-major; count per frame → cu_l2.
        rf = time_idx.reshape(B, P * t_kept)  # (B, P*T_kept) real frame per token
        perm_l2 = rf.argsort(dim=1, stable=True)  # (B, P*T_kept) flat → time-major
        cnt = F.one_hot(rf, T).sum(dim=1).to(torch.int32)  # (B, T) tokens per (clip, frame)
        seg_l2 = cnt.reshape(B * T)  # (B*T,)
        cu_l2 = torch.zeros(B * T + 1, dtype=torch.int32, device=device)
        cu_l2[1:] = seg_l2.cumsum(0).to(torch.int32)
        cu_l2_drop = _drop_offsets(cu_l2)
        max_seqlen_l2 = int(cnt.max().item()) if cnt.numel() else N

    return PackPlan(
        order=order,
        depth=depth,
        time_idx=time_idx,
        cu_seqlens=cu,
        cu_seqlens_drop=cu_drop,
        max_seqlen=geom.max_c * t_kept,
        n_selected=P,
        t_kept=t_kept,
        n_tokens=B * P * t_kept,
        grid_pos=grid_pos,
        n_shafts=S,
        perm_l2=perm_l2,
        cu_l2=cu_l2,
        cu_l2_drop=cu_l2_drop,
        max_seqlen_l2=max_seqlen_l2,
    )


def gather_cells(x_full: Tensor, order: Tensor, time_idx: Tensor) -> Tensor:
    """(B, N, T, d) full grid → (B, P, T_kept, d) compacted visible cells.

    Gather the P selected contacts (``order``), then per contact its shaft's kept
    frames (``time_idx``). Full-grid callers pass ``time_idx = arange(T)`` (a no-op on
    the time axis) ⇒ this reduces to the contact-only gather.
    """
    b, _, t, d = x_full.shape
    p, tk = order.shape[1], time_idx.shape[2]
    xc = x_full.gather(1, order[:, :, None, None].expand(b, p, t, d))  # (B, P, T, d)
    return xc.gather(2, time_idx[:, :, :, None].expand(b, p, tk, d))  # (B, P, T_kept, d)


def scatter_cells(
    x_packed: Tensor, order: Tensor, time_idx: Tensor, n_full: int, t_full: int
) -> Tensor:
    """(B, P, T_kept, d) compacted → (B, n_full, t_full, d) full grid; unwritten
    (masked) cells stay 0. Inverse of :func:`gather_cells`."""
    b, p, tk, d = x_packed.shape
    out = x_packed.new_zeros(b, n_full, t_full, d)
    # scatter along the full T axis within the selected contacts, then place contacts.
    contact_buf = x_packed.new_zeros(b, p, t_full, d)
    contact_buf.scatter_(2, time_idx[:, :, :, None].expand(b, p, tk, d), x_packed)
    return out.scatter_(1, order[:, :, None, None].expand(b, p, t_full, d), contact_buf)


def gather_tokens(x_full: Tensor, order: Tensor) -> Tensor:
    """(B, N, T, d) full grid → (B, P, T, d) packed, selecting ``order`` per clip
    (contact-level, full T). Used by the full-grid teacher/predictor path."""
    _, _, t, d = x_full.shape
    return x_full.gather(1, order[:, :, None, None].expand(-1, -1, t, d))


def scatter_tokens(x_packed: Tensor, order: Tensor, n_full: int) -> Tensor:
    """(B, P, T, d) packed → (B, n_full, T, d) full grid; unselected rows stay 0
    (contact-level, full T). Inverse of :func:`gather_tokens`."""
    b, _, t, d = x_packed.shape
    out = x_packed.new_zeros(b, n_full, t, d)
    return out.scatter_(1, order[:, :, None, None].expand(-1, -1, t, d), x_packed)
