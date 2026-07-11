"""v14_converged_v3 — varlen pack plan for the ragged L1 attention (#24).

The L1 block attends jointly over (contact, time) WITHIN a shaft, block-diagonal
by shaft. The padded ``(n_shafts, max_c)`` gather (``geometry.py``) runs on any
backend but pays the max_c zero-pad on every linear + attention. This module
builds the RAGGED alternative: the selected contacts of every clip laid END TO
END in shaft-grouped order, with a ``cu_seqlens`` index marking each
``(clip, shaft)`` block — the flat layout ``flash_attn_varlen_func`` consumes, and
that the SDPA block-diagonal reference reproduces exactly (dual backend, ``attention``).

Two flavours share one builder:

  * ONLINE encoder — ``visible`` = the per-clip ``~mask`` (B, N). Each clip selects
    its OWN ``M_vis`` visible contacts, so the plan is per-forward and per-clip; a
    whole-masked shaft contributes a ZERO-length segment (kept, not filtered — the
    segment count stays a per-session constant B·S ⇒ compile-safe).
  * TEACHER / PREDICTOR — ``visible=None`` ⇒ all valid contacts, shared across the
    batch and static per session (build once, cache).

HARD CONSTRAINT (memo project-v3-pipeline-build-contract, Ben's RoPE catch): the
plan carries ``depth`` = the RAW clinical contact index (drop/mask gaps preserved),
NEVER a condensed 0..P. Only ``|depth_i − depth_j|`` within a shaft matters and the
gap always counts. ``order`` maps each packed slot back to its contact in N, so the
predictor re-inserts every mask-query at its TRUE depth.

Compile-safety: the selection is a FIXED-shape sort (``n_selected`` is a per-session
constant — ``M_vis`` masked / ``N`` unmasked — NOT a data-dependent ``.nonzero()``),
and ``cu_seqlens`` is a constant-length (B·S+1) tensor whose VALUES vary per step.
So the plan recompiles never; only its contents move.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from speech_decoding.models.v14_converged_v3.geometry import L1Geometry


@dataclass(frozen=True)
class PackPlan:
    """Ragged per-clip layout for one contact selection (visible or all-valid)."""

    order: Tensor  # (B, P) long — selected contacts per clip, shaft-grouped, into N
    depth: Tensor  # (B, P) long — clinical depth per selected contact (gaps kept)
    cu_seqlens: Tensor  # (B*S+1,) int32 — per (clip, shaft) block bound, TOKEN units
    cu_seqlens_drop: Tensor  # (n_real+1,) int64 — cu_seqlens with the zero-length (whole-
    # masked) shafts removed, for the GPU njt path. Compacting the offsets is a data-
    # dependent masked_select (the surviving-block count varies per step) ⇒ ONE CPU sync;
    # done HERE, once per plan, so the ~9-block encoder / ~8-block predictor towers reuse
    # it sync-free instead of each L1 block re-dropping (≈12 hidden syncs/call otherwise).
    max_seqlen: int  # max block length in tokens (static upper bound = max_c * T)
    n_selected: int  # P — selected contacts per clip (constant per session)
    n_tokens: int  # B * P * T — total packed tokens (constant per session)
    grid_pos: Tensor  # (B, P) long — flat (shaft*max_c + slot_in_shaft) grid cell per
    # packed slot. The SDPA-per-block attention (varlen.sdpa_block_diag_packed) scatters
    # the packed q/k/v into the padded (B, S*max_c, T) grid at these cells, runs one dense
    # cuDNN SDPA per (clip, shaft) block (masked pad keys), and gathers back — same grid
    # + key set as the padded L1Block._attn oracle, so per-contact-identical, but with the
    # linears still on the packed P (keeps #24's visible-gather saving).
    n_shafts: int  # S — shaft count (grid row count); grid width = max_c = max_seqlen // T


def build_pack_plan(
    geom: L1Geometry,
    *,
    n_time: int,
    batch: int,
    n_selected: int,
    visible: Tensor | None = None,
) -> PackPlan:
    """Build the varlen pack plan for ``batch`` clips over ``n_time`` slots.

    ``visible`` (B, N) bool selects the ONLINE encoder's per-clip visible contacts;
    ``None`` selects every valid contact (teacher/predictor, shared across the batch).
    ``n_selected`` = P is the per-clip selected count (a per-session CONSTANT the
    caller supplies — ``M_vis`` masked or ``N`` unmasked — so the selection stays
    fixed-shape and never syncs on ``visible.sum()``).

    Shaft-grouped order comes for free: contacts are gathered in ``(shaft, slot)``
    order via ``geom.gather_idx[geom.valid]``, and the per-clip selection preserves
    that order, so each clip's packed tokens are already contiguous by shaft.
    """
    device = geom.gather_idx.device
    S = geom.n_shafts
    T = n_time
    B = batch
    P = n_selected

    # canonical (shaft-grouped) order of the N valid contacts and their per-contact
    # shaft id + clinical depth, all aligned to that order.
    canon = geom.gather_idx[geom.valid]  # (N,) long — contacts in (shaft, slot) order
    depth_canon = geom.depth[geom.valid]  # (N,) long — clinical depth, gaps preserved
    N = int(canon.shape[0])
    shaft_of_contact = geom.shaft_of_contact  # (N,) long — shaft id per contact-in-N

    if visible is None:
        vis_canon = torch.ones((B, N), dtype=torch.bool, device=device)
        vis_full = torch.ones((B, shaft_of_contact.shape[0]), dtype=torch.bool, device=device)
    else:
        vis_canon = visible[:, canon]  # (B, N) bool in canonical order
        vis_full = visible  # (B, N) bool in contact order (for the per-shaft count)

    # FIXED-shape selection: key = canonical position where visible, else N (a sentinel
    # that sorts last). The first P sorted keys are exactly the P visible positions in
    # ascending (shaft-grouped) order. No data-dependent .nonzero() ⇒ compile-safe.
    ar = torch.arange(N, device=device)
    key = torch.where(vis_canon, ar[None, :].expand(B, N), N)
    pos = key.sort(dim=1).values[:, :P]  # (B, P) canonical positions of the P selected
    order = canon[pos]  # (B, P) long — selected contacts in N, shaft-grouped
    depth = depth_canon[pos]  # (B, P) long — clinical depth per selected slot

    # flat padded-grid cell (shaft*max_c + slot_in_shaft) per selected slot, for the
    # SDPA-per-block scatter. canon iterates ``geom.valid`` in row-major (shaft, slot)
    # order, so the j-th valid cell's flat grid index = grid_flat[valid][j]; index that
    # by ``pos`` to get the cell of each packed slot. Visible-but-holey shafts keep their
    # ORIGINAL slot positions (holes stay empty, masked out) — matching the padded oracle.
    grid_flat = torch.arange(S * geom.max_c, device=device)  # (S*max_c,)
    canon_grid = grid_flat[geom.valid.reshape(-1)]  # (N,) grid cell per canon position
    grid_pos = canon_grid[pos]  # (B, P) long — grid cell per packed slot

    # per-(clip, shaft) visible count → segment lengths in TOKEN units (Vs * T). Kept
    # for ALL S shafts (a whole-masked shaft ⇒ Vs=0 ⇒ zero-length block, not filtered)
    # so cu_seqlens has the per-session-constant length B*S+1.
    onehot = F.one_hot(shaft_of_contact, S).to(torch.float32)  # (N, S)
    v_per = (vis_full.to(torch.float32) @ onehot).round().to(torch.long)  # (B, S)
    seg_tok = (v_per * T).reshape(B * S)  # (B*S,) tokens per (clip, shaft) block
    cu = torch.zeros(B * S + 1, dtype=torch.int32, device=device)
    cu[1:] = seg_tok.cumsum(0).to(torch.int32)

    # Precompute the compacted offsets ONCE (drop zero-length whole-masked shafts): the
    # njt path can't feed empty jagged rows to the varlen backward (NaN grad), and the
    # boolean compaction is a single data-dependent CPU sync — paid here per plan, not
    # per L1 block. int64 for nested_tensor_from_jagged.
    off = cu.to(torch.int64)
    keep = off[1:] != off[:-1]
    cu_drop = torch.cat([off[:1], off[1:][keep]])

    return PackPlan(
        order=order,
        depth=depth,
        cu_seqlens=cu,
        cu_seqlens_drop=cu_drop,
        max_seqlen=geom.max_c * T,
        n_selected=P,
        n_tokens=B * P * T,
        grid_pos=grid_pos,
        n_shafts=S,
    )


def gather_tokens(x_full: Tensor, order: Tensor) -> Tensor:
    """(B, N, T, d) full grid → (B, P, T, d) packed, selecting ``order`` per clip."""
    _, _, t, d = x_full.shape
    return x_full.gather(1, order[:, :, None, None].expand(-1, -1, t, d))


def scatter_tokens(x_packed: Tensor, order: Tensor, n_full: int) -> Tensor:
    """(B, P, T, d) packed → (B, n_full, T, d) full grid; unselected rows stay 0.

    Inverse of :func:`gather_tokens` — writes each packed slot back to its true
    contact row (``order``). The predictor uses this to lift encoder-visible latents
    into the all-N buffer before overwriting masked rows with the mask query.
    """
    b, _, t, d = x_packed.shape
    out = x_packed.new_zeros(b, n_full, t, d)
    return out.scatter_(1, order[:, :, None, None].expand(-1, -1, t, d), x_packed)
