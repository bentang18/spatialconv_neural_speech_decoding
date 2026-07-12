"""v14_converged_v3 — dual-axis (space × time) block masking (Phase 5, rewritten).

Masking-axis inversion (Ben 2026-07-12): the old scheme masked whole CONTACTS
(tube along TIME). The new scheme masks on BOTH axes independently, per sensor:

  SPACE (which contacts, across all time) — the montage / cross-sensor task:
    • WHOLE-SHAFT tier: a STOCHASTIC number of shafts (``K ~ Binomial(S,
      whole_shaft_frac)`` clamped to the montage feasibility ceiling ``K_max``)
      masked 100% — reconstructable only cross-sensor. Kept a PURE removal (the
      ~0.30 along-shaft common mode would leak through one survivor).
    • INTRA tier: every other shaft masked with contiguous width-``w_s`` depth-blocks
      (floor 4 = along-shaft HGA autocorr length), random starts, OVERLAPS ALLOWED
      (union of blocks → varied run/gap structure = regularization, data2vec-2.0
      philosophy). Snap to EXACTLY ``D = round(space_frac·N)`` masked contacts via
      one global priority argsort — whole-shaft contacts held (never trimmed, stay a
      true 100% montage target), intra contacts filled/trimmed by cover-rank.
      NO keep-alive floor: because the whole-shaft count is now stochastic, a shaft
      that saturates to 100% via the intra tier is simply another whole drop, a legal
      outcome — the ≥1-visible rule only existed to protect a PINNED whole count.

  TIME (which frames, per surviving shaft) — the temporal-prediction task:
    • Contiguous width-``w_t`` time-blocks (floor 7 = HGA |STFT| support: a masked
      frame is fully hidden only ≥ support frames from any visible frame), random
      starts, overlaps allowed. Snap PER SHAFT to EXACTLY ``T_mask = round(time_frac·T)``
      masked frames (constant ``T_kept`` ⇒ the buffer stays static). Independent per
      (row, shaft) ⇒ HETEROGENEOUS inter-sensor timing (at any t some sensors visible,
      some masked) — the L2 / montage pressure the homogeneous-intra outer product
      otherwise loses.
    • AT-LEAST-ONE-SENSOR-LIVES (Ben 2026-07-12): a real frame t masked across EVERY
      live shaft would leave a masked cell (c,t) with no same-t visible neighbour for
      the predictor to reconstruct from. Guarantee ≥1 live sensor per t WITHOUT
      breaking the exact-``T_mask`` (static) invariant: assign each t a single
      "guardian" shaft (``V`` = live shafts this row) via a random per-row permutation
      (``perm_rank % V`` — balanced so each live-index guards ``⌊T/V⌋..⌈T/V⌉`` frames,
      but RANDOMLY scattered, NOT the ``t mod V`` diagonal, which was a phase-locked,
      memorizable positional crutch) — and FORBID that shaft from masking t. Guardian
      frames are held visible, the remaining ``T_mask`` masked frames are block-sampled
      as usual. Exactly one guardian per t ⇒ coverage ≥1 everywhere, by construction.
      Feasible for ``V ≥ 2`` (guardian frames ``≈T/V ≤ T_kept``); ``V=1`` (a single
      live shaft — combinatorially impossible at ``space_frac 0.5`` on a real montage)
      would over-constrain and is asserted against, not special-cased.

Compose (in the objective) as a per-sensor OUTER PRODUCT: cell (contact c of shaft s,
frame t) is VISIBLE iff contact c is spatially kept AND frame t is kept for shaft s.
Visible cells per surviving shaft form a clean ``C_kept × T_kept`` rectangle ⇒ the
online encoder's varlen blocks stay rectangular, njt-fast, and STATIC:

  • space snapped to exactly D  ⇒ N−D visible contacts (constant P for the encoder).
  • time snapped to exactly T_mask per shaft ⇒ T_kept visible frames per shaft (constant).
  • ⇒ online tokens = (N−D)·T_kept and loss denominator = N·T − (N−D)·T_kept are
    per-session CONSTANTS (compile once). Only WHICH cells are masked varies per step.

FULLY VECTORIZED (cover-rank + argsort, no python loop over shafts/blocks/rows),
over R independent rows (one per clip in the batch).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from speech_decoding.models.v14_converged_v3.geometry import L1Geometry


@dataclass(frozen=True)
class V3MaskConfig:
    space_frac: float = 0.50  # D = round(space_frac·N) contacts masked (whole + intra)
    time_frac: float = 0.5  # T_mask = round(time_frac·T) frames masked per shaft
    whole_shaft_frac: float = 0.10  # E[K]; K ~ Binomial(S, frac) clamped to K_max
    block_w_space: int = 4  # depth-block width (contacts); floor 4 = HGA along-shaft autocorr
    block_w_time: int = 7  # time-block width (frames); floor 7 = HGA |STFT| support


@dataclass(frozen=True)
class V3Masks:
    """One batch of sampled masks (R rows). All counts are per-row/per-shaft EXACT."""

    contact_mask: Tensor  # (R, N) bool — True = spatially masked contact. Exactly D/row.
    frame_mask: Tensor  # (R, S, T) bool — True = temporally masked frame. Exactly T_mask/(row,shaft).
    whole_contact: Tensor  # (R, N) bool — True where the contact's shaft was wholly dropped (⊆ contact_mask).


def _k_max(cs: Tensor, d: int) -> int:
    """Largest number of shafts whose total contacts ≤ D (whole-shaft feasibility
    ceiling). Any K ≤ K_max RANDOM shafts sum ≤ the K largest ≤ K_max largest ≤ D, so
    clamping the stochastic whole count to K_max keeps whole-contacts ≤ D for EVERY
    draw ⇒ the snap never has to trim a whole shaft (which would leak the common mode)."""
    sizes = torch.sort(cs, descending=True).values
    csum = torch.cumsum(sizes, 0)
    return int((csum <= d).sum().item())


def assert_mask_feasible(geom: L1Geometry, cfg: V3MaskConfig = V3MaskConfig()) -> None:
    """Fail LOUD at session setup on a degenerate (montage, cfg). Reads scalar counts,
    so call ONCE at setup — never inside the compiled forward (graph-break)."""
    valid = geom.valid  # (S, C) bool
    n = int(valid.sum().item())
    d = round(cfg.space_frac * n)
    if not (0 < d < n):
        raise ValueError(f"space_frac={cfg.space_frac} ⇒ D={d} not in (0, N={n})")
    cs = valid.sum(1)  # (S,) contacts per shaft
    if _k_max(cs, d) < 1 and cfg.whole_shaft_frac > 0:
        raise ValueError(
            f"whole-shaft infeasible: no shaft fits under D={d} (largest shaft "
            f"{int(cs.max())} > D). Lower whole_shaft_frac to 0 or raise space_frac."
        )


def _cover_rank(valid: Tensor, width: int, n_rows: int, generator: torch.Generator) -> Tensor:
    """Contiguous-block cover-rank over ``(U, L)`` units (shafts) of length L.

    Per (row, unit): scatter width-``width`` blocks at random start ranks (starts from
    ``-(width-1)`` so the shallow/early edge is coverable — else it under-masks),
    OVERLAPS ALLOWED. ``cover_rank[i]`` = the min block start-rank among all spans
    covering position i (lower ⇒ covered by an earlier-placed block). Returns
    ``(R, U, L)`` float, ``inf`` on invalid positions. The caller takes the lowest-rank
    positions per unit (space: globally to D; time: per shaft to T_mask) — the union of
    the lowest-ranked overlapping blocks, contiguous by construction."""
    dev = valid.device
    u, length = valid.shape
    p = width - 1
    n_start = length + p
    starts = torch.arange(-p, length, device=dev)  # (n_start,)
    start_rank = (
        torch.rand(n_rows, u, n_start, generator=generator, device=dev).argsort(2).argsort(2)
    )  # (R, U, n_start) random 0-based rank per candidate start
    s_idx = starts[None, None, :, None]  # (1,1,n_start,1)
    c_idx = torch.arange(length, device=dev)[None, None, None, :]  # (1,1,1,L)
    cover = (s_idx <= c_idx) & (c_idx < s_idx + width) & valid[None, :, None, :]  # (R,U,n_start,L)
    big = n_start + 1
    ranks = torch.where(cover, start_rank[:, :, :, None], big)
    cover_rank = ranks.min(dim=2).values.float()  # (R, U, L)
    return torch.where(valid[None].expand(n_rows, u, length), cover_rank, float("inf"))


def sample_masks(
    geom: L1Geometry,
    n_contacts: int,
    *,
    n_time: int,
    n_rows: int,
    generator: torch.Generator,
    cfg: V3MaskConfig = V3MaskConfig(),
) -> V3Masks:
    """Sample ``n_rows`` independent dual-axis masks (see module docstring)."""
    r, s, c = n_rows, geom.n_shafts, geom.max_c
    n, t = n_contacts, n_time
    valid = geom.valid  # (S, C)
    dev = valid.device
    cs = valid.sum(1)  # (S,)
    d = round(cfg.space_frac * n)
    t_mask = round(cfg.time_frac * t)
    k_max = _k_max(cs, d)

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=generator, device=dev)

    # ── SPACE ────────────────────────────────────────────────────────────────
    # whole-shaft tier: stochastic count K ~ Binomial(S, frac) clamped to K_max, then
    # K random shafts. clamp keeps whole-contacts ≤ D for every draw (see _k_max).
    k = (rand(r, s) < cfg.whole_shaft_frac).sum(1).clamp(max=k_max)  # (R,)
    ws_rank = rand(r, s).argsort(1).argsort(1)  # (R, S) random 0-based shaft rank
    whole = ws_rank < k[:, None]  # (R, S) bool

    cover_space = _cover_rank(valid, cfg.block_w_space, r, generator)  # (R, S, C)
    valid_g = valid[None].expand(r, s, c)
    whole_g = whole[:, :, None].expand(r, s, c) & valid_g
    nonwhole_g = valid_g & ~whole_g
    # priority: whole [0,1) (always taken) < non-whole [1, 2+n_start) by cover-rank
    # (lowest-rank = block cells first) with a random tiebreak that round-robins across
    # shafts, so no single shaft is exhausted before the others contribute a block.
    r0 = rand(r, s, c)
    pri = torch.full((r, s, c), float("inf"), device=dev)
    pri = torch.where(whole_g, r0, pri)  # [0, 1)
    pri = torch.where(nonwhole_g, 1.0 + cover_space + r0, pri)  # [1, 2+n_start)
    sel_idx = pri.reshape(r, s * c).argsort(1)[:, :d]  # (R, D) grid cells, all finite/valid
    gidx_flat = geom.gather_idx.reshape(-1)  # (S*C,) → contact index in N
    target = gidx_flat[sel_idx]  # (R, D) distinct contact indices per row
    contact_mask = torch.zeros(r, n, dtype=torch.bool, device=dev)
    contact_mask.scatter_(1, target, True)

    # ── TIME (per shaft, snap to exactly T_mask, ≥1 live sensor per t) ─────────
    # A shaft is LIVE this row iff it keeps ≥1 contact (whole drops AND intra
    # saturation both kill it — no keep-alive floor). Guardianship is shared across
    # live shafts only; dead shafts have no visible cells so their frame mask is moot.
    # per-shaft visible-contact count via one-hot matmul (vectorized, no python loop).
    sc_onehot = torch.zeros(n, s, device=dev)
    sc_onehot[torch.arange(n, device=dev), geom.shaft_of_contact] = 1.0
    vis_per_shaft = (~contact_mask).to(torch.float32) @ sc_onehot  # (R, S)
    live_shaft = vis_per_shaft > 0.5  # (R, S) bool
    v_live = live_shaft.sum(1).clamp(min=1)  # (R,) live-shaft count (≥1 by construction)
    # local live-index of each shaft = # live shafts strictly before it in the row.
    lidx = (live_shaft.cumsum(1) - 1).clamp(min=0)  # (R, S) only meaningful where live
    # Balanced-RANDOM guardian (Ben 2026-07-12): a deterministic ``t % V == lidx``
    # forces a rigid diagonal of always-visible cells (phase-locked to t=0) — a
    # memorizable positional scaffold that systematically exempts a (t, shaft = t mod V)
    # sublattice from the temporal task. Instead assign each frame its guardian via a
    # random per-row permutation. ``perm_rank`` is a bijection 0..T-1, so ``% v_live`` is
    # still perfectly balanced (live-index j guards ⌊T/V⌋ or ⌈T/V⌉ frames — the SAME
    # per-shaft counts as the deterministic version ⇒ exact-T_mask static invariant and
    # ≥1-coverage preserved identically) but the guarded frames are randomly scattered,
    # not a diagonal, killing the positional crutch.
    perm_rank = rand(r, t).argsort(1).argsort(1)  # (R, T) random bijection 0..T-1 per row
    guardian_local = perm_rank % v_live[:, None]  # (R, T) balanced live-index labels, scattered
    guardian = guardian_local[:, None, :] == lidx[:, :, None]  # (R, S, T)
    forbid = guardian & live_shaft[:, :, None]  # (R, S, T) held-visible frames

    valid_time = torch.ones(s, t, dtype=torch.bool, device=dev)  # every frame valid
    cover_time = _cover_rank(valid_time, cfg.block_w_time, r, generator)  # (R, S, T)
    cover_time = torch.where(forbid, torch.full_like(cover_time, float("inf")), cover_time)
    time_rank = cover_time.argsort(-1).argsort(-1)  # (R, S, T) 0-based; forbid → last
    frame_mask = time_rank < t_mask  # (R, S, T) exactly T_mask True per (row, shaft)

    whole_contact = whole[:, geom.shaft_of_contact] & contact_mask  # (R, N)
    return V3Masks(contact_mask=contact_mask, frame_mask=frame_mask, whole_contact=whole_contact)
