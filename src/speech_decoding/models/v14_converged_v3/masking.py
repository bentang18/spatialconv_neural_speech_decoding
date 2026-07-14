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
    • Contiguous width-``w_t`` time-blocks (floor 4 = 125 ms @ 32 Hz; a masked frame is
      fully hidden only ≥ support frames from any visible frame, and HGA's support is
      2 slots — nperseg 128 @ hop 64. The old floor of 7 cited "HGA |STFT| support" but
      that is 2, not 7; see V3MaskConfig.block_w_time), random
      starts, overlaps allowed. Snap PER SHAFT to EXACTLY ``T_mask = round(time_frac·T)``
      masked frames (constant ``T_kept`` ⇒ the buffer stays static). Independent per
      (row, shaft) ⇒ HETEROGENEOUS inter-sensor timing (at any t some sensors visible,
      some masked) — the L2 / montage pressure the homogeneous-intra outer product
      otherwise loses.
    • NO GUARDIAN (M14, 2026-07-15 — the "at-least-one-sensor-lives" rule is DELETED).
      It forbade one randomly-chosen live shaft per t from masking that frame, so that
      every t kept ≥1 visible sensor. M14 measured what it actually did, and it was
      WORSE THAN DEAD WEIGHT: because the forbidden frames are scattered through every
      shaft's candidate set, they SHRED the time blocks. Realized run length, target vs
      realized: w=4 → 4.32, w=8 → 5.83, w=12 → 6.63. It SATURATES the realized run at
      ~6 no matter how wide a block you ask for — i.e. it would have silently prevented
      any wide block from ever burying a cell deep enough to beat the STFT window
      overlap, which is the ENTIRE POINT of the block width. It re-creates the very leak
      the width exists to kill. Guardian OFF: w=12 → 11.65, as designed.
      What it bought: a same-t spatial path for 0.03% of masked cells (dead-frame rate
      0.000488 over 13 montages ≈ 0.5^V at V≈11 live shafts — the sampler and the
      arithmetic agree to 3 figures). Those cells are not "unpredictable": a masked cell
      at a dead frame still has its OWN shaft's other frames and every other shaft's
      other frames. It has no SAME-t neighbour, which is a much weaker deprivation than
      the rule's name suggests.
      STATIC-SAFE, and that is provable rather than hoped: the guardian only pushed
      frames to the back of the cover-rank sort, BEFORE the snap
      ``frame_mask = time_rank < t_mask``. Exactly ``T_mask`` frames are masked per
      (row, shaft) with it or without it, so ``T_kept``, ``D``, the online token count
      ``(N-D)·T_kept`` and the loss denominator are all unchanged. Deleting it also
      deletes the ``V ≥ 2`` feasibility constraint, so the sampler gets strictly more
      robust on degenerate montages.

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
    space_frac: float = 0.60  # D = round(space_frac·N) contacts masked (whole + intra)
    time_frac: float = 0.5  # T_mask = round(time_frac·T) frames masked per shaft
    whole_shaft_frac: float = 0.10  # E[K]; K ~ Binomial(S, frac) clamped to K_max
    # ⚠️ NOT JUSTIFIED BY ANY MEASUREMENT. The comment this replaces claimed "floor 4 = HGA
    # along-shaft autocorr", which is FALSE: M2 measured HGA depth-lag ACF at 0.34 / 0.21 /
    # 0.18 for lags 1/2/3 — gone by lag 2, not 4. M14 then measured the lever directly and it
    # is WEAK: own-shaft copy-ability of a masked contact is 0.237 / 0.192 / 0.134 at
    # w_s = 1 / 4 / 8 (HGA, 13 montages). So 4 is neither right nor wrong; it is INERTIA.
    # It is also the wrong QUESTION: a temporal leak is a TRANSFORM ARTIFACT (the visible
    # STFT window literally contains the masked samples — copying it teaches nothing, and it
    # must be defeated), whereas a spatial correlation is PHYSICS (a neighbouring contact
    # genuinely sees a correlated population — predicting it IS the cross-sensor task). Do
    # not "widen w_s to kill the leak"; there is no spatial leak to kill.
    block_w_space: int = 4  # depth-block width (contacts). See above: inertia, not evidence.
    # B6 (Ben, 2026-07-15): 7 -> 4. The old comment claimed "floor 7 = HGA |STFT| support",
    # which is wrong: at the SHARED hop=64 grid HGA (nperseg 128) has support 2 slots, not 7.
    # M14 (2026-07-15) then found the real mechanism: the leak is STFT WINDOW OVERLAP, and the
    # overlap factor is nperseg/hop = 16 (SLOW) / 4 (MID) / 2 (HGA) on the shared 32 Hz grid.
    # Under the PER-BAND token rates (HGA 32 Hz, MID 16 Hz, SLOW 4 Hz) every band's token hop
    # is nperseg/2, so the overlap factor is EXACTLY 2 for all three, and a width-4 block puts
    # its deepest cells at margin 2 = zero sample overlap. That is why 4 is right, in every
    # band's OWN grid. Realized geometry (M14, guardian OFF): w=4 -> run 5.57.
    block_w_time: int = 4  # time-block width, in each band's OWN grid; deepest cell at margin 2


@dataclass(frozen=True)
class V3Masks:
    """One batch of sampled masks (R rows). All counts are per-row/per-shaft EXACT."""

    contact_mask: Tensor  # (R, N) bool — True = spatially masked contact. Exactly D/row.
    frame_mask: Tensor  # (R, S, T) bool — True = temporally masked frame. Exactly T_mask/(row,shaft).
    whole_contact: Tensor  # (R, N) bool — True where the contact's shaft was wholly dropped (⊆ contact_mask).


def _k_max(cs: Tensor, d: int) -> Tensor:
    """Largest number of shafts whose total contacts ≤ D (whole-shaft feasibility
    ceiling). Any K ≤ K_max RANDOM shafts sum ≤ the K largest ≤ K_max largest ≤ D, so
    clamping the stochastic whole count to K_max keeps whole-contacts ≤ D for EVERY
    draw ⇒ the snap never has to trim a whole shaft (which would leak the common mode).

    Returns a 0-dim tensor (NO host sync) so ``sample_masks`` can call it inside the
    compiled forward and feed it straight to ``.clamp(max=…)``; the setup-time
    feasibility check wraps it in ``int()`` where a Python scalar is wanted."""
    sizes = torch.sort(cs, descending=True).values
    csum = torch.cumsum(sizes, 0)
    return (csum <= d).sum()


def assert_mask_feasible(geom: L1Geometry, cfg: V3MaskConfig = V3MaskConfig()) -> None:
    """Fail LOUD at session setup on a degenerate (montage, cfg). Reads scalar counts,
    so call ONCE at setup — never inside the compiled forward (graph-break)."""
    valid = geom.valid  # (S, C) bool
    n = int(valid.sum().item())
    d = round(cfg.space_frac * n)
    if not (0 < d < n):
        raise ValueError(f"space_frac={cfg.space_frac} ⇒ D={d} not in (0, N={n})")
    cs = valid.sum(1)  # (S,) contacts per shaft
    if int(_k_max(cs, d)) < 1 and cfg.whole_shaft_frac > 0:
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

    # ── TIME (per shaft, snap to exactly T_mask) ──────────────────────────────
    # NO GUARDIAN (M14, 2026-07-15). Every frame is a candidate on every shaft; blocks
    # are placed by cover-rank and the lowest-ranked T_mask frames win. The deleted
    # guardian used to force one live shaft per t to the back of this sort, which capped
    # the realized run at ~6 however wide the block — see the module docstring. Blank
    # frames (no shaft visible at t) are now possible at ~0.05% of frames and that is
    # ACCEPTED: such a cell still has its own shaft's other frames and every other
    # shaft's other frames; it lacks only a SAME-t neighbour.
    valid_time = torch.ones(s, t, dtype=torch.bool, device=dev)  # every frame valid
    cover_time = _cover_rank(valid_time, cfg.block_w_time, r, generator)  # (R, S, T)
    time_rank = cover_time.argsort(-1).argsort(-1)  # (R, S, T) 0-based
    frame_mask = time_rank < t_mask  # (R, S, T) exactly T_mask True per (row, shaft)

    whole_contact = whole[:, geom.shaft_of_contact] & contact_mask  # (R, N)
    return V3Masks(contact_mask=contact_mask, frame_mask=frame_mask, whole_contact=whole_contact)
