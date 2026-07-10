"""v14_converged_v3 — electrode-unit time-tube masking (Phase 5).

Memo project-v14-converged-v3-sensor-architecture, two-tier per-SHAFT scheme
(Ben 2026-07-09, supersedes the earlier global-budget cover-rank):

  • WHOLE-SENSOR tier (the patient-invariant augmentation): ``whole_shaft_frac`` of
    the shafts are masked 100% — pulled entirely out of the L1 encoder so the only
    way to reconstruct them is cross-sensor (the L2 / predictor-offload task). Kept
    a PURE removal (not I-JEPA's [0.85,1.0]): the ~0.30 along-shaft common mode
    never decorrelates, so one visible contact would leak it to the whole shaft and
    turn a cross-sensor target back into a trivial within-shaft copy.
  • WITHIN-SENSOR tier: every surviving shaft is masked at its OWN ratio
    ``r ~ Uniform[r_lo, r_hi]`` (each shaft is its own 1-D image, V-JEPA-per-image),
    realised as contiguous depth-blocks of width ``Uniform{block_w_lo..hi}``. The
    floor ``block_w_lo=4`` is the measured along-shaft HGA autocorrelation length:
    the local excess above the common mode lives at lag 1 (+0.177) and is gone by
    lag 2 (+0.047), so ≥4-wide keeps the trivially-copyable lag-1 edge fraction
    (2/W) ≤ 50%. Per-shaft ratios kill the global clumping (some shafts ending up
    ~100% masked) that a single pooled budget produced.

Constant held-out count ``M = round(mask_frac·N)`` per row ⇒ static shapes (compile
once per session). The per-shaft ratios are drawn to AVERAGE ~M, then a single
priority argsort reconciles to EXACTLY M — whole shafts are held (never trimmed),
the marginal within-shaft block contacts absorb any over/undershoot.

Vectorized end-to-end (the memo's "ALWAYS VECTORIZE" rule): argsort-of-rand for the
whole-shaft pick, a COVER-RANK argsort for the contiguous blocks, a per-shaft
top-``k`` select, and one reconciliation argsort. No python loop over shafts /
blocks / rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from speech_decoding.models.v14_converged_v3.geometry import L1Geometry


@dataclass(frozen=True)
class V3MaskConfig:
    mask_frac: float = 0.575  # M = round(mask_frac·N); = whole(0.15)+(1−whole)·mean r(0.5), lets overall dip <0.60
    whole_shaft_frac: float = 0.15  # fraction of shafts masked 100% (patient-invariant)
    block_w_lo: int = 4  # Uniform{lo..hi}; floor 4 = along-shaft HGA autocorr length
    block_w_hi: int = 8
    r_lo: float = 0.30  # per-shaft within-sensor ratio ~ Uniform[r_lo, r_hi]
    r_hi: float = 0.70


def sample_contact_mask(
    geom: L1Geometry,
    n_contacts: int,
    *,
    n_rows: int,
    generator: torch.Generator,
    cfg: V3MaskConfig = V3MaskConfig(),
) -> Tensor:
    """Sample ``n_rows`` independent per-contact masks → ``(R, N)`` bool.

    True = held out (an SSL target, time-tubed). Exactly ``M = round(mask_frac·N)``
    True per row.
    """
    R, S, C = n_rows, geom.n_shafts, geom.max_c
    N = n_contacts
    M = round(cfg.mask_frac * N)
    valid = geom.valid  # (S, C) bool
    gidx_flat = geom.gather_idx.reshape(-1)  # (S*C,) long
    dev = valid.device
    Cs = valid.sum(1)  # (S,) contacts per shaft

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=generator, device=dev)

    # --- whole-sensor tier: round(frac·S) shafts, each masked 100% ---
    n_ws = round(cfg.whole_shaft_frac * S)
    ws_rank = rand(R, S).argsort(1).argsort(1)  # (R, S) 0-based random rank
    whole = ws_rank < n_ws  # (R, S) bool

    # --- per-shaft within-sensor target count: r ~ Uniform[r_lo, r_hi] ---
    r = cfg.r_lo + (cfg.r_hi - cfg.r_lo) * rand(R, S)  # (R, S)
    k_s = torch.round(r * Cs[None].float()).long()  # (R, S) target masked count
    k_s = torch.where(whole, Cs[None].expand(R, S), k_s)  # whole shafts → full
    k_s = k_s.clamp(max=Cs[None].expand(R, S))

    # --- contiguous block cover-rank (variable width, negative starts) ---
    # Starts run from -(w_hi-1)..C-1 so contact 0 is reachable by a block starting
    # "before" it (span s covers c iff s ≤ c < s+w) — else the shallow edge is
    # under-masked. cover_rank[c] = min random start-rank among spans covering c.
    P = cfg.block_w_hi - 1
    n_start = C + P
    starts = torch.arange(-P, C, device=dev)  # (n_start,)
    w = torch.randint(
        cfg.block_w_lo, cfg.block_w_hi + 1, (R, S, n_start), generator=generator, device=dev
    )
    start_rank = rand(R, S, n_start).argsort(2).argsort(2)  # (R, S, n_start)
    s_idx = starts[None, None, :, None]
    c_idx = torch.arange(C, device=dev)[None, None, None, :]
    cover = (s_idx <= c_idx) & (c_idx < s_idx + w[:, :, :, None]) & valid[None, :, None, :]
    BIG = n_start + 1
    ranks = torch.where(cover, start_rank[:, :, :, None], BIG)  # (R,S,n_start,C)
    cover_rank = ranks.min(dim=2).values.float()  # (R, S, C); finite for valid contacts

    # --- per-shaft select the k_s lowest-cover-rank contacts. Tie-break by DEPTH
    # (not random): contacts sharing a cover_rank are the same block, so a depth
    # tiebreak takes them as a contiguous shallow-prefix — a random tiebreak would
    # scramble which of them get picked and punch length-1 holes (orphans, the
    # trivially lag-1-interpolatable case floor-4 exists to prevent). The negative-
    # start blocks already flatten the shallow edge, so no random jitter is needed. ---
    c_pos = torch.arange(C, device=dev).float()
    cr = torch.where(
        valid[None].expand(R, S, C),
        cover_rank + c_pos[None, None, :] / (C + 1),  # depth tiebreak → contiguous partial block
        torch.full((R, S, C), float(2 * BIG), device=dev),
    )
    wsrank = cr.argsort(2).argsort(2)  # (R, S, C) 0 = lowest cover_rank on the shaft
    sel = (wsrank < k_s[:, :, None]) & valid[None].expand(R, S, C)  # selected within-shaft

    # --- reconcile to EXACTLY M via one priority argsort ---
    # tiers: whole (always in) < within-selected (trim highest wsrank first) <
    # within-unselected (pad pool) < invalid. Taking the M smallest holds the count
    # constant while whole shafts are never trimmed.
    whole_c = whole[:, :, None].expand(R, S, C) & valid[None].expand(R, S, C)
    within_sel = sel & ~whole_c
    within_unsel = valid[None].expand(R, S, C) & ~sel & ~whole_c
    r0 = rand(R, S, C)
    pri = torch.full((R, S, C), float("inf"), device=dev)
    pri = torch.where(whole_c, 0.5 * r0, pri)  # [0, 0.5)
    pri = torch.where(within_sel, 1.0 + wsrank.float() / (C + 1), pri)  # [1, 2)
    pri = torch.where(within_unsel, 2.0 + wsrank.float() / (C + 1), pri)  # [2, 3) extend blocks, not scatter
    pri = pri.reshape(R, S * C)

    sel_idx = pri.argsort(dim=1)[:, :M]  # (R, M) grid-cell indices, all finite/valid
    target = gidx_flat[sel_idx]  # (R, M) contact indices (distinct per row)
    mask = torch.zeros(R, N, dtype=torch.bool, device=dev)
    mask.scatter_(1, target, True)
    return mask
