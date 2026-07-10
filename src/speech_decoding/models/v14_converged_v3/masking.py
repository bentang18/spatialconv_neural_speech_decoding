"""v14_converged_v3 — electrode-unit time-tube masking (Phase 5).

Memo project-v14-converged-v3-sensor-architecture. The mask is a per-ELECTRODE
boolean (a masked electrode is time-tubed = hidden across ALL slots — the
objective broadcasts it over the clock). Distribution, small→large spatial
extent: within-shaft contiguous contact-blocks (w~Uniform{block_w_lo..hi}, the
MAJORITY mass) ⊂ whole-shaft masks (the THIN TAIL that trains the offloaded
predictor L2). Constant held-out count ``M = round(mask_frac·N)`` per row ⇒ static
shapes (compile once per session).

Vectorized end-to-end (the memo's "ALWAYS VECTORIZE" rule): one COVER-RANK
argsort, generalising v2's ``_hga_fill_not_trim`` to (a) variable block width and
(b) a whole-shaft priority tier. No python loop over shafts / blocks / rows.

Cover-rank recipe: a contact's block priority = the min random start-rank among
the spans covering it (span ``s`` covers contact ``c`` iff ``s ≤ c < s+w_s``);
selecting the smallest-``M`` by (tier, cover_rank, position) ≡ adding whole spans
in random order until the budget is hit and trimming the last — contiguous blocks,
exact count, no scan. The whole-shaft tier sits strictly below every block
priority, so designated shafts fill first (large-extent tail), blocks fill the
remainder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from speech_decoding.models.v14_converged_v3.geometry import L1Geometry


@dataclass(frozen=True)
class V3MaskConfig:
    mask_frac: float = 0.60  # ⇒ M = round(mask_frac·N) held out
    block_w_lo: int = 4  # Uniform{lo..hi} inclusive; floor 4 = along-shaft HGA autocorr
    block_w_hi: int = 8
    whole_shaft_frac: float = 0.15  # fraction of shafts wholly masked (thin tail)


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

    # --- whole-shaft tier: round(frac·S) shafts per row, argsort-of-rand top-k ---
    n_ws = round(cfg.whole_shaft_frac * S)
    ws_rank = (
        torch.rand(R, S, generator=generator, device=dev).argsort(1).argsort(1)
    )  # (R, S) 0-based
    whole = ws_rank < n_ws  # (R, S) bool

    # --- within-shaft block cover-rank (variable width) ---
    # Start positions run from -(w_hi-1) to C-1. The negative starts let contact 0
    # be covered by a block starting "before" it (clamped to the shaft), so every
    # contact — boundary or interior — is reachable by the same w candidate spans.
    # Without them, contact 0 is coverable ONLY by a start at 0 (span s covers c iff
    # s ≤ c < s+w ⇒ shallow-end deficit), masking it at ~0.36 vs ~0.67 interior.
    P = cfg.block_w_hi - 1  # max left-extension
    n_start = C + P
    starts = torch.arange(-P, C, device=dev)  # (n_start,) actual start positions
    w = torch.randint(
        cfg.block_w_lo, cfg.block_w_hi + 1, (R, S, n_start), generator=generator, device=dev
    )  # per candidate start
    start_rank = (
        torch.rand(R, S, n_start, generator=generator, device=dev).argsort(2).argsort(2)
    )  # (R, S, n_start) random rank of each candidate start
    s_idx = starts[None, None, :, None]  # start axis
    c_idx = torch.arange(C, device=dev)[None, None, None, :]  # covered axis
    cover = (
        (s_idx <= c_idx)
        & (c_idx < s_idx + w[:, :, :, None])
        & valid[None, :, None, :]
    )  # (R, S, n_start, C_c)
    BIG = n_start + 1
    ranks = torch.where(cover, start_rank[:, :, :, None], BIG)  # (R,S,n_start,C_c)
    cover_rank = ranks.min(dim=2).values  # (R, S, C) min over starts; finite for valid

    # --- composite priority: tier ⊳ (whole < block); ties broken at RANDOM ---
    # A deterministic position tiebreak (+pos) would systematically pick the shallow
    # end of the marginal block, re-introducing a mild depth gradient. A uniform
    # [0,1) tiebreak (< the tier gap of C) leaves marginal selection depth-unbiased.
    pos = torch.arange(C, device=dev)[None, None, :].expand(R, S, C).float()
    rand_tie = torch.rand(R, S, C, generator=generator, device=dev)  # [0, 1)
    ws_rank_f = ws_rank[:, :, None].float().expand(R, S, C)
    whole_pri = ws_rank_f * C + pos  # in [0, S*C); whole shafts fully selected anyway
    block_pri = float(S * C) + cover_rank.float() * C + rand_tie  # ≥ S*C, tiers disjoint
    priority = torch.where(whole[:, :, None].expand(R, S, C), whole_pri, block_pri)
    priority = priority.masked_fill(~valid[None].expand(R, S, C), float("inf"))
    priority = priority.reshape(R, S * C)

    # --- pick the M smallest-priority contacts, scatter onto the contact axis ---
    sel = priority.argsort(dim=1)[:, :M]  # (R, M) grid-cell indices, all valid
    target = gidx_flat[sel]  # (R, M) contact indices (distinct per row)
    mask = torch.zeros(R, N, dtype=torch.bool, device=dev)
    mask.scatter_(1, target, True)
    return mask
