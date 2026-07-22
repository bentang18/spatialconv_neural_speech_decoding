"""v14_converged_v3 — unified space × time block masking (Phase 5, r4 global rewrite).

ONE primitive, per-band budgets (Ben-locked 2026-07-15, contract project-r4-contract §2).
Everything is "union of contiguous ``_cover_rank`` blocks on a lattice, snapped to an exact
per-unit count" — the same call serves both axes; only the lattice and snap-policy differ.

  SPACE (which contacts, all bands, all time) — the within-shaft prediction task.
    PER-SHAFT BALANCED (Ben 2026-07-15: a global top-D let one shaft starve another —
    measured std 0.18, min 0.00, occasional 0/N shafts). Each shaft masks EXACTLY
    ``d_s = round(space_frac·n_s)`` of its OWN contacts via contiguous width-``block_w_space``
    depth-blocks (``_cover_rank`` within shaft, random starts, overlaps allowed). Balanced by
    construction; keep-alive is automatic (``d_s ≤ n_s−1`` leaves ≥1 visible/shaft — no argmax
    hack). WHOLE-SHAFT tier (class W): a stochastic K ~ Binomial(S, whole_shaft_frac) clamped to
    K_max sets d_s=n_s (full). DROPPED in Design B (frac=0); machinery kept (dead at 0). Static
    count (Σ d_s) holds for Design B (whole=0); with whole>0 the per-row total varies (legacy).

  TIME (which frames, GLOBAL across shafts) — the temporal-prediction task.
    The old per-shaft-INDEPENDENT time mask existed only to pressure the deleted encoder L2, so the
    time mask is GLOBAL (same band-b frames hidden for every shaft). ONE unified rule (Ben-locked
    2026-07-15, replacing the blackout mechanism): **each band is masked INDEPENDENTLY, in contiguous
    blocks of ``block_w_band`` (=4) of its OWN tokens, snapped to ~``*_mask_frac``.** Same rule, three
    grids — HGA 4×31 ms, MID 4×62 ms, SLOW 4×250 ms blocks. 4 own-tokens is the leak-safe minimum
    (M14 overlap-factor-2 on each band's decimated grid ⇒ the 2 interior tokens of a width-4 block
    have no visible same-band neighbor sharing samples; the objective's margin-gate scores those).

      • HGA (32 Hz, T tokens): blocks ≈125 ms, snapped to ``hga_mask_frac``.
      • MID (16 Hz, T/2 tokens): blocks ≈250 ms, snapped to ``mid_mask_frac``.
      • SLOW (4 Hz, T/8 tokens): blocks ≈1 s, snapped to ``slow_mask_frac``. SLOW is NOW a first-class
        masked band (symmetric with the others), no longer a mere blackout input-drop.

    WHY no blackout: (a) SLOW is on the loss now (uniform margin-gated rule), so it needs no special
    input-drop; (b) independent ~50% masking of all three bands makes a slot land all-three-masked
    ~12% of the time — those emergent empty windows ARE the perceiver's temporal-prediction pressure,
    handled by the soft-global (RoPE-localized, NOT hard-windowed) Stage-1 cross-attention, no
    special-casing. Per-band decode already gives per-band temporal masking its weight.

    Per-band masked counts are per-session CONSTANTS ⇒ compiled shapes fixed:
      HGA = round(hga_mask_frac·T),  MID = round(mid_mask_frac·T/2),  SLOW = round(slow_mask_frac·T/8).
      Bands are masked INDEPENDENTLY; empty-window slots emerge stochastically (shapes stay static
      because each band's own count is fixed).

Compose (in the objective) as an OUTER PRODUCT: a (contact c, band b, token t_b) is VISIBLE iff
contact c is spatially kept AND band-b token t_b is not temporally masked. Global time mask ⇒
every surviving shaft sees the SAME visible rectangle per band ⇒ online varlen blocks stay
rectangular and STATIC. time_pos = RoPE (contract §2, LOCKED).

FULLY VECTORIZED (cover-rank + argsort, no python loop over shafts/blocks/rows).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from speech_decoding.models.v14_converged_v3.geometry import L1Geometry

# Band lattice strides on the shared 32 Hz clock — MUST match stem.PER_BAND_SPECS
# ((7,8),(6,2),(7,1)) = SLOW 8, MID 2, HGA 1. The latent-slot grid == the SLOW grid.
SLOW_STRIDE = 8
MID_STRIDE = 2


@dataclass(frozen=True)
class V3MaskConfig:
    space_frac: float = 0.50  # each shaft masks d_s = round(space_frac·n_s) of its OWN contacts.
    whole_shaft_frac: float = 0.0  # E[K]; K ~ Binomial(S, frac) clamped to K_max. Design B: 0.
    keep_alive: bool = True  # Design B: reserve ≥1 visible contact per non-whole shaft (d_s≤n_s−1).
    block_w_space: int = 4  # depth-block width (contacts).
    # ── TIME (global; each band masked INDEPENDENTLY on its OWN grid, ONE unified rule) ──
    hga_mask_frac: float = 0.50  # HGA masked on its 32 Hz grid (Ben: 50%).
    mid_mask_frac: float = 0.50  # MID masked on its 16 Hz grid (Ben: 50% for symmetry).
    slow_mask_frac: float = 0.50  # SLOW masked on its 4 Hz grid (Ben 2026-07-15: 50%, first-class band).
    block_w_band: int = 4  # leak-safe block width in a band's OWN tokens (M14 margin 2); same all 3.
    temporal_mask_frac: float = 0.50  # r5 ONLY: single early-fused 32 Hz grid masked (T7-tunable).
    # r5 temporal block width (tokens) on the 32 Hz grid. 6 tokens = 187.5 ms (Ben 2026-07-22,
    # "150-200 ms range"). Accept-the-bleed ⇒ NO leak floor (unlike block_w_band's M14 margin);
    # width is a task-difficulty knob. Frontend RF = 5 frames = ±1 token, so a width-W block
    # leaks only at its 2 edges (interior W-2 tokens fully clean) — edge share 2/W = 33% at W=6.
    # T7 τ-autocorr may retune. Separate from block_w_band so r4 stays byte-identical.
    temporal_block_w: int = 6


@dataclass(frozen=True)
class V3Masks:
    """One batch of sampled masks (R rows). Per-band counts are per-row EXACT constants.

    Time masks are GLOBAL across shafts (no S axis): the same band-b frames are hidden for every
    surviving contact. All three band masks are INDEPENDENT and symmetric — ``slow_mask`` is SLOW's
    own 4 Hz temporal mask (SLOW token k == latent slot k), NOT a blackout indicator. A latent slot
    is empty exactly where all three band masks cover it — an emergent (not engineered) event."""

    contact_mask: Tensor  # (R, N) bool — spatially masked contact.
    whole_contact: Tensor  # (R, N) bool — contact whose shaft was wholly dropped (⊆ contact_mask).
    hga_mask: Tensor  # (R, T_hga) bool — HGA masked on its 32 Hz grid. Exactly round(hga_frac·T).
    mid_mask: Tensor  # (R, T_mid) bool — MID masked on its 16 Hz grid. Exactly round(mid_frac·T/2).
    slow_mask: Tensor  # (R, T_slow) bool — SLOW masked on its 4 Hz grid. Exactly round(slow_frac·T/8).


def _k_max(cs: Tensor, d: int | Tensor) -> Tensor:
    """Largest number of shafts whose total contacts ≤ D (whole-shaft feasibility ceiling).
    ``d`` may be a Python int or a 0-dim tensor (``csum <= d`` broadcasts either)."""
    sizes = torch.sort(cs, descending=True).values
    csum = torch.cumsum(sizes, 0)
    return (csum <= d).sum()


def assert_mask_feasible(geom: L1Geometry, cfg: V3MaskConfig = V3MaskConfig()) -> None:
    """Fail LOUD at session setup on a degenerate (montage, cfg). Setup-time only (host sync)."""
    valid = geom.valid  # (S, C) bool
    n = int(valid.sum().item())
    cs = valid.sum(1)  # (S,) contacts per shaft
    d_s = torch.round(cfg.space_frac * cs.float()).long()
    if cfg.keep_alive:
        d_s = torch.minimum(d_s, (cs - 1).clamp(min=0))
    if int(d_s.sum()) == 0 or int(d_s.sum()) >= n:
        raise ValueError(
            f"space_frac={cfg.space_frac} keep_alive={cfg.keep_alive} ⇒ Σd_s={int(d_s.sum())} "
            f"not in (0, N={n}); every shaft size ≤1 leaves nothing to mask under keep-alive?"
        )
    d = int(d_s.sum())
    if int(_k_max(cs, d)) < 1 and cfg.whole_shaft_frac > 0:
        raise ValueError(
            f"whole-shaft infeasible: no shaft fits under D={d} (largest shaft "
            f"{int(cs.max())} > D). Lower whole_shaft_frac to 0 or raise space_frac."
        )


def assert_time_feasible(n_time: int, cfg: V3MaskConfig = V3MaskConfig()) -> None:
    """Fail LOUD if the time config is degenerate for this clip length. Setup-time only."""
    if n_time % SLOW_STRIDE != 0:
        raise ValueError(f"n_time={n_time} not a multiple of SLOW_STRIDE={SLOW_STRIDE}")
    if n_time % MID_STRIDE != 0:
        raise ValueError(f"n_time={n_time} not a multiple of MID_STRIDE={MID_STRIDE}")
    grids = {
        "HGA": (n_time, cfg.hga_mask_frac),
        "MID": (n_time // MID_STRIDE, cfg.mid_mask_frac),
        "SLOW": (n_time // SLOW_STRIDE, cfg.slow_mask_frac),
    }
    for name, (length, frac) in grids.items():
        cnt = round(frac * length)
        if not (0 <= cnt <= length):
            raise ValueError(f"{name}: round({frac}·{length})={cnt} not in [0,{length}]")
        if cnt > 0 and cfg.block_w_band > length:
            raise ValueError(
                f"{name} grid length {length} < block_w_band={cfg.block_w_band}: clip too short "
                f"for a leak-safe {name} block (need n_time ≥ {cfg.block_w_band}·stride)"
            )


def _cover_rank(valid: Tensor, width: int, n_rows: int, generator: torch.Generator) -> Tensor:
    """Contiguous-block cover-rank over ``(U, L)`` units of length L.

    Per (row, unit): scatter width-``width`` blocks at random start ranks (starts from
    ``-(width-1)`` so the early edge is coverable), OVERLAPS ALLOWED. ``cover_rank[i]`` = the min
    block start-rank among all spans covering position i. Returns ``(R, U, L)`` float, ``inf`` on
    invalid positions. Callers take the lowest-rank positions per unit (contiguous by construction)."""
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
    """Sample ``n_rows`` independent unified space × time masks (see module docstring)."""
    r, s, c = n_rows, geom.n_shafts, geom.max_c
    n, t = n_contacts, n_time
    valid = geom.valid  # (S, C)
    dev = valid.device
    cs = valid.sum(1)  # (S,)

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=generator, device=dev)

    # ── SPACE (per-shaft balanced) ─────────────────────────────────────────────
    d_s_base = torch.round(cfg.space_frac * cs.float()).long()  # (S,) per-shaft target
    if cfg.keep_alive:
        d_s_base = torch.minimum(d_s_base, (cs - 1).clamp(min=0))  # reserve ≥1 visible/shaft
    d = d_s_base.sum()  # 0-dim tensor — kept on-device (no host sync); _k_max broadcasts it
    k_max = _k_max(cs, d)
    # whole-shaft tier (legacy; frac=0 in Design B): whole shafts mask ALL n_s.
    k = (rand(r, s) < cfg.whole_shaft_frac).sum(1).clamp(max=k_max)  # (R,)
    ws_rank = rand(r, s).argsort(1).argsort(1)  # (R, S) random 0-based shaft rank
    whole = ws_rank < k[:, None]  # (R, S) bool
    d_s = torch.where(whole, cs[None].expand(r, s), d_s_base[None].expand(r, s))  # (R, S)

    cover_space = _cover_rank(valid, cfg.block_w_space, r, generator)  # (R, S, C) inf on invalid
    key = cover_space + rand(r, s, c)  # block-rank + random tiebreak; inf on invalid → sorts last
    within_rank = key.argsort(-1).argsort(-1)  # (R, S, C) 0-based rank within shaft
    grid_mask = within_rank < d_s[:, :, None]  # (R, S, C) exactly d_s per (row, shaft); valid-only

    # map (S,C) grid → (N,) contacts. valid cells carry a distinct contact index in gather_idx.
    vpos = valid.reshape(-1).nonzero(as_tuple=True)[0]  # (N,) valid grid positions
    vcontact = geom.gather_idx.reshape(-1)[vpos]  # (N,) permutation of 0..N−1
    contact_mask = torch.zeros(r, n, dtype=torch.bool, device=dev)
    contact_mask[:, vcontact] = grid_mask.reshape(r, -1)[:, vpos]
    whole_contact = torch.zeros(r, n, dtype=torch.bool, device=dev)
    whole_g = whole[:, :, None].expand(r, s, c) & valid[None].expand(r, s, c)
    whole_contact[:, vcontact] = whole_g.reshape(r, -1)[:, vpos]

    # ── TIME (global; each band masked INDEPENDENTLY on its own leak-safe grid) ─
    if t % SLOW_STRIDE != 0:
        raise ValueError(f"n_time={t} not a multiple of SLOW_STRIDE={SLOW_STRIDE}")
    if t % MID_STRIDE != 0:
        raise ValueError(f"n_time={t} not a multiple of MID_STRIDE={MID_STRIDE}")
    t_mid = t // MID_STRIDE
    n_slots = t // SLOW_STRIDE

    def _band_mask(length: int, frac: float) -> Tensor:
        """~frac of a band's own grid, in contiguous width-block_w_band blocks (leak-safe)."""
        ones = torch.ones(1, length, dtype=torch.bool, device=dev)
        cover = _cover_rank(ones, cfg.block_w_band, r, generator).squeeze(1)  # (R, length)
        cnt = round(frac * length)
        return cover.argsort(-1).argsort(-1) < cnt  # (R, length) exactly cnt masked

    hga_mask = _band_mask(t, cfg.hga_mask_frac)  # (R, T)   exactly round(hga_frac·T)
    mid_mask = _band_mask(t_mid, cfg.mid_mask_frac)  # (R, T/2) exactly round(mid_frac·T/2)
    slow_mask = _band_mask(n_slots, cfg.slow_mask_frac)  # (R, T/8) exactly round(slow_frac·T/8)

    return V3Masks(
        contact_mask=contact_mask,
        whole_contact=whole_contact,
        hga_mask=hga_mask,
        mid_mask=mid_mask,
        slow_mask=slow_mask,
    )


# ── r5 (Chang 2-stream) single early-fused band masking ─────────────────────────
@dataclass(frozen=True)
class V3MasksR5:
    """r5 masks — ONE early-fused 32 Hz band (no SLOW/MID/HGA split).

    SPACE (``contact_mask``) is the SAME per-shaft balanced tier as ``V3Masks``. TIME is
    PER-SHAFT INDEPENDENT on the one 32 Hz grid — each ``(row, shaft)`` masks its OWN
    ``round(temporal_mask_frac·T)`` timepoints in contiguous width-``temporal_block_w`` blocks
    (L1-only shaft-local ⇒ each shaft is its own axiom). Shape-neutral: the count is identical
    per shaft, so ``m_vis`` is unchanged — only WHICH timepoints a shaft hides differs.
    Downstream (``pack_r4.token_flags_r5``) scores EVERY masked token (``in_loss == masked``,
    accept-the-bleed) — no margin gate."""

    contact_mask: Tensor  # (R, N) bool — spatially masked contact.
    temporal_mask: Tensor  # (R, S, T) bool — per-shaft. Exactly round(frac·T) per (row, shaft).


def sample_masks_r5(
    geom: L1Geometry,
    n_contacts: int,
    *,
    n_time: int,
    n_rows: int,
    generator: torch.Generator,
    cfg: V3MaskConfig = V3MaskConfig(),
) -> V3MasksR5:
    """Sample ``n_rows`` r5 masks: per-shaft balanced SPACE + ONE 32 Hz temporal mask.

    The SPACE tier is a faithful copy of :func:`sample_masks` (kept SEPARATE so the
    seed-sensitive locked r4 sampler is untouched). r5 has a SINGLE band, so there is one
    temporal mask (``temporal_mask_frac``) instead of the SLOW/MID/HGA trio."""
    r, s, c = n_rows, geom.n_shafts, geom.max_c
    n, t = n_contacts, n_time
    valid = geom.valid  # (S, C)
    dev = valid.device
    cs = valid.sum(1)  # (S,)

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=generator, device=dev)

    # ── SPACE (per-shaft balanced) — faithful copy of sample_masks ──────────────
    d_s_base = torch.round(cfg.space_frac * cs.float()).long()
    if cfg.keep_alive:
        d_s_base = torch.minimum(d_s_base, (cs - 1).clamp(min=0))
    d = d_s_base.sum()
    k_max = _k_max(cs, d)
    k = (rand(r, s) < cfg.whole_shaft_frac).sum(1).clamp(max=k_max)  # (R,)
    ws_rank = rand(r, s).argsort(1).argsort(1)  # (R, S)
    whole = ws_rank < k[:, None]  # (R, S)
    d_s = torch.where(whole, cs[None].expand(r, s), d_s_base[None].expand(r, s))

    cover_space = _cover_rank(valid, cfg.block_w_space, r, generator)  # (R, S, C)
    key = cover_space + rand(r, s, c)
    within_rank = key.argsort(-1).argsort(-1)  # (R, S, C)
    grid_mask = within_rank < d_s[:, :, None]  # (R, S, C) exactly d_s per (row, shaft)

    vpos = valid.reshape(-1).nonzero(as_tuple=True)[0]  # (N,)
    vcontact = geom.gather_idx.reshape(-1)[vpos]  # (N,)
    contact_mask = torch.zeros(r, n, dtype=torch.bool, device=dev)
    contact_mask[:, vcontact] = grid_mask.reshape(r, -1)[:, vpos]

    # ── TIME (per-shaft INDEPENDENT 32 Hz masks) ────────────────────────────────
    # L1-only shaft-local ⇒ each shaft is its own axiom: give every (row, shaft) its OWN
    # contiguous-block temporal mask, not one shared across the pack's shafts. Shape-neutral —
    # cnt is identical per shaft, so m_vis = (T−cnt)·Σ(visible contacts) is unchanged; only
    # WHICH timepoints a shaft hides differs. width = temporal_block_w (6 = 187.5 ms @ 32 Hz);
    # accept-the-bleed ⇒ no margin gate downstream.
    ones = torch.ones(s, t, dtype=torch.bool, device=dev)
    cover = _cover_rank(ones, cfg.temporal_block_w, r, generator)  # (R, S, T)
    cnt = round(cfg.temporal_mask_frac * t)
    temporal_mask = cover.argsort(-1).argsort(-1) < cnt  # (R, S, T) exactly cnt per (row, shaft)

    return V3MasksR5(contact_mask=contact_mask, temporal_mask=temporal_mask)
