"""B36 paradigm-B masked-JEPA mask samplers (WS-B3/B4, 6/03 masking lock).

Ratified by Ben 2026-06-03 (``reports/b36_masking_handoff_2026_06_03.md``),
superseding the shipped per-cell-0.5 (M2) and time-block-0.3 (M4) defaults.
Two staged JEPA terms, one sampler each:

  * **M2 — P1 front-end** (:func:`sample_m2_mask`, default ``"bands"``):
    structured 1D spectro-temporal bands over the ``(B, C, F_p, T_p)`` token
    grid — a masked *time-band* hides ALL freq patches at those time columns,
    a masked *freq-band* hides ALL time patches at those freq rows; the masked
    set is their UNION (SpecAugment / A-JEPA TF-aware shape). Held-out ratio
    0.50, split symmetrically across the two axes via ``a = b = 1 - sqrt(1 -
    r)`` (so union coverage ``1 - (1-a)(1-b) = r``). Independent per electrode
    (no cross-electrode pathway at P1). The ``"random"`` sister
    (:func:`sample_token_mask`) is the per-cell Bernoulli must-beat baseline.

  * **M4 — P2 parcel** (:func:`sample_m4_mask`, default ``"tube"``): mask a
    whole covered parcel across ALL ``T_p`` time-patches (a "tube") on a
    uniform-random 0.20 SUBSET of covered parcels. The parcel axis is
    unordered, so selection is uniform-random, not a spatial block. The
    ``"time_block"`` sister (:func:`sample_parcel_time_block_mask`) is the
    contiguous-time-block shape.

Convention: a ``True`` entry means MASKED — a prediction target on the loss
side, and (for the M4 drop set) an electrode-time cell removed from the
visible student input so the front-end never encodes it (leakage-free).
``visible = ~mask``. Every sampler takes an explicit :class:`torch.Generator`,
so a given seed reproduces the mask bit-for-bit.

**Coupling law** (:func:`validate_m4_coupling`): mask shape and predictor
scope move as a pair, or the SSL task leaks — ``tube`` ↔ ``cross_time``
(default; the parcel is gone at every timestep so cross-time attention has
nothing to copy) and ``time_block`` ↔ ``co_temporal`` (sister; the parcel
survives at other times but the predictor cannot reach them) are shortcut-free;
``time_block`` + ``cross_time`` is the **H1 leak** (masked-at-t, visible-at-t±1,
trivial cross-time interpolation) and is rejected.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# ----------------------------------------------------------------------------
# M2 — P1 front-end spectro-temporal mask (WS-B3)
# ----------------------------------------------------------------------------


def sample_token_mask(
    shape: tuple[int, ...],
    *,
    mask_ratio: float,
    generator: torch.Generator,
    device: torch.device | str | None = None,
) -> Tensor:
    """Per-element Bernoulli mask over a front-end token grid — the M2
    ``"random"`` sister (R-m2-random, the structured-bands must-beat baseline).

    ``shape`` is the ``(B, C, F_p, T_p)`` grid; each cell is independently
    masked with probability ``mask_ratio``. Returns a bool tensor of ``shape``
    where ``True`` == masked. The draw uses ``generator`` (same seed → same
    mask). ``device`` overrides the output device (defaults to the generator's
    device).
    """
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1]; got {mask_ratio}")
    u = torch.rand(shape, generator=generator, device=generator.device)
    mask = u < mask_ratio
    if device is not None:
        mask = mask.to(device)
    return mask


def _sample_axis_bands(
    B: int,
    C: int,
    *,
    axis_len: int,
    n_valid: int,
    frac: float,
    floor: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    """``(B, C, axis_len)`` bool: ``~frac·n_valid`` cells masked as contiguous
    bands of width ≥ ``floor``, placed NON-overlapping (touching allowed) with
    uniform-random starts inside the valid prefix ``[0, n_valid)``.

    Non-overlapping placement makes the realized masked count exact at
    ``n_bands·width`` (no overlap shortfall). That realized fraction
    ``n_bands·width / n_valid`` equals the target ``frac`` only when ``width``
    divides ``round(frac·n_valid)``; otherwise it rounds DOWN by up to one
    band-width (e.g. ``frac·n_valid = 5``, ``width = 2`` → ``n_bands = 2`` → 4
    masked, not 5). On the locked grids (T_p even, F_p with floor 1) it lands
    on target. Each band has width ``floor`` (the minimum the 6/03 lock
    allows). ``n_valid ≤ axis_len`` carves out the corpus-valid freq prefix
    (B36 C5); time always uses ``n_valid = axis_len``.
    """
    if n_valid <= 0 or frac <= 0.0:
        return torch.zeros((B, C, axis_len), dtype=torch.bool, device=device)
    width = min(max(1, floor), n_valid)
    n_target = int(round(frac * n_valid))
    if n_target <= 0:
        return torch.zeros((B, C, axis_len), dtype=torch.bool, device=device)
    n_bands = max(1, n_target // width)
    n_bands = min(n_bands, n_valid // width)
    if n_bands <= 0:
        return torch.zeros((B, C, axis_len), dtype=torch.bool, device=device)
    # Non-overlapping fixed-width interval placement: pick n_bands sorted
    # offsets t in [0, n_valid - n_bands·width], then start_i = t_i + i·width.
    # Touching (t_i == t_{i-1}) is allowed; true overlap is impossible.
    free = n_valid - n_bands * width
    t = torch.randint(
        0, free + 1, (B, C, n_bands), generator=generator, device=device
    )
    t, _ = t.sort(dim=-1)
    offsets = torch.arange(n_bands, device=device) * width  # (n_bands,)
    starts = t + offsets  # (B, C, n_bands)
    idx = torch.arange(axis_len, device=device).view(1, 1, 1, axis_len)
    lo = starts.unsqueeze(-1)  # (B, C, n_bands, 1)
    band = (idx >= lo) & (idx < lo + width)  # (B, C, n_bands, axis_len)
    return band.any(dim=2)  # (B, C, axis_len)


def sample_token_band_mask(
    shape: tuple[int, int, int, int],
    *,
    held_out_ratio: float = 0.50,
    generator: torch.Generator,
    time_band_floor: int = 2,
    freq_band_floor: int = 1,
    time_freq_split: tuple[float, float] | None = None,
    freq_patch_valid: Tensor | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Structured 1D spectro-temporal band mask over a ``(B, C, F_p, T_p)``
    grid (M2 default — WS-B3).

    A masked time-band hides every freq patch at those time columns; a masked
    freq-band hides every time patch at those freq rows; the returned mask is
    the UNION. Scale-based, so it is grid-size agnostic (works at any ``T_p``).

    Parameters
    ----------
    shape
        ``(B, C, F_p, T_p)`` front-end token grid.
    held_out_ratio
        Target union masked fraction ``r`` (default 0.50). Symmetric split sets
        the per-axis fraction ``a = b = 1 - sqrt(1 - r)``.
    time_freq_split
        ``(a, b)`` per-axis masked fractions, overriding the symmetric default
        (the ``R-m2-time-weighted-split`` sister, e.g. ``(0.40, 0.20)``).
    time_band_floor, freq_band_floor
        Minimum band width on each axis (lock: time ≥ 2 patches ≥ 250 ms,
        freq ≥ 1 patch ≈ 1 octave). Bands are realized at exactly this width.
    freq_patch_valid
        Optional ``(F_p,)`` bool (corpus-valid freq prefix, B36 C5). Freq-bands
        are confined to the valid rows; ``None`` (BT, all valid) → full ``F_p``.
        Must mark a contiguous prefix ``[0, n_valid)``; a non-prefix mask or a
        per-clip ``(B, F_p)`` mask raises (per-clip is the deferred WS-H path).
    """
    if not 0.0 <= held_out_ratio <= 1.0:
        raise ValueError(f"held_out_ratio must be in [0, 1]; got {held_out_ratio}")
    B, C, F_p, T_p = shape
    g_dev = generator.device

    if time_freq_split is None:
        a = b = 1.0 - math.sqrt(1.0 - held_out_ratio)
    else:
        a, b = time_freq_split
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError(f"time_freq_split fractions must be in [0, 1]; got {time_freq_split}")

    if freq_patch_valid is None:
        n_freq_valid = F_p
    else:
        fpv = freq_patch_valid.to(torch.bool)
        # The band sampler places ONE shared freq-row layout per batch, so it
        # confines to a single shared valid prefix. A per-clip (B, F_p) mask
        # (mixed-corpus batches) needs per-element n_valid in _sample_axis_bands
        # — that's the deferred WS-H extension (see v14_blockers.md B36-C5 row).
        if fpv.dim() == 2:
            raise NotImplementedError(
                f"freq_patch_valid {tuple(fpv.shape)}: per-clip (B, F_p) "
                "confinement is not supported by the M2 band sampler yet (it "
                "places one shared band layout per batch). Pass a shared "
                "(F_p,) prefix for within-corpus batches; WS-H extends "
                "_sample_axis_bands to per-element n_valid for mixed-corpus."
            )
        if fpv.dim() != 1 or fpv.shape[0] != F_p:
            raise ValueError(
                f"freq_patch_valid must be (F_p,) = ({F_p},); got {tuple(fpv.shape)}"
            )
        n_freq_valid = int(fpv.sum())
        # n_valid-as-count only confines to valid rows if they ARE the prefix
        # [0, n_valid) (C5 corpus-valid prefix). Reject a non-prefix mask
        # rather than silently masking invalid rows below a valid one.
        if not bool(fpv[:n_freq_valid].all()):
            raise ValueError(
                "freq_patch_valid must mark a contiguous prefix [0, n_valid) "
                f"(C5 corpus-valid prefix); got non-prefix {fpv.tolist()}"
            )

    time_band = _sample_axis_bands(
        B, C, axis_len=T_p, n_valid=T_p, frac=a,
        floor=time_band_floor, generator=generator, device=g_dev,
    )  # (B, C, T_p)
    freq_band = _sample_axis_bands(
        B, C, axis_len=F_p, n_valid=n_freq_valid, frac=b,
        floor=freq_band_floor, generator=generator, device=g_dev,
    )  # (B, C, F_p)

    # Union: cell (f, t) masked iff its freq-row OR its time-col is masked.
    mask = freq_band.unsqueeze(-1) | time_band.unsqueeze(-2)  # (B, C, F_p, T_p)
    if device is not None:
        mask = mask.to(device)
    return mask


def sample_m2_mask(
    shape: tuple[int, int, int, int],
    *,
    mask_type: str = "bands",
    held_out_ratio: float = 0.50,
    generator: torch.Generator,
    time_band_floor: int = 2,
    freq_band_floor: int = 1,
    time_freq_split: tuple[float, float] | None = None,
    freq_patch_valid: Tensor | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """M2 front-end mask dispatcher (WS-B3). ``"bands"`` (default, 6/03 lock)
    → :func:`sample_token_band_mask`; ``"random"`` (R-m2-random sister) →
    per-cell Bernoulli at ``held_out_ratio``."""
    if mask_type == "bands":
        return sample_token_band_mask(
            shape, held_out_ratio=held_out_ratio, generator=generator,
            time_band_floor=time_band_floor, freq_band_floor=freq_band_floor,
            time_freq_split=time_freq_split, freq_patch_valid=freq_patch_valid,
            device=device,
        )
    if mask_type == "random":
        return sample_token_mask(
            shape, mask_ratio=held_out_ratio, generator=generator, device=device,
        )
    raise ValueError(f"unknown m2 mask_type={mask_type!r}; expected 'bands' or 'random'")


# ----------------------------------------------------------------------------
# M2 — 2STFT dual-band front-end mask (FE 2STFT, 6/12 lock)
# ----------------------------------------------------------------------------


def _sample_axis_band_multiset(
    B: int,
    C: int,
    *,
    axis_len: int,
    widths: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    """``(B, C, axis_len)`` bool: place ``len(widths)`` NON-overlapping blocks of
    the GIVEN ``widths`` (a multiset; per-``(B, C)`` ORDER and POSITIONS both
    randomized) at random offsets in ``[0, axis_len)``. The realized masked count
    is EXACTLY ``sum(widths)`` — blocks never overlap (touching allowed).

    This is the variable-width sibling of :func:`_sample_axis_bands`: there the
    6/03 lock fixes a single floor and lets ``n_bands`` float; here the 2STFT
    high-band lock (6/12) fixes the block-width MULTISET ({3,3,2} = 8 of 16) and
    so the placement must honor those exact widths rather than a uniform floor.
    """
    widths = tuple(int(w) for w in widths)
    n_blocks = len(widths)
    total = sum(widths)
    if n_blocks == 0 or total <= 0:
        return torch.zeros((B, C, axis_len), dtype=torch.bool, device=device)
    if total > axis_len:
        raise ValueError(
            f"sum(widths)={total} exceeds axis_len={axis_len}: blocks cannot be "
            f"placed non-overlapping (widths={widths})"
        )
    # Random ORDER per (B, C): argsort of uniform noise == a random permutation.
    order = torch.argsort(
        torch.rand((B, C, n_blocks), generator=generator, device=device), dim=-1
    )
    w_all = torch.tensor(widths, dtype=torch.long, device=device)  # (n_blocks,)
    w = w_all[order]  # (B, C, n_blocks) — widths in random per-row order
    prefix = torch.cumsum(w, dim=-1) - w  # (B, C, n_blocks) exclusive cumulative width
    # Random POSITIONS: with t_i = start_i - prefix_i, non-overlap holds iff the
    # t_i are non-decreasing and t_{n-1} ≤ free; so draw n_blocks values in
    # [0, free], sort, and set start_i = t_i + prefix_i. Then
    # start_{i+1} - (start_i + w_i) = t_{i+1} - t_i ≥ 0 (gap) and the last block
    # ends at t_{n-1} + total ≤ free + total = axis_len.
    free = axis_len - total
    t = torch.randint(
        0, free + 1, (B, C, n_blocks), generator=generator, device=device
    )
    t, _ = t.sort(dim=-1)
    starts = t + prefix  # (B, C, n_blocks)
    idx = torch.arange(axis_len, device=device).view(1, 1, 1, axis_len)
    lo = starts.unsqueeze(-1)  # (B, C, n_blocks, 1)
    hi = (starts + w).unsqueeze(-1)  # (B, C, n_blocks, 1)
    band = (idx >= lo) & (idx < hi)  # (B, C, n_blocks, axis_len)
    return band.any(dim=2)  # (B, C, axis_len)


def _sample_axis_anchor_dilate(
    B: int,
    C: int,
    *,
    axis_len: int,
    anchor_frac: float,
    width: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    """``(B, C, axis_len)`` bool: anchor-dilate mask with OVERLAPS allowed (Ben
    2026-06-13 high-band regime). Pick ``round(anchor_frac·axis_len)`` DISTINCT
    anchor positions uniformly at random per ``(B, C)``, then mask the half-open
    block ``[a, a+width)`` at every anchor ``a`` — "mask the position and the
    next ``width-1``". Anchor blocks MAY overlap (two adjacent anchors share a
    cell); the returned mask is their UNION, so the realized masked count is
    DATA-DEPENDENT in ``[n_anchor, min(n_anchor·width, axis_len)]`` (it equals
    ``n_anchor·width`` only when no two blocks touch). A block running off the
    end is clamped by the union (``idx < axis_len``), so an anchor at the last
    position masks just itself ("the next" does not exist).

    The sibling of :func:`_sample_axis_band_multiset` (disjoint, exact count):
    there the high-band lock fixes non-overlapping widths; here the anchor
    regime samples a fixed FRACTION of positions and dilates with overlap, which
    is closer to a SpecAugment time-mask than a tiled block layout. Positions
    are fully randomized per row (independent draw from ``generator``).
    """
    if axis_len <= 0 or anchor_frac <= 0.0 or width <= 0:
        return torch.zeros((B, C, axis_len), dtype=torch.bool, device=device)
    width = min(int(width), axis_len)
    n_anchor = int(round(anchor_frac * axis_len))
    n_anchor = max(1, min(n_anchor, axis_len))
    # DISTINCT anchors per (B, C): argsort of uniform noise is a random
    # permutation; its first n_anchor entries are a uniform random subset.
    perm = torch.argsort(
        torch.rand((B, C, axis_len), generator=generator, device=device), dim=-1
    )
    anchors = perm[..., :n_anchor]                       # (B, C, n_anchor)
    idx = torch.arange(axis_len, device=device).view(1, 1, 1, axis_len)
    lo = anchors.unsqueeze(-1)                            # (B, C, n_anchor, 1)
    # mask [a, a+width); the union (.any) folds in any overlap and clamps the
    # tail at axis_len (idx never reaches axis_len), so no boundary wrap.
    block = (idx >= lo) & (idx < lo + width)             # (B, C, n_anchor, axis_len)
    return block.any(dim=2)                              # (B, C, axis_len)


def sample_m2_dual_band_mask(
    B: int,
    K: int,
    *,
    F_p_low: int,
    T_low_p: int,
    F_p_high: int,
    T_high_p: int,
    low_freq_frac: float = 0.43,
    low_freq_floor: int = 3,
    high_time_widths: tuple[int, ...] = (3, 3, 2),
    high_time_anchor_frac: float | None = None,
    high_time_anchor_width: int | None = None,
    generator: torch.Generator,
    device: torch.device | str | None = None,
) -> Tensor:
    """2STFT M2 front-end mask over the FLAT per-parcel token sequence
    ``(B, K, S)`` with ``S = F_p_low·T_low_p + F_p_high·T_high_p`` (the union tap
    built by ``V14Encoder._forward_dual_band`` — low block first, then high, each
    laid out time-OUTER / freq-INNER). ``True`` == masked.

    The two baskets carry DIFFERENT masking styles (handoff §5):

      * **low** (slow STFT — spectral resolution) — a *freq-block tube*: one
        contiguous freq-block of width ``low_freq_floor`` (default 3 of
        ``F_p_low``=7) masked across ALL ``T_low_p`` time-patches (≈ 3/7).
      * **high** (fast STFT — temporal resolution) — a *time-block tube* across
        ALL ``F_p_high`` freq-patches, in ONE of two styles:
          - DEFAULT (disjoint multiset): blocks of the ``high_time_widths``
            multiset (default {3,3,2} = 8 of 16), non-overlapping.
          - ANCHOR-DILATE (Ben 2026-06-13, opt-in via BOTH
            ``high_time_anchor_frac`` + ``high_time_anchor_width``): sample that
            FRACTION of time positions as anchors, dilate each to ``width`` (mask
            the position and the next ``width-1``), OVERLAPS allowed → union.
            e.g. frac 0.30 + width 2 on T_high_p=80 masks ~50% of time-cols.

    Sampled INDEPENDENTLY PER PARCEL — Ben override 2026-06-12 of the handoff's
    per-electrode wording: the M2 tap is already post-pool, so there is no
    electrode axis to mask; the parcel grid is the finest available, and each
    ``(B, K)`` parcel draws its own placement from ``generator`` (seed-
    reproducible). The low tube contributes ``low_freq_floor·T_low_p`` masked
    tokens and the high tube ``sum(high_time_widths)·F_p_high`` — on the locked
    grid both are 24, the balanced ~24/24 the §6 pooled-mean loss expects.
    """
    g_dev = generator.device

    # --- low band: ONE freq-block (width=floor) across ALL low time-patches ---
    low_freq_band = _sample_axis_bands(
        B, K, axis_len=F_p_low, n_valid=F_p_low, frac=low_freq_frac,
        floor=low_freq_floor, generator=generator, device=g_dev,
    )  # (B, K, F_p_low) — True freq-rows masked at every time
    # Flat low layout is time-OUTER / freq-INNER (i = t·F_p_low + f). Masking ALL
    # time ⇒ broadcast the freq-band over T_low_p, then flatten t-outer/f-inner.
    low_flat = (
        low_freq_band.unsqueeze(2)               # (B, K, 1, F_p_low)
        .expand(B, K, T_low_p, F_p_low)          # (B, K, T_low_p, F_p_low)
        .reshape(B, K, T_low_p * F_p_low)        # (B, K, S_low)
    )

    # --- high band: time-block tube across ALL high freq-patches ---
    # ANCHOR-DILATE (overlaps allowed) when BOTH anchor params are set, else the
    # locked disjoint multiset. Anchor mode is the Ben 2026-06-13 regime: sample
    # `frac` of time positions, dilate each to `width`, union the overlaps.
    if high_time_anchor_frac is not None and high_time_anchor_width is not None:
        high_time_band = _sample_axis_anchor_dilate(
            B, K, axis_len=T_high_p, anchor_frac=high_time_anchor_frac,
            width=high_time_anchor_width, generator=generator, device=g_dev,
        )  # (B, K, T_high_p) — overlap union of dilated anchors
    else:
        high_time_band = _sample_axis_band_multiset(
            B, K, axis_len=T_high_p, widths=high_time_widths,
            generator=generator, device=g_dev,
        )  # (B, K, T_high_p) — True time-cols masked at every freq
    high_flat = (
        high_time_band.unsqueeze(-1)             # (B, K, T_high_p, 1)
        .expand(B, K, T_high_p, F_p_high)        # (B, K, T_high_p, F_p_high)
        .reshape(B, K, T_high_p * F_p_high)      # (B, K, S_high)
    )

    mask = torch.cat([low_flat, high_flat], dim=2)  # (B, K, S)
    if device is not None:
        mask = mask.to(device)
    return mask


# ----------------------------------------------------------------------------
# M4 — P2 parcel×time mask (WS-B4)
# ----------------------------------------------------------------------------


def _electrode_time_drop(support: Tensor, parcel_time_mask: Tensor) -> Tensor:
    """``(B, C, T_p)`` bool: electrode ``c`` is dropped at time ``t`` iff its
    DK parcel is masked there. Derived from the one-hot ``support`` so every
    dropped cell maps to a masked parcel by construction."""
    onehot = (support > 0).to(parcel_time_mask.device).float()  # (B, C, K)
    return (
        torch.einsum("bck,bkt->bct", onehot, parcel_time_mask.float()) > 0
    )


def sample_parcel_tube_mask(
    support: Tensor,
    *,
    n_time_patches: int,
    mask_ratio: float = 0.20,
    n_min_visible: int = 3,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Parcel-tube mask + electrode-time drop (M4 default — WS-B4).

    Masks a uniform-random SUBSET of COVERED parcels, each across ALL ``T_p``
    time-patches (a "tube"). The count is
    ``clamp(round(mask_ratio · N_covered), 1, N_covered - n_min_visible)`` per
    clip → 0 only for N ≤ 3 (can't keep ``n_min_visible``=3 visible and mask
    any); at real coverage N ≥ 14 the clamp is inert → ``round(0.20·N)`` masked.
    Uncovered K-slots are never masked and never targets.

    Returns
    -------
    parcel_time_mask
        ``(B, K, T_p)`` bool — ``True`` == masked target (whole tube per picked
        parcel).
    electrode_time_drop
        ``(B, C, T_p)`` bool — electrode dropped from the visible student input
        (its parcel is masked); for a tube this is an electrode-level drop.
    """
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1]; got {mask_ratio}")
    if support.dim() != 3:
        raise ValueError(f"support must be (B, C, K); got shape {tuple(support.shape)}")
    if n_time_patches < 1:
        raise ValueError(f"n_time_patches must be >= 1; got {n_time_patches}")
    if n_min_visible < 0:
        raise ValueError(f"n_min_visible must be >= 0; got {n_min_visible}")

    B, _C, K = support.shape
    T_p = n_time_patches
    g_dev = generator.device

    covered = (support.sum(dim=1) > 0).to(g_dev)  # (B, K) bool
    n_covered = covered.sum(dim=1)  # (B,) long

    # mask_count = clamp(round(ratio·N), 1, N − n_min_visible); → 0 only when
    # N ≤ n_min_visible (can't mask while keeping n_min_visible visible).
    target = torch.round(mask_ratio * n_covered.float()).long()  # (B,)
    upper = (n_covered - n_min_visible).clamp(min=0)  # (B,)
    mask_count = torch.minimum(target.clamp(min=1), upper)  # (B,)

    # Rank covered parcels by a random key (uncovered → +inf, so they rank last
    # and are never selected since mask_count ≤ N_covered); mask the lowest.
    rand = torch.rand((B, K), generator=generator, device=g_dev)
    rand = rand.masked_fill(~covered, float("inf"))
    rank = rand.argsort(dim=1).argsort(dim=1)  # (B, K) rank of each parcel
    parcel_mask = rank < mask_count.unsqueeze(1)  # (B, K)

    parcel_time_mask = parcel_mask.unsqueeze(-1).expand(B, K, T_p)
    parcel_time_mask = parcel_time_mask.to(support.device)
    drop = _electrode_time_drop(support, parcel_time_mask)
    return parcel_time_mask, drop


def sample_parcel_time_block_mask(
    support: Tensor,
    *,
    n_time_patches: int,
    mask_ratio: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Per-covered-parcel contiguous time-block mask + electrode-time drop —
    the M4 ``"time_block"`` sister (R-time-block; pairs with a ``co_temporal``
    predictor, NEVER ``cross_time`` — see :func:`validate_m4_coupling`).

    For every COVERED parcel a single contiguous time block of length
    ``round(mask_ratio · T_p)`` (≥ 1) is masked at a uniformly random offset;
    uncovered parcels are never masked. Returns the ``(B, K, T_p)`` parcel mask
    and the derived ``(B, C, T_p)`` electrode-time drop.
    """
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1]; got {mask_ratio}")
    if support.dim() != 3:
        raise ValueError(f"support must be (B, C, K); got shape {tuple(support.shape)}")
    if n_time_patches < 1:
        raise ValueError(f"n_time_patches must be >= 1; got {n_time_patches}")

    B, _C, K = support.shape
    T_p = n_time_patches
    g_dev = generator.device

    covered = (support.sum(dim=1) > 0).to(g_dev)  # (B, K) bool
    block_len = min(T_p, max(1, round(mask_ratio * T_p)))
    max_start = T_p - block_len
    if max_start > 0:
        starts = torch.randint(0, max_start + 1, (B, K), generator=generator, device=g_dev)
    else:
        starts = torch.zeros((B, K), dtype=torch.long, device=g_dev)

    t_idx = torch.arange(T_p, device=g_dev).view(1, 1, T_p)
    starts_ = starts.unsqueeze(-1)
    block = (t_idx >= starts_) & (t_idx < starts_ + block_len)  # (B, K, T_p)
    parcel_time_mask = (block & covered.unsqueeze(-1)).to(support.device)
    drop = _electrode_time_drop(support, parcel_time_mask)
    return parcel_time_mask, drop


def sample_m4_mask(
    support: Tensor,
    *,
    n_time_patches: int,
    mask_type: str = "tube",
    mask_ratio: float = 0.20,
    n_min_visible: int = 3,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """M4 parcel mask dispatcher (WS-B4). ``"tube"`` (default, 6/03 lock) →
    :func:`sample_parcel_tube_mask`; ``"time_block"`` (R-time-block sister) →
    :func:`sample_parcel_time_block_mask`; ``"mix"`` is a deferred sister."""
    if mask_type == "tube":
        return sample_parcel_tube_mask(
            support, n_time_patches=n_time_patches, mask_ratio=mask_ratio,
            n_min_visible=n_min_visible, generator=generator,
        )
    if mask_type == "time_block":
        return sample_parcel_time_block_mask(
            support, n_time_patches=n_time_patches, mask_ratio=mask_ratio,
            generator=generator,
        )
    if mask_type == "mix":
        raise NotImplementedError(
            "M4 'mix' (Brain-JEPA tube+time-block with matched predictors) is a "
            "deferred sister — see reports/b36_masking_handoff_2026_06_03.md."
        )
    raise ValueError(f"unknown m4 mask_type={mask_type!r}; expected 'tube' or 'time_block'")


# ----------------------------------------------------------------------------
# B37 composite mask — M2 band ∧ M4 tube in ONE sampler (D7)
# ----------------------------------------------------------------------------


def sample_composite_mask(
    support: Tensor,
    *,
    n_freq_patches: int,
    n_time_patches: int,
    m2_held_out_ratio: float = 0.50,
    m4_mask_ratio: float = 0.20,
    m2_mask_type: str = "bands",
    m4_mask_type: str = "tube",
    m4_n_min_visible: int = 3,
    generator: torch.Generator,
    m2_time_band_floor: int = 2,
    m2_freq_band_floor: int = 1,
    m2_time_freq_split: tuple[float, float] | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """B37 joint composite mask (D7): one masked student forward, deep-supervised
    at M2 (within-parcel bands) AND M4 (whole-parcel tubes).

    The mean-before pool runs the stem PER-PARCEL (D2), so the M2 band mask is
    per-PARCEL — ``(B, K, F_p, T_p)`` — NOT per-electrode. Sampling order:

    1. **tube (M4):** drop a uniform-random subset of COVERED parcels @
       ``m4_mask_ratio`` across ALL time (:func:`sample_m4_mask`). A tubed parcel
       is masked for every ``(f, t)``.
    2. **band (M2):** within each SURVIVING (non-tubed) parcel, drop structured
       1-D spectro-temporal bands @ ``m2_held_out_ratio`` over the ``(F_p, T_p)``
       plane (:func:`sample_m2_mask` over the per-parcel grid). Bands on TUBED
       parcels are zeroed — a tubed parcel is wholly an M4 target, never an M2
       one (the M2 predictor would have no visible context there).

    Effective visible content ≈ ``(1 - m4_mask_ratio) · (1 - m2_held_out_ratio)``
    ≈ 0.80 × 0.50 ≈ **40%** of the field (ratio compounding; lower the ratios if
    the encoder starves — spec §3 caveat). One shared ``generator`` drives both
    draws sequentially, so a fixed seed is fully reproducible.

    Returns
    -------
    token_mask
        ``(B, K, F_p, T_p)`` bool — M2 band targets (``True`` = masked), nonzero
        only on surviving parcels. Fed to the encoder to zero the masked cells
        upstream of the per-parcel token blocks AND scored by the M2 loss.
    parcel_time_mask
        ``(B, K, T_p)`` bool — M4 tube targets (``True`` = masked whole parcel).
        Fed to the encoder to drop the tubed parcels (and exclude them from the
        cross-parcel latent) AND scored by the M4 loss.
    """
    if support.dim() != 3:
        raise ValueError(f"support must be (B, C, K); got shape {tuple(support.shape)}")
    B, _C, K = support.shape
    F_p, T_p = n_freq_patches, n_time_patches

    # 1. tube — which covered parcels are dropped whole (all freq, all time).
    parcel_time_mask, _drop = sample_m4_mask(
        support,
        n_time_patches=T_p,
        mask_type=m4_mask_type,
        mask_ratio=m4_mask_ratio,
        n_min_visible=m4_n_min_visible,
        generator=generator,
    )
    # A tube is whole-time, so ``any`` == ``all`` over t for a tubed parcel.
    tubed = parcel_time_mask.any(dim=-1)  # (B, K) bool
    # SURVIVING = covered AND not tubed: the only parcels the M2 task touches.
    # Uncovered K-slots carry no signal; tubed parcels are wholly M4 targets.
    covered = support.sum(dim=1) > 0  # (B, K) bool
    surviving = covered & ~tubed      # (B, K) bool

    # 2. band — per-parcel spectro-temporal bands over (F_p, T_p), then confine
    #    to SURVIVING parcels (the band sampler is per-row axis-agnostic, so the
    #    parcel axis K substitutes for the electrode axis C cleanly).
    token_mask = sample_m2_mask(
        (B, K, F_p, T_p),
        mask_type=m2_mask_type,
        held_out_ratio=m2_held_out_ratio,
        generator=generator,
        time_band_floor=m2_time_band_floor,
        freq_band_floor=m2_freq_band_floor,
        time_freq_split=m2_time_freq_split,
        device=device,
    )
    token_mask = token_mask & surviving.to(token_mask.device).view(B, K, 1, 1)
    # Return both masks on one device. ``token_mask`` honors ``device=``; the
    # tube (built from ``support``) rides ``support.device`` — mirror it so a
    # caller passing ``device != support.device`` gets a same-device pair.
    return token_mask, parcel_time_mask.to(token_mask.device)


# ----------------------------------------------------------------------------
# Coupling guard (M4 mask shape ↔ predictor scope)
# ----------------------------------------------------------------------------

_M4_VALID_COUPLINGS = frozenset({("tube", "cross_time"), ("time_block", "co_temporal")})


def validate_m4_coupling(mask_type: str, predictor_scope: str) -> None:
    """Raise unless the M4 (mask_type, predictor_scope) pair is shortcut-free.

    Valid: ``tube`` ↔ ``cross_time`` (default), ``time_block`` ↔ ``co_temporal``
    (sister). ``time_block`` + ``cross_time`` is the H1 leak; ``tube`` +
    ``co_temporal`` is over-hard. Call at config/build time.
    """
    if (mask_type, predictor_scope) not in _M4_VALID_COUPLINGS:
        if (mask_type, predictor_scope) == ("time_block", "cross_time"):
            why = "time_block+cross_time = the H1 leak (masked-at-t target is visible at t±1)"
        elif (mask_type, predictor_scope) == ("tube", "co_temporal"):
            why = "tube+co_temporal is over-hard (a whole-time tube leaves nothing co-temporal to attend)"
        else:
            why = "tube must pair with cross_time and time_block with co_temporal"
        raise ValueError(
            f"invalid M4 coupling (mask_type={mask_type!r}, "
            f"predictor_scope={predictor_scope!r}): {why}. "
            f"Allowed: {sorted(_M4_VALID_COUPLINGS)}."
        )


__all__ = [
    "sample_token_mask",
    "sample_token_band_mask",
    "sample_m2_mask",
    "sample_parcel_tube_mask",
    "sample_parcel_time_block_mask",
    "sample_m4_mask",
    "sample_composite_mask",
    "validate_m4_coupling",
]
