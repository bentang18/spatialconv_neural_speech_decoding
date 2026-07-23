"""v14_converged_v3 r4 — flat per-band varlen pack plan (Design B, no L2).

r4 tokens are RAGGED per (contact, band): each valid contact carries SLOW ``T/8`` +
MID ``T/2`` + HGA ``T`` tokens on the shared 32 Hz lattice (stem.PerBandStem). The
L1-only encoder/predictor attend WITHIN a shaft over ALL of that shaft's tokens
(contract project-r4-contract-2026-07-15, option (a): one packed varlen sequence per
shaft, bands mix natively). This module lays those tokens out FLAT, shaft-contiguous,
with per-token RoPE coords + ``cu_seqlens`` — the contract the flat varlen kernels
(``varlen.varlen_block_diag_attention``) already consume. NO L2, NO rectangle.

Token order (shaft-major, then canonical contact within shaft, then band-major
SLOW/MID/HGA, then band-token index) — shaft-contiguous so ``cu_seqlens`` delimits
each (clip, shaft) block. Every array below is in this one flat order.

Compile-safety (same discipline as packing.py): per-band masked COUNTS are static
(``round(frac·T_b)``) and per-shaft spatial masking is balanced (``d_s`` fixed), so
token counts — hence ``cu_seqlens`` VALUES and every packed shape — are per-session
CONSTANTS; only WHICH tokens are masked varies per clip.

MASK SEMANTICS (contract §2):
  * ``masked`` (predictor query) — a token iff its CONTACT is spatially masked OR its
    (band, time) is temporally masked (global per band).
  * ``in_loss`` (margin-gated, #26) — a masked token scored by JEPA iff it is
    leak-safe: a spatially-masked contact has NO visible same-band frame of its OWN
    (leak-proof by construction), else its temporal position must sit >=2 tokens from
    the nearest visible same-band token in its OWN band grid (overlap-factor-2 ⇒
    margin 2 = zero raw-sample overlap, M14).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from speech_decoding.models.v14_converged_v3.geometry import L1Geometry
from speech_decoding.models.v14_converged_v3.masking import V3Masks
from speech_decoding.models.v14_converged_v3.stem import PER_BAND_SPECS

# band axis order (matches stem.PER_BAND_SPECS and PerBandStem.forward): SLOW, MID, HGA.
BAND_STRIDES: tuple[int, ...] = tuple(st for _, st in PER_BAND_SPECS)
N_BANDS = len(PER_BAND_SPECS)
MARGIN = 2  # leak-safe margin in a band's OWN grid (overlap factor 2 ⇒ 2 = no overlap)


@dataclass(frozen=True)
class R4Grid:
    """Clip-INDEPENDENT flat full-grid geometry (one token per valid (contact,band,pos))."""

    depth: Tensor  # (total,) long — clinical contact index per token (L1 index-RoPE)
    time_pos: Tensor  # (total,) long — shared-32Hz lattice position (L1 time-RoPE)
    contact: Tensor  # (total,) long — contact index into N (for gather/scatter)
    band: Tensor  # (total,) long — 0=SLOW 1=MID 2=HGA
    bandpos: Tensor  # (total,) long — band-token index j in [0, T_b)
    shaft: Tensor  # (total,) long — shaft id per token
    cu_seqlens: Tensor  # (S+1,) int32 — per-shaft block bounds, TOKEN units
    cu_seqlens_drop: Tensor  # (S'+1,) int64 — cu minus zero-length shafts (njt)
    max_seqlen: int  # longest shaft block, tokens (static bound)
    total: int  # N * K_full — total tokens (constant per session)
    k_full: int  # tokens per contact = sum_b T_b
    band_lengths: tuple[int, ...]  # (T_slow, T_mid, T_hga)


@dataclass(frozen=True)
class VisiblePack:
    """Per-clip compaction of the VISIBLE (~masked) tokens for the ONLINE encoder.

    The flat block-diagonal kernel has NO key-mask (``varlen.varlen_block_diag_attention``
    attends over every token in a shaft block), so a token is excluded from the encoder's
    attention ONLY by leaving the packed sequence — physical drop. Which tokens survive
    differs per clip, so coords are ``(B, M_vis)``; the per-shaft visible COUNT is a
    per-session constant (contract §2: exact ``d_s`` spatial + exact per-band temporal
    masking), so ``cu_seqlens`` VALUES and ``M_vis`` are constants ⇒ shapes never recompile.
    """

    idx: Tensor  # (B, M_vis) long — full-grid column per packed visible token (grid order)
    depth: Tensor  # (B, M_vis) long — clinical contact index (L1 index-RoPE coord)
    time_pos: Tensor  # (B, M_vis) long — shared-32Hz lattice position (L1 time-RoPE coord)
    parcel: Tensor  # (B, M_vis) long — parcel id per visible token
    cu_seqlens: Tensor  # (S+1,) int32 — per-shaft visible block bounds (static, clip-shared)
    max_seqlen: int  # longest shaft visible block, tokens (static bound)
    m_vis: int  # visible tokens per clip (per-session constant)


def band_token_counts(n_time: int) -> tuple[int, ...]:
    """(T_slow, T_mid, T_hga) = n_time // stride per band."""
    for st in BAND_STRIDES:
        if n_time % st != 0:
            raise ValueError(f"n_time {n_time} not a multiple of band stride {st}")
    return tuple(n_time // st for st in BAND_STRIDES)


def _drop_offsets(cu: Tensor) -> Tensor:
    off = cu.to(torch.int64)
    keep = off[1:] != off[:-1]
    return torch.cat([off[:1], off[1:][keep]])


def build_r4_grid(
    geom: L1Geometry, *, n_time: int, max_seqlen: int | None = None
) -> R4Grid:
    """Build the clip-independent flat full-grid layout for one session.

    ``max_seqlen`` (longest shaft block, tokens) is a per-session CONSTANT; pass the cached
    value to skip its ``.item()`` host sync (the compiled per-step path). ``None`` ⇒ derive
    it here (one sync — the standalone/first-call fallback)."""
    device = geom.gather_idx.device
    S, max_c = geom.n_shafts, geom.max_c
    lengths = band_token_counts(n_time)  # (T_slow, T_mid, T_hga)
    k_full = int(sum(lengths))

    # canonical (shaft-major, slot) contact order.
    rows = torch.arange(S, device=device)[:, None].expand(S, max_c)[geom.valid]  # (N,)
    canon = geom.gather_idx[geom.valid]  # (N,) contact index into N
    depth_canon = geom.depth[geom.valid]  # (N,) clinical depth
    n = int(canon.shape[0])

    # the per-contact (band, bandpos, time_pos) block — identical for every contact.
    band_blk = torch.cat(
        [torch.full((L,), b, dtype=torch.long, device=device) for b, L in enumerate(lengths)]
    )  # (k_full,)
    pos_blk = torch.cat(
        [torch.arange(L, device=device) for L in lengths]
    )  # (k_full,) band-token index
    lat_blk = torch.cat(
        [torch.arange(L, device=device) * st for L, st in zip(lengths, BAND_STRIDES)]
    )  # (k_full,) lattice position = j * stride

    # tile over the N contacts (contact-major ⇒ shaft-contiguous).
    depth = depth_canon.repeat_interleave(k_full)
    contact = canon.repeat_interleave(k_full)
    shaft = rows.repeat_interleave(k_full)
    band = band_blk.repeat(n)
    bandpos = pos_blk.repeat(n)
    time_pos = lat_blk.repeat(n)

    # per-shaft token counts = n_s * k_full.
    n_per_shaft = geom.valid.sum(dim=1).to(torch.long)  # (S,)
    seg = (n_per_shaft * k_full).to(torch.int32)  # (S,)
    cu = torch.zeros(S + 1, dtype=torch.int32, device=device)
    cu[1:] = seg.cumsum(0).to(torch.int32)
    if max_seqlen is None:
        max_seqlen = int((n_per_shaft.max() * k_full).item())

    return R4Grid(
        depth=depth, time_pos=time_pos, contact=contact, band=band, bandpos=bandpos,
        shaft=shaft, cu_seqlens=cu, cu_seqlens_drop=_drop_offsets(cu),
        max_seqlen=max_seqlen, total=n * k_full, k_full=k_full, band_lengths=lengths,
    )


def pack_band_tokens(band_tokens: Sequence[Tensor], grid: R4Grid) -> Tensor:
    """Flatten per-band token tuples into the flat grid token order.

    ``band_tokens`` = (SLOW, MID, HGA) each ``(B, N, T_b, d)`` over the FULL N contacts
    (``stem.PerBandStem.forward`` output). Returns ``(B, total, d)`` in EXACTLY the order
    ``build_r4_grid`` lays out — shaft-major canonical contact, then band-major SLOW/MID/HGA,
    then band-token index — so ``x_flat[:, t]`` is the token whose (contact, band, bandpos)
    is (``grid.contact[t]``, ``grid.band[t]``, ``grid.bandpos[t]``). No copy of the layout
    logic: the canonical contact order is read back off ``grid.contact`` (constant within
    each k_full block), and bands are concatenated in spec order (== ``grid.band_lengths``).

    Band-count agnostic: the expected stream count is ``len(grid.band_lengths)`` — 3 for the
    r4/fine-HGA grids (SLOW, MID, HGA), 1 for the r5 single early-fused band (``build_r5_grid``).
    r4 behaviour is byte-identical (its grids always carry 3 band_lengths)."""
    if len(band_tokens) != len(grid.band_lengths):
        raise ValueError(
            f"expected {len(grid.band_lengths)} band token streams, got {len(band_tokens)}"
        )
    b0 = band_tokens[0]
    B, d = b0.shape[0], b0.shape[-1]
    n = grid.total // grid.k_full
    canon = grid.contact.reshape(n, grid.k_full)[:, 0]  # (n,) contact index, canonical order
    blocks: list[Tensor] = []
    for band, (tok, length) in enumerate(zip(band_tokens, grid.band_lengths)):
        if tok.shape[2] != length:
            raise ValueError(
                f"band {band} has {tok.shape[2]} tokens, grid expects {length}"
            )
        blocks.append(tok[:, canon])  # (B, n, T_b, d) reindexed to canonical contacts
    x = torch.cat(blocks, dim=2)  # (B, n, k_full, d) — per contact [SLOW; MID; HGA]
    return x.reshape(B, grid.total, d)


def build_visible_pack(
    grid: R4Grid, masked: Tensor, parcel_packed: Tensor, *,
    m_vis: int | None = None, max_seqlen: int | None = None,
) -> VisiblePack:
    """Compact each clip's VISIBLE (~masked) tokens into a flat, shaft-contiguous layout.

    Fixed-shape selection (packing.py:124 technique, compile-safe — no ``.nonzero()``):
    visible columns keep their grid index, masked columns get sentinel ``total``; an
    ascending sort floats the ``M_vis`` visible columns to the front in grid order (so the
    packed set stays shaft-contiguous). ``M_vis`` and the longest visible shaft block
    (``max_seqlen``) are per-session constants; pass the cached values to skip their
    ``.item()`` host syncs (the compiled per-step path). ``None`` ⇒ derive here (one sync
    each — the standalone/first-call fallback).
    """
    B, total = masked.shape
    device = masked.device
    visible = ~masked
    if m_vis is None:
        m_vis = int(visible.sum(1).max().item())
    ar = torch.arange(total, device=device)
    key = torch.where(visible, ar[None, :].expand(B, total), total)
    idx = key.sort(dim=1).values[:, :m_vis]  # (B, M_vis) visible columns, grid order
    S = int(grid.cu_seqlens.numel() - 1)
    per_shaft = torch.zeros(B, S, dtype=torch.long, device=device)
    per_shaft.scatter_add_(1, grid.shaft[None, :].expand(B, total), visible.long())
    seg = per_shaft[0].to(torch.int32)  # static across clips (per-session constant)
    cu = torch.zeros(S + 1, dtype=torch.int32, device=device)
    cu[1:] = seg.cumsum(0).to(torch.int32)
    if max_seqlen is None:
        max_seqlen = int(seg.max().item())
    return VisiblePack(
        idx=idx,
        depth=grid.depth[idx],
        time_pos=grid.time_pos[idx],
        parcel=parcel_packed[idx],
        cu_seqlens=cu,
        max_seqlen=max_seqlen,
        m_vis=m_vis,
    )


def scatter_visible(h_vis: Tensor, idx: Tensor, total: int, fill: Tensor) -> Tensor:
    """(B, M_vis, d) visible latents → (B, total, d) full grid; masked slots take ``fill``.

    Inverse of the :func:`build_visible_pack` gather: places ``fill`` (the mask query,
    broadcast over every slot) then overwrites the visible slots with their encoded latents
    at ``idx``. The predictor then runs the FULL grid so the mask query gets its target's
    RoPE/parcel position (JEPA mask-token + positional-embed)."""
    B, m_vis, d = h_vis.shape
    # ``fill`` is the mask-query PARAMETER — it stays fp32 under bf16 autocast (params are not
    # autocast-downcast, like nn.Embedding), whereas ``h_vis`` is the encoder's autocast-bf16
    # output. scatter_ requires self.dtype == src.dtype, so cast the grid to the scatter
    # source's dtype (bf16 under autocast; no-op under fp32) — same pattern as towers' parcel
    # embed .to(x.dtype). Without this the compiled bf16 forward hard-errors on the scatter.
    out = fill.to(h_vis.dtype).expand(B, total, d).clone()
    out.scatter_(1, idx[:, :, None].expand(B, m_vis, d), h_vis)
    return out


def _dist_to_visible(mask: Tensor) -> Tensor:
    """(B, L) bool band-mask → (B, L) int distance to the nearest VISIBLE (False) entry.
    Visible positions get 0. Fully-masked rows get L (treated as leak-safe)."""
    B, L = mask.shape
    device = mask.device
    idx = torch.arange(L, device=device)
    BIG = L + 1
    vis_pos = torch.where(~mask, idx[None, :].expand(B, L), torch.full((B, L), BIG, device=device))
    # nearest visible to the left (running max of seen visible positions) and right.
    left = torch.cummax(torch.where(~mask, idx[None, :].expand(B, L), torch.full((B, L), -BIG, device=device)), dim=1).values
    right = torch.flip(torch.cummin(torch.flip(vis_pos, [1]), dim=1).values, [1])
    dist_left = idx[None, :] - left  # to nearest visible on the left
    dist_right = right - idx[None, :]  # to nearest visible on the right
    dist = torch.minimum(dist_left, dist_right).clamp(min=0)
    return dist.to(torch.long)


def token_flags(
    grid: R4Grid, masks: V3Masks
) -> tuple[Tensor, Tensor]:
    """(masked, in_loss) each (B, total) bool over the flat full grid.

    ``masks.contact_mask`` (B, N) — spatially-masked contacts. ``masks.{slow,mid,hga}_mask``
    (B, T_b) — GLOBAL per-band temporal masks. See MASK SEMANTICS in the module docstring.
    """
    band_masks = (masks.slow_mask, masks.mid_mask, masks.hga_mask)
    B = masks.contact_mask.shape[0]
    device = grid.contact.device

    contact_masked = masks.contact_mask[:, grid.contact]  # (B, total)

    temporal_masked = torch.zeros(B, grid.total, dtype=torch.bool, device=device)
    temporal_in_loss = torch.zeros(B, grid.total, dtype=torch.bool, device=device)
    for b, bm in enumerate(band_masks):
        sel = grid.band == b  # (total,) tokens of band b
        pos = grid.bandpos[sel]  # (n_b,) band-token indices
        temporal_masked[:, sel] = bm[:, pos]
        margin_ok = bm & (_dist_to_visible(bm) >= MARGIN)  # (B, T_b)
        temporal_in_loss[:, sel] = margin_ok[:, pos]

    masked = contact_masked | temporal_masked
    # spatially-masked contact ⇒ leak-proof (no own visible same-band frame); else the
    # temporal margin gate decides.
    in_loss = contact_masked | (temporal_masked & temporal_in_loss)
    return masked, in_loss


# ── r6 (3-band native-rate) PER-SHAFT band masks + MARGIN gate ───────────────────
def token_flags_r6(grid: R4Grid, masks) -> tuple[Tensor, Tensor]:
    """(masked, in_loss) each (B, total) bool for r6's 3-band grid with PER-SHAFT band masks.

    r6 = r4's margin-gated leak-safety (``in_loss`` ⊆ ``masked``, M14 margin 2 — KEPT, unlike
    r5's accept-the-bleed) on r5-style PER-SHAFT band masks. Reuses ``build_r4_grid`` (the grid
    is identical — native bake changes only WHERE bins come from, not token positions). The only
    change from :func:`token_flags`: each band mask is ``(B, S, T_b)`` (per-shaft) not ``(B, T_b)``
    (global), so a token reads its OWN shaft's row (``bm[:, grid.shaft, grid.bandpos]``) and the
    margin gate runs per-shaft (``_dist_to_visible`` over each ``(B·S, T_b)`` row independently).
    Reduces EXACTLY to :func:`token_flags` when every shaft shares one mask (test_pack_r6).
    ``masks`` (``masking.V3MasksR6``): ``contact_mask`` (B, N) + {slow,mid,hga}_mask (B, S, T_b)."""
    band_masks = (masks.slow_mask, masks.mid_mask, masks.hga_mask)  # each (B, S, T_b)
    B = masks.contact_mask.shape[0]
    device = grid.contact.device

    contact_masked = masks.contact_mask[:, grid.contact]  # (B, total)

    temporal_masked = torch.zeros(B, grid.total, dtype=torch.bool, device=device)
    temporal_in_loss = torch.zeros(B, grid.total, dtype=torch.bool, device=device)
    for b, bm in enumerate(band_masks):  # bm (B, S, T_b)
        sel = grid.band == b  # (total,) tokens of band b
        sh = grid.shaft[sel]  # (n_b,) each token's shaft
        pos = grid.bandpos[sel]  # (n_b,) band-token index
        temporal_masked[:, sel] = bm[:, sh, pos]  # (B, n_b) per-shaft mask at token's (shaft,pos)
        s_dim, t_b = bm.shape[1], bm.shape[2]
        flat = bm.reshape(B * s_dim, t_b)  # margin gate PER (row, shaft), each its own grid
        margin_ok = (flat & (_dist_to_visible(flat) >= MARGIN)).reshape(B, s_dim, t_b)
        temporal_in_loss[:, sel] = margin_ok[:, sh, pos]  # (B, n_b)

    masked = contact_masked | temporal_masked
    # spatially-masked contact ⇒ leak-proof (no own visible same-band frame); else margin gate.
    in_loss = contact_masked | (temporal_masked & temporal_in_loss)
    return masked, in_loss


# ── r5 (Chang 2-stream) single-band grid + accept-the-bleed flags ────────────────
R5_STRIDE = 1  # one early-fused band, stride-1 lattice: token index == 32 Hz frame index.


def build_r5_grid(
    geom: L1Geometry, *, n_time: int, max_seqlen: int | None = None
) -> R4Grid:
    """r5 flat full-grid — ONE early-fused band at 32 Hz, stride-1 lattice.

    The r5 stem (``EarlyFusionStem``) emits a SINGLE token stream at 32 Hz (T tokens per
    contact), so the grid is the degenerate one-band case of :func:`build_r4_grid`: band 0
    for every token, ``bandpos == time_pos == token index`` in [0, T), ``k_full == T``. The
    R4Grid contract is unchanged (RoPE coords ``depth``/``time_pos``, ``cu_seqlens``,
    shaft-contiguous order) so pack/pe/encoder are byte-identical to the multi-band path.

    ``max_seqlen`` is a per-session CONSTANT; pass the cached value to skip its ``.item()``
    host sync (compiled path). ``None`` ⇒ derive here (one sync — standalone fallback)."""
    device = geom.gather_idx.device
    S, max_c = geom.n_shafts, geom.max_c
    k_full = int(n_time)

    # canonical (shaft-major, slot) contact order — same as build_r4_grid.
    rows = torch.arange(S, device=device)[:, None].expand(S, max_c)[geom.valid]  # (N,)
    canon = geom.gather_idx[geom.valid]  # (N,) contact index into N
    depth_canon = geom.depth[geom.valid]  # (N,) clinical depth
    n = int(canon.shape[0])

    pos_blk = torch.arange(k_full, device=device)  # (T,) token index == lattice (stride 1)

    depth = depth_canon.repeat_interleave(k_full)
    contact = canon.repeat_interleave(k_full)
    shaft = rows.repeat_interleave(k_full)
    band = torch.zeros(n * k_full, dtype=torch.long, device=device)  # single band 0
    bandpos = pos_blk.repeat(n)
    time_pos = pos_blk.repeat(n)  # stride 1 ⇒ lattice position == token index

    n_per_shaft = geom.valid.sum(dim=1).to(torch.long)  # (S,)
    seg = (n_per_shaft * k_full).to(torch.int32)  # (S,)
    cu = torch.zeros(S + 1, dtype=torch.int32, device=device)
    cu[1:] = seg.cumsum(0).to(torch.int32)
    if max_seqlen is None:
        max_seqlen = int((n_per_shaft.max() * k_full).item())

    return R4Grid(
        depth=depth, time_pos=time_pos, contact=contact, band=band, bandpos=bandpos,
        shaft=shaft, cu_seqlens=cu, cu_seqlens_drop=_drop_offsets(cu),
        max_seqlen=max_seqlen, total=n * k_full, k_full=k_full, band_lengths=(k_full,),
    )


def token_flags_r5(grid: R4Grid, masks) -> tuple[Tensor, Tensor]:
    """(masked, in_loss) each (B, total) bool for the r5 single-band grid.

    ACCEPT THE BLEED (memo project-r5-config-canonical-2026-07-21): ``in_loss == masked`` —
    EVERY masked token is scored, NO margin gate (the M14 margin gate is r4-only and stays
    there). A token is masked iff its CONTACT is spatially masked OR its OWN SHAFT's temporal
    mask covers its ``time_pos``. ``masks`` (``masking.V3MasksR5``): ``contact_mask`` (B, N),
    ``temporal_mask`` (B, S, T) — per-shaft. Returns the same tensor for both flags."""
    contact_masked = masks.contact_mask[:, grid.contact]  # (B, total)
    # per-shaft temporal: pick each token's OWN shaft mask at its 32 Hz index (bandpos).
    temporal_masked = masks.temporal_mask[:, grid.shaft, grid.bandpos]  # (B, total)
    masked = contact_masked | temporal_masked
    return masked, masked


# ── v3r5nf (no-fusion) two stride-1 bands + per-stream accept-the-bleed flags ─────
def build_r5nf_grid(
    geom: L1Geometry, *, n_time: int, time_stride: int = 1, max_seqlen: int | None = None
) -> R4Grid:
    """v3r5nf flat full-grid — TWO bands (0=HGA, 1=LFS), BOTH stride-``time_stride``.

    The v3r5nf stem (``NoFusionStem``) emits TWO token streams (T tokens each per contact), so
    the grid is the two-band case with BOTH bands sharing one lattice: ``band_lengths == (T,
    T)``, ``k_full == 2T``, both HGA and LFS at a given (contact, token) sharing the SAME
    position (band embed the only distinguisher). ``bandpos`` is the token index within band
    (0..T−1) — the temporal-MASK index; ``time_pos = bandpos·time_stride`` is the RoPE lattice
    position. ``time_stride=1`` (v3r5nf, 32 Hz tokens): time_pos == bandpos, byte-identical.
    ``time_stride=2`` (v3r5nffast, 16 Hz tokens): each token sits at its true 32 Hz-lattice
    position (bandpos p → lattice 2p), so RoPE geometry matches the 32 Hz arm (nf_token_geometry).
    Token order per contact is band-major HGA then LFS: ``[HGA 0..T−1, LFS 0..T−1]`` — the same
    shaft-major / canonical-contact / band-major layout as ``build_r4_grid``. The R4Grid contract
    is unchanged so pack/pe/encoder are byte-identical.

    ``max_seqlen`` is a per-session CONSTANT; pass the cached value to skip its ``.item()``
    host sync (compiled path). ``None`` ⇒ derive here (one sync — standalone fallback)."""
    device = geom.gather_idx.device
    S, max_c = geom.n_shafts, geom.max_c
    T = int(n_time)
    lengths = (T, T)  # (HGA, LFS) both stride-1 @32 Hz
    k_full = 2 * T

    # canonical (shaft-major, slot) contact order — same as build_r4_grid.
    rows = torch.arange(S, device=device)[:, None].expand(S, max_c)[geom.valid]  # (N,)
    canon = geom.gather_idx[geom.valid]  # (N,) contact index into N
    depth_canon = geom.depth[geom.valid]  # (N,) clinical depth
    n = int(canon.shape[0])

    # per-contact block: band-major [HGA 0..T-1, LFS 0..T-1], both stride 1.
    band_blk = torch.cat(
        [torch.full((T,), b, dtype=torch.long, device=device) for b in range(2)]
    )  # (2T,)
    pos_blk = torch.cat([torch.arange(T, device=device) for _ in range(2)])  # (2T,)

    depth = depth_canon.repeat_interleave(k_full)
    contact = canon.repeat_interleave(k_full)
    shaft = rows.repeat_interleave(k_full)
    band = band_blk.repeat(n)
    bandpos = pos_blk.repeat(n)  # token index within band (0..T-1) — the temporal-MASK index
    time_pos = pos_blk.repeat(n) * time_stride  # RoPE lattice position (bandpos·stride)

    n_per_shaft = geom.valid.sum(dim=1).to(torch.long)  # (S,)
    seg = (n_per_shaft * k_full).to(torch.int32)  # (S,)
    cu = torch.zeros(S + 1, dtype=torch.int32, device=device)
    cu[1:] = seg.cumsum(0).to(torch.int32)
    if max_seqlen is None:
        max_seqlen = int((n_per_shaft.max() * k_full).item())

    return R4Grid(
        depth=depth, time_pos=time_pos, contact=contact, band=band, bandpos=bandpos,
        shaft=shaft, cu_seqlens=cu, cu_seqlens_drop=_drop_offsets(cu),
        max_seqlen=max_seqlen, total=n * k_full, k_full=k_full, band_lengths=lengths,
    )


def token_flags_r5nf(grid: R4Grid, masks) -> tuple[Tensor, Tensor]:
    """(masked, in_loss) each (B, total) bool for the v3r5nf two-band grid.

    ACCEPT THE BLEED (``in_loss == masked``, no margin gate — the M14 gate is r4-only). Each
    stream masked INDEPENDENTLY: a token is masked iff its CONTACT is spatially masked OR its
    OWN SHAFT's temporal mask covers its ``bandpos``, using that token's stream's fields.
    ``masks`` (``masking.V3MasksR5NF``): {hga,lfs}_contact_mask (B, N), {hga,lfs}_temporal_mask
    (B, S, T). Compile-safe per-stream select over the flat grid (no data-dependent gather):
    compute BOTH streams then ``where`` by band. Returns the same tensor for both flags."""
    hga_c = masks.hga_contact_mask[:, grid.contact]  # (B, total)
    lfs_c = masks.lfs_contact_mask[:, grid.contact]  # (B, total)
    hga_t = masks.hga_temporal_mask[:, grid.shaft, grid.bandpos]  # (B, total)
    lfs_t = masks.lfs_temporal_mask[:, grid.shaft, grid.bandpos]  # (B, total)
    is_hga = (grid.band == 0)[None]  # (1, total)
    contact_masked = torch.where(is_hga, hga_c, lfs_c)
    temporal_masked = torch.where(is_hga, hga_t, lfs_t)
    masked = contact_masked | temporal_masked
    return masked, masked
