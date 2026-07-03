"""Converged-architecture v2 — 2-band magnitude frontend + set-pool SSL.

NEW module (the impl plan's locked decision 1): old :mod:`v14_converged` is
untouched (still runs 3STFT). v2 re-derives BANDS / tokenizer / pool / latent /
predictors / masks / loss FRESH against the design memos and reuses only LEAF
primitives from :mod:`v14_encoder` by import. This file (P2.1) scaffolds the
band geometry + token layout; stems / set-pool / latent / predictors / masks /
loss land in P2.2+.

Frontend (memory project_frontend_chang_2band_hga_lfs_2026_06_25, fs=2048):
  LFS  N=1024 hop=512  2–56 Hz  |STFT| 28 bins → 3 LOG freq-patches 4/8/16 bins
       (2-8 / 10-24 / 26-56 Hz), tk=2 → 6 tok/1s, 30/5s
  HGA  N=128  hop=64   64–160 Hz |STFT|  7 bins → 1 fold freq-patch (7→1), tk=2
       → 16 tok/1s, 80/5s
Totals 22 tok/1s, 110/5s (per electrode). Both MAGNITUDE (in_channels=1; no
phase, no beta). RoPE shares one physical-time clock; LFS:HGA stride = 1024:128
= 8:1 (500 ms : 62.5 ms). 4 freq-patches total (1 HGA + 3 LFS) carry the freq
identity (4 freq-embeds at the frontend; the pool's K/V read-operator typing is
``pool_op``: 1 shared / 2 band-tied / 4 per-patch — see ``cell_operator_index``).

Unlike :class:`v14_converged.BandSpec` (uniform ``kernel_freq``), LFS's 3 log
groups have DIFFERENT bin counts (4/8/16) → a single ``fk`` cannot express them.
:class:`BandSpecV2` therefore carries explicit per-patch bin counts.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from speech_decoding.models.v14_encoder import (
    NEG_INF_MASK_VALUE,
    _JointTokenBlock,
    _PatchStem,
    _rope_freqs,
)

FS_HZ: int = 2048


@dataclass(frozen=True)
class BandSpecV2:
    """One band of the converged-v2 2-band magnitude frontend.

    ``freq_patch_bins`` lists the in-band rfft bins folded into each freq-patch
    (HGA ``(7,)`` = one 7→1 fold; LFS ``(4, 8, 16)`` = three log groups). The sum
    is the band's ``torch.stft`` freq-bin count the cache stores. ``n_time_frames``
    is the in-band frame count over the clip (``center=True`` → ``1 + n_samples //
    hop``); ``kernel_time`` (= tk = 2) sets the non-overlapping time-patch stride.
    Magnitude only ⇒ ``in_channels = 1``.
    """

    name: str
    nperseg: int                       # N
    hop: int                           # N // 2
    freq_patch_bins: tuple[int, ...]   # rfft bins per freq-patch (HGA (7,); LFS (4,8,16))
    n_time_frames: int                 # stft frames over the clip (center=True)
    kernel_time: int = 2               # tk
    in_channels: int = 1               # magnitude

    @property
    def n_freq_bins(self) -> int:
        return sum(self.freq_patch_bins)

    @property
    def n_freq_patches(self) -> int:
        return len(self.freq_patch_bins)

    @property
    def n_time_patches(self) -> int:
        return (self.n_time_frames - self.kernel_time) // self.kernel_time + 1

    @property
    def n_tokens(self) -> int:
        return self.n_freq_patches * self.n_time_patches

    @property
    def time_patch_stride_samples(self) -> int:
        # Non-overlapping patches advance by tk·hop samples (= N at hop = N/2).
        return self.kernel_time * self.hop


# The locked 2-band ladder (frontend memo). n_time_frames encodes the 1 s clip;
# bands_for_clip_len() retimes the TIME axis for the 5 s SSL clip.
LFS = BandSpecV2("lfs", 1024, 512, freq_patch_bins=(4, 8, 16), n_time_frames=5)
HGA = BandSpecV2("hga", 128, 64, freq_patch_bins=(7,), n_time_frames=33)
BANDS_V2: tuple[BandSpecV2, ...] = (LFS, HGA)  # low → high concat order

N_TOKENS_V2: int = sum(b.n_tokens for b in BANDS_V2)              # 6 + 16 = 22 @1s
N_FREQ_PATCHES_V2: int = sum(b.n_freq_patches for b in BANDS_V2)  # 3 + 1 = 4


def bands_for_clip_len(
    clip_len_s: float, fs: int = FS_HZ, base: tuple[BandSpecV2, ...] = BANDS_V2,
) -> tuple[BandSpecV2, ...]:
    """The locked 2-band ladder retimed for a ``clip_len_s`` clip.

    SSL pretrains on 5 s (110 tokens); 1 s eval = 22 (the ``BANDS_V2`` constants).
    Only ``n_time_frames`` is clip-length-dependent — ``1 + n_samples // hop`` for
    ``center=True`` — so token order, freq grid, and the RoPE clock are preserved.
    5 s ⇒ LFS 21 / HGA 161 frames → 10 / 80 time-patches → 30 + 80 = 110 tokens."""
    n_samples = int(round(clip_len_s * fs))
    out: list[BandSpecV2] = []
    for b in base:
        n_frames = 1 + n_samples // b.hop
        if n_frames < b.kernel_time:
            raise ValueError(
                f"clip_len_s={clip_len_s}s gives band {b.name!r} only {n_frames} "
                f"stft frames (< kernel_time {b.kernel_time}); clip too short"
            )
        out.append(replace(b, n_time_frames=n_frames))
    return tuple(out)


def band_slot_mults(bands: tuple[BandSpecV2, ...] = BANDS_V2) -> list[int]:
    """Per-band RoPE-clock multiplier = time-patch stride ÷ finest stride.

    The shared clock unit is the finest band's stride (HGA, 128 samples = 62.5 ms).
    Every band's stride must be an integer multiple, else the bands cannot share
    one RoPE clock. Locked 2-band ⇒ LFS 1024 / HGA 128 = 8 : 1. Raises if not."""
    min_stride = min(b.time_patch_stride_samples for b in bands)
    mults: list[int] = []
    for b in bands:
        mult, rem = divmod(b.time_patch_stride_samples, min_stride)
        if rem != 0:
            raise ValueError(
                f"band {b.name!r} time-patch stride {b.time_patch_stride_samples} "
                f"is not an integer multiple of the finest stride {min_stride}; "
                f"the bands cannot share one RoPE clock"
            )
        mults.append(mult)
    return mults


def token_metadata(
    bands: tuple[BandSpecV2, ...] = BANDS_V2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Geometry-fixed per-token ``(band_id, freq_patch_id, time_slot)`` longs.

    Single source for the tokenizer, the 4 frontend freq-embeds, the pool's
    freq-specific weight select, and the latent. Token order: bands concat
    LFS→HGA, each flattened ``(F_p, T_p)`` row-major (freq-patch-major,
    time-minor). ``freq_patch_id ∈ [0, 4)`` indexes the 4 freq-patches
    (LFS 0/1/2 + HGA 3); ``time_slot = mult · time_patch_idx`` puts both bands
    on the shared HGA-stride clock."""
    mults = band_slot_mults(bands)
    band_id: list[int] = []
    freq_patch_id: list[int] = []
    time_slot: list[int] = []
    patch_base = 0
    for bi, (b, mult) in enumerate(zip(bands, mults)):
        for fp in range(b.n_freq_patches):
            for tp in range(b.n_time_patches):
                band_id.append(bi)
                freq_patch_id.append(patch_base + fp)
                time_slot.append(mult * tp)
        patch_base += b.n_freq_patches
    return (
        torch.tensor(band_id, dtype=torch.long),
        torch.tensor(freq_patch_id, dtype=torch.long),
        torch.tensor(time_slot, dtype=torch.long),
    )


def _freq_patch_slices(band: BandSpecV2) -> list[tuple[int, int]]:
    """Per-freq-patch ``(lo, hi)`` bin slices over the band's freq axis.

    HGA ``(7,)`` → ``[(0, 7)]`` (one 7→1 fold); LFS ``(4, 8, 16)`` →
    ``[(0, 4), (4, 12), (12, 28)]`` (the 3 log groups). Each slice feeds its own
    ``_PatchStem`` with ``kernel_freq = hi − lo`` ⇒ each folds to exactly 1 patch."""
    slices: list[tuple[int, int]] = []
    lo = 0
    for g in band.freq_patch_bins:
        slices.append((lo, lo + g))
        lo += g
    return slices


class TwoBandTokenizerV2(nn.Module):
    """v2 per-electrode frontend tokenizer (Stage 1) — 2-band magnitude.

    One ``_PatchStem`` per freq-patch GROUP (NOT per band): HGA = 1 stem
    (``kernel_freq=7`` fold), LFS = 3 stems (``kernel_freq=4/8/16`` over the log
    groups). The uniform-fk :class:`v14_converged.ThreeBandTokenizer` can't
    express LFS's non-uniform groups, so each group gets its own stem over its
    bin slice; every stem folds its bins to 1 freq-patch and strides time by
    ``kernel_time=2``. Electrodes ride the batch dim — isolated, no cross-electrode
    mixing.

    Forward (magnitude, ``in_channels=1``; whole-session robust-z applied at load):
      - ``lfs``: ``(B, C, 28, T_lfs)`` ``|STFT|``
      - ``hga``: ``(B, C, 7,  T_hga)`` ``|STFT|``
    Output ``tokens`` ``(B, C, S, d)`` in band-then-(freq-patch-major, time) order
    (LFS g0/g1/g2 then HGA). Geometry metadata exposed as buffers ``band_id``,
    ``freq_patch_id`` ∈ [0, 4), ``time_slot`` (shared HGA-stride clock).
    """

    def __init__(self, d_model: int, bands: tuple[BandSpecV2, ...] = BANDS_V2) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.d_model = d_model
        self.bands = bands
        band_slot_mults(bands)  # assert the bands share one RoPE clock

        # One stem per freq-patch group, flattened in token order (band-major,
        # freq-patch-major). _stem_band / _stem_slice carry which band + bin slice
        # each stem reads — plain python lists (geometry-fixed, not parameters).
        self.stems = nn.ModuleList()
        self._stem_band: list[int] = []
        self._stem_slice: list[tuple[int, int]] = []
        self._stem_name: list[str] = []
        for bi, b in enumerate(bands):
            for gi, (lo, hi) in enumerate(_freq_patch_slices(b)):
                self.stems.append(
                    _PatchStem(
                        d_model,
                        kernel_freq=hi - lo,
                        kernel_time=b.kernel_time,
                        in_channels=b.in_channels,
                    )
                )
                self._stem_band.append(bi)
                self._stem_slice.append((lo, hi))
                self._stem_name.append(f"{b.name}_fp{gi}")

        band_id, freq_patch_id, time_slot = token_metadata(bands)
        self.register_buffer("band_id", band_id, persistent=False)
        self.register_buffer("freq_patch_id", freq_patch_id, persistent=False)
        self.register_buffer("time_slot", time_slot, persistent=False)

        # Per-stem mean per-token L2 norm (detached monitor): with 4 separate
        # stems, one running hot would silently dominate the additive latent.
        self.last_band_token_norm: dict[str, Tensor] = {}

    def forward(self, lfs: Tensor, hga: Tensor) -> Tensor:
        """``(B,C,28,T_lfs) / (B,C,7,T_hga)`` → tokens ``(B, C, S, d)``."""
        band_inputs = (lfs, hga)
        for bi, b in enumerate(self.bands):
            F = band_inputs[bi].shape[-2]
            if F != b.n_freq_bins:
                raise ValueError(
                    f"band {b.name!r} input has {F} freq bins, expected "
                    f"{b.n_freq_bins}"
                )
        per_patch: list[Tensor] = []
        norms: dict[str, Tensor] = {}
        for stem, bi, (lo, hi), name in zip(
            self.stems, self._stem_band, self._stem_slice, self._stem_name
        ):
            x = band_inputs[bi][:, :, lo:hi, :]              # (B, C, g, T)
            out = stem(x)                                    # (B, C, 1, T_p, d)
            B, C, F_p, T_p, d = out.shape
            if F_p != 1:
                raise ValueError(
                    f"stem {name!r} produced {F_p} freq-patches, expected 1 "
                    f"(kernel_freq must fold its bin group to one patch)"
                )
            norms[name] = out.detach().norm(dim=-1).mean()
            per_patch.append(out.reshape(B, C, T_p, d))      # (B, C, T_p, d)
        self.last_band_token_norm = norms
        return torch.cat(per_patch, dim=2)                   # (B, C, S, d)


class FrontendEncoderV2(nn.Module):
    """Stage 1 (converged-v2): per-electrode ISOLATED 2-band frontend transformer.

    Tokenize → add the learned freq-patch tag (4 patches) → ``n_layers`` joint
    freq×time self-attention blocks (RoPE on the shared HGA-stride physical-time
    clock) → LayerNorm. Electrodes ride the batch dim throughout: no
    cross-electrode pathway (Stage-1 isolation). ``key_mask`` ``(B, C, S)`` bool
    (``True`` = attendable) gives the student a leak-free M2-visibility forward
    identical to physically dropping the masked cells, but batchable.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        *,
        bands: tuple[BandSpecV2, ...] = BANDS_V2,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        head_dim = d_model // n_heads
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        self.tokenizer = TwoBandTokenizerV2(d_model, bands)
        n_fp = sum(b.n_freq_patches for b in bands)

        self.freq_embed = nn.Parameter(torch.empty(n_fp, d_model))
        nn.init.trunc_normal_(self.freq_embed, std=0.02)

        n_slots = int(self.tokenizer.time_slot.max().item()) + 1
        self.register_buffer("key_rope", _rope_freqs(head_dim, n_slots), persistent=False)

        self.blocks = nn.ModuleList(
            [_JointTokenBlock(d_model, n_heads, qk_norm=qk_norm) for _ in range(n_layers)]
        )
        self.ln_out = nn.LayerNorm(d_model)

    def encode_tokens(
        self,
        tok: Tensor,
        freq_patch_id: Tensor,
        time_slot: Tensor,
        *,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        """Run +freq-embed → blocks → LN over pre-tokenized cells ``(B,C,L,d)``.

        ``freq_patch_id``/``time_slot`` index the freq-embed + RoPE tables and may
        be the canonical ``(S,)`` (shared across electrodes — the dense path) OR
        per-electrode ``(B,C,L)`` (the STATIC path, where parcel-uniform M2 leaves a
        DIFFERENT visible-cell set per parcel ⇒ per-row gather + per-row RoPE). This
        split is what lets the packed static forward reuse the exact dense block
        stack — the basis of the dense==static equivalence."""
        B, C, L, d = tok.shape
        x = tok + self.freq_embed[freq_patch_id]                 # (B,C,L,d)
        if time_slot.dim() == 1:
            rope = self.key_rope[:, time_slot, :]                # (2, L, head_dim) shared
        else:
            rs = self.key_rope[:, time_slot, :]                  # (2, B, C, L, head_dim)
            rope = rs.reshape(2, B * C, L, rs.shape[-1])         # per-row
        x = x.reshape(B * C, L, d)
        km = None if key_mask is None else key_mask.reshape(B * C, L)
        for blk in self.blocks:
            x = blk(x, rope, km)
        return self.ln_out(x).reshape(B, C, L, d)

    def forward(
        self,
        lfs: Tensor,
        hga: Tensor,
        *,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        """``(B,C,28,T_lfs)/(B,C,7,T_hga)`` → per-electrode features ``(B, C, S, d)``."""
        tok = self.tokenizer(lfs, hga)                            # (B, C, S, d)
        return self.encode_tokens(
            tok, self.tokenizer.freq_patch_id, self.tokenizer.time_slot, key_mask=key_mask
        )


@dataclass(frozen=True)
class M2MaskConfigV2:
    """Parcel-uniform M2 mask config (frontend memo, locked 2026-06-26).

    HGA = wav2vec2 FILL-NOT-TRIM time-spans, exact ``n_mask_hga = round(frac_hga·
    T_hga)`` cells (``frac_hga=0.50`` ⇒ 8 @1s / 40 @5s). LFS = freq-tube, fixed 1
    of 3 log groups (uniform-random which) ⇒ exactly ``T_lfs`` cells. Total
    ``n_mask = n_mask_hga + T_lfs`` constant per parcel (the static invariant).
    ``hg_start_rate`` is vestigial under fill-not-trim (we fill to the exact
    target from all candidate starts) — kept only as documentation of the span
    regime; ``hg_span`` (the granularity) is load-bearing."""

    frac_hga: float = 0.50
    hg_span: int = 3
    n_lfs_tube_groups: int = 1


def n_mask_hga(band_hga: BandSpecV2, cfg: M2MaskConfigV2 = M2MaskConfigV2()) -> int:
    return round(cfg.frac_hga * band_hga.n_time_patches)


def _hga_fill_not_trim(
    R: int, T: int, target: int, span: int, generator: torch.Generator
) -> Tensor:
    """Vectorized wav2vec2 fill-not-trim over ``R`` rows → ``(R, T)`` bool, exactly
    ``target`` True per row, span-granular with the LAST span tail-trimmed.

    NO sequential loop. Each cell's COVER-RANK = the min random rank among the
    spans that cover it (a cell is covered by start ``s`` iff ``s ≤ t < s+span``).
    Adding spans in random order until the union hits ``target`` and trimming the
    last span's tail ≡ selecting the ``target`` cells with the smallest
    ``(cover_rank, time)`` — cells of fully-added spans (small rank) first, then a
    time-ordered prefix of the last span. One argsort, no scan over spans."""
    if target > T:
        raise ValueError(f"n_mask_hga target {target} exceeds T_hga {T}")
    if target == 0:
        return torch.zeros(R, T, dtype=torch.bool)
    allowed = T - span + 1
    if allowed < 1:
        raise ValueError(f"hg_span {span} exceeds T_hga {T}")
    # Random rank of each candidate start (double argsort → integer 0..allowed-1).
    order_rank = torch.rand(R, allowed, generator=generator).argsort(dim=1).argsort(dim=1)
    s = torch.arange(allowed)[:, None]
    t = torch.arange(T)[None, :]
    cover = (s <= t) & (t < s + span)                          # (allowed, T) bool
    big = allowed + 1
    ranks = torch.where(cover[None], order_rank[:, :, None], big)  # (R, allowed, T)
    cover_rank = ranks.min(dim=1).values                       # (R, T) — every cell covered
    composite = cover_rank * T + torch.arange(T)               # (cover_rank, time) lexicographic
    sel = composite.argsort(dim=1)[:, :target]                 # (R, target) smallest
    mask = torch.zeros(R, T, dtype=torch.bool)
    mask.scatter_(1, sel, True)
    return mask


def _sample_m2_rows(
    R: int,
    bands: tuple[BandSpecV2, ...],
    generator: torch.Generator,
    cfg: M2MaskConfigV2,
) -> Tensor:
    """Core M2 sampler over ``R`` independent rows → ``(R, S)`` bool, constant count
    ``n_mask = n_mask_hga + T_lfs`` per row. Row = one parcel (uniform) or one
    electrode (heterogeneous) — the caller reshapes. Token order matches
    :func:`token_metadata`: LFS groups 0/1/2 (each ``T_lfs``) then HGA (``T_hga``)."""
    lfs, hga = bands
    if (lfs.name, hga.name) != ("lfs", "hga"):
        raise ValueError(f"expected (lfs, hga) bands, got {(lfs.name, hga.name)}")
    T_lfs = lfs.n_time_patches
    n_lfs_groups = lfs.n_freq_patches
    target_hga = n_mask_hga(hga, cfg)

    # LFS freq-tube: pick cfg.n_lfs_tube_groups of the 3 groups, mask all their time.
    grp = torch.rand(R, n_lfs_groups, generator=generator).argsort(dim=1)[
        :, : cfg.n_lfs_tube_groups
    ]                                                          # (R, n_tube_groups)
    lfs_mask = torch.zeros(R, n_lfs_groups, T_lfs, dtype=torch.bool)
    lfs_mask.scatter_(1, grp[:, :, None].expand(-1, -1, T_lfs), True)
    lfs_mask = lfs_mask.reshape(R, n_lfs_groups * T_lfs)       # (R, n_lfs cells)

    hga_mask = _hga_fill_not_trim(R, hga.n_time_patches, target_hga, cfg.hg_span, generator)
    return torch.cat([lfs_mask, hga_mask], dim=1)             # (R, S)


def sample_m2_masks_v2(
    B: int,
    P: int,
    bands: tuple[BandSpecV2, ...],
    generator: torch.Generator,
    cfg: M2MaskConfigV2 = M2MaskConfigV2(),
) -> Tensor:
    """Parcel-uniform M2 masks ``(B, P, S)`` bool (True = held out / an M2 target).

    Constant count ``n_mask = n_mask_hga + T_lfs`` per parcel. VECTORIZED over all
    ``B·P`` parcels — no per-parcel/per-electrode python loop."""
    return _sample_m2_rows(B * P, bands, generator, cfg).reshape(B, P, -1)


def sample_m2_masks_hetero_v2(
    B: int,
    C: int,
    bands: tuple[BandSpecV2, ...],
    generator: torch.Generator,
    cfg: M2MaskConfigV2 = M2MaskConfigV2(),
) -> Tensor:
    """Run-B heterogeneous per-ELECTRODE M2 masks ``(B, C, S)`` bool — same constant
    count per row as the parcel-uniform sampler, but each electrode masks its OWN
    cells independently. Within a parcel, electrodes now disagree on which cells are
    held out, so the pool must be told to skip masked electrode-cells (its ``block``
    arg) or the zero-filled held-out token leaks into the seeds. VECTORIZED over
    ``B·C``. Membership is NOT consulted — padded electrodes get masks too; the
    caller intersects with the voltage/support mask when it applies the block."""
    return _sample_m2_rows(B * C, bands, generator, cfg).reshape(B, C, -1)


def sample_parcel_tube_v2(
    B: int, P: int, tube_ratio: float, generator: torch.Generator
) -> Tensor:
    """Per-clip tubed-parcel mask ``(B, P)`` bool, constant count ``n_tube`` per
    clip. ``n_tube = round(tube_ratio·P)`` clamped to ``[1, P−1]`` (≥1 tubed for an
    M4 target, ≥1 untubed for context). VECTORIZED over B (argsort-of-rand =
    per-row randperm). A 1-parcel session tubes nothing (M4 inert)."""
    if P < 2:
        return torch.zeros(B, P, dtype=torch.bool)
    n_tube = max(1, min(round(tube_ratio * P), P - 1))
    idx = torch.rand(B, P, generator=generator).argsort(dim=1)[:, :n_tube]  # (B, n_tube)
    tube = torch.zeros(B, P, dtype=torch.bool)
    tube.scatter_(1, idx, True)
    return tube


def electrode_drop_count(membership: Tensor, drop_frac: float, min_keep: int) -> Tensor:
    """Per-parcel whole-electrode drop count ``n_drop(p)`` ``(P,)`` long (Run-B).

    ``n_drop = clamp(round(drop_frac·n_elec), 0, max(n_elec − min_keep, 0))`` per
    parcel — a survivor FLOOR: at least ``min_keep`` electrodes always stay visible,
    so a parcel with ``n_elec ≤ min_keep`` is EXEMPT (drops nothing, serves as pure
    visible context). Deterministic in ``n_elec`` ⇒ constant per parcel across clips
    (only WHICH electrodes drop varies) ⇒ the static-shape contract holds: the total
    dropped-per-clip ``Σ_p n_drop(p)`` is clip-invariant."""
    if not 0.0 <= drop_frac <= 1.0:
        raise ValueError(f"drop_frac must be in [0,1], got {drop_frac}")
    if min_keep < 1:
        raise ValueError(f"min_keep must be ≥1, got {min_keep}")
    n_elec = membership.sum(1)                                    # (P,) long
    upper = (n_elec - min_keep).clamp(min=0)                      # survivor floor
    raw = torch.round(drop_frac * n_elec.to(torch.float32)).long()
    return raw.clamp(min=0).minimum(upper)                        # (P,)


def _empty_cell_mask(pool_block: Tensor, membership: Tensor) -> Tensor:
    """``(B,P,S)`` bool — a parcel-cell is EMPTY when EVERY member electrode is
    blocked there (Run-B hetero: dropped or M2-masked), so the pool produces only
    the finite-sentinel null. Threaded as a key-block up the stack (latent + M4 —
    a null no live token attends is a JEPA *drop*, not a mask token) and as a 0/1
    melec loss weight. ``pool_block`` ``(B,S,C)`` True = electrode blocked at cell;
    ``membership`` ``(P,C)`` bool."""
    vis = (~pool_block).to(torch.float32)                        # (B,S,C) visible
    cnt = torch.einsum("bsc,pc->bsp", vis, membership.to(torch.float32))
    return (cnt < 0.5).permute(0, 2, 1).contiguous()             # (B,P,S)


def sample_electrode_drop_v2(
    B: int, membership: Tensor, drop_frac: float, min_keep: int, generator: torch.Generator
) -> Tensor:
    """Per-clip whole-electrode drop mask ``(B, C)`` bool (True = dropped = an
    M-elec reconstruction target for the Run-B pool head).

    Each parcel drops EXACTLY ``electrode_drop_count(p)`` of its own electrodes,
    chosen uniformly at random per clip (constant count ⇒ static shapes). Dropped
    electrodes are always parcel members; membership is a partition so the per-parcel
    selections are disjoint. VECTORIZED over ``B`` and ``P`` (argsort-of-rand =
    per-(clip,parcel) randperm), no python loop over parcels/electrodes."""
    P, C = membership.shape
    ndrop = electrode_drop_count(membership, drop_frac, min_keep)  # (P,)
    # Rank each parcel's members by a per-(clip,parcel) random key; non-members sort
    # last (+inf) so they are never among the first ``ndrop`` picks.
    scores = torch.rand(B, P, C, generator=generator).masked_fill(
        ~membership[None], float("inf")
    )
    rank = scores.argsort(dim=2).argsort(dim=2)                    # (B,P,C) 0-based rank
    drop_pp = rank < ndrop[None, :, None]                          # (B,P,C) first ndrop members
    return (drop_pp & membership[None]).any(dim=1)                 # (B,C) partition ⇒ exact


@dataclass(frozen=True)
class StaticShapesV2:
    """Per-rank static shape header for one converged-v2 step. All fields are
    session-constant (fixed by ``--same-session-ranks``); per-clip variation is
    ONLY *which* parcels tube + *which* cells mask — constant-count, pure gathers.
    The derived gather lengths size every downstream flatten/scatter."""

    b: int
    c: int
    P: int
    n_p: tuple[int, ...]  # electrodes per parcel, len P, Σ = c
    k: int
    S: int
    n_mask: int
    s_vis: int
    n_tube: int
    P_vis: int

    @property
    def latent_student(self) -> int:
        return self.P_vis * self.k * self.s_vis

    @property
    def latent_teacher(self) -> int:
        return self.P * self.k * self.S

    @property
    def m2_q(self) -> int:
        return self.b * self.c * self.n_mask

    @property
    def m4_q(self) -> int:
        # LEGACY M4 query count: {tubed parcels: all S cells} ∪ {untubed: the
        # n_mask masked cells}, ×k seeds. (Run-A M4 is tubed-only ⇒ ``m4_q_tubed``.)
        return self.b * (self.n_tube * self.k * self.S + self.P_vis * self.k * self.n_mask)

    @property
    def m4_q_tubed(self) -> int:
        # Run-A M4 query count: tubed parcels only, all S cells, ×k seeds.
        return self.b * self.n_tube * self.k * self.S

    @property
    def m3_q(self) -> int:
        # Run-A M3 query count: untubed parcels' n_mask M2-masked cells, ×k seeds.
        return self.b * self.P_vis * self.k * self.n_mask


def compute_static_shapes_v2(
    m2_mask: Tensor,
    tube_mask: Tensor,
    membership: Tensor,
    bands: tuple[BandSpecV2, ...],
    *,
    k: int,
    cfg: M2MaskConfigV2 = M2MaskConfigV2(),
) -> StaticShapesV2:
    """Build + VALIDATE the static header. FAILS LOUD if the masks are not
    count-uniform (parcel-uniform M2 + fill-not-trim must guarantee an EXACT
    ``n_mask`` per parcel; the tube an exact ``n_tube`` per clip) or if membership
    is not a clean electrode→parcel partition — any of these silently breaks the
    static-shape assumption every downstream gather relies on."""
    if m2_mask.dtype != torch.bool or tube_mask.dtype != torch.bool:
        raise ValueError("m2_mask and tube_mask must be bool")
    B, P, S = m2_mask.shape
    Pm, C = membership.shape
    lfs, hga = bands
    if Pm != P:
        raise ValueError(f"membership P {Pm} != m2_mask P {P}")
    if tuple(tube_mask.shape) != (B, P):
        raise ValueError(f"tube_mask {tuple(tube_mask.shape)} != (B,P)=({B},{P})")
    S_expected = sum(b.n_tokens for b in bands)
    if S != S_expected:
        raise ValueError(f"m2_mask S {S} != bands S {S_expected}")

    # membership must be an exact partition: each electrode in exactly one parcel.
    col = membership.sum(0)
    if not torch.equal(col, torch.ones_like(col)):
        raise ValueError("membership is not a partition (some electrode in ≠1 parcel)")
    n_p = membership.sum(1)
    if (n_p < 1).any():
        raise ValueError("membership has an empty parcel (n_p<1) — pass active parcels only")

    expected_n_mask = n_mask_hga(hga, cfg) + lfs.n_time_patches
    counts = m2_mask.sum(-1)  # (B,P)
    if not (counts == expected_n_mask).all():
        bad = counts[counts != expected_n_mask]
        raise ValueError(
            f"M2 count not uniform: expected {expected_n_mask}/parcel, "
            f"saw {sorted(set(bad.tolist()))} — fill-not-trim or freq-tube broke"
        )

    tube_counts = tube_mask.sum(-1)  # (B,)
    n_tube = int(tube_counts[0])
    if not (tube_counts == n_tube).all():
        raise ValueError(
            f"tube count not uniform across clips: saw {sorted(set(tube_counts.tolist()))}"
        )

    return StaticShapesV2(
        b=B,
        c=C,
        P=P,
        n_p=tuple(int(x) for x in n_p.tolist()),
        k=k,
        S=S,
        n_mask=expected_n_mask,
        s_vis=S - expected_n_mask,
        n_tube=n_tube,
        P_vis=P - n_tube,
    )


def cell_operator_index(
    bands: tuple[BandSpecV2, ...] = BANDS_V2, *, op_mode: str = "band"
) -> Tensor:
    """``(S,)`` long → which pool reading-operator (`W_K`/`W_V`) each cell uses.

    ``op_mode`` selects the band-typing granularity of the pool's K/V read:
      - ``"shared"`` (``n_op=1``): every cell uses ONE operator (all zeros). The
        pool is a plain PMA / Perceiver pooler — band identity already rides in
        via separate cells + the frontend's band-colored tokens, so a shared
        read is the simplest grounded default (Run-B first arm).
      - ``"band"`` (``n_op=2``): operator = ``band_id`` — the 3 LFS log groups
        SHARE operator 0, HGA gets operator 1. (Run-A/legacy default.)
      - ``"patch"`` (``n_op=4``): operator = ``freq_patch_id`` (LFS 0/1/2, HGA 3)
        — the untied ablation, each freq-patch its own K/V.
    The exact integer labels are immaterial; what matters is the grouping +
    that ``W_K`` is sized to ``n_operators(op_mode)``."""
    band_id, freq_patch_id, _ = token_metadata(bands)
    if op_mode == "shared":
        return torch.zeros_like(band_id)
    if op_mode == "band":
        return band_id.clone()
    if op_mode == "patch":
        return freq_patch_id.clone()
    raise ValueError(f"op_mode must be shared|band|patch, got {op_mode!r}")


def n_operators(op_mode: str = "band") -> int:
    return {"shared": 1, "band": 2, "patch": 4}[op_mode]


def active_parcels(parcel_of_electrode: Tensor) -> tuple[Tensor, Tensor]:
    """Active parcels for one subject from per-electrode parcel labels.

    Returns ``(parcel_labels (P,), membership (P, C) bool)`` where ``P`` =
    distinct labels present (the ``~16``, NOT the DKT total), ``parcel_labels``
    are the DKT label ids (index into the universal ``embed_p^q`` table), and
    ``membership[p, e]`` = electrode ``e`` ∈ parcel ``p``. Every active parcel has
    ≥1 electrode by construction ⇒ no empty rows."""
    parcel_labels = torch.unique(parcel_of_electrode)            # sorted, P
    membership = parcel_of_electrode[None, :] == parcel_labels[:, None]
    return parcel_labels, membership


class RelativeGeometry(nn.Module):
    """Native-RAS relative positional bias for the Run-B spatial pool (2026-07-01).

    Maps a centroid-relative offset Δ (mm, native subject space) to a per-head
    additive attention-logit bias. Δ is normalized by ``sigma_mm`` (the MEASURED
    ~12 mm RMS within-parcel centroid offset — cohort of 7, 116 parcels), Fourier-
    featurized at ``n_freqs`` geometric wavelengths spanning ``wl_min_sigma``·σ
    (~6 mm — resolves adjacent electrodes for the reconstruction query; measured NN
    spacing ~3.2 mm) to ``wl_max_sigma``·σ (~30 mm ≈ the 95th-pct 24.7 mm parcel
    extent — the coarsest STRUCTURED scale), PLUS the raw linear offset. The band
    stops at the parcel extent because longer-range monotonic decay + direction are
    already carried by the raw linear term, so wider Fourier octaves just duplicate
    it. A tiny MLP → ``n_heads`` scalars.

    ISOTROPIC (one σ, all axes): the P-axis anisotropy (std ~2× L/I) is a shaft-
    sampling artifact, NOT coupling physics, so the mm-metric stays axis-symmetric
    for cross-subject transfer. Δ is relative ⇒ translation-invariant by
    construction; per-axis Fourier preserves DIRECTION (sin is odd), which the
    reconstruction query needs to disambiguate opposite equidistant electrodes.
    Last layer zero-init ⇒ the bias starts at exactly 0 (pool = pure content
    attention at init; geometry is learned in).
    """

    def __init__(
        self,
        n_heads: int,
        *,
        sigma_mm: float = 12.0,
        n_freqs: int = 4,
        wl_min_sigma: float = 0.5,
        wl_max_sigma: float = 2.5,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if n_freqs < 1:
            raise ValueError(f"n_freqs must be ≥1, got {n_freqs}")
        if not 0.0 < wl_min_sigma < wl_max_sigma:
            raise ValueError(f"need 0<wl_min_sigma<wl_max_sigma, got {wl_min_sigma},{wl_max_sigma}")
        self.n_heads = n_heads
        self.sigma_mm = float(sigma_mm)
        # geometric wavelengths (in σ units) → angular frequencies ω=2π/λ.
        wl = torch.logspace(math.log10(wl_min_sigma), math.log10(wl_max_sigma), n_freqs)
        self.register_buffer("omega", 2.0 * math.pi / wl)            # (n_freqs,)
        feat_dim = 3 * (2 * n_freqs + 1)                            # per axis: raw + sin,cos×nf
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(), nn.Linear(hidden, n_heads)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def features(self, delta: Tensor) -> Tensor:
        """``(..., 3)`` mm offset → ``(..., 3·(2·n_freqs+1))`` Fourier+linear features."""
        u = delta / self.sigma_mm                                   # (...,3) normalized
        ang = u[..., None] * self.omega                             # (...,3,n_freqs)
        feats = torch.cat([u[..., None], ang.sin(), ang.cos()], dim=-1)  # (...,3,2nf+1)
        return feats.flatten(-2)                                    # (...,3·(2nf+1))

    def forward(self, delta: Tensor) -> Tensor:
        """``(..., 3)`` mm offset → ``(..., n_heads)`` per-head additive logit bias."""
        return self.mlp(self.features(delta))

    @property
    def feat_dim(self) -> int:
        return 3 * (self.omega.numel() * 2 + 1)


class ElectrodeReconHead(nn.Module):
    """Run-B reconstruction-decode head (2026-07-01) — the MNI-free identity.

    Perceiver-IO-style positional decode: reconstruct a DROPPED electrode's teacher
    frontend feature from its parcel's ``k`` pool seeds, addressed by the electrode's
    native-RAS centroid-relative offset (the query carries WHICH electrode; the k
    seeds carry the pooled field). One masked-free cross-attention block — a single
    positional query attends to the k seeds. Because the query is a continuous
    relational offset (not a discrete slot), the pool is FORCED to encode a spatial
    field rich enough to interpolate held-out electrodes ⇒ retain the detail it
    currently averages away. Per (dropped-electrode, cell) row is independent; the
    cell identity rides in WHICH seeds (``seeds`` are already the pool at that cell).
    """

    def __init__(
        self, d_model: int, n_heads: int, geom_feat_dim: int, *, n_op: int = 1
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_op = n_op
        # Positional query = learned base + a projection of the offset's Fourier
        # features (the SAME featurization the pool bias uses, so both geometry
        # readers share one frequency basis).
        self.q_base = nn.Parameter(torch.zeros(d_model))
        self.q_pos = nn.Linear(geom_feat_dim, d_model, bias=False)
        # One pre-norm decode block: cross-attention (query → k seeds) + FFN. The
        # single cross-attention keeps the pool pressure honest (no multi-hop over
        # seeds), while the FFN gives the readout the nonlinearity a bare linear
        # seed-interpolation lacks — so a well-encoded pool is actually recoverable
        # (else the loss floors out and stops pressuring the pool). Not stacked:
        # a deeper decoder would reconstruct FROM a lossy pool and relieve the tax.
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)      # query = band-agnostic
        # BAND-TYPED K/V + output projections (``n_op`` distinct weights, gathered per
        # cell by ``cell_patch``) — the multiplicative inverse of the pool's band-typed
        # W_K/W_V: the seeds are band-specific (the pool typed them), the target is
        # band-specific, so a band-BLIND readout would have to undo n_op encodings with
        # one map. Bands are abundant (every cell is one of n_op groups) ⇒ no rare-
        # bucket staleness (unlike a per-parcel typing). Applied as ``x @ W[op]``.
        self.W_k = nn.Parameter(torch.empty(n_op, d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(n_op, d_model, d_model))
        self.out_head = nn.Parameter(torch.empty(n_op, d_model, d_model))
        for o in range(n_op):
            nn.init.xavier_uniform_(self.W_k[o])
            nn.init.xavier_uniform_(self.W_v[o])
            nn.init.xavier_uniform_(self.out_head[o])
        self.W_o = nn.Linear(d_model, d_model, bias=False)     # head-merge, band-agnostic
        self.ln_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(                              # standard 4x FFN
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def _band_proj(self, x: Tensor, weight: Tensor, band: Tensor | None) -> Tensor:
        """``x @ weight[band]`` — per-row band-typed linear. ``weight`` ``(n_op,d,d)``,
        ``band`` ``(N,)`` the cell's operator id. Masked-SUM over the ≤4 ops (grad-safe,
        no in-place, no ``(N,d,d)`` gather); ``n_op=1`` is the plain single op (band
        ignored ⇒ may be None)."""
        if self.n_op == 1:
            return x @ weight[0]
        if band is None:
            raise ValueError(f"band-typed recon (n_op={self.n_op}) needs a band index")
        out = x @ weight[0] * (band == 0).view(band.shape[0], *([1] * (x.dim() - 1)))
        for op in range(1, self.n_op):
            m = (band == op).view(band.shape[0], *([1] * (x.dim() - 1)))
            out = out + (x @ weight[op]) * m
        return out

    def forward(
        self, seeds: Tensor, offset_feats: Tensor, band: Tensor | None = None
    ) -> Tensor:
        """``seeds`` ``(N, k, d)`` the query electrode's parcel seeds at its cell;
        ``offset_feats`` ``(N, F)`` = ``RelativeGeometry.features(centroid-relative
        offset)``; ``band`` ``(N,)`` long = the cell's K/V operator id (``cell_patch``),
        required iff ``n_op>1``. Returns ``(N, d)`` the reconstructed feature."""
        N, k, d = seeds.shape
        H, hd = self.n_heads, self.head_dim
        q = self.q_base + self.q_pos(offset_feats)              # (N, d)
        qn, sn = self.ln_q(q), self.ln_kv(seeds)               # pre-norm
        qh = self.W_q(qn).reshape(N, 1, H, hd).transpose(1, 2)  # (N, H, 1, hd)
        kh = self._band_proj(sn, self.W_k, band).reshape(N, k, H, hd).transpose(1, 2)
        vh = self._band_proj(sn, self.W_v, band).reshape(N, k, H, hd).transpose(1, 2)
        # Explicit (unmasked, no-dropout) attention — mathematically identical to
        # F.scaled_dot_product_attention but NOT routed through the fused dispatcher.
        # The recon query is degenerate (q_len=1, kv_len=k=2); the flash/cuDNN fused
        # kernels the dispatcher picks for it mis-launch on GPU (cudaErrorInvalid-
        # Configuration). At k=2 the small matmul + softmax is trivially cheap and
        # backend-independent (CPU/GPU/compile identical).
        attn = torch.softmax(qh @ kh.transpose(-2, -1) / (hd ** 0.5), dim=-1)  # (N,H,1,k)
        ctx = attn @ vh                                        # (N, H, 1, hd)
        # Perceiver-IO BasicDecoder: query residual OFF (the query is a positional
        # ADDRESS, not content to preserve), MLP residual ON, then band-typed readout.
        h = self.W_o(ctx.transpose(1, 2).reshape(N, d))        # no query residual
        h = h + self.ffn(self.ln_ff(h))                        # MLP + residual
        return self._band_proj(h, self.out_head, band)         # band-typed readout


class SetPoolV2(nn.Module):
    """Stage 2 set-pool (PMA) — a parcel's electrode SET → ``k`` seeds, per cell.

    Block-diagonal masked cross-attention, ONE per-(f,t)-CELL aggregation:
    seed ``(p, j)`` at cell ``s`` attends ONLY to parcel ``p``'s electrodes' tokens
    at that SAME cell (membership mask; time/freq structural, not a key tag).

    Query ``= base_j + embed_p^q`` is FREQ- and TIME-agnostic (shared ``W_Q``;
    "who's asking" = parcel + seed). Keys/values are FREQUENCY-SPECIFIC via
    distinct WEIGHTS (NOT embeddings — in a per-cell pool an additive freq embed
    on keys cancels in softmax): ``k_e = W_K^{(op)} x_e``, gathered per cell from
    a stacked ``(n_op, d, d)`` by ``cell_patch``. TIED DEFAULT ``n_op=2`` (HGA |
    LFS); UNTIE → 4 is the first ablation. The whole cell axis is VECTORIZED —
    operators gathered + projected by einsum, no python loop over cells/parcels/
    electrodes.

    Forward (student passes its ``s_vis`` visible cells, teacher the full ``S`` —
    same module, ``S`` is just the cell count):
      - ``x`` ``(B, C, S, d)`` per-electrode tokens
      - ``membership`` ``(P, C)`` bool, ``parcel_labels`` ``(P,)`` long (DKT ids)
      - ``cell_patch`` ``(S,)`` long ∈ [0, n_op)
    Output ``(B, P, k, S, d)``. The pool computes ALL ``P`` parcels (incl. soon-
    tubed; block-diag ⇒ leak-free) — the latent gathers the untubed ones.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        k: int = 2,
        n_parcels: int,
        n_op: int = 2,
        ln_out: bool = False,
        mab: bool = False,
        parcel_embed: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.k = k
        self.n_op = n_op
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Queries: base seed color (k), SHARED across parcels (the k probes are
        # universal — same learned seeds everywhere, the cross-subject-transfer
        # property). Run-A/legacy adds a per-parcel query embed (coarse anatomical
        # anchor); Run-B DROPS it (``parcel_embed=False``) so the pool is a UNIVERSAL
        # geometric operator — parcels differ only by electrode geometry (membership +
        # relational bias + band), not a learned per-region prior that goes stale on a
        # held-out subject; anatomical identity is re-injected downstream at M4.
        self.base_seed = nn.Parameter(torch.empty(k, d_model))
        nn.init.trunc_normal_(self.base_seed, std=0.02)
        self.embed_pq: nn.Parameter | None = None
        if parcel_embed:
            self.embed_pq = nn.Parameter(torch.empty(n_parcels, d_model))
            nn.init.trunc_normal_(self.embed_pq, std=0.02)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)

        # Stacked frequency-specific reading operators (gathered per cell).
        self.W_K = nn.Parameter(torch.empty(n_op, d_model, d_model))
        self.W_V = nn.Parameter(torch.empty(n_op, d_model, d_model))
        for o in range(n_op):  # one-time init (n_op≤4), not the hot path
            nn.init.xavier_uniform_(self.W_K[o])
            nn.init.xavier_uniform_(self.W_V[o])

        self.out = nn.Linear(d_model, d_model, bias=False)
        # Run-A: terminal LayerNorm so the pool output (the M3 target) is unit-
        # scale, matching frontend/latent ``ln_out``. Default OFF ⇒ legacy pool
        # (no LN) resumes byte-identical.
        self.ln_out = nn.LayerNorm(d_model) if ln_out else None
        # Run-B: complete the bare PMA into a proper pre-norm MAB — seed/query
        # residual + FFN — matching Set-Transformer PMA (1810.00825) and the
        # Perceiver encoder cross-attention (2103.03206), which both carry an FFN
        # and a query residual. Gated so Run-A's bare pool resumes byte-identical.
        self.mab = mab
        if mab:
            self.ln_q = nn.LayerNorm(d_model)
            self.ln_kv = nn.LayerNorm(d_model)
            self.ln_ff = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )

    def forward(
        self,
        x: Tensor,                # (B, C, S, d)
        membership: Tensor,       # (P, C) bool
        parcel_labels: Tensor,    # (P,) long  DKT ids
        cell_patch: Tensor,       # (S,) long  ∈ [0, n_op)
        *,
        block: Tensor | None = None,      # (B, S, C) bool — Run-B: electrode blocked at cell
        geom_bias: Tensor | None = None,  # (H, Pk, C) float — Run-B: per-head relational bias
    ) -> Tensor:
        B, C, S, d = x.shape
        P = membership.shape[0]
        H, hd, k = self.n_heads, self.head_dim, self.k
        if cell_patch.shape != (S,):
            raise ValueError(f"cell_patch must be (S={S},), got {tuple(cell_patch.shape)}")

        # Queries (P, k, d) → W_Q → (P·k, H, hd). Cell-independent. MAB pre-norms
        # the query (raw q_tok kept for the seed residual below). Run-B: no per-parcel
        # embed ⇒ the shared base_seed broadcast over all parcels.
        q_tok = self.base_seed[None, :, :].expand(P, k, d)
        if self.embed_pq is not None:
            q_tok = self.embed_pq[parcel_labels][:, None, :] + q_tok
        q_in = self.ln_q(q_tok) if self.mab else q_tok
        q = self.W_Q(q_in).reshape(P * k, H, hd)

        # Per-cell freq-specific key/value projection — gather operator, einsum.
        # MAB pre-norms the electrode tokens before the K/V projection.
        xk = self.ln_kv(x) if self.mab else x
        WK = self.W_K[cell_patch]                                # (S, d, d)
        WV = self.W_V[cell_patch]                                # (S, d, d)
        keys = torch.einsum("bcsd,sde->bcse", xk, WK)           # (B, C, S, d)
        vals = torch.einsum("bcsd,sde->bcse", xk, WV)

        # Batch the cell axis: (B, S, C, H, hd).
        keys = keys.permute(0, 2, 1, 3).reshape(B * S, C, H, hd)
        vals = vals.permute(0, 2, 1, 3).reshape(B * S, C, H, hd)
        qh = q[None].expand(B * S, P * k, H, hd).permute(0, 2, 1, 3)  # (BS, H, Pk, hd)
        kh = keys.permute(0, 2, 1, 3)                                 # (BS, H, C, hd)
        vh = vals.permute(0, 2, 1, 3)

        # Block-diagonal additive bias (Pk, C): 0 = attend, -1e4 = block. Finite
        # sentinel (not -inf) keeps a fully-blocked row a uniform finite softmax,
        # NaN-free in fwd+bwd on every backend (see _MultiHeadCrossAttention).
        mem_pk = membership.repeat_interleave(k, dim=0)              # (Pk, C)
        bias = torch.zeros(P * k, C, dtype=qh.dtype, device=x.device)
        bias = bias.masked_fill(~mem_pk, NEG_INF_MASK_VALUE)[None, None]  # (1,1,Pk,C)
        # Run-B additive contributions (both None ⇒ byte-identical to the base pool):
        #  * geom_bias (H,Pk,C): per-head relational seed→electrode bias. Non-member
        #    logits are already -1e4 ⇒ geom there is inert. Broadcast over (B·S).
        #  * block (B,S,C): whole-electrode drop + heterogeneous M2 mask — the pool
        #    must not read a dropped/masked electrode-cell (else its zero-filled or
        #    held-out feature leaks in). Additive -1e4 per (B·S). Broadcast over Pk.
        if geom_bias is not None:
            bias = bias + geom_bias[None].to(qh.dtype)               # (1,H,Pk,C)
        if block is not None:
            if tuple(block.shape) != (B, S, C):
                raise ValueError(f"block must be (B,S,C)=({B},{S},{C}), got {tuple(block.shape)}")
            blk = torch.zeros(B, S, C, dtype=qh.dtype, device=x.device)
            blk = blk.masked_fill(block, NEG_INF_MASK_VALUE)
            bias = bias + blk.reshape(B * S, 1, 1, C)                # (B·S,1,1,C) broadcast
        ctx = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias)  # (BS,H,Pk,hd)
        ctx = ctx.permute(0, 2, 1, 3).reshape(B * S, P * k, d)

        # No-coverage rows (none for active parcels, kept for safety) → 0.
        no_cov = ~mem_pk.any(dim=-1)                                # (Pk,)
        ctx = ctx.masked_fill(no_cov[None, :, None], 0.0)
        out = self.out(ctx)                                         # (BS, Pk, d)
        if self.mab:
            out = q_tok.reshape(P * k, d)[None] + out               # seed/query residual
            out = out + self.ffn(self.ln_ff(out))                  # FFN residual
        if self.ln_out is not None:
            out = self.ln_out(out)
        return out.reshape(B, S, P, k, d).permute(0, 2, 3, 1, 4)    # (B, P, k, S, d)


def _max_time_slot(bands: tuple[BandSpecV2, ...]) -> int:
    _, _, time_slot = token_metadata(bands)
    return int(time_slot.max().item())


class LatentEncoderV2(nn.Module):
    """Stage 3 latent — global self-attention over the parcel SEEDS (cost center).

    Input is the pooled seeds ``(B, P, k, S, d)`` (NOT electrodes — v2 pools first,
    so the latent is SMALLER: ``P·k·S`` ≈ 12·2·60 vs the old ``C·S``). Bridge:
    add ``embed_p^pos`` (the latent's learned parcel positional tag, broadcast
    across ``k`` seeds + ``S`` cells) then flatten to one ``P·k·S`` token set and
    run ``n_layers`` all-pairs joint self-attention blocks. RoPE keys off the
    seed's TIME (``cell_time_slot`` on the shared HGA-stride clock); the seed's
    frequency identity is carried in via the residual from the frontend embed
    (NOT re-added here). Freq is already structural in the seed — the pool was
    per-cell — so only parcel(+) and time(RoPE) positions are added at the latent.

    The student passes its gathered ``P_vis`` untubed parcels × ``s_vis`` visible
    cells; the teacher passes all ``P`` × ``S``. ``key_mask`` ``(B,P,k,S)`` bool
    (``True`` = attendable) supports a dense/padded path; the static hot path
    gathers instead and passes ``None``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_parcels: int,
        *,
        ssl_bands: tuple[BandSpecV2, ...] | None = None,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        head_dim = d_model // n_heads
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")

        # embed_p^pos — the latent's own parcel tag (separate from the pool's
        # embed_p^q), universal DKT-sized table.
        self.parcel_embed = nn.Embedding(n_parcels, d_model)
        nn.init.trunc_normal_(self.parcel_embed.weight, std=0.02)

        # Size the RoPE table to the LONGEST clock the model sees (5 s SSL by
        # default); the 1 s eval slots are a subset ⇒ safe to gather.
        if ssl_bands is None:
            ssl_bands = bands_for_clip_len(5.0)
        n_slots = _max_time_slot(ssl_bands) + 1
        self.register_buffer("key_rope", _rope_freqs(head_dim, n_slots), persistent=False)

        self.blocks = nn.ModuleList(
            [_JointTokenBlock(d_model, n_heads, qk_norm=qk_norm) for _ in range(n_layers)]
        )
        self.ln_out = nn.LayerNorm(d_model)

    def forward(
        self,
        seeds: Tensor,            # (B, P, k, S, d)
        parcel_labels: Tensor | None,    # (P,) long DKT ids, OR None if pre-tagged
        cell_time_slot: Tensor,   # (S,) long  RoPE clock per cell
        *,
        key_mask: Tensor | None = None,  # (B, P, k, S) bool, True = attendable
    ) -> Tensor:
        """``parcel_labels=None`` means the seeds already carry ``embed_p^pos``
        (added pre-gather). The static hot path tags ALL P parcels at the pool
        output — shared ``(P,)`` labels — THEN gathers the untubed parcels per clip,
        so the latent never needs per-clip labels: it runs pure self-attention on
        the pre-tagged, parcel-gathered seeds (the parcel axis is gathered, the cell
        axis + RoPE are untouched)."""
        B, P, k, S, d = seeds.shape
        if parcel_labels is None:
            x = seeds
        else:
            x = seeds + self.parcel_embed(parcel_labels)[None, :, None, None, :]
        x = x.reshape(B, P * k * S, d)
        # RoPE slots: shared (S,) (teacher / uniform cells) tiled across P·k, OR
        # per-row (B, P·k·S) when the static path's visible cells DIFFER per parcel.
        if cell_time_slot.dim() == 1:
            if cell_time_slot.shape != (S,):
                raise ValueError(
                    f"cell_time_slot must be (S={S},), got {tuple(cell_time_slot.shape)}"
                )
            slot = cell_time_slot.repeat(P * k)                  # (P·k·S,)
            rope = self.key_rope[:, slot, :]                     # (2, N, head_dim) shared
        else:
            if cell_time_slot.shape != (B, P * k * S):
                raise ValueError(
                    f"per-row cell_time_slot must be (B, P·k·S)=({B},{P * k * S}), "
                    f"got {tuple(cell_time_slot.shape)}"
                )
            rs = self.key_rope[:, cell_time_slot, :]             # (2, B, N, head_dim)
            rope = rs                                            # per-row
        km = None if key_mask is None else key_mask.reshape(B, P * k * S)
        for blk in self.blocks:
            x = blk(x, rope, km)
        return self.ln_out(x).reshape(B, P, k, S, d)


class JepaPredictorV2(nn.Module):
    """I-JEPA joint predictor for v2 — serves BOTH M2 (shallow) and M4 (deep).

    Queries (held-out positions) are built from a shared learnable mask token +
    a learned freq-patch tag, optionally + a parcel tag (M4) + a seed tag (M4),
    RoPE-positioned in time. They are concatenated with the projected visible
    CONTEXT and run through ``n_layers`` joint self-attention blocks (each query
    attends to all context + the queries co-resolve); the query rows are read out
    and projected (raw, NO LayerNorm before the L1 — the TARGET is LN'd instead,
    loss memo Refinement 5) to the teacher feature dim ``d_model``.

    ONE class, two configs:
      - **M2** (per-electrode, electrodes ride the batch dim): ``n_parcels=None,
        k=None`` ⇒ query = mask + freq + RoPE, NO parcel/seed (stage-1 isolated).
        Context = the electrode's ``s_vis`` visible FRONTEND tokens.
      - **M4** (over the latent): ``n_parcels`` + ``k`` set ⇒ query = mask + freq
        + parcel + seed + RoPE. Context = the student LATENT seeds. ONE predictor,
        extended query set {tubed: all S cells} ∪ {untubed: M2-masked cells}.

    All query/context metadata is per-row ``(B, L)`` (B = whatever rides the
    batch dim — real B·C electrodes for M2, B clips for M4). ``ctx_key_mask``
    ``(B, Lc)`` drops padded/invalid context; the static hot path gathers ⇒ None.
    """

    def __init__(
        self,
        d_model: int,
        pred_dim: int,
        n_heads: int,
        n_layers: int,
        *,
        n_parcels: int | None = None,
        k: int | None = None,
        ssl_bands: tuple[BandSpecV2, ...] | None = None,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if pred_dim % n_heads != 0:
            raise ValueError(f"pred_dim={pred_dim} not divisible by n_heads={n_heads}")
        head_dim = pred_dim // n_heads
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")

        self.ctx_proj = nn.Linear(d_model, pred_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.freq_embed = nn.Parameter(torch.empty(N_FREQ_PATCHES_V2, pred_dim))
        nn.init.trunc_normal_(self.freq_embed, std=0.02)

        self.parcel_embed = (
            nn.Embedding(n_parcels, pred_dim) if n_parcels is not None else None
        )
        if self.parcel_embed is not None:
            nn.init.trunc_normal_(self.parcel_embed.weight, std=0.02)
        if k is not None:
            self.seed_embed = nn.Parameter(torch.empty(k, pred_dim))
            nn.init.trunc_normal_(self.seed_embed, std=0.02)
        else:
            self.seed_embed = None

        if ssl_bands is None:
            ssl_bands = bands_for_clip_len(5.0)
        n_slots = _max_time_slot(ssl_bands) + 1
        self.register_buffer("key_rope", _rope_freqs(head_dim, n_slots), persistent=False)
        self.blocks = nn.ModuleList(
            [_JointTokenBlock(pred_dim, n_heads, qk_norm=qk_norm) for _ in range(n_layers)]
        )
        self.head = nn.Linear(pred_dim, d_model, bias=False)  # raw pred, NO LN

    def _build_queries(
        self,
        q_freq: Tensor,        # (B, Lq)  RoPE-time added in forward, not here
        q_parcel: Tensor | None,
        q_seed: Tensor | None,
    ) -> Tensor:
        q = self.mask_token + self.freq_embed[q_freq]            # (B, Lq, pred)
        if self.parcel_embed is not None:
            if q_parcel is None:
                raise ValueError("this predictor has a parcel tag; pass q_parcel")
            q = q + self.parcel_embed(q_parcel)
        elif q_parcel is not None:
            raise ValueError("this predictor has no parcel tag; q_parcel must be None")
        if self.seed_embed is not None:
            if q_seed is None:
                raise ValueError("this predictor has a seed tag; pass q_seed")
            q = q + self.seed_embed[q_seed]
        elif q_seed is not None:
            raise ValueError("this predictor has no seed tag; q_seed must be None")
        return q

    def forward(
        self,
        ctx: Tensor,             # (B, Lc, d_model)
        ctx_slot: Tensor,        # (B, Lc) long
        q_slot: Tensor,          # (B, Lq) long
        q_freq: Tensor,          # (B, Lq) long
        *,
        q_parcel: Tensor | None = None,   # (B, Lq) long  (M4)
        q_seed: Tensor | None = None,     # (B, Lq) long  (M4)
        ctx_key_mask: Tensor | None = None,  # (B, Lc) bool, True = real
    ) -> Tensor:
        """Predict ``(B, Lq, d_model)`` at the query positions from the context."""
        B, Lc, _ = ctx.shape
        Lq = q_slot.shape[1]
        q = self._build_queries(q_freq, q_parcel, q_seed)          # (B, Lq, pred)
        c = self.ctx_proj(ctx)                                     # (B, Lc, pred)
        tokens = torch.cat([c, q], dim=1)                          # (B, Lc+Lq, pred)

        slots = torch.cat([ctx_slot, q_slot], dim=1)               # (B, Lc+Lq)
        rope = self.key_rope[:, slots, :]                          # (2, B, L, head_dim)

        key_mask: Tensor | None = None
        if ctx_key_mask is not None:
            q_keep = torch.ones(B, Lq, dtype=torch.bool, device=ctx.device)
            key_mask = torch.cat([ctx_key_mask, q_keep], dim=1)    # (B, L)

        for blk in self.blocks:
            tokens = blk(tokens, rope, key_mask)
        return self.head(tokens[:, Lc:])                           # (B, Lq, d_model)


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, tau: float) -> None:
    """In-place EMA: ``θ_t ← τ·θ_t + (1−τ)·θ_s`` on params; buffers copied.

    The JEPA momentum target update (mirrors ``v14_converged.update_teacher``):
    foreach mul/add over the param lists (one fused kernel, no python loop),
    then a hard copy of every buffer (RoPE tables etc. — non-learned, must track
    the student exactly). ``tau`` is the teacher's retention (e.g. 0.9992)."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}")
    t_params = list(teacher.parameters())
    s_params = list(student.parameters())
    if len(t_params) != len(s_params):
        raise ValueError(
            f"teacher/student param count mismatch: {len(t_params)} vs {len(s_params)}"
        )
    torch._foreach_mul_(t_params, tau)
    torch._foreach_add_(t_params, s_params, alpha=1.0 - tau)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)


def _ln_target(t: Tensor) -> Tensor:
    """LayerNorm-noaffine over the feature dim + stop-grad — the V-JEPA-2.1 target
    transform that puts the shallow (frontend) and deep (latent) teacher targets on
    one O(1) per-scalar scale so the shared denominator balances COUNT not magnitude
    (no depth λ). ``eps`` matches ``F.layer_norm`` default."""
    return F.layer_norm(t.detach(), (t.shape[-1],))


def converged_v2_loss(
    m2_pred: Tensor,
    m2_target: Tensor,
    m4_pred: Tensor,
    m4_target: Tensor,
    m4_weight: Tensor,
    m4_tubed: Tensor,
) -> dict[str, Tensor]:
    """Shared-denominator dual-depth JEPA loss (locked Ben 2026-06-25).

    ``L = [ Σ_{M2} Σ_d|p̂−t| + Σ_{M4} (w_p·n_p) Σ_d|p̂−t̄| ] / [ (|M2| + Σ_{M4} w_p·n_p)·d ]``

    A weighted MEAN over every predicted scalar: each M2 frontend cell weight 1,
    each M4 latent query weight ``w_p·n_p`` (= ``m4_weight``), the weight in BOTH
    numerator and denominator. That keeps ``L`` magnitude-interpretable — a constant
    per-scalar error ``e`` yields exactly ``e`` regardless of the weights (the
    legibility property the old weighted-num / unweighted-denom scheme broke). Both
    teacher targets are LN-noaffine'd + detached (no depth λ).

    All query tensors are pre-flattened ``(Q, d)``; the forward owns the gather and
    fills ``m4_weight`` (per-query ``w_p·n_p``) and ``m4_tubed`` (cross-parcel vs
    within-parcel-inpaint, for the scale diagnostic). Returns ``loss`` plus per-term
    means + ratios for the Refinement-5 scale assertions."""
    d = m2_pred.shape[-1]
    if m4_pred.shape[-1] != d:
        raise ValueError(f"m2/m4 feature dim mismatch: {d} vs {m4_pred.shape[-1]}")
    for name, w in (("m4_weight", m4_weight), ("m4_tubed", m4_tubed)):
        if w.shape[0] != m4_pred.shape[0]:
            raise ValueError(f"{name} length {w.shape[0]} != m4 queries {m4_pred.shape[0]}")

    err_m2 = (m2_pred - _ln_target(m2_target)).abs().sum(-1)  # (Q2,)
    err_m4 = (m4_pred - _ln_target(m4_target)).abs().sum(-1)  # (Q4,)

    num = err_m2.sum() + (m4_weight * err_m4).sum()
    den = (err_m2.new_tensor(float(err_m2.numel())) + m4_weight.sum()) * d
    loss = num / den

    def _wmean(e: Tensor, w: Tensor) -> Tensor:
        wsum = w.sum()
        if wsum == 0:
            return e.new_zeros(())
        return (w * e).sum() / (wsum * d)

    tubed = m4_tubed.bool()
    diag = {
        "loss": loss,
        "loss_m2": err_m2.mean() / d if err_m2.numel() else loss.new_zeros(()),
        "loss_m4": _wmean(err_m4, m4_weight),
        "loss_m4_tubed": _wmean(err_m4[tubed], m4_weight[tubed]),
        "loss_m4_untubed": _wmean(err_m4[~tubed], m4_weight[~tubed]),
    }
    diag["ratio_m2_m4"] = diag["loss_m2"] / diag["loss_m4"].clamp_min(1e-12)
    diag["ratio_tubed_untubed"] = diag["loss_m4_tubed"] / diag["loss_m4_untubed"].clamp_min(
        1e-12
    )
    return diag


def converged_v2_loss_per_head(
    m2_pred: Tensor,
    m2_target: Tensor,
    m3_pred: Tensor,
    m3_target: Tensor,
    m4_pred: Tensor,
    m4_target: Tensor,
    *,
    w_m2: float = 1.0,
    w_m3: float = 1.0,
    w_m4: float = 1.0,
    m3_weight: Tensor | None = None,
    m4_weight: Tensor | None = None,
    melec_pred: Tensor | None = None,
    melec_target: Tensor | None = None,
    w_melec: float = 1.0,
    melec_weight: Tensor | None = None,
) -> dict[str, Tensor]:
    """Run-A per-head JEPA loss (Ben 2026-06-27): three INDEPENDENT
    self-normalized L1 means on RAW (detached) EMA targets, combined with
    explicit weights (V-JEPA convention: equal). NO target LayerNorm (the pool/
    frontend/latent already end in ``ln_out``), NO shared denominator, NO
    count-weighting::

        L_m2 = |m2_pred − sg(t_front [masked])|.mean()   # mean over its OWN (Q×d)
        L_m3 = |m3_pred − sg(t_seeds [masked])|.mean()   # pool output, masked cells
        L_m4 = |m4_pred − sg(t_latent[tubed ])|.mean()   # tubed parcels
        loss = w_m2·L_m2 + w_m3·L_m3 + w_m4·L_m4

    Each ``.mean()`` divides by that head's own scalar count, so a head with few
    queries isn't drowned by one with many — the per-head balance the shared
    denominator lacked. All pred/target tensors are pre-flattened ``(Q, d)``.

    ``m3_weight``/``m4_weight`` (optional, shape ``(Q,)``) turn that head's flat
    mean into a CONVEX electrode-support-weighted mean ``Σ w·rowloss / Σ w``
    (rowloss = mean over ``d``). Convex ⇒ same scale as the flat mean, so the
    weighted heads stay comparable to the unweighted M2. None ⇒ flat mean."""
    d = m2_pred.shape[-1]
    for name, t in (("m3_pred", m3_pred), ("m4_pred", m4_pred)):
        if t.shape[-1] != d:
            raise ValueError(f"{name} feature dim {t.shape[-1]} != m2 dim {d}")

    def _head(pred: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
        if pred.numel() == 0:
            return pred.new_zeros(())
        err = (pred - target.detach()).abs()
        if weight is None:
            return err.mean()
        rowloss = err.mean(-1)                          # (Q,) mean over d
        return (rowloss * weight).sum() / weight.sum().clamp_min(1e-12)

    l_m2 = _head(m2_pred, m2_target)
    l_m3 = _head(m3_pred, m3_target, m3_weight)
    l_m4 = _head(m4_pred, m4_target, m4_weight)
    loss = w_m2 * l_m2 + w_m3 * l_m3 + w_m4 * l_m4
    diag = {
        "loss": loss,
        "loss_m2": l_m2,
        "loss_m3": l_m3,
        "loss_m4": l_m4,
    }
    # Run-B: masked-electrode reconstruction head (replaces M3; another
    # self-normalized L1 mean on the RAW detached teacher frontend feature).
    if melec_pred is not None:
        if melec_target is None:
            raise ValueError("melec_pred given but melec_target is None")
        if melec_pred.shape[-1] != d:
            raise ValueError(f"melec_pred feature dim {melec_pred.shape[-1]} != m2 dim {d}")
        # ``melec_weight`` (Run-B complete-grid): 0/1 per query cell — 0 at cells
        # whose survivor-seed is fully empty (no signal to reconstruct from), so the
        # self-normalized mean divides by LIVE cells only (don't supervise the
        # information-theoretically impossible).
        l_melec = _head(melec_pred, melec_target, melec_weight)
        loss = loss + w_melec * l_melec
        diag["loss"] = loss
        diag["loss_melec"] = l_melec
        diag["ratio_melec_m4"] = l_melec / l_m4.clamp_min(1e-12)
        # Overlay the recon (third-head) loss onto Run-A's "m3" slot so the wandb
        # dashboards compare the third SSL term across runs. The M3 head is OFF in
        # Run-B (its l_m3=0), so without this loss_m3/ratio_m3_m4 would plot a flat
        # zero over Run-A's m3 curve. The training loss is UNCHANGED (built from
        # l_melec above); this only re-labels the diagnostic.
        diag["loss_m3"] = l_melec
    diag["ratio_m2_m4"] = l_m2 / l_m4.clamp_min(1e-12)
    diag["ratio_m3_m4"] = diag["loss_m3"] / l_m4.clamp_min(1e-12)
    return diag


def _select_idx(mask: Tensor, count: int) -> Tensor:
    """Positions of the first ``count`` True entries along the last axis, ascending.

    The caller guarantees an EXACT ``count`` True per row (the static header's
    parcel-uniform M2 / fixed tube / fill-not-trim invariants), so this is a pure
    gather-index builder — used for visible cells, masked cells, and untubed
    parcels alike. ``argsort(descending, stable)`` floats the True positions to the
    front in their original order; the final ``sort`` re-ascends the kept prefix."""
    order = mask.int().argsort(dim=-1, descending=True, stable=True)
    return order[..., :count].sort(dim=-1).values


@dataclass(frozen=True)
class V14ConvergedV2Config:
    """Architecture + science knobs for :class:`V14ConvergedV2`.

    Architecture dims (``d_model`` … ``n_parcels``) have NO defaults — the caller
    justifies them per the discuss-before-code rule. The science knobs carry the
    locked first-run values (impl-plan + memos): ``k=2`` seeds/parcel,
    ``tube_ratio=0.25``, ``ema_tau=0.9992``. ``pool_op`` (the pool's K/V
    band-typing) auto-resolves by run mode — see :attr:`pool_op_resolved`."""

    d_model: int
    n_heads: int
    frontend_layers: int
    latent_layers: int
    m2_pred_layers: int
    m4_pred_layers: int
    pred_dim: int
    n_parcels: int
    k: int = 2
    tube_ratio: float = 0.25
    # Pool K/V read-operator typing. ``None`` ⇒ auto (see ``pool_op_resolved``):
    # ``"band"`` (n_op=2, LFS | HGA) for every run mode — the physiology-backed
    # default (LFS↔HGA is the one real spatial-regime boundary; n_op=1 is pool-band-
    # blind, n_op=4 over-splits LFS). Explicit ``"shared"|"band"|"patch"`` (1/2/4)
    # overrides — ``"shared"``/``"patch"`` are the tie/untie ablations.
    pool_op: str | None = None
    ema_tau: float = 0.9992
    # --- Run-A bundle (all default to LEGACY behavior for checkpoint resume) ---
    # ``m3_pred_layers`` is the master switch: set it (e.g. 6) to enable the Run-A
    # architecture — the M3 pool-inpaint head, a terminal LayerNorm on the pool,
    # M4 restricted to TUBED parcels only, and the per-head self-normalized loss
    # (raw EMA targets). Left None ⇒ the legacy assembly resumes byte-identical
    # (M4 = tubed∪untubed, shared-denominator ``_ln_target`` loss, no pool LN).
    m3_pred_layers: int | None = None
    qk_norm: bool = False
    w_m2: float = 1.0
    w_m3: float = 1.0
    w_m4: float = 1.0
    # Electrode-count weighting for M3/M4 (Run-A only): each parcel's per-head
    # loss is weighted by its electrode-support count n_elec(p)=membership[p].sum,
    # combined as a CONVEX weighted mean (÷ Σ n_elec) — a precision / inverse-
    # variance prior (a parcel pooled from more electrodes is a lower-variance
    # target) that also matches M2's implicit electrode-density weighting. Convex
    # ⇒ scale-preserving: M2≈M3≈M4 stay comparable. M2 stays unweighted (already
    # per-electrode-cell). Default False = flat per-head means.
    support_weight: bool = False
    # --- Run-B bundle (masked-electrode-prediction pool head) ---------------
    # ``m3_drop_frac`` (ρ) is the Run-B master switch: set it (e.g. 0.3) ⇒ the M3
    # pool-inpaint head is REPLACED by the per-parcel masked-electrode-prediction
    # head — whole electrodes are dropped from each parcel (survivor floor
    # ``m3_min_keep``) and their teacher FRONTEND feature is reconstructed from the
    # parcel's k pool seeds, queried by the electrode's native-RAS centroid-relative
    # offset via a relative attention bias. Fixes the pooling tax (the pool must
    # encode a spatial field rich enough to interpolate held-out electrodes). None ⇒
    # Run-B off (Run-A/legacy assembly unchanged). ``w_melec`` weights the head's
    # self-normalized L1 term. ``sigma_mm`` (12.0, measured RMS centroid offset) +
    # ``geom_n_freqs`` size the Fourier relative-geometry features. Requires Run-A
    # (``m3_pred_layers`` set) — Run-B reshapes that head, it is not standalone.
    m3_drop_frac: float | None = None
    m3_min_keep: int = 3
    w_melec: float = 1.0
    sigma_mm: float = 12.0
    geom_n_freqs: int = 5
    # Run-B M2 masking granularity. False ⇒ parcel-uniform (all electrodes in a
    # parcel hold out the SAME cells — the Run-A default, broadcast to electrodes).
    # True ⇒ per-electrode HETEROGENEOUS (each electrode holds out its OWN cells;
    # the pool then blocks each electrode's masked cells so its zero-filled token
    # never leaks into the seeds). Only consulted in Run-B. Swept flag — the
    # parcel-uniform arm is kept so the prototype A/B can attribute the delta.
    m2_hetero: bool = False

    @property
    def run_a(self) -> bool:
        """True ⇒ the Run-A bundle is active (M3 + pool LN + tubed-only M4 +
        per-head loss). Gated on the M3 head's presence."""
        return self.m3_pred_layers is not None

    @property
    def run_b(self) -> bool:
        """True ⇒ the Run-B masked-electrode-prediction head replaces M3. Gated on
        ``m3_drop_frac``; requires Run-A (it reshapes the M3 head)."""
        return self.m3_drop_frac is not None

    @property
    def pool_op_resolved(self) -> str:
        """Pool K/V operator mode, defaulting to ``"band"`` (n_op=2, LFS | HGA) for
        every run mode when ``pool_op`` is None. The pool needs SOME multiplicative
        band-typing — an additive freq-embed is cell-common-mode and cancels in the
        per-cell pooling softmax, so n_op=1 (``"shared"``) leaves the pool band-blind.
        But the only spatial-regime boundary the physiology supports at mm–cm ECoG
        spacing is LFS↔HGA (focal broadband high-gamma tracks local firing vs
        distributed low-freq: Muller 2016 J Neural Eng; Ray & Maunsell 2011 PLoS
        Biol); the 3-way LFS split (``"patch"``, n_op=4) rides a smooth, modest,
        partly non-monotonic coherence gradient (Bullock 1995) and starves its ops
        (~2 tokens each vs HGA's 16) — an opt-in ablation, not the default. Run-A/
        legacy already used ``"band"`` (byte-identical). Explicit ``pool_op`` wins."""
        if self.pool_op is not None:
            return self.pool_op
        return "band"


@dataclass(frozen=True)
class _SessionLayout:
    """Session-constant geometry for one homogeneous batch (cache once/session)."""

    labels: Tensor       # (P,)  DKT parcel ids
    membership: Tensor   # (P, C) bool
    parcel_idx: Tensor   # (C,)  each electrode's row in `labels`
    freq_id: Tensor      # (S,)  freq-patch id ∈ [0, 4)
    slot: Tensor         # (S,)  RoPE time slot (shared HGA clock)
    cell_patch: Tensor   # (S,)  pool operator id ∈ [0, n_op)


class V14ConvergedV2(nn.Module):
    """Converged-v2 SSL model — the full student+teacher assembly (#276 / P3.1).

    One step: 2-band magnitude frontend → set-pool (k seeds/parcel/cell) → latent
    self-attention over seeds, with a dual-depth multi-level JEPA loss (M2 = shallow
    frontend target, M4 = deep latent target). The student runs the DROP-NOT-PAD
    static path (pack visible cells / gather untubed parcels) so every tensor is
    session-constant; the teacher (EMA, frozen) runs full shapes. Masks are accepted
    as args (sampled by the dataloader so the ~455 ms CPU mask bubble stays out of
    ``_step``); :meth:`sample_masks` is the matching sampler.

    The 7-stage flow + the pool-needs-canonical-S finding are in memory
    ``project_converged_v2_assembly_forward_2026_06_26``. Built on the P2 primitives
    whose dense==static equivalence is proven cell-by-cell + parcel-by-parcel.
    """

    def __init__(
        self, cfg: V14ConvergedV2Config, *, bands: tuple[BandSpecV2, ...] = BANDS_V2
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.base_bands = bands
        # Build every submodule against the LONGEST clock (5 s SSL) so the RoPE
        # tables cover the 5 s slots; 1 s eval slots are a subset (safe to gather).
        self.ssl_bands = bands_for_clip_len(5.0, base=bands)
        n_op = n_operators(cfg.pool_op_resolved)

        self.frontend = FrontendEncoderV2(
            cfg.d_model, cfg.n_heads, cfg.frontend_layers, bands=self.ssl_bands,
            qk_norm=cfg.qk_norm,
        )
        self.pool = SetPoolV2(
            cfg.d_model, cfg.n_heads, k=cfg.k, n_parcels=cfg.n_parcels, n_op=n_op,
            ln_out=cfg.run_a or cfg.run_b, mab=cfg.run_b,
            parcel_embed=not cfg.run_b,
        )
        self.latent = LatentEncoderV2(
            cfg.d_model, cfg.n_heads, cfg.latent_layers, cfg.n_parcels,
            ssl_bands=self.ssl_bands, qk_norm=cfg.qk_norm,
        )
        self.m2_predictor = JepaPredictorV2(
            cfg.d_model, cfg.pred_dim, cfg.n_heads, cfg.m2_pred_layers,
            ssl_bands=self.ssl_bands, qk_norm=cfg.qk_norm,
        )
        self.m4_predictor = JepaPredictorV2(
            cfg.d_model, cfg.pred_dim, cfg.n_heads, cfg.m4_pred_layers,
            n_parcels=cfg.n_parcels, k=cfg.k, ssl_bands=self.ssl_bands,
            qk_norm=cfg.qk_norm,
        )
        # Run-A: M3 pool-inpaint head (untubed parcels, M2-masked cells, pool
        # target). Parcel- + seed-tagged like M4. None in legacy mode.
        self.m3_predictor: JepaPredictorV2 | None = None
        if cfg.m3_pred_layers is not None and not cfg.run_b:
            self.m3_predictor = JepaPredictorV2(
                cfg.d_model, cfg.pred_dim, cfg.n_heads, cfg.m3_pred_layers,
                n_parcels=cfg.n_parcels, k=cfg.k, ssl_bands=self.ssl_bands,
                qk_norm=cfg.qk_norm,
            )

        # Run-B: native-RAS relative geometry (shared by the pool seed→electrode
        # bias and the recon query), learned per-seed spatial offsets, and the
        # masked-electrode reconstruction head. Replaces the M3 pool-inpaint head.
        self.geom: RelativeGeometry | None = None
        self.recon: ElectrodeReconHead | None = None
        self.pool_seed_offset: nn.Parameter | None = None
        if cfg.run_b:
            self.geom = RelativeGeometry(
                cfg.n_heads, sigma_mm=cfg.sigma_mm, n_freqs=cfg.geom_n_freqs
            )
            self.recon = ElectrodeReconHead(
                cfg.d_model, cfg.n_heads, self.geom.feat_dim, n_op=n_op
            )
            # k seed probes, centroid-relative mm frame; zero-init ⇒ all seeds start
            # at the parcel centroid (pure content pool at step 0).
            self.pool_seed_offset = nn.Parameter(torch.zeros(cfg.k, 3))

        # Canonical weight init: unify every tower Linear/Embedding under one
        # scheme (trunc_normal 0.02 — GPT-2/ViT/MAE), replacing PyTorch's default
        # kaiming_uniform(a=√5). ``apply`` only visits child MODULES, so the raw-
        # Parameter inits (xavier K/V, base_seed, q_base, pool_seed_offset,
        # mask/seed/freq embeds) are untouched; the one casualty is the geom
        # adaLN-zero gate (a Linear), re-asserted below. Runs before the teacher
        # deep-copy so the frozen towers inherit the corrected init.
        self.apply(self._init_weights)
        for mod in self.modules():
            if isinstance(mod, RelativeGeometry):
                nn.init.zeros_(mod.mlp[-1].weight)
                nn.init.zeros_(mod.mlp[-1].bias)

        # EMA teacher — frozen deep copies of the three target-side towers.
        self.teacher_frontend = copy.deepcopy(self.frontend)
        self.teacher_pool = copy.deepcopy(self.pool)
        self.teacher_latent = copy.deepcopy(self.latent)
        for m in (self.teacher_frontend, self.teacher_pool, self.teacher_latent):
            for p in m.parameters():
                p.requires_grad_(False)

        self.m2_mask_cfg = M2MaskConfigV2()

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Canonical trunc_normal(0.02) for every Linear/Embedding (GPT-2/ViT/MAE).
        Deliberate raw-Parameter inits are not child modules ⇒ untouched by
        ``apply``; the geom adaLN-zero gate is a Linear and is re-asserted by the
        caller after ``apply``."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    # -- mask sampling (dataloader-side; keeps the bubble out of _step) ---------
    def sample_masks(
        self,
        B: int,
        membership: Tensor,
        bands: tuple[BandSpecV2, ...],
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        """``(m2_mask (B,P,S), tube_mask (B,P))`` for one homogeneous batch."""
        P = membership.shape[0]
        m2 = sample_m2_masks_v2(B, P, bands, generator, self.m2_mask_cfg)
        tube = sample_parcel_tube_v2(B, P, self.cfg.tube_ratio, generator)
        return m2, tube

    def sample_drop(
        self, B: int, membership: Tensor, generator: torch.Generator
    ) -> Tensor:
        """Run-B whole-electrode drop ``(B, C)`` bool (True = dropped). Count-uniform
        per parcel (survivor floor ``m3_min_keep``) so shapes stay static. Sampled
        dataloader-side, passed to :meth:`forward` as ``drop_mask``."""
        if not self.cfg.run_b:
            raise ValueError("sample_drop is Run-B only (set m3_drop_frac)")
        assert self.cfg.m3_drop_frac is not None
        return sample_electrode_drop_v2(
            B, membership, self.cfg.m3_drop_frac, self.cfg.m3_min_keep, generator
        )

    def sample_m2_hetero(
        self,
        B: int,
        C: int,
        bands: tuple[BandSpecV2, ...],
        generator: torch.Generator,
    ) -> Tensor:
        """Run-B per-electrode HETEROGENEOUS M2 mask ``(B, C, S)`` bool. Same
        constant per-row hold-out count as the parcel-uniform sampler (SAME
        ``m2_mask_cfg``) ⇒ ``s_vis``/``n_mask`` static shapes are unchanged; only
        WHICH cells each electrode holds out differs. Sampled dataloader-side,
        passed to :meth:`forward` as ``m2_elec_mask``."""
        if not (self.cfg.run_b and self.cfg.m2_hetero):
            raise ValueError("sample_m2_hetero is Run-B + m2_hetero only")
        return sample_m2_masks_hetero_v2(B, C, bands, generator, self.m2_mask_cfg)

    # -- session geometry -------------------------------------------------------
    def session_layout(
        self, parcel_of_electrode: Tensor, bands: tuple[BandSpecV2, ...]
    ) -> _SessionLayout:
        device = parcel_of_electrode.device
        labels, membership = active_parcels(parcel_of_electrode)
        parcel_idx = membership.int().argmax(0)                   # (C,)
        _, freq_id, slot = token_metadata(bands)
        cell_patch = cell_operator_index(bands, op_mode=self.cfg.pool_op_resolved)
        return _SessionLayout(
            labels=labels,
            membership=membership,
            parcel_idx=parcel_idx,
            freq_id=freq_id.to(device),
            slot=slot.to(device),
            cell_patch=cell_patch.to(device),
        )

    # -- M4 extended query set --------------------------------------------------
    def _m4_indices(
        self,
        tube_mask: Tensor,
        m2_mask: Tensor,
        sh: StaticShapesV2,
        *,
        tubed_only: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build the M4 held-out query index set ``(pos, cell, seed, tubed)``.

        Legacy query set = {tubed parcels: ALL S cells} ∪ {untubed parcels: their
        n_mask M2-masked cells}, ×k seeds. ``tubed_only=True`` (Run-A) drops the
        untubed block — M4 then predicts ONLY the tubed parcels (the untubed
        masked-cell inpaint moves to the M3 head). Flatten order = block
        (tubed‖untubed) then parcel-major / seed-mid / cell-minor. ``pos`` is the
        P-axis position, ``cell`` the S-axis position, ``seed`` the seed id;
        ``tubed`` (Lq,) flags the cross-parcel block. The matching target gather
        (``(pos·k+seed)·S+cell`` into the flattened teacher latent) is built in
        :meth:`forward` — the alignment is proven by the oracle test."""
        B = tube_mask.shape[0]
        S, k = sh.S, sh.k
        n_tube, P_vis, n_mask = sh.n_tube, sh.P_vis, sh.n_mask
        device = tube_mask.device
        arS = torch.arange(S, device=device)
        arK = torch.arange(k, device=device)

        tube_idx = _select_idx(tube_mask, n_tube)                 # (B, n_tube)

        # tubed block (B, n_tube, k, S): all S cells of each tubed parcel.
        t_pos = tube_idx[:, :, None, None].expand(B, n_tube, k, S)
        t_cell = arS[None, None, None, :].expand(B, n_tube, k, S)
        t_seed = arK[None, None, :, None].expand(B, n_tube, k, S)

        if tubed_only:
            pos, cell, seed = (
                t_pos.reshape(B, -1),
                t_cell.reshape(B, -1),
                t_seed.reshape(B, -1),
            )
            tubed = torch.ones(pos.shape[1], dtype=torch.bool, device=device)
            return pos, cell, seed, tubed

        # untubed block (B, P_vis, k, n_mask): the parcel's own M2-masked cells.
        untubed_idx = _select_idx(~tube_mask, P_vis)              # (B, P_vis)
        pmask_idx = _select_idx(m2_mask, n_mask)                  # (B, P, n_mask)
        u_pos = untubed_idx[:, :, None, None].expand(B, P_vis, k, n_mask)
        u_cell_pp = torch.gather(
            pmask_idx, 1, untubed_idx[:, :, None].expand(B, P_vis, n_mask)
        )                                                         # (B, P_vis, n_mask)
        u_cell = u_cell_pp[:, :, None, :].expand(B, P_vis, k, n_mask)
        u_seed = arK[None, None, :, None].expand(B, P_vis, k, n_mask)

        pos = torch.cat([t_pos.reshape(B, -1), u_pos.reshape(B, -1)], dim=1)
        cell = torch.cat([t_cell.reshape(B, -1), u_cell.reshape(B, -1)], dim=1)
        seed = torch.cat([t_seed.reshape(B, -1), u_seed.reshape(B, -1)], dim=1)
        Lt, Lu = n_tube * k * S, P_vis * k * n_mask
        tubed = torch.cat(
            [
                torch.ones(Lt, dtype=torch.bool, device=device),
                torch.zeros(Lu, dtype=torch.bool, device=device),
            ]
        )
        return pos, cell, seed, tubed

    def _m3_indices(
        self, tube_mask: Tensor, m2_mask: Tensor, sh: StaticShapesV2
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build the M3 held-out query index set ``(pos, cell, seed)`` — each
        UNTUBED parcel's n_mask M2-masked cells, ×k seeds.

        Identical structure to the legacy M4 untubed block, but M3 targets the
        teacher POOL output ``t_seeds`` (not the latent). The untubed-parcel order
        matches ``_select_idx(~tube_mask, P_vis)`` used to gather the M3 context
        (``s_in``) in :meth:`forward`, so context parcel p ↔ query parcel p. The
        target gather is ``(pos·k+seed)·S+cell`` into the flattened ``t_seeds``."""
        B = tube_mask.shape[0]
        S, k = sh.S, sh.k
        P_vis, n_mask = sh.P_vis, sh.n_mask
        device = tube_mask.device
        arK = torch.arange(k, device=device)

        untubed_idx = _select_idx(~tube_mask, P_vis)             # (B, P_vis)
        pmask_idx = _select_idx(m2_mask, n_mask)                 # (B, P, n_mask)
        u_pos = untubed_idx[:, :, None, None].expand(B, P_vis, k, n_mask)
        u_cell_pp = torch.gather(
            pmask_idx, 1, untubed_idx[:, :, None].expand(B, P_vis, n_mask)
        )                                                        # (B, P_vis, n_mask)
        u_cell = u_cell_pp[:, :, None, :].expand(B, P_vis, k, n_mask)
        u_seed = arK[None, None, :, None].expand(B, P_vis, k, n_mask)
        return u_pos.reshape(B, -1), u_cell.reshape(B, -1), u_seed.reshape(B, -1)

    # -- Run-B native-RAS geometry ---------------------------------------------
    def _centroid_relative(self, coords: Tensor, membership: Tensor, parcel_idx: Tensor):
        """``(centroid (P,3), u_e (C,3))`` — per-parcel centroid + each electrode's
        centroid-relative native-RAS offset (the translation-free frame the bias +
        recon query both read)."""
        memf = membership.to(coords.dtype)                        # (P,C)
        centroid = (memf @ coords) / memf.sum(1, keepdim=True).clamp_min(1.0)  # (P,3)
        u_e = coords - centroid[parcel_idx]                       # (C,3)
        return centroid, u_e

    def _pool_geom_bias(self, u_e: Tensor, membership: Tensor) -> Tensor:
        """Per-head seed→electrode relative bias ``(H, P·k, C)`` for the pool. Seed
        ``j``'s probe sits at ``seed_offset_j`` (centroid frame); its bias vs
        electrode ``e`` is ``geom(seed_offset_j − u_e)``. Independent of parcel (each
        seed reuses the same k offsets), so compute ``(k,C,H)`` and tile over P;
        cross-parcel pairs are membership-masked in the pool anyway."""
        assert self.geom is not None and self.pool_seed_offset is not None
        P, C = membership.shape
        k = self.cfg.k
        delta = self.pool_seed_offset[:, None, :] - u_e[None, :, :]  # (k,C,3)
        geom_k = self.geom(delta)                                 # (k,C,H)
        H = geom_k.shape[-1]
        geom_pk = geom_k[None].expand(P, k, C, H).reshape(P * k, C, H)
        return geom_pk.permute(2, 0, 1).contiguous()             # (H, P·k, C)

    def _melec_recon(
        self,
        s_seeds: Tensor,      # (B,P,k,S,d) student seeds (dropped electrodes blocked)
        t_front: Tensor,      # (B,C,S,d) teacher frontend (the recon target source)
        drop_mask: Tensor,    # (B,C) bool, True = dropped
        pvis_idx: Tensor,     # (B,P,s_vis) each parcel's visible-cell indices
        parcel_idx: Tensor,   # (C,) each electrode's parcel row
        u_e: Tensor,          # (C,3) centroid-relative offsets
        membership: Tensor,   # (P,C) bool
        cell_patch: Tensor,   # (S,) long — each cell's band op id (recon K/V typing)
        sh: StaticShapesV2,
    ) -> tuple[Tensor, Tensor]:
        """Reconstruct each DROPPED electrode's teacher frontend feature at its
        parcel's VISIBLE cells (seeds are real there) from the survivor-only seeds,
        addressed by the electrode's native-RAS offset. Returns flat
        ``(pred (N,d), target (N,d))`` with ``N = B·D·s_vis``, ``D`` = the static
        per-clip drop count. All gathers are static (``D`` fixed by membership)."""
        assert self.recon is not None and self.geom is not None
        assert self.cfg.m3_drop_frac is not None
        B, C, S, d = t_front.shape
        k, s_vis = sh.k, sh.s_vis
        nd = electrode_drop_count(membership, self.cfg.m3_drop_frac, self.cfg.m3_min_keep)
        D = int(nd.sum().item())
        if D == 0:                                               # nothing droppable
            z = t_front.new_zeros(0, d)
            return z, z
        drop_idx = _select_idx(drop_mask, D)                    # (B,D) electrode ids
        pd = parcel_idx[drop_idx]                               # (B,D) their parcels
        dvis = torch.gather(pvis_idx, 1, pd[:, :, None].expand(B, D, s_vis))  # (B,D,s_vis)

        # context seeds: gather parcel pd, then that parcel's visible cells.
        seeds_p = torch.gather(
            s_seeds, 1, pd[:, :, None, None, None].expand(B, D, k, S, d)
        )                                                       # (B,D,k,S,d)
        seeds_ctx = torch.gather(
            seeds_p, 3, dvis[:, :, None, :, None].expand(B, D, k, s_vis, d)
        )                                                       # (B,D,k,s_vis,d)
        seeds_ctx = seeds_ctx.permute(0, 1, 3, 2, 4).reshape(B * D * s_vis, k, d)

        # target: teacher frontend of the dropped electrode at those visible cells.
        tgt_e = torch.gather(t_front, 1, drop_idx[:, :, None, None].expand(B, D, S, d))
        tgt = torch.gather(
            tgt_e, 2, dvis[:, :, :, None].expand(B, D, s_vis, d)
        ).reshape(B * D * s_vis, d)

        # positional query features (per dropped electrode, broadcast over cells).
        feats_e = self.geom.features(u_e)                       # (C,F)
        feats = feats_e[drop_idx]                               # (B,D,F)
        Fdim = feats.shape[-1]
        feats = feats[:, :, None, :].expand(B, D, s_vis, Fdim).reshape(B * D * s_vis, Fdim)
        band = cell_patch[dvis].reshape(B * D * s_vis)          # (B·D·s_vis,) band per cell
        pred = self.recon(seeds_ctx, feats, band)              # (B·D·s_vis, d)
        return pred, tgt

    def _melec_recon_full(
        self,
        s_seeds: Tensor,      # (B,P,k,S,d) student seeds (dropped electrodes blocked)
        t_front: Tensor,      # (B,C,S,d) teacher frontend (the recon target source)
        drop_mask: Tensor,    # (B,C) bool, True = dropped
        empty_ps: Tensor,     # (B,P,S) bool, True = parcel-cell has no visible seed
        parcel_idx: Tensor,   # (C,) each electrode's parcel row
        u_e: Tensor,          # (C,3) centroid-relative offsets
        membership: Tensor,   # (P,C) bool
        cell_patch: Tensor,   # (S,) long — each cell's band op id (recon K/V typing)
        sh: StaticShapesV2,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Complete-grid (Run-B hetero) recon: reconstruct each DROPPED electrode's
        teacher frontend at ALL ``S`` cells from the survivor seeds at that cell,
        addressed by its native-RAS offset. Under hetero the pool exposes the full
        grid (the union of the survivors' per-electrode masks covers it), so there is
        no parcel-visible subset to gather — we predict every cell and mask the LOSS
        at the rare empty cell (all survivors masked there ⇒ nothing to interpolate).
        Returns ``(pred (N,d), target (N,d), weight (N,))`` with ``N = B·D·S`` and
        ``weight`` 0/1 (0 at empty cells)."""
        assert self.recon is not None and self.geom is not None
        assert self.cfg.m3_drop_frac is not None
        B, C, S, d = t_front.shape
        k = sh.k
        nd = electrode_drop_count(membership, self.cfg.m3_drop_frac, self.cfg.m3_min_keep)
        D = int(nd.sum().item())
        if D == 0:                                               # nothing droppable
            z = t_front.new_zeros(0, d)
            return z, z, t_front.new_zeros(0)
        drop_idx = _select_idx(drop_mask, D)                    # (B,D) electrode ids
        pd = parcel_idx[drop_idx]                               # (B,D) their parcels

        # context seeds: the dropped electrode's parcel, ALL S cells (complete grid).
        seeds_ctx = torch.gather(
            s_seeds, 1, pd[:, :, None, None, None].expand(B, D, k, S, d)
        )                                                       # (B,D,k,S,d)
        seeds_ctx = seeds_ctx.permute(0, 1, 3, 2, 4).reshape(B * D * S, k, d)

        # target: teacher frontend of the dropped electrode at all S cells.
        tgt = torch.gather(
            t_front, 1, drop_idx[:, :, None, None].expand(B, D, S, d)
        ).reshape(B * D * S, d)

        # positional query features (per dropped electrode, broadcast over cells).
        feats_e = self.geom.features(u_e)                       # (C,F)
        feats = feats_e[drop_idx]                               # (B,D,F)
        Fdim = feats.shape[-1]
        feats = feats[:, :, None, :].expand(B, D, S, Fdim).reshape(B * D * S, Fdim)
        band = cell_patch[None, None, :].expand(B, D, S).reshape(B * D * S)  # band per cell
        pred = self.recon(seeds_ctx, feats, band)              # (B·D·S, d)

        # loss weight: 0 where the dropped electrode's parcel-cell has no survivor
        # (empty seed) — don't supervise an impossible reconstruction.
        empty_d = torch.gather(empty_ps, 1, pd[:, :, None].expand(B, D, S))  # (B,D,S)
        weight = (~empty_d).to(pred.dtype).reshape(B * D * S)
        return pred, tgt, weight

    # -- the SSL step -----------------------------------------------------------
    def forward(
        self,
        lfs: Tensor,                  # (B, C, 28, T_lfs)  |STFT|, robust-z'd
        hga: Tensor,                  # (B, C, 7,  T_hga)
        parcel_of_electrode: Tensor,  # (C,) long  DKT label per electrode
        m2_mask: Tensor,              # (B, P, S) bool, True = M2 target
        tube_mask: Tensor,            # (B, P) bool, True = tubed parcel
        *,
        clip_len_s: float,
        coords: Tensor | None = None,      # (C,3) native-RAS mm — Run-B only
        drop_mask: Tensor | None = None,   # (B,C) bool, True = dropped — Run-B only
        m2_elec_mask: Tensor | None = None,  # (B,C,S) per-electrode M2 mask — Run-B hetero
        return_taps: bool = False,
    ) -> dict[str, Tensor]:
        if self.cfg.run_b and (coords is None or drop_mask is None):
            raise ValueError("Run-B forward requires coords (C,3) and drop_mask (B,C)")
        if self.cfg.run_b and self.cfg.m2_hetero and m2_elec_mask is None:
            raise ValueError("Run-B m2_hetero forward requires m2_elec_mask (B,C,S)")
        bands = bands_for_clip_len(clip_len_s, base=self.base_bands)
        lay = self.session_layout(parcel_of_electrode, bands)
        labels, membership, parcel_idx = lay.labels, lay.membership, lay.parcel_idx
        freq_id, slot, cell_patch = lay.freq_id, lay.slot, lay.cell_patch
        sh = compute_static_shapes_v2(
            m2_mask, tube_mask, membership, bands, k=self.cfg.k, cfg=self.m2_mask_cfg
        )
        B, C, P, S, d = sh.b, sh.c, sh.P, sh.S, self.cfg.d_model
        k, s_vis, n_mask = sh.k, sh.s_vis, sh.n_mask
        P_vis = sh.P_vis

        # === TEACHER (EMA, full shapes, no mask, detached) =====================
        # Route through encode_tokens with per-clip (S,) metadata so the teacher is
        # clip-len-parameterized (frontend.forward would use build-time buffers).
        with torch.no_grad():
            t_tok = self.teacher_frontend.tokenizer(lfs, hga)         # (B,C,S,d)
            t_front = self.teacher_frontend.encode_tokens(t_tok, freq_id, slot)
            t_seeds = self.teacher_pool(t_front, membership, labels, cell_patch)
            t_latent = self.teacher_latent(t_seeds, labels, slot)     # (B,P,k,S,d)

        # === STUDENT stage 1: frontend PACKED over visible cells ===============
        # Parcel-uniform (Run-A default): broadcast the parcel mask to its electrodes.
        # Heterogeneous (Run-B m2_hetero): each electrode holds out its OWN cells —
        # same per-row count ⇒ s_vis/n_mask unchanged. The pool then blocks each
        # electrode's masked cell (stage 4) so its zero-filled token can't leak.
        if m2_elec_mask is not None:
            if tuple(m2_elec_mask.shape) != (B, C, S):
                raise ValueError(
                    f"m2_elec_mask must be (B,C,S)=({B},{C},{S}), got {tuple(m2_elec_mask.shape)}"
                )
            elec_mask = m2_elec_mask                                  # (B,C,S) heterogeneous
        else:
            elec_mask = m2_mask[:, parcel_idx, :]                     # (B,C,S) parcel-broadcast
        visible = ~elec_mask
        vis_idx = _select_idx(visible, s_vis)                        # (B,C,s_vis)
        mask_idx = _select_idx(elec_mask, n_mask)                    # (B,C,n_mask)
        tok = self.frontend.tokenizer(lfs, hga)                       # (B,C,S,d)
        gtok = torch.gather(tok, 2, vis_idx[..., None].expand(B, C, s_vis, d))
        gfreq = freq_id[vis_idx]                                      # (B,C,s_vis)
        gslot = slot[vis_idx]
        s_front_vis = self.frontend.encode_tokens(gtok, gfreq, gslot)  # (B,C,s_vis,d)

        # === stage 2: M2 predictor (per-electrode; B·C rows) ===================
        ctx2 = s_front_vis.reshape(B * C, s_vis, d)
        ctx2_slot = gslot.reshape(B * C, s_vis)
        q2_slot = slot[mask_idx].reshape(B * C, n_mask)
        q2_freq = freq_id[mask_idx].reshape(B * C, n_mask)
        m2_pred = self.m2_predictor(ctx2, ctx2_slot, q2_slot, q2_freq).reshape(
            B, C, n_mask, d
        )
        m2_target = torch.gather(
            t_front, 2, mask_idx[..., None].expand(B, C, n_mask, d)
        )                                                            # (B,C,n_mask,d)

        # === stage 3: SCATTER visible features back to canonical S =============
        s_front_full = t_front.new_zeros(B, C, S, d)
        s_front_full.scatter_(2, vis_idx[..., None].expand(B, C, s_vis, d), s_front_vis)

        # === stage 4: pool over S (cheap; masked-cell seeds garbage-but-unused) =
        pool_block: Tensor | None = None
        pool_geom: Tensor | None = None
        u_e: Tensor | None = None
        if self.cfg.run_b:
            assert coords is not None and drop_mask is not None
            if tuple(coords.shape) != (C, 3):
                raise ValueError(f"coords must be (C={C},3), got {tuple(coords.shape)}")
            if tuple(drop_mask.shape) != (B, C):
                raise ValueError(f"drop_mask must be (B={B},C={C}), got {tuple(drop_mask.shape)}")
            _, u_e = self._centroid_relative(coords, membership, parcel_idx)
            pool_geom = self._pool_geom_bias(u_e, membership)         # (H,P·k,C)
            # Dropped electrodes are invisible to the STUDENT pool at every cell ⇒
            # its seeds pool the survivors only (the recon must interpolate them).
            pool_block = drop_mask[:, None, :].expand(B, S, C)        # (B,S,C)
            # Heterogeneous M2: also skip each electrode's OWN masked cells — its
            # token is zero-filled there (scattered visible-only), so pooling it
            # would leak a zero into a parcel-visible seed. Union with the drop.
            if m2_elec_mask is not None:
                pool_block = pool_block | m2_elec_mask.permute(0, 2, 1)  # (B,S,C)
        s_seeds = self.pool(
            s_front_full, membership, labels, cell_patch,
            block=pool_block, geom_bias=pool_geom,
        )                                                            # (B,P,k,S,d)

        # Complete-grid (Run-B hetero): different electrodes hold out DIFFERENT cells,
        # so their union covers the grid — the pool exposes the full S field and NO
        # parcel cell is dropped (unlike Run-A / uniform, which pvis-drop the masked
        # cells). A parcel-cell with no visible member (all survivors masked there) is
        # EMPTY = the finite-sentinel null; key-block it up the stack (latent + M4 —
        # a null no live token attends is a JEPA drop, not a mask token) and drop it
        # from the melec loss.
        hetero = self.cfg.run_b and m2_elec_mask is not None
        empty_ps: Tensor | None = None
        lat_key_mask: Tensor | None = None
        ctx4_key_mask: Tensor | None = None
        melec_weight: Tensor | None = None
        if hetero:
            assert pool_block is not None
            empty_ps = _empty_cell_mask(pool_block, membership)      # (B,P,S)

        # === stage 5: tag seeds, gather untubed parcels ========================
        if hetero:
            # Complete grid: keep ALL S cells (no pvis drop); the empty cells are
            # key-blocked in the latent/M4 instead. RoPE slots are the shared (S,)
            # clock — every parcel exposes the same full cell set.
            assert empty_ps is not None
            s_tag = s_seeds + self.latent.parcel_embed(labels)[None, :, None, None, :]
            untubed_idx = _select_idx(~tube_mask, P_vis)             # (B,P_vis)
            gu = untubed_idx[:, :, None, None, None].expand(B, P_vis, k, S, d)
            s_in = torch.gather(s_tag, 1, gu)                        # (B,P_vis,k,S,d)
            s_lat = S
            lat_slot: Tensor = slot                                 # (S,) shared clock
            ctx4_slot = slot.repeat(P_vis * k)[None].expand(B, P_vis * k * S)
            attend_unt = torch.gather(
                ~empty_ps, 1, untubed_idx[:, :, None].expand(B, P_vis, S)
            )                                                       # (B,P_vis,S)
            lat_key_mask = attend_unt[:, :, None, :].expand(B, P_vis, k, S)
            ctx4_key_mask = lat_key_mask.reshape(B, P_vis * k * S)
        else:
            # pvis path (Run-A + Run-B-uniform): drop each parcel's M2-masked cells
            # (Run-A's M3 predicts them; uniform masks every electrode identically so
            # the masked cell has no visible seed anyway).
            pvis_idx = _select_idx(~m2_mask, s_vis)                  # (B,P,s_vis)
            gpv = pvis_idx[:, :, None, :, None].expand(B, P, k, s_vis, d)
            s_seeds_vis = torch.gather(s_seeds, 3, gpv)              # (B,P,k,s_vis,d)
            s_tag = (
                s_seeds_vis
                + self.latent.parcel_embed(labels)[None, :, None, None, :]
            )
            untubed_idx = _select_idx(~tube_mask, P_vis)            # (B,P_vis)
            gu = untubed_idx[:, :, None, None, None].expand(B, P_vis, k, s_vis, d)
            s_in = torch.gather(s_tag, 1, gu)                       # (B,P_vis,k,s_vis,d)
            # per-row latent RoPE slots = the untubed parcels' visible-cell slots.
            pvis_slot = slot[pvis_idx]                              # (B,P,s_vis)
            untubed_slot = torch.gather(
                pvis_slot, 1, untubed_idx[:, :, None].expand(B, P_vis, s_vis)
            )                                                       # (B,P_vis,s_vis)
            lat_slot = (
                untubed_slot[:, :, None, :].expand(B, P_vis, k, s_vis).reshape(
                    B, P_vis * k * s_vis
                )
            )
            s_lat = s_vis
            ctx4_slot = lat_slot

        # === stage 6: latent self-attention over the seeds =====================
        s_latent = self.latent(s_in, None, lat_slot, key_mask=lat_key_mask)  # (B,P_vis,k,s_lat,d)

        # === stage 7: M4 (+ Run-A M3) predictor(s) ============================
        ctx4 = s_latent.reshape(B, P_vis * k * s_lat, d)
        t_latent_flat = t_latent.reshape(B, P * k * S, d)
        m3_pred: Tensor | None = None
        m3_target: Tensor | None = None
        melec_pred: Tensor | None = None
        melec_target: Tensor | None = None

        if self.cfg.run_b:
            # Run-B: Run-A's tubed-only M4, but the M3 pool-inpaint head is REPLACED
            # by the masked-electrode reconstruction (M-elec). M2 + M4 unchanged.
            pos, cell, seed, _tubed = self._m4_indices(
                tube_mask, m2_mask, sh, tubed_only=True
            )
            m4_pred = self.m4_predictor(
                ctx4, ctx4_slot, slot[cell], freq_id[cell],
                q_parcel=labels[pos], q_seed=seed,
                ctx_key_mask=ctx4_key_mask,          # hetero: drop empty context cells
            )
            flat4 = (pos * k + seed) * S + cell
            Lq4 = flat4.shape[1]
            m4_target = torch.gather(
                t_latent_flat, 1, flat4[..., None].expand(B, Lq4, d)
            )
            assert u_e is not None and drop_mask is not None
            if hetero:
                assert empty_ps is not None
                melec_pred, melec_target, melec_weight = self._melec_recon_full(
                    s_seeds, t_front, drop_mask, empty_ps, parcel_idx, u_e,
                    membership, cell_patch, sh
                )
            else:
                melec_pred, melec_target = self._melec_recon(
                    s_seeds, t_front, drop_mask, pvis_idx, parcel_idx, u_e,
                    membership, cell_patch, sh
                )
            m4_w = None
            if self.cfg.support_weight:
                n_elec = membership.sum(-1).to(m4_pred.dtype)
                m4_w = n_elec[pos].reshape(-1)
            empty = m2_pred.new_zeros(0, d)
            out = converged_v2_loss_per_head(
                m2_pred.reshape(-1, d), m2_target.reshape(-1, d),
                empty, empty,                                    # M3 replaced by M-elec
                m4_pred.reshape(-1, d), m4_target.reshape(-1, d),
                w_m2=self.cfg.w_m2, w_m3=0.0, w_m4=self.cfg.w_m4, m4_weight=m4_w,
                melec_pred=melec_pred, melec_target=melec_target,
                w_melec=self.cfg.w_melec, melec_weight=melec_weight,
            )
        elif self.cfg.run_a:
            # M4 predicts the TUBED parcels only (all S cells); the untubed
            # masked-cell inpaint is the separate M3 head (pool target).
            pos, cell, seed, _tubed = self._m4_indices(
                tube_mask, m2_mask, sh, tubed_only=True
            )
            m4_pred = self.m4_predictor(
                ctx4, ctx4_slot, slot[cell], freq_id[cell],
                q_parcel=labels[pos], q_seed=seed,
            )                                                        # (B,Lq4,d)
            flat4 = (pos * k + seed) * S + cell
            Lq4 = flat4.shape[1]
            m4_target = torch.gather(
                t_latent_flat, 1, flat4[..., None].expand(B, Lq4, d)
            )

            # M3: each UNTUBED parcel's M2-masked cells, predicting the teacher
            # POOL output ``t_seeds``. Context = ``s_in`` (the latent's input =
            # the untubed parcels' tagged visible-cell pool seeds) — same parcel
            # order ⇒ context parcel p ↔ query parcel p. ``lat_slot`` is its
            # per-row RoPE clock (the untubed parcels' visible-cell slots).
            assert self.m3_predictor is not None  # guaranteed by cfg.run_a
            ctx3 = s_in.reshape(B, P_vis * k * s_vis, d)
            m3_pos, m3_cell, m3_seed = self._m3_indices(tube_mask, m2_mask, sh)
            m3_pred = self.m3_predictor(
                ctx3, lat_slot, slot[m3_cell], freq_id[m3_cell],
                q_parcel=labels[m3_pos], q_seed=m3_seed,
            )                                                        # (B,Lq3,d)
            flat3 = (m3_pos * k + m3_seed) * S + m3_cell
            Lq3 = flat3.shape[1]
            t_seeds_flat = t_seeds.reshape(B, P * k * S, d)
            m3_target = torch.gather(
                t_seeds_flat, 1, flat3[..., None].expand(B, Lq3, d)
            )

            # Electrode-support weights: each query row weighted by its parcel's
            # electrode count n_elec(p) = membership[p].sum(). Convex weighted
            # mean in the loss (÷ Σ n_elec) ⇒ scale-preserving precision prior.
            m3_w: Tensor | None = None
            m4_w: Tensor | None = None
            if self.cfg.support_weight:
                n_elec = membership.sum(-1).to(m3_pred.dtype)        # (P,)
                m3_w = n_elec[m3_pos].reshape(-1)                    # (Q3,)
                m4_w = n_elec[pos].reshape(-1)                       # (Q4,)
            out = converged_v2_loss_per_head(
                m2_pred.reshape(-1, d), m2_target.reshape(-1, d),
                m3_pred.reshape(-1, d), m3_target.reshape(-1, d),
                m4_pred.reshape(-1, d), m4_target.reshape(-1, d),
                w_m2=self.cfg.w_m2, w_m3=self.cfg.w_m3, w_m4=self.cfg.w_m4,
                m3_weight=m3_w, m4_weight=m4_w,
            )
        else:
            # LEGACY: M4 over {tubed: all S} ∪ {untubed: masked cells}; shared-
            # denominator loss with ``_ln_target``. Resumes byte-identical.
            pos, cell, seed, tubed = self._m4_indices(tube_mask, m2_mask, sh)
            q_parcel = labels[pos]                                   # (B,Lq) DKT label
            q_freq = freq_id[cell]
            q_slot = slot[cell]
            m4_pred = self.m4_predictor(
                ctx4, ctx4_slot, q_slot, q_freq, q_parcel=q_parcel, q_seed=seed
            )                                                        # (B,Lq,d)
            flat = (pos * k + seed) * S + cell                       # (B,Lq) into P·k·S
            Lq = flat.shape[1]
            m4_target = torch.gather(
                t_latent_flat, 1, flat[..., None].expand(B, Lq, d)
            )                                                        # (B,Lq,d)
            n_p = membership.sum(1)                                   # (P,) long
            m4_weight = n_p[pos].float()                             # (B,Lq)  w_p=1
            out = converged_v2_loss(
                m2_pred.reshape(-1, d),
                m2_target.reshape(-1, d),
                m4_pred.reshape(-1, d),
                m4_target.reshape(-1, d),
                m4_weight.reshape(-1),
                tubed[None, :].expand(B, Lq).reshape(-1),
            )

        if return_taps:
            # Near-free detached SSL-health taps for the monitor callback. The loss
            # path above is unchanged — these are additive + `.detach()`'d, so a
            # tap-on step is bit-identical to a tap-off step (verified in tests).
            # Frontend/latent taps keep their native per-cell shapes (the callback
            # flattens to (N, d)); pred/target taps are the SAME flattened (Q, d)
            # rows the loss scores, so their stats match what the model predicts.
            out["_tap_teacher_frontend"] = t_front.detach()       # (B,C,S,d)
            out["_tap_student_latent"] = s_latent.detach()        # (B,P_vis,k,s_vis,d)
            out["_tap_m2_pred"] = m2_pred.reshape(-1, d).detach()  # (Q2,d)
            out["_tap_m2_target"] = m2_target.reshape(-1, d).detach()
            out["_tap_m4_pred"] = m4_pred.reshape(-1, d).detach()  # (Q4,d)
            out["_tap_m4_target"] = m4_target.reshape(-1, d).detach()
            if m3_pred is not None and m3_target is not None:
                out["_tap_m3_pred"] = m3_pred.reshape(-1, d).detach()   # (Q3,d)
                out["_tap_m3_target"] = m3_target.reshape(-1, d).detach()
                out["_tap_teacher_pool"] = t_seeds.detach()       # (B,P,k,S,d)
            if melec_pred is not None and melec_target is not None:
                out["_tap_melec_pred"] = melec_pred.detach()      # (N,d)
                out["_tap_melec_target"] = melec_target.detach()
                if hetero:
                    assert empty_ps is not None and melec_weight is not None
                    out["_tap_empty_cell"] = empty_ps.detach()    # (B,P,S) True=empty
                    out["_tap_melec_weight"] = melec_weight.detach()  # (N,) 0/1
                # Run-B pool tap: the SAME pool_rankme/pool_feat_std source Run-A emits
                # from its M3 block (the teacher-pool "M3 target" = the pooling-tax rank
                # diagnostic). t_seeds is computed unconditionally in the teacher block,
                # so emit it here too — else pool_rankme/feat_std never log for Run-B and
                # can't overlay Run-A. Detached ⇒ training bit-identical.
                out["_tap_teacher_pool"] = t_seeds.detach()       # (B,P,k,S,d)
            # Flag for the monitor: Run-A/Run-B regress RAW targets (no _ln_target),
            # so explained-variance must be measured against raw targets. Emitted
            # only in Run-A/Run-B; legacy omits it and the monitor's `.get()` defaults
            # to the legacy `_ln_target` behavior (keeps the legacy tap-key set).
            if self.cfg.run_a or self.cfg.run_b:
                out["_tap_raw_targets"] = t_front.new_tensor(1.0)
        return out

    @torch.no_grad()
    def encode_clip_taps(
        self,
        lfs: Tensor,                  # (B, C, 28, T_lfs)  |STFT|, robust-z'd
        hga: Tensor,                  # (B, C, 7,  T_hga)
        parcel_of_electrode: Tensor,  # (C,) long  DKT label per electrode
        *,
        clip_len_s: float,
        use_teacher: bool = False,
    ) -> dict[str, Tensor]:
        """Clean, mask-free clip encoding for the offline dev linear probe.

        The student (or EMA-teacher) towers run over the FULL token grid with NO M2/M4
        mask — i.e. exactly the teacher path of :meth:`forward` (the lines that build
        ``t_front`` / ``t_seeds`` / ``t_latent``), which is the only mask-free encode in
        the module. Returns the per-electrode frontend tap ``(B,C,S,d)``, the per-parcel
        POOL output ``seeds`` ``(B,P,k,S,d)`` (the M3 head's target surface, post-pool /
        pre-latent, terminal-LayerNorm'd) and the per-parcel latent tap ``(B,P,k,S,d)``
        (the M4 head's target surface), plus the session layout
        (``labels`` ``(P,)`` / ``membership`` ``(P,C)`` / ``parcel_idx`` ``(C,)``) the
        probe needs to pool electrodes→parcels and pick present parcels. No loss, no
        grad — purely for probing a trained checkpoint."""
        bands = bands_for_clip_len(clip_len_s, base=self.base_bands)
        lay = self.session_layout(parcel_of_electrode, bands)
        fe = self.teacher_frontend if use_teacher else self.frontend
        pool = self.teacher_pool if use_teacher else self.pool
        lat = self.teacher_latent if use_teacher else self.latent
        tok = fe.tokenizer(lfs, hga)                              # (B,C,S,d)
        front = fe.encode_tokens(tok, lay.freq_id, lay.slot)     # (B,C,S,d)
        seeds = pool(front, lay.membership, lay.labels, lay.cell_patch)  # (B,P,k,S,d)
        latent = lat(seeds, lay.labels, lay.slot)                # (B,P,k,S,d)
        # M3 attentive surface: the bare pool output carries NO parcel tag (the pool
        # is per-cell), so a single-query attentive readout that pools ACROSS parcels
        # would be parcel-blind. Re-add the latent's OWN parcel embed (`embed_p^pos`,
        # the same table the latent tags its input with at line 793) so the M3 token
        # carries the model's parcel id — no new readout embedding. The bare `pool`
        # is kept untagged for the keep-S ridge rung (per-parcel weight blocks make
        # the tag structural there; tagging would only shift the floor's numbers).
        # M4 (`latent`) already rode through that embed internally ⇒ no re-add.
        pool_tagged = seeds + lat.parcel_embed(lay.labels)[None, :, None, None, :]
        return {
            "frontend": front,
            "pool": seeds,
            "pool_tagged": pool_tagged,
            "latent": latent,
            "labels": lay.labels,
            "membership": lay.membership,
            "parcel_idx": lay.parcel_idx,
        }

    @torch.no_grad()
    def ema_step(self) -> None:
        """Advance the EMA teacher one step (call AFTER the optimizer step)."""
        ema_update(self.teacher_frontend, self.frontend, self.cfg.ema_tau)
        ema_update(self.teacher_pool, self.pool, self.cfg.ema_tau)
        ema_update(self.teacher_latent, self.latent, self.cfg.ema_tau)
