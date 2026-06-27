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
identity (4 freq-embeds at the frontend; the pool's freq-specific weights map
these → 2 reading operators tied-default / 4 untied-ablation).

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
            [_JointTokenBlock(d_model, n_heads) for _ in range(n_layers)]
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


def sample_m2_masks_v2(
    B: int,
    P: int,
    bands: tuple[BandSpecV2, ...],
    generator: torch.Generator,
    cfg: M2MaskConfigV2 = M2MaskConfigV2(),
) -> Tensor:
    """Parcel-uniform M2 masks ``(B, P, S)`` bool (True = held out / an M2 target).

    Constant count ``n_mask = n_mask_hga + T_lfs`` per parcel. VECTORIZED over all
    ``B·P`` parcels — no per-parcel/per-electrode python loop. Token order matches
    :func:`token_metadata`: LFS groups 0/1/2 (each ``T_lfs``) then HGA (``T_hga``)."""
    lfs, hga = bands
    if (lfs.name, hga.name) != ("lfs", "hga"):
        raise ValueError(f"expected (lfs, hga) bands, got {(lfs.name, hga.name)}")
    R = B * P
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

    m2 = torch.cat([lfs_mask, hga_mask], dim=1)               # (R, S)
    return m2.reshape(B, P, -1)


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
        # {tubed parcels: all S cells} ∪ {untubed: the n_mask masked cells}, ×k seeds.
        return self.b * (self.n_tube * self.k * self.S + self.P_vis * self.k * self.n_mask)


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
    bands: tuple[BandSpecV2, ...] = BANDS_V2, *, tie_lfs: bool = True
) -> Tensor:
    """``(S,)`` long → which pool reading-operator (`W_K`/`W_V`) each cell uses.

    TIED DEFAULT (`tie_lfs=True`, `n_op=2`): operator = ``band_id`` — the 3 LFS
    log groups SHARE one operator (0), HGA its own (1). UNTIED ablation
    (`tie_lfs=False`, `n_op=4`): operator = ``freq_patch_id`` (LFS 0/1/2, HGA 3).
    The exact integer labels are immaterial; what matters is the grouping +
    that ``W_K`` is sized to ``n_operators(tie_lfs)``."""
    band_id, freq_patch_id, _ = token_metadata(bands)
    return band_id.clone() if tie_lfs else freq_patch_id.clone()


def n_operators(tie_lfs: bool = True) -> int:
    return 2 if tie_lfs else 4


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

        # Queries: base seed color (k) + per-parcel query embed (universal,
        # DKT-sized) + shared W_Q. Freq/time-agnostic.
        self.base_seed = nn.Parameter(torch.empty(k, d_model))
        nn.init.trunc_normal_(self.base_seed, std=0.02)
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

    def forward(
        self,
        x: Tensor,                # (B, C, S, d)
        membership: Tensor,       # (P, C) bool
        parcel_labels: Tensor,    # (P,) long  DKT ids
        cell_patch: Tensor,       # (S,) long  ∈ [0, n_op)
    ) -> Tensor:
        B, C, S, d = x.shape
        P = membership.shape[0]
        H, hd, k = self.n_heads, self.head_dim, self.k
        if cell_patch.shape != (S,):
            raise ValueError(f"cell_patch must be (S={S},), got {tuple(cell_patch.shape)}")

        # Queries (P, k, d) → W_Q → (P·k, H, hd). Cell-independent.
        q_tok = self.embed_pq[parcel_labels][:, None, :] + self.base_seed[None, :, :]
        q = self.W_Q(q_tok).reshape(P * k, H, hd)

        # Per-cell freq-specific key/value projection — gather operator, einsum.
        WK = self.W_K[cell_patch]                                # (S, d, d)
        WV = self.W_V[cell_patch]                                # (S, d, d)
        keys = torch.einsum("bcsd,sde->bcse", x, WK)             # (B, C, S, d)
        vals = torch.einsum("bcsd,sde->bcse", x, WV)

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
        ctx = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias)  # (BS,H,Pk,hd)
        ctx = ctx.permute(0, 2, 1, 3).reshape(B * S, P * k, d)

        # No-coverage rows (none for active parcels, kept for safety) → 0.
        no_cov = ~mem_pk.any(dim=-1)                                # (Pk,)
        ctx = ctx.masked_fill(no_cov[None, :, None], 0.0)
        out = self.out(ctx)                                         # (BS, Pk, d)
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
            [_JointTokenBlock(d_model, n_heads) for _ in range(n_layers)]
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
            [_JointTokenBlock(pred_dim, n_heads) for _ in range(n_layers)]
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
    ``tube_ratio=0.25``, ``tie_lfs=True`` (n_op=2, untie→4 is the first ablation),
    ``ema_tau=0.9992``."""

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
    tie_lfs: bool = True
    ema_tau: float = 0.9992


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
        n_op = n_operators(cfg.tie_lfs)

        self.frontend = FrontendEncoderV2(
            cfg.d_model, cfg.n_heads, cfg.frontend_layers, bands=self.ssl_bands
        )
        self.pool = SetPoolV2(
            cfg.d_model, cfg.n_heads, k=cfg.k, n_parcels=cfg.n_parcels, n_op=n_op
        )
        self.latent = LatentEncoderV2(
            cfg.d_model, cfg.n_heads, cfg.latent_layers, cfg.n_parcels,
            ssl_bands=self.ssl_bands,
        )
        self.m2_predictor = JepaPredictorV2(
            cfg.d_model, cfg.pred_dim, cfg.n_heads, cfg.m2_pred_layers,
            ssl_bands=self.ssl_bands,
        )
        self.m4_predictor = JepaPredictorV2(
            cfg.d_model, cfg.pred_dim, cfg.n_heads, cfg.m4_pred_layers,
            n_parcels=cfg.n_parcels, k=cfg.k, ssl_bands=self.ssl_bands,
        )

        # EMA teacher — frozen deep copies of the three target-side towers.
        self.teacher_frontend = copy.deepcopy(self.frontend)
        self.teacher_pool = copy.deepcopy(self.pool)
        self.teacher_latent = copy.deepcopy(self.latent)
        for m in (self.teacher_frontend, self.teacher_pool, self.teacher_latent):
            for p in m.parameters():
                p.requires_grad_(False)

        self.m2_mask_cfg = M2MaskConfigV2()

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

    # -- session geometry -------------------------------------------------------
    def session_layout(
        self, parcel_of_electrode: Tensor, bands: tuple[BandSpecV2, ...]
    ) -> _SessionLayout:
        device = parcel_of_electrode.device
        labels, membership = active_parcels(parcel_of_electrode)
        parcel_idx = membership.int().argmax(0)                   # (C,)
        _, freq_id, slot = token_metadata(bands)
        cell_patch = cell_operator_index(bands, tie_lfs=self.cfg.tie_lfs)
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
        self, tube_mask: Tensor, m2_mask: Tensor, sh: StaticShapesV2
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build the M4 held-out query index set ``(pos, cell, seed, tubed)``.

        Query set = {tubed parcels: ALL S cells} ∪ {untubed parcels: their n_mask
        M2-masked cells}, ×k seeds. Flatten order = block (tubed‖untubed) then
        parcel-major / seed-mid / cell-minor. ``pos`` is the P-axis position (for
        the teacher gather + n_p weight), ``cell`` the S-axis position, ``seed`` the
        seed id; ``tubed`` (Lq,) flags the cross-parcel block. The matching target
        gather (``(pos·k+seed)·S+cell`` into the flattened teacher latent) is built
        in :meth:`forward` from these — the alignment is proven by the oracle test."""
        B = tube_mask.shape[0]
        S, k = sh.S, sh.k
        n_tube, P_vis, n_mask = sh.n_tube, sh.P_vis, sh.n_mask
        device = tube_mask.device
        arS = torch.arange(S, device=device)
        arK = torch.arange(k, device=device)

        tube_idx = _select_idx(tube_mask, n_tube)                 # (B, n_tube)
        untubed_idx = _select_idx(~tube_mask, P_vis)              # (B, P_vis)
        pmask_idx = _select_idx(m2_mask, n_mask)                  # (B, P, n_mask)

        # tubed block (B, n_tube, k, S): all S cells of each tubed parcel.
        t_pos = tube_idx[:, :, None, None].expand(B, n_tube, k, S)
        t_cell = arS[None, None, None, :].expand(B, n_tube, k, S)
        t_seed = arK[None, None, :, None].expand(B, n_tube, k, S)

        # untubed block (B, P_vis, k, n_mask): the parcel's own M2-masked cells.
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
        return_taps: bool = False,
    ) -> dict[str, Tensor]:
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
        elec_mask = m2_mask[:, parcel_idx, :]                         # (B,C,S) True=masked
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
        s_seeds = self.pool(s_front_full, membership, labels, cell_patch)  # (B,P,k,S,d)

        # === stage 5: gather visible seeds/parcel, tag, gather untubed parcels ==
        pvis_idx = _select_idx(~m2_mask, s_vis)                       # (B,P,s_vis)
        gpv = pvis_idx[:, :, None, :, None].expand(B, P, k, s_vis, d)
        s_seeds_vis = torch.gather(s_seeds, 3, gpv)                   # (B,P,k,s_vis,d)
        s_tag = (
            s_seeds_vis
            + self.latent.parcel_embed(labels)[None, :, None, None, :]
        )
        untubed_idx = _select_idx(~tube_mask, P_vis)                  # (B,P_vis)
        gu = untubed_idx[:, :, None, None, None].expand(B, P_vis, k, s_vis, d)
        s_in = torch.gather(s_tag, 1, gu)                            # (B,P_vis,k,s_vis,d)

        # per-row latent RoPE slots = the untubed parcels' visible-cell slots.
        pvis_slot = slot[pvis_idx]                                   # (B,P,s_vis)
        untubed_slot = torch.gather(
            pvis_slot, 1, untubed_idx[:, :, None].expand(B, P_vis, s_vis)
        )                                                            # (B,P_vis,s_vis)
        lat_slot = (
            untubed_slot[:, :, None, :].expand(B, P_vis, k, s_vis).reshape(
                B, P_vis * k * s_vis
            )
        )

        # === stage 6: latent self-attention over the gathered seeds ============
        s_latent = self.latent(s_in, None, lat_slot)                 # (B,P_vis,k,s_vis,d)

        # === stage 7: M4 predictor (extended held-out query set) ===============
        ctx4 = s_latent.reshape(B, P_vis * k * s_vis, d)
        ctx4_slot = lat_slot
        pos, cell, seed, tubed = self._m4_indices(tube_mask, m2_mask, sh)
        q_parcel = labels[pos]                                       # (B,Lq) DKT label
        q_freq = freq_id[cell]
        q_slot = slot[cell]
        m4_pred = self.m4_predictor(
            ctx4, ctx4_slot, q_slot, q_freq, q_parcel=q_parcel, q_seed=seed
        )                                                            # (B,Lq,d)
        flat = (pos * k + seed) * S + cell                          # (B,Lq) into P·k·S
        t_latent_flat = t_latent.reshape(B, P * k * S, d)
        Lq = flat.shape[1]
        m4_target = torch.gather(
            t_latent_flat, 1, flat[..., None].expand(B, Lq, d)
        )                                                            # (B,Lq,d)
        n_p = membership.sum(1)                                       # (P,) long
        m4_weight = n_p[pos].float()                                 # (B,Lq)  w_p=1

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
        the module. Returns the per-electrode frontend tap ``(B,C,S,d)`` and the
        per-parcel latent tap ``(B,P,k,S,d)``, plus the session layout
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
        return {
            "frontend": front,
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
