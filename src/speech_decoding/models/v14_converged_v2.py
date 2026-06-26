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

    def forward(
        self,
        lfs: Tensor,
        hga: Tensor,
        *,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        """``(B,C,28,T_lfs)/(B,C,7,T_hga)`` → per-electrode features ``(B, C, S, d)``."""
        tok = self.tokenizer(lfs, hga)                            # (B, C, S, d)
        B, C, S, d = tok.shape
        tok = tok + self.freq_embed[self.tokenizer.freq_patch_id]  # + (S, d)
        rope = self.key_rope[:, self.tokenizer.time_slot, :]      # (2, S, head_dim)
        x = tok.reshape(B * C, S, d)
        km = None if key_mask is None else key_mask.reshape(B * C, S)
        for blk in self.blocks:
            x = blk(x, rope, km)
        return self.ln_out(x).reshape(B, C, S, d)


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
        parcel_labels: Tensor,    # (P,) long  DKT ids
        cell_time_slot: Tensor,   # (S,) long  RoPE clock per cell
        *,
        key_mask: Tensor | None = None,  # (B, P, k, S) bool, True = attendable
    ) -> Tensor:
        B, P, k, S, d = seeds.shape
        if cell_time_slot.shape != (S,):
            raise ValueError(
                f"cell_time_slot must be (S={S},), got {tuple(cell_time_slot.shape)}"
            )
        x = seeds + self.parcel_embed(parcel_labels)[None, :, None, None, :]
        x = x.reshape(B, P * k * S, d)
        # token order is (p, j, s) p-major s-minor ⇒ tile cell slots across P·k.
        slot = cell_time_slot.repeat(P * k)                       # (P·k·S,)
        rope = self.key_rope[:, slot, :]                          # (2, N, head_dim)
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
