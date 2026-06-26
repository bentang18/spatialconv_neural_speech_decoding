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

from dataclasses import dataclass, replace

import torch
from torch import Tensor

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
