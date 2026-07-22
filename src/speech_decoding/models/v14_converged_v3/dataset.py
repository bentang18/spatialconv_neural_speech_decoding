"""v14_converged_v3 — V3SessionDataset (Phase D2/D4).

Map-style dataset over the per-session continuous |STFT| spec caches (memo
project-v3-pipeline-build-contract-2026-07-10, Option B: a fresh lightweight reader
of the ``.npy`` band caches, NOT the heavyweight ``MultiStftView``/neuralset path).

Each dataset index is one random-CONTINUOUS clip draw from a session. ``__getitem__``
(1) samples a guard-2-valid ``t0`` (``clip_sampler`` — reject-free span complement),
(2) memmap-reads the 3 band ``.npy`` caches at the survivor rows (``keep_idx``) over
``[t0, t0+clip_frames)``, (3) applies the session's FROZEN robust-z (the reused
``SessionRobustZNormalizer.from_stats``, its median/σ sliced to survivors at spec
build), and returns a ``V3ClipSample`` for the session-homogeneous collate.

Per-index session identity feeds the reused ``_SessionGroupedBatchSampler`` (via
``session_key_list``) so every batch is one session ⇒ shared geom/parcel, static
shapes, no electrode padding. ``clips_per_session`` sets the per-epoch clip budget;
``set_epoch`` re-seeds the per-index draw so each epoch sees fresh windows.

The per-session ``V3SessionSpec`` (band paths + frozen stats + setup + bad spans) is
assembled by ``build_session_spec`` from already-loaded fixtures; the braintreebank
file-format adapter that produces those fixtures from disk is a thin layer validated
on DeltaAI at F2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import gcd

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# Default per-band read rate relative to the 32 Hz clip clock: (num, den) s.t. a band
# frame index = 32Hz_index · num // den. Uniform (1,1) = arm0 (all bands cached @32 Hz,
# stem decimates). Native fine-HGA = ((1,8),(1,2),(4,1)) (SLOW 4 Hz / MID 16 Hz / HGA
# 128 Hz), extracted at native rate — the stem consumes them directly (memo
# project-fine-hga-bt-rebake-tasklist-2026-07-21).
UNIFORM_BAND_RATES: tuple[tuple[int, int], ...] = ((1, 1), (1, 1), (1, 1))

# r5 Chang 2-stream: both bands (v3hga, v3lfs) cached @64 Hz = 2× the 32 Hz clip clock;
# the EarlyFusionStem pools 2 frames → 1 token (stride-2). So each band reads 2× frames.
R5_BAND_RATES: tuple[tuple[int, int], ...] = ((2, 1), (2, 1))
# Fine-HGA OFAT native rates vs the 32 Hz clip clock (SLOW 4 Hz = 1/8, MID 16 Hz = 1/2,
# HGA 128 Hz = 4/1) — the SINGLE source of truth the dispatch --frontend flag reads.
NATIVE_FINE_BAND_RATES: tuple[tuple[int, int], ...] = ((1, 8), (1, 2), (4, 1))


def _start_align(band_rates: Sequence[tuple[int, int]]) -> int:
    """Binding clip-start alignment = lcm of band denominators (t0 must be ≡0 mod it
    so every t0·num//den is an integer landing on a shared window lattice)."""
    align = 1
    for _num, den in band_rates:
        align = align * den // gcd(align, den)
    return align


def reference_n_frames(
    total_frames: Sequence[int], band_rates: Sequence[tuple[int, int]]
) -> int:
    """The 32 Hz clip-clock length from per-band NATIVE frame counts.

    Each band supports ``total_frames_b · den // num`` 32 Hz frames; the common clock
    is the min (so no band read goes out of bounds), floored to the clip-start
    alignment so the last window start stays aligned. Uniform rates ((1,1)) reduce to
    ``min(total_frames)`` — identical to taking any single band's count when equal.
    """
    caps = [tf * den // num for tf, (num, den) in zip(total_frames, band_rates)]
    align = _start_align(band_rates)
    return (min(caps) // align) * align

from speech_decoding.extractors.normalize import SessionRobustZNormalizer
from speech_decoding.models.v14_converged_v3.batch import V3ClipSample
from speech_decoding.models.v14_converged_v3.clip_sampler import sample_clip_start
from speech_decoding.models.v14_converged_v3.session_setup import V3SessionSetup


@dataclass(frozen=True)
class V3SessionSpec:
    """Per-session fixtures the dataset draws clips from (survivor-aligned)."""

    session_key: tuple  # (subject_id, trial_id)
    band_paths: tuple[str, ...]  # 3 × .npy path, each (C_full, F_band, T_total) fp32
    band_stats: tuple[tuple[Tensor, Tensor], ...]  # 3 × (median, sigma) sliced to (N,F,1)
    band_norms: tuple[SessionRobustZNormalizer, ...]  # frozen robust-z per band
    keep_idx: Tensor  # (N,) long — survivor rows into the .npy C dim
    setup: V3SessionSetup
    n_frames: int  # T_total of the caches
    bad_spans_s: list[tuple[float, float]]  # guard-2 spans, hop-invariant seconds


def build_session_spec(
    *,
    session_key: tuple,
    band_paths: Sequence[str],
    band_stats: Sequence[tuple[Tensor, Tensor]],
    setup: V3SessionSetup,
    n_frames: int,
    bad_spans_s: Sequence[tuple[float, float]],
    sigma_floor: float = 1e-6,
    winsor: float | Sequence[float] | None = None,
) -> V3SessionSpec:
    """Slice the full-C robust-z stats to survivors and freeze a normalizer per band.

    ``band_stats`` arrive at the caches' full ``C`` rows (as stored in ``.stats.npz``);
    ``keep_idx`` selects the survivor rows so median/σ align to the clip rows the
    mmap read returns. ``winsor``/``sigma_floor`` are applied at transform (reused
    ``SessionRobustZNormalizer`` contract), not baked into σ. ``winsor`` may be a
    single |z| cap broadcast to all bands, or a per-band sequence (v3 default
    SLOW/MID 15, HGA 20 — HGA's heavier tails get the looser clamp).
    """
    keep = setup.keep_idx
    sliced = tuple((med[keep], sig[keep]) for med, sig in band_stats)
    if winsor is None or isinstance(winsor, (int, float)):
        winsor_bands: tuple[float | None, ...] = (winsor,) * len(sliced)
    else:
        winsor_bands = tuple(winsor)
        if len(winsor_bands) != len(sliced):
            raise ValueError(
                f"winsor sequence length {len(winsor_bands)} != n_bands {len(sliced)}"
            )
    norms = tuple(
        SessionRobustZNormalizer.from_stats(
            median=med, sigma=sig, sigma_floor=sigma_floor, winsor=w
        )
        for (med, sig), w in zip(sliced, winsor_bands)
    )
    return V3SessionSpec(
        session_key=session_key,
        band_paths=tuple(band_paths),
        band_stats=sliced,
        band_norms=norms,
        keep_idx=keep,
        setup=setup,
        n_frames=int(n_frames),
        bad_spans_s=list(bad_spans_s),
    )


class V3SessionDataset(Dataset):
    """Random-continuous clips over per-session spec caches (guard-2 + robust-z)."""

    def __init__(
        self,
        sessions: Sequence[V3SessionSpec],
        *,
        clips_per_session: int,
        clip_frames: int,
        fps: float,
        seed: int = 0,
        band_rates: Sequence[tuple[int, int]] = UNIFORM_BAND_RATES,
    ) -> None:
        self.sessions = list(sessions)
        self.clips_per_session = int(clips_per_session)
        self.clip_frames = int(clip_frames)
        self.fps = float(fps)
        self._seed = int(seed)
        self.band_rates = tuple((int(n), int(d)) for n, d in band_rates)
        # band-count agnostic: rates must align to each session's bands (3 for r4/arm0
        # slow/mid/hga, 2 for r5 v3hga/v3lfs). __getitem__ zips over band_paths.
        for s in self.sessions:
            if len(s.band_paths) != len(self.band_rates):
                raise ValueError(
                    f"session {s.session_key}: {len(s.band_paths)} bands != "
                    f"{len(self.band_rates)} band_rates"
                )
        self.start_align = _start_align(self.band_rates)
        # invariant: every per-band clip length clip_frames·num//den must be integer,
        # else the band clip has an undefined boundary. Fail loud at construction.
        for num, den in self.band_rates:
            if (self.clip_frames * num) % den != 0:
                raise ValueError(
                    f"clip_frames {self.clip_frames} × {num}/{den} not integer — band "
                    f"clip length undefined; clip_frames must be divisible by {den}"
                )
        self._epoch = 0
        self._index: list[tuple[int, int]] = [
            (si, k)
            for si in range(len(self.sessions))
            for k in range(self.clips_per_session)
        ]
        self._last_t0: int | None = None  # test/monitor hook: the t0 last drawn

    def __len__(self) -> int:
        return len(self._index)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def session_key_list(self) -> list[tuple]:
        """Per-index session key for the _SessionGroupedBatchSampler."""
        return [self.sessions[si].session_key for si, _ in self._index]

    def _draw_seed(self, index: int) -> int:
        # deterministic per (base seed, epoch, index) — fresh windows each epoch,
        # reproducible across ranks/workers that share seed + epoch.
        return (
            self._seed * 1_000_003
            + self._epoch * 10_000_019
            + index * 2_654_435_761
        ) & 0x7FFFFFFF

    def __getitem__(self, index: int) -> V3ClipSample:
        si, _ = self._index[index]
        s = self.sessions[si]
        g = torch.Generator().manual_seed(self._draw_seed(index))
        t0 = sample_clip_start(
            n_frames=s.n_frames,
            clip_frames=self.clip_frames,
            bad_spans_s=s.bad_spans_s,
            fps=self.fps,
            generator=g,
            start_align=self.start_align,
        )
        self._last_t0 = t0
        keep = s.keep_idx.numpy()
        end = t0 + self.clip_frames
        bands = []
        for path, norm, (num, den) in zip(s.band_paths, s.band_norms, self.band_rates):
            lo, hi = t0 * num // den, end * num // den  # native-rate band window
            mm = np.load(path, mmap_mode="r")  # (C_full, F_band, T_band)
            clip = np.asarray(mm[keep, :, lo:hi], dtype=np.float32)  # (N, F, T_band)
            del mm
            bands.append(norm.transform(torch.from_numpy(clip)))
        return V3ClipSample(
            bands=tuple(bands), setup=s.setup, session_key=s.session_key
        )
