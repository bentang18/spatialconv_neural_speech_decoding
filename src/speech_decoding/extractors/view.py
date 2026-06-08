"""V14 view extractors: single-STFT (legacy) + Multi-STFT (T1.5, 5/22 pivot).

T1.5 / 5/22 spec lock (project_v14_imindbench_multistft_pivot_2026_05_22.md),
FE-RAW-1 amendment (project_v14_fe_raw_frontend_f50_2026_06_04, LANDED):
v14's front-end-of-record is ``MultiStftView`` — 3 STFTs at common
hop=128 @ 2048 Hz (16 Hz front-end rate → 8 Hz latent after the (5,2)
patch stem; iMINDBench-standard hop, hop=128 re-lock 2026-06-03;
Nperseg=1024/512/256). ``front_end="raw"`` (DEFAULT) keeps the RAW
``|STFT|`` bins — F=50 (low k2–15 + mid k8–37 + hi k19–24) — and is the
executed front-end of record. The 30-bin ⅓-octave filterbank with per-bin
STFT routing (k0–k14 low, k15–k21 mid, k22–k29 hi) is DEMOTED to
``front_end="fbank"`` (the ``R-filterbank-30bin`` P1 low-data sister).
``LogStftView`` (single-STFT) is preserved as the F-single-STFT sister
cell and the fast smoke-test front-end; per B32 it is still the wired
dispatch default ``electrode_tokens_extractor`` until the Multi-STFT
cache lands.

5/25 swap: ``apply_log`` defaults to ``False`` (abs magnitude, no log
compression) — matches iMINDBench's Multi-STFT track which standardizes raw
``|STFT|`` per (channel, freq, session) without log. The D-SigLIP/Whisper
input-scale-matching rationale that motivated log compression died on 5/22
(DINOv3 dropped, Stage 3a deleted; distillation now targets the Whisper-L8
hidden state, not the log-mel input). ``F-log-amplitude`` is the demoted
sister cell — set ``apply_log=True`` to recover the pre-5/25 behavior.

Per-corpus valid-bin mask: ``multi_stft_raw_valid_bin_mask(low_hz, high_hz)``
— boolean (F=50,) tensor for the ``front_end="raw"`` default
(``multi_stft_valid_bin_mask`` is the (F=30,) ⅓-octave-filterbank analog for
the ``front_end="fbank"`` sister) — flagging which bins are spanned by a
corpus's recording passband. Used by SSL losses (input fill 0, key mask
−∞, L_recon target mask excludes invalid bins).


Inherits :class:`speech_decoding.extractors.reference.CARIeegExtractor` (which
itself inherits NeuralSet's :class:`neuralset.extractors.neuro.IeegExtractor`),
so the v14 default preprocessing chain ``N1 × R2 × F1 × A.0`` runs in front of
the STFT:

  * ``N1 train-set z-score``  → ``scaler="StandardScaler"``
  * ``R2 shaft CAR``          → ``car="shaft"``
  * ``F1 60 Hz notch``        → ``notch_filter=60.0``
  * ``A.0 [0, 1]s anchor``    → ``start=0.0, duration=1.0`` at the segmenter

Then applies a torch-STFT byte-equivalent to the upstream Neuroprobe baseline
helper ``preprocess_stft(..., preprocess="stft_abs")``. Log compression
(``log(x + eps)``) is opt-in via ``apply_log=True``.

``LogStftView`` (single-STFT) output shape: ``(C, F_bin=38, T_bin=17)``
per 1-s @ 2048 Hz Ieeg trigger window (nperseg=512, hop=128). Time-last
per NeuralSet's :class:`~neuralset.base.TimedArray` convention
(``frequency = T_bin / duration = 17 Hz``). ``MultiStftView`` instead
emits ``(C, F=50, T_bin)`` (``front_end="raw"`` default; the demoted
``front_end="fbank"`` sister emits ``(C, F=30, T_bin)``) at the common
hop=128 → 16 Hz front-end rate (hop=128 re-lock 2026-06-03). The v14 encoder
transposes to ``(C, T_bin, F_bin)`` internally via its ``time_last_input`` flag.

STFT params source: Neuroprobe Section D (page 18) — ``nperseg=512``,
``poverlap=0.75``, ``window=hann``, freq range ``0-150 Hz``, sample rate
``2048 Hz``. Frame count math: ``torch.stft(center=True)`` reflect-pads
``nperseg//2 = 256`` samples each side → padded length 2560 →
``1 + (2560 - 512) // 128 = 17`` frames. Freq bin math: ``rfftfreq(512,
1/2048)`` has bins at ``[0, 4, 8, ..., 1024]`` Hz; keeping ``0 ≤ f ≤ 150``
selects 38 bins.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import time
import typing as tp
import uuid
from pathlib import Path

import numpy as np
import torch
from neuralset.base import TimedArray
from pydantic import model_validator

from speech_decoding.extractors.quality import lof_bad_channel_mask
from speech_decoding.extractors.reference import CARIeegExtractor
from speech_decoding.extractors.normalize import SessionRobustZNormalizer

logger = logging.getLogger(__name__)


def _reset_infra_uid_cache(infra) -> None:
    """Clear exca's MEMOIZED uid on a freshly model_copy(deep=True)'d infra so it
    recomputes from its own (post-update) config. ``model_copy(deep=True)`` deep-
    copies the memo (``_uid`` and ``_state.uid``); if the source infra's uid was
    ever computed before the copy, the copy would otherwise return the SOURCE's
    stale uid — silently writing to the wrong cache namespace. Touches only ``_uid``
    and ``_state.uid`` (NOT ``_uid_string``, the format template, which ``uid()``
    needs). Defensive: a no-op if exca's internals move, never raises — worst case
    reverts to the order-dependent behavior that is correct on today's path."""
    priv = getattr(infra, "__pydantic_private__", None) or {}
    if "_uid" in priv:
        priv["_uid"] = None
    state = priv.get("_state")
    if state is not None and hasattr(state, "uid"):
        state.uid = None


# ---------------------------------------------------------------------------
# Multi-STFT helpers (T1.5)
# ---------------------------------------------------------------------------

MULTI_STFT_N_BINS: int = 30
MULTI_STFT_F0_HZ: float = 1.0
MULTI_STFT_OCTAVE_STEP: float = 1.0 / 3.0          # log ⅓-octave spacing
MULTI_STFT_HALF_BW_OCTAVES: float = 0.5            # triangular kernel half-width
# Routing per the 5/22 spec: which STFT each output bin sources from.
# Index 0 = STFT_low (Nperseg=1024), 1 = STFT_mid (Nperseg=512), 2 = STFT_hi (Nperseg=256).
MULTI_STFT_ROUTING: tuple[int, ...] = tuple([0] * 15 + [1] * 7 + [2] * 8)


# ---------------------------------------------------------------------------
# FE-RAW-1 (2026-06-04): raw |STFT| bins replace the const-Q filterbank as the
# live front end. Spec: reports/fe_raw_frontend_handoff_2026_06_04.md §4.2.
#
# Non-overlapping crossovers at 32 / 152 Hz, drop-delta floor 4 Hz, cap 192 Hz.
# Each (nperseg, k_start, k_end) selects the inclusive rfft-bin slice for one
# source STFT; ``k = f / Δf`` with Δf = sample_rate / nperseg. At 2048 Hz:
#   low  (Δf=2)  k2 (4 Hz)  … k15 (30 Hz)   → 14 bins
#   mid  (Δf=4)  k8 (32 Hz) … k37 (148 Hz)  → 30 bins
#   hi   (Δf=8)  k19 (152)  … k24 (192 Hz)  → 6 bins
# Concatenation order low → mid → hi gives a strictly-increasing, piecewise-
# uniform (2/4/8 Hz) freq axis of length F_RAW = 50.
# ---------------------------------------------------------------------------
F_RAW: int = 50
# (stft_idx, k_start, k_end inclusive). stft_idx 0=low(1024) 1=mid(512) 2=hi(256).
MULTI_STFT_RAW_BINS: tuple[tuple[int, int, int], ...] = (
    (0, 2, 15),
    (1, 8, 37),
    (2, 19, 24),
)


def multi_stft_raw_bin_freqs_hz(
    *,
    sample_rate: int = 2048,
    nperseg_low: int = 1024,
    nperseg_mid: int = 512,
    nperseg_hi: int = 256,
    raw_bins: tuple[tuple[int, int, int], ...] = MULTI_STFT_RAW_BINS,
) -> torch.Tensor:
    """Center frequencies of the F=50 raw-|STFT| bins, in concat order
    (low → mid → hi). Strictly increasing by construction (§4.2)."""
    nperseg_by_stft = {0: nperseg_low, 1: nperseg_mid, 2: nperseg_hi}
    freqs: list[float] = []
    for s_idx, k0, k1 in raw_bins:
        rfftfreq = torch.fft.rfftfreq(nperseg_by_stft[s_idx], d=1.0 / sample_rate)
        freqs.extend(float(rfftfreq[k]) for k in range(k0, k1 + 1))
    return torch.tensor(freqs)


def multi_stft_raw_valid_bin_mask(
    *,
    passband_low_hz: float,
    passband_high_hz: float,
    sample_rate: int = 2048,
    nperseg_low: int = 1024,
    nperseg_mid: int = 512,
    nperseg_hi: int = 256,
    raw_bins: tuple[tuple[int, int, int], ...] = MULTI_STFT_RAW_BINS,
) -> torch.Tensor:
    """Per-corpus valid-bin mask for the raw front end (FE-RAW-1 §4.4) — the
    raw analog of :func:`multi_stft_valid_bin_mask`. A raw bin is valid iff its
    center frequency lies in ``[passband_low_hz, passband_high_hz]`` (point-
    frequency test, replacing the filterbank's triangular-overlap criterion).

    BT / D-cohort (broadband) → all 50 valid. SWEC / AJILE12 (≤ 120 Hz) → 37
    valid (low 14 + mid k8–k30 = 32–120 Hz) / 13 invalid (mid k31–k37 + hi).
    """
    freqs = multi_stft_raw_bin_freqs_hz(
        sample_rate=sample_rate, nperseg_low=nperseg_low,
        nperseg_mid=nperseg_mid, nperseg_hi=nperseg_hi, raw_bins=raw_bins,
    )
    return (freqs >= passband_low_hz) & (freqs <= passband_high_hz)


def multi_stft_bin_centers_hz(
    *,
    n_bins: int = MULTI_STFT_N_BINS,
    f0_hz: float = MULTI_STFT_F0_HZ,
    octave_step: float = MULTI_STFT_OCTAVE_STEP,
) -> torch.Tensor:
    """Filterbank center frequencies — ``f0 * 2^(k * step)`` for ``k ∈ [0, n_bins)``."""
    k = torch.arange(n_bins).float()
    return f0_hz * torch.pow(2.0, k * octave_step)


def multi_stft_valid_bin_mask(
    *,
    passband_low_hz: float,
    passband_high_hz: float,
    n_bins: int = MULTI_STFT_N_BINS,
    f0_hz: float = MULTI_STFT_F0_HZ,
    octave_step: float = MULTI_STFT_OCTAVE_STEP,
    half_bw_octaves: float = MULTI_STFT_HALF_BW_OCTAVES,
) -> torch.Tensor:
    """Per-corpus valid-bin mask: bin k is valid iff its triangular support
    has *any* overlap with the recording passband. Matches the 5/19 SWEC
    audit memory ("SWEC 0.5–120 Hz → trains filterbank bins k0–k21"): the
    "any overlap" criterion is what hits k=21 at center 128 Hz with its
    lower edge ~90 Hz still inside the 0.5–120 Hz passband.
    """
    centers = multi_stft_bin_centers_hz(n_bins=n_bins, f0_hz=f0_hz, octave_step=octave_step)
    bw_factor = 2.0 ** half_bw_octaves
    lo_edge = centers / bw_factor
    hi_edge = centers * bw_factor
    return (hi_edge >= passband_low_hz) & (lo_edge <= passband_high_hz)


@functools.lru_cache(maxsize=8)
def _build_multi_stft_filterbank(
    sample_rate: int,
    nperseg_low: int,
    nperseg_mid: int,
    nperseg_hi: int,
    n_bins: int,
    f0_hz: float,
    octave_step: float,
    half_bw_octaves: float,
    routing: tuple[int, ...],
) -> dict[int, "tuple[torch.Tensor, torch.Tensor]"]:
    """Triangular log-octave filterbank weights per source STFT.

    Returns ``{stft_idx: (weights, out_bin_indices)}`` where
    ``weights[i, j]`` is the contribution of STFT-bin ``i`` to output bin
    ``out_bin_indices[j]``. Each output bin is L1-normalized across its
    source STFT-bins so that broadband-flat input produces unit-magnitude
    output bins (modulo numerical precision).
    """
    if len(routing) != n_bins:
        raise ValueError(
            f"routing has {len(routing)} entries, expected {n_bins}"
        )
    centers = multi_stft_bin_centers_hz(
        n_bins=n_bins, f0_hz=f0_hz, octave_step=octave_step,
    )
    bw_factor = 2.0 ** half_bw_octaves
    lo_edges = centers / bw_factor
    hi_edges = centers * bw_factor

    nperseg_by_stft = {0: nperseg_low, 1: nperseg_mid, 2: nperseg_hi}
    freqs_by_stft = {
        s: torch.fft.rfftfreq(np_, d=1.0 / sample_rate)
        for s, np_ in nperseg_by_stft.items()
    }

    per_stft_w: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
    per_stft_idx: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for k in range(n_bins):
        s_idx = routing[k]
        freqs = freqs_by_stft[s_idx]
        center = centers[k].item()
        lo = lo_edges[k].item()
        hi = hi_edges[k].item()
        # Triangular weights peaked at center.
        left = (freqs - lo) / max(center - lo, 1e-12)
        right = (hi - freqs) / max(hi - center, 1e-12)
        weights = torch.minimum(left, right).clamp(min=0.0)
        total = weights.sum()
        if float(total) > 0.0:
            weights = weights / total
        per_stft_w[s_idx].append(weights)
        per_stft_idx[s_idx].append(k)

    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for s_idx in (0, 1, 2):
        if per_stft_w[s_idx]:
            stacked = torch.stack(per_stft_w[s_idx], dim=-1)              # (F_stft_in, n_assigned)
            idx = torch.tensor(per_stft_idx[s_idx], dtype=torch.long)
            result[s_idx] = (stacked, idx)
    return result


def _multi_stft_view(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    hop_length: int,
    nperseg_low: int,
    nperseg_mid: int,
    nperseg_hi: int,
    n_bins: int,
    f0_hz: float,
    octave_step: float,
    half_bw_octaves: float,
    routing: tuple[int, ...],
    log_eps: float,
    apply_log: bool = False,
) -> torch.Tensor:
    """Compute the 3-STFT + ⅓-octave filterbank view.

    Input shape ``(..., C, T_samples)``; output shape ``(..., C, n_bins, T_bin)``
    with ``T_bin`` shared across STFTs (common hop_length). When
    ``apply_log=True``, returns ``log(filterbank_energy + log_eps)``; default
    ``apply_log=False`` returns linear-magnitude filterbank energies (5/25
    swap, iMINDBench-parity).
    """
    nps_by_stft = {0: nperseg_low, 1: nperseg_mid, 2: nperseg_hi}
    # Run each STFT. (Variable is named ``mag_stfts`` post-5/25: the per-STFT
    # tensor is |X|, not log|X|; log compression is applied at the very end
    # only when ``apply_log=True``.)
    mag_stfts: dict[int, torch.Tensor] = {}
    for s_idx, nps in nps_by_stft.items():
        win = torch.hann_window(nps, device=waveform.device)
        wf = waveform
        if wf.shape[-1] < nps:
            # NeuralSet's prepare() probes with sub-second inputs — pad so
            # torch.stft(center=True) can reflect-pad without crashing. Real
            # 1 s windows never hit this branch.
            wf = torch.nn.functional.pad(wf, (0, nps - wf.shape[-1]))
        spec = torch.stft(
            wf,
            n_fft=nps,
            hop_length=hop_length,
            win_length=nps,
            window=win,
            return_complex=True,
            normalized=False,
            center=True,
        )
        mag_stfts[s_idx] = torch.abs(spec)                                # (..., F_stft, T)

    # Cross-STFT sanity: time axes should match. The largest Nperseg with
    # center=True pads by Nperseg//2 each side, so frame counts can drift by 1
    # vs the smallest if the input is shorter than half the largest window.
    T_bins = [int(mag_stfts[s].shape[-1]) for s in (0, 1, 2)]
    T = min(T_bins)
    for s in (0, 1, 2):
        if mag_stfts[s].shape[-1] != T:
            mag_stfts[s] = mag_stfts[s][..., :T]

    fbank = _build_multi_stft_filterbank(
        sample_rate=sample_rate,
        nperseg_low=nperseg_low,
        nperseg_mid=nperseg_mid,
        nperseg_hi=nperseg_hi,
        n_bins=n_bins,
        f0_hz=f0_hz,
        octave_step=octave_step,
        half_bw_octaves=half_bw_octaves,
        routing=routing,
    )

    out_shape = list(waveform.shape[:-1]) + [n_bins, T]
    out = torch.zeros(out_shape, dtype=mag_stfts[0].dtype, device=waveform.device)
    for s_idx, (weights, idx) in fbank.items():
        # spec: (..., F_stft, T); weights: (F_stft, n_assigned)
        # → (..., n_assigned, T)
        weights = weights.to(out.dtype).to(out.device)
        banded = torch.einsum("fa,...ft->...at", weights, mag_stfts[s_idx])
        out.index_copy_(dim=-2, index=idx.to(out.device), source=banded)

    if apply_log:
        return torch.log(out + log_eps)
    return out


def _multi_stft_raw_view(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    hop_length: int,
    nperseg_low: int,
    nperseg_mid: int,
    nperseg_hi: int,
    raw_bins: tuple[tuple[int, int, int], ...],
    log_eps: float,
    apply_log: bool = False,
) -> torch.Tensor:
    """FE-RAW-1 (§4.2): raw |STFT| bins, no filterbank.

    Runs the same three STFTs as :func:`_multi_stft_view` (Hann, ``center=True``,
    ``normalized=False``, value axis = raw ``|STFT|``) and concatenates the
    exact rfft-bin slices in ``raw_bins`` (inclusive k ranges), low → mid → hi.
    Input ``(..., C, T_samples)`` → output ``(..., C, F_RAW, T_bin)`` with
    ``T_bin`` shared across STFTs (common hop). The STFT call is byte-identical
    to the const-Q path; only the bin *selection* differs.
    """
    nps_by_stft = {0: nperseg_low, 1: nperseg_mid, 2: nperseg_hi}
    mag_stfts: dict[int, torch.Tensor] = {}
    for s_idx, nps in nps_by_stft.items():
        win = torch.hann_window(nps, device=waveform.device)
        wf = waveform
        if wf.shape[-1] < nps:
            # NeuralSet's prepare() probes with sub-second inputs — pad so
            # torch.stft(center=True) can reflect-pad without crashing. Real
            # 1 s windows never hit this branch.
            wf = torch.nn.functional.pad(wf, (0, nps - wf.shape[-1]))
        spec = torch.stft(
            wf,
            n_fft=nps,
            hop_length=hop_length,
            win_length=nps,
            window=win,
            return_complex=True,
            normalized=False,
            center=True,
        )
        mag_stfts[s_idx] = torch.abs(spec)                                # (..., F_stft, T)

    # Common-hop time axes should match; trim the rare center-pad off-by-one.
    T = min(int(mag_stfts[s].shape[-1]) for s in (0, 1, 2))

    bands: list[torch.Tensor] = []
    for s_idx, k0, k1 in raw_bins:
        bands.append(mag_stfts[s_idx][..., k0 : k1 + 1, :T])              # (..., k1-k0+1, T)
    out = torch.cat(bands, dim=-2)                                        # (..., F_RAW, T)

    if apply_log:
        return torch.log(out + log_eps)
    return out


def _multi_stft_raw_view_chunked(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    hop_length: int,
    nperseg_low: int,
    nperseg_mid: int,
    nperseg_hi: int,
    raw_bins: tuple[tuple[int, int, int], ...],
    log_eps: float,
    apply_log: bool = False,
    chunk_frames: int = 960,
) -> torch.Tensor:
    """Memory-bounded whole-recording raw |STFT|, byte-identical to one
    un-chunked :func:`_multi_stft_raw_view` over the full ``waveform``.

    Computing a single ``torch.stft`` over a multi-hour recording allocates
    ``O(C · F · T)`` for the complex spectrogram (tens of GB). This processes
    the recording in ``chunk_frames``-wide output blocks via overlap-save: each
    block reads ``half = nperseg_low // 2`` extra REAL samples of context on
    each side, runs the standard ``_multi_stft_raw_view`` on that padded
    segment, then keeps only the frames whose full analysis window was real
    (trimming the ``center=True`` reflect-contaminated boundary frames). The
    only reflect frames retained are at the TRUE recording ends, where the
    segment edge coincides with the global edge and the reflection is exactly
    what the un-chunked call produces.

    Correctness hinges on ``half`` being a multiple of ``hop_length`` (true for
    the 1024/512/256 @ hop=128 lock: 512 = 4·128) so every segment starts on a
    sample that is a multiple of ``hop_length`` — only then does segment frame
    ``g`` map to global frame ``seg_start // hop + g``. Pinned to the un-chunked
    result by ``test_multi_stft_raw_view_chunked_matches_unchunked``.

    Input ``(..., C, T_samples)`` → output ``(..., C, F_RAW, T_bin)`` with
    ``T_bin = 1 + T_samples // hop_length`` (the ``center=True`` frame count).
    """
    if hop_length <= 0 or nperseg_low % hop_length != 0:
        raise ValueError(
            f"chunked whole-movie STFT needs nperseg_low ({nperseg_low}) "
            f"divisible by hop_length ({hop_length}) so segment starts land on "
            "the global hop grid"
        )
    n_samples = int(waveform.shape[-1])
    total_frames = 1 + n_samples // hop_length
    half = nperseg_low // 2  # widest window's reach; a multiple of hop_length

    def _view(seg: torch.Tensor) -> torch.Tensor:
        return _multi_stft_raw_view(
            seg,
            sample_rate=sample_rate,
            hop_length=hop_length,
            nperseg_low=nperseg_low,
            nperseg_mid=nperseg_mid,
            nperseg_hi=nperseg_hi,
            raw_bins=raw_bins,
            log_eps=log_eps,
            apply_log=apply_log,
        )

    # Whole recording fits in one pass (or is too short to chunk): direct call.
    if total_frames <= chunk_frames or n_samples <= 2 * half + nperseg_low:
        return _view(waveform)

    blocks: list[torch.Tensor] = []
    f = 0
    while f < total_frames:
        f_end = min(f + chunk_frames, total_frames)
        # Real-sample span covering output frames [f, f_end), padded by `half`
        # of REAL context each side; clamp to the recording (true-edge reflect
        # then matches the global call). lo stays a multiple of hop_length.
        lo = max(0, (f * hop_length) - half)
        hi = min(n_samples, (f_end - 1) * hop_length + half + 1)
        # A short tail segment (< nperseg_low) would trip _multi_stft_raw_view's
        # zero-pad probe branch, corrupting the widest window. Grow the segment
        # leftward (staying on the hop grid so frame alignment holds) so every
        # window sees a full-length input.
        if hi - lo < nperseg_low:
            lo = ((max(0, hi - nperseg_low)) // hop_length) * hop_length
        seg = waveform[..., lo:hi]
        spec_seg = _view(seg)  # (..., F, 1 + (hi-lo)//hop)
        g0 = f - lo // hop_length
        g1 = f_end - lo // hop_length
        blocks.append(spec_seg[..., g0:g1])
        f = f_end
    return torch.cat(blocks, dim=-1)


def _log_stft_view(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    nperseg: int,
    poverlap: float,
    min_freq_hz: float,
    max_freq_hz: float,
    log_eps: float,
    apply_log: bool = False,
) -> torch.Tensor:
    """Compute the upstream-byte-parity STFT magnitude.

    Matches Neuroprobe ``preprocess_stft(..., "stft_abs")``. When
    ``apply_log=True``, returns ``log(|X| + log_eps)`` for byte parity with
    the pre-5/25 v14 default. Input shape ``(..., C, T_samples)``; output
    shape ``(..., C, F_bin, T_bin)``.
    """

    noverlap = int(nperseg * poverlap)
    hop_length = nperseg - noverlap
    window = torch.hann_window(nperseg, device=waveform.device)

    # NeuralSet's `prepare()` probes extractors with a 0.001s window (~2
    # samples @ 2048 Hz). torch.stft(center=True) reflect-pads by nperseg//2
    # and needs input_len >= pad_size + 1. Real 1 s windows (2048 samples)
    # never hit this branch — pad only the introspection probe so output
    # shapes can be discovered, then real calls remain bit-identical.
    if waveform.shape[-1] < nperseg:
        pad_amount = nperseg - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    complex_stft = torch.stft(
        waveform,
        n_fft=nperseg,
        hop_length=hop_length,
        win_length=nperseg,
        window=window,
        return_complex=True,
        normalized=False,
        center=True,
    )
    freqs = torch.fft.rfftfreq(nperseg, d=1.0 / sample_rate)
    keep = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
    complex_stft = complex_stft[..., keep, :]
    mag = torch.abs(complex_stft)
    if apply_log:
        return torch.log(mag + log_eps)
    return mag


class LogStftView(CARIeegExtractor):
    """v14 I2 single-STFT view on top of CARIeegExtractor's waveform pipeline.

    Class name is historical — defaults to abs magnitude (no log) post-5/25
    swap. Set ``apply_log=True`` for the pre-5/25 byte-parity behavior. The
    sister-cell role is ``F-single-STFT`` (vs the Multi-STFT default).

    Per-event output: ``(C, F_bin, T_bin)`` STFT magnitude, time-last.
    Defaults match Neuroprobe Section D + v14 recipe lock (2026-05-12). When
    ``c_max`` is set, the leading channel dim pads with zeros to ``c_max``
    so per-batch collation aligns with ``V14DKHardSupportExtractor`` and
    ``ElectrodeValidMask``.
    """

    sample_rate_hz: int = 2048
    stft_nperseg: int = 512
    stft_poverlap: float = 0.75
    stft_max_freq_hz: float = 150.0
    stft_min_freq_hz: float = 0.0
    stft_log_eps: float = 1e-6
    apply_log: bool = False
    c_max: int | None = None

    def n_time_bins_for_duration(self, duration_s: float) -> int:
        """STFT frame count for a ``duration_s`` window: ``1 + L // hop`` where
        ``hop = nperseg·(1 − poverlap)`` and ``L = round(duration_s · fs)``
        (``torch.stft(center=True)``). 1 s @ 2048 Hz → 17."""
        hop = self.stft_nperseg - int(self.stft_nperseg * self.stft_poverlap)
        n_samples = int(round(duration_s * self.sample_rate_hz))
        return 1 + n_samples // hop

    def _get_timed_array(
        self, event, start: float, duration: float,
    ) -> TimedArray:
        waveform_ta = super()._get_timed_array(event, start, duration)
        waveform_t = torch.from_numpy(np.asarray(waveform_ta.data)).float()
        spec = _log_stft_view(
            waveform_t,
            sample_rate=int(float(waveform_ta.frequency)),
            nperseg=self.stft_nperseg,
            poverlap=self.stft_poverlap,
            min_freq_hz=self.stft_min_freq_hz,
            max_freq_hz=self.stft_max_freq_hz,
            log_eps=self.stft_log_eps,
            apply_log=self.apply_log,
        )
        if self.c_max is not None:
            c_event = int(spec.shape[0])
            if c_event > self.c_max:
                raise ValueError(
                    f"event has {c_event} electrodes which exceeds c_max={self.c_max}"
                )
            padded = torch.zeros(
                self.c_max, int(spec.shape[1]), int(spec.shape[2]), dtype=spec.dtype,
            )
            padded[:c_event] = spec
            spec = padded
        n_time_bins = int(spec.shape[-1])
        new_frequency = float(n_time_bins) / float(duration)
        return TimedArray(
            frequency=new_frequency,
            start=start,
            duration=duration,
            data=spec.cpu().numpy(),
        )


class _SpecCacheEntry(tp.NamedTuple):
    """One session's whole-movie raw-|STFT| memmap-cache record. Immutable +
    picklable so it survives both Linux fork and pydantic's spawn pickling of
    private attrs (the ``_session_stats`` inheritance pattern)."""

    path: str
    ch_names: tuple[str, ...]
    total_frames: int
    sample_rate: int


class MultiStftView(CARIeegExtractor):
    """v14 Multi-STFT front-end on top of CARIeegExtractor's waveform pipeline.
    Replaces ``LogStftView`` as the v14 default; single-STFT lives on as the
    F-single-STFT sister cell.

    FE-RAW-1 (2026-06-04, ``reports/fe_raw_frontend_handoff_2026_06_04.md``):
    ``front_end="raw"`` (default) emits the §4.2 RAW ``|STFT|`` bins — F=50, no
    filterbank — non-overlapping crossovers at 32/152 Hz, drop-delta 4 Hz, cap
    192 Hz. The FE-filterbank-sweep (2026-06-03) showed the const-Q filterbank
    hurts cross-subject transfer (RANK4 −0.013…−0.019), so the lossy pooling is
    dropped and the learned Conv2d patch stem does the pooling instead. The
    const-Q ⅓-octave filterbank is DEMOTED, not deleted — ``front_end="fbank"``
    recovers it (the F=30 P1 low-data-regularizer sister, handoff §8).

    Per-event output:
      * raw   → ``(C, F=50, T_bin)`` raw |STFT| bins, time-last.
      * fbank → ``(C, F=30, T_bin)`` ⅓-octave magnitude filterbank, time-last
                (k0–k14 from low, k15–k21 from mid, k22–k29 from hi; centered at
                ``2^(k/3)`` Hz, ~1 → ~813 Hz; see ``MULTI_STFT_ROUTING``).

    Three internal STFTs at common hop=128 @ 2048 Hz (Nperseg=1024/512/256).
    Hop=128 matches iMINDBench's standardized Multi-STFT and yields a 16 Hz
    front-end frame rate; the downstream patch stem halves the time axis to an
    8 Hz latent, matching the Whisper teacher (50→8 Hz pool, B33) with identity
    passthrough at Phase-3 (hop=128 re-lock 2026-06-03).

    5/25 swap: ``apply_log`` defaults to ``False`` — iMINDBench-parity raw
    magnitude. Set ``apply_log=True`` for the F-log-amplitude sister cell
    (pre-5/25 default behavior).
    """

    # FE-RAW-1: front-end mode. "raw" (default) = F=50 raw |STFT| bins; "fbank"
    # = the demoted F=30 const-Q filterbank sister.
    front_end: tp.Literal["raw", "fbank"] = "raw"
    sample_rate_hz: int = 2048
    # FE-01 (hop=128 re-lock 2026-06-03, reverses B20 v4 hop=256): hop=128 @
    # 2048 Hz → 16 Hz front-end frame rate. Matches iMINDBench's standardized
    # Multi-STFT (Fig 8 "all STFT sections use a hop length of 128 timepoints";
    # PopT-v2 uses the same 128-sample hop) — the config behind the 0.663 gate.
    # The (3,2)-stride patch stem then halves the time axis to an 8 Hz latent
    # (T_p), which equals the Whisper teacher pool target (50→8 Hz, B33) → the
    # P3 student is a true identity-rate passthrough, no resample either side.
    # R-hop-256 (the old B20 default) lives on as a sister cell.
    hop_length: int = 128
    nperseg_low: int = 1024
    nperseg_mid: int = 512
    nperseg_hi: int = 256
    n_fbank_bins: int = MULTI_STFT_N_BINS
    fbank_f0_hz: float = MULTI_STFT_F0_HZ
    fbank_octave_step: float = MULTI_STFT_OCTAVE_STEP
    fbank_half_bw_octaves: float = MULTI_STFT_HALF_BW_OCTAVES
    log_eps: float = 1e-6
    apply_log: bool = False
    c_max: int | None = None
    # C3 (WS-C, B13 lock): per-(electrode, freq-bin, session) robust-z over the
    # filterbank output. When True, ``prepare`` fits a ``SessionRobustZNormalizer``
    # once per session over that session's *own* full recording (B13: median/MAD
    # is physical-unit calibration — impedance × amp-gain — fit identically train
    # and eval, NOT fit-on-train-only) and ``_get_timed_array`` applies the frozen
    # stats per clip. Replaces the dropped ``scaler="StandardScaler"`` (raw-voltage,
    # whole-recording, non-robust). Default False so the synthetic capstone /
    # fast_dev_run smoke (per-token encoder LN handles scale) stays untouched; the
    # dispatch flips it True for real runs.
    session_robust_z: bool = False
    session_z_sigma_floor: float = 1e-6
    # STFT-fit chunk for the session stat pass: the full recording's filterbank
    # is computed in ``session_z_chunk_s``-wide slices (one cached-raw read per
    # session, STFT'd chunk-by-chunk to bound peak memory) then concatenated
    # along time before the median/MAD fit. Numerically inert for a median over
    # ~10^5 frames: only the handful of center-pad frames at each chunk seam
    # differ from a single continuous STFT (<0.5% of frames).
    session_z_chunk_s: float = 60.0
    # Pydantic doesn't allow tuple[int, ...] as a default freely — keep
    # the routing as a class-level constant pulled from the module.
    fbank_routing: tp.ClassVar[tuple[int, ...]] = MULTI_STFT_ROUTING
    # FE-RAW-1: raw-bin selection (stft_idx, k_start, k_end inclusive) for the
    # ``front_end="raw"`` path. Class-level (data-determining but fixed).
    raw_bins: tp.ClassVar[tuple[tuple[int, int, int], ...]] = MULTI_STFT_RAW_BINS
    # Per-session frozen normalizers, keyed by ``event._splittable_event_uid()``
    # (the session key the cached raw is keyed on). Mutated once in ``prepare`` on
    # the main process, read per-clip in ``_get_timed_array`` — the same
    # prepare-fills-private-state / workers-inherit pattern as ``_channels`` on
    # the MneRaw base (Linux fork inherits parent memory; pydantic pickles private
    # attrs under spawn).
    _session_stats: dict[str, SessionRobustZNormalizer] = {}
    # False until ``_fit_session_robust_z`` has run. ``MneRaw.prepare`` fires a
    # ``duration=0.001`` shape-probe clip THROUGH this view *before* the fit, so
    # ``_apply_session_robust_z`` must skip while this is False (the probe only
    # needs the output shape, which robust-z leaves unchanged). Real training
    # clips run with it True, so a genuinely-missing session still fails loud.
    _stats_ready: bool = False

    # D2 (T1.7, project_v14_preproc_recipe_2026_05_12 L.0 amendment): per-session
    # MNE-LOF bad-channel detection. When True, ``prepare`` runs
    # ``find_bad_channels_lof(threshold, n_neighbors)`` once per session over that
    # session's FILTERED, PRE-CAR voltage (the fit loads a ``car=None`` sibling, so
    # LOF input = notch+HPF applied, no CAR: a bad channel must not pollute the CAR
    # reference, and CAR's rank-deficiency would distort LOF's neighbor structure),
    # records the flagged channels, and ``_preprocess_raw`` then stamps
    # ``raw.info["bads"]`` per clip so the inherited ``drop_bads`` step removes them
    # BEFORE shaft-CAR. So the LOF-relevant order holds: filtered → LOF → drop →
    # shaftCAR. NOTE the inherited clip path (CARIeegExtractor._preprocess_raw)
    # applies shaft-CAR BEFORE notch/HPF (pick → drop_bads → CAR → MneRaw filter) —
    # a pre-existing CAR-before-filter ordering, harmless because CAR and the linear
    # notch/HPF commute, flagged to Ben separately. Requires ``drop_bads=True`` and
    # ``lof_report_path`` (model_validator). Default False so the synthetic capstone
    # / fast_dev_run smoke is untouched; the BT dispatch flips it True. BEN REVIEW
    # GATE: the per-session/per-subject drop counts are logged loud and written to
    # JSON for review before this is relied on in a scored run ("report back how
    # many channels are dropped").
    lof_bad_channels: bool = False
    lof_threshold: float = 1.5
    lof_n_neighbors: int = 20
    lof_ch_type: str = "seeg"
    lof_report_path: str | None = None
    # session key (``event._splittable_event_uid()``) -> LOF-flagged bad channel
    # names. Mirrors the ``_session_stats`` fill-in-prepare / read-per-clip pattern.
    _session_bads: dict[str, list[str]] = {}
    _bads_ready: bool = False

    # ------------------------------------------------------------------ #
    # Whole-movie raw-|STFT| feature cache (Ben-approved 2026-06-05).      #
    # ------------------------------------------------------------------ #
    # Default None = OFF: the per-clip path recomputes the STFT and the
    # robust-z fit uses the seam-chunked ``_fit_session_robust_z`` loop — both
    # byte-for-byte the pre-cache behavior, so live runs are untouched unless
    # this is set. When set, ``prepare`` materializes each session's
    # SESSION-channel-order whole-recording raw |STFT| once (seam-free via
    # ``_multi_stft_raw_view_chunked``) under
    # ``{spec_cache_dir}/{infra.uid()}/{session_uid}.npy`` (fp32 + ``.json``
    # sidecar) and ``_get_timed_array`` slices that memmap per clip instead of
    # re-STFT'ing. The namespace is ``infra.uid()`` (the data-determining pydantic
    # fields: notch/filter/car/lof/STFT config) PLUS a digest of the
    # data-determining ClassVars ``raw_bins``/``fbank_routing`` (which infra.uid()
    # cannot see) — see ``_spec_cache_namespace``, so an FE-RAW bin change cannot
    # silently hit a stale spec. The M0 optimizer sweep (#45) thus computes the
    # spec ONCE and every later same-config job memmaps it; sessions already on
    # disk are skipped (cross-job reuse), each per-session file keyed by the
    # SHA-1 of the session uid (``_spec_cache_stem``) so path separators / JSON
    # metacharacters / the 255-byte filename limit can never break the write.
    #
    # Three Ben-accepted feature forks vs the per-clip path (benign / SSL-neutral
    # or -positive): (1) clip edges read real neighbouring audio rather than
    # ``torch.stft`` center-reflect padding; (2) clip frames snap to the global
    # hop grid (≤ hop/2 = 31 ms at hop=128); (3) the robust-z fit is seam-free.
    # Stats are fit on the on-disk fp32 frames in BOTH the build and the cross-job
    # cache-hit path, so every sweep job's stats are bit-identical regardless of
    # who built the cache. Only supported for the v14 default front_end="raw",
    # apply_log=False, and no waveform-domain transforms (baseline/scale/clamp),
    # which ``_validate_spec_cache_config`` enforces at construction.
    spec_cache_dir: str | None = None
    # session key (``event._splittable_event_uid()``) -> cache record, filled in
    # ``prepare`` on the main process and inherited by forked DataLoader workers
    # (the same fill-in-prepare / read-per-clip pattern as ``_session_stats``).
    _spec_cache_index: dict[str, _SpecCacheEntry] = {}
    _spec_ready: bool = False

    @model_validator(mode="after")
    def _validate_spec_cache_config(self) -> "MultiStftView":
        # The cache path bypasses ``super()._get_timed_array`` (the waveform read
        # + scatter + STFT) and slices a precomputed |STFT| instead. That is only
        # equivalent to the per-clip path for the v14 default config: raw bins,
        # no log (so the global-channel scatter background |STFT|(0)=0 matches the
        # per-clip path's STFT-of-zero rows), and no waveform-domain transforms
        # (baseline subtraction / scale_factor / clamp all act pre-STFT on the
        # waveform, which the cache never sees). Fail loud at construction rather
        # than silently desync features.
        if self.spec_cache_dir is not None:
            if self.front_end != "raw":
                raise ValueError(
                    "MultiStftView.spec_cache_dir is only supported for "
                    f"front_end='raw' (got {self.front_end!r}): the fbank scatter "
                    "background is not implemented for the cache path."
                )
            if self.apply_log:
                raise ValueError(
                    "MultiStftView.spec_cache_dir requires apply_log=False: the "
                    "global-channel scatter fills absent rows with |STFT|(0)=0, "
                    "which only matches the per-clip path when log is off."
                )
            for field in ("baseline", "scale_factor", "clamp"):
                if getattr(self, field, None) is not None:
                    raise ValueError(
                        f"MultiStftView.spec_cache_dir does not support {field!r}: "
                        "it is a waveform-domain transform applied before the STFT, "
                        "which the precomputed-spectrogram cache path bypasses."
                    )
        return self

    @model_validator(mode="after")
    def _validate_lof_config(self) -> "MultiStftView":
        # Fail at construction, not silently mid-run. LOF that flags-but-never-
        # drops, or drops-but-never-reports, both defeat the point.
        if self.lof_bad_channels:
            if not self.drop_bads:
                raise ValueError(
                    "MultiStftView.lof_bad_channels=True requires drop_bads=True "
                    "so the LOF-flagged channels are actually removed before "
                    "shaft-CAR (otherwise LOF flags but never drops)."
                )
            if not self.lof_report_path:
                raise ValueError(
                    "MultiStftView.lof_bad_channels=True requires lof_report_path "
                    "set: the per-session/per-subject drop counts MUST be written "
                    "to JSON for the Ben review gate ('report back how many "
                    "channels are dropped'). Set a run-scoped path."
                )
        return self

    def _exclude_from_cache_uid(self) -> list[str]:
        # ``lof_report_path`` and ``spec_cache_dir`` are OUTPUT / FEATURE-CACHE
        # LOCATION knobs, not data-determining: two runs pointing them at different
        # paths must hit the SAME multi-TB STFT cache AND produce the SAME run uid.
        # ``lof_report_path`` is where the LOF drop counts are written;
        # ``spec_cache_dir`` (#80) is where the whole-movie raw-|STFT| memmap lives
        # — the per-clip features it slices are byte-identical to the recompute path
        # (parity-tested), so arming the cache is identity-transparent (same uid,
        # just skips the per-run whole-movie recompute). The other ``lof_*`` fields
        # DO determine which channels get dropped, so they stay in the uid.
        # (Default-OFF ``lof_bad_channels`` / default-None ``spec_cache_dir`` are
        # excluded by exclude_defaults, so an un-armed run perturbs nothing.)
        return super()._exclude_from_cache_uid() + ["lof_report_path", "spec_cache_dir"]

    def n_time_bins_for_duration(self, duration_s: float) -> int:
        """STFT frame count for a ``duration_s`` window: ``1 + L // hop`` where
        ``L = round(duration_s · sample_rate_hz)`` (``torch.stft(center=True)``).
        At hop=128: 5 s → 81, 1 s → 17 — the nominal 16 Hz × duration (80 / 16)
        plus the one center-pad frame. The (3,2)-stride patch stem floors both
        onto the load-bearing patch counts T_p = 40 / 8 (8 Hz latent; RoPE
        downward-generalizes from 40 to 8, D8). The dispatch uses this to size
        the encoder's ``n_time_bins`` (RoPE ceiling) from a phase-conditional
        ``clip_len``."""
        n_samples = int(round(duration_s * self.sample_rate_hz))
        return 1 + n_samples // self.hop_length

    def _spec_from_waveform(
        self, waveform_t: torch.Tensor, sample_rate: int,
    ) -> torch.Tensor:
        """``(C, T)`` waveform → ``(C, F, T_bin)``, with this view's exact
        STFT config. ``front_end="raw"`` (default) emits the F=50 raw |STFT|
        bins; ``front_end="fbank"`` the demoted F=30 const-Q filterbank. Single
        source of truth shared by the per-clip path and the session-stat fit, so
        fit-time and apply-time features are byte-identical."""
        if self.front_end == "raw":
            return _multi_stft_raw_view(
                waveform_t,
                sample_rate=sample_rate,
                hop_length=self.hop_length,
                nperseg_low=self.nperseg_low,
                nperseg_mid=self.nperseg_mid,
                nperseg_hi=self.nperseg_hi,
                raw_bins=self.raw_bins,
                log_eps=self.log_eps,
                apply_log=self.apply_log,
            )
        return _multi_stft_view(
            waveform_t,
            sample_rate=sample_rate,
            hop_length=self.hop_length,
            nperseg_low=self.nperseg_low,
            nperseg_mid=self.nperseg_mid,
            nperseg_hi=self.nperseg_hi,
            n_bins=self.n_fbank_bins,
            f0_hz=self.fbank_f0_hz,
            octave_step=self.fbank_octave_step,
            half_bw_octaves=self.fbank_half_bw_octaves,
            routing=self.fbank_routing,
            log_eps=self.log_eps,
            apply_log=self.apply_log,
        )

    def _get_timed_array(
        self, event, start: float, duration: float,
    ) -> TimedArray:
        if self.spec_cache_dir is not None and self._spec_ready:
            # Whole-movie cache armed: slice the precomputed memmap instead of
            # reading the waveform and re-STFT'ing it. _stats_ready/_spec_ready
            # are both False during the prepare shape-probe, so the probe still
            # falls through to the recompute path below (it needs the real STFT
            # only to learn the output shape).
            return self._cached_clip(event, start, duration)
        waveform_ta = super()._get_timed_array(event, start, duration)
        waveform_t = torch.from_numpy(np.asarray(waveform_ta.data)).float()
        spec = self._spec_from_waveform(
            waveform_t, int(float(waveform_ta.frequency)),
        )
        return self._finalize_clip(event, spec, start, duration)

    def _finalize_clip(
        self, event, spec: torch.Tensor, start: float, duration: float,
    ) -> TimedArray:
        """Shared clip tail for both the recompute and cache read paths: frozen
        robust-z (global-channel order), c_max pad, and the TimedArray wrap. Spec
        arrives in ``(C_global, F, T)`` global-channel order in BOTH paths (the
        recompute path scatters the waveform pre-STFT; the cache path scatters the
        sliced spec), so the robust-z stats (also scattered to global) broadcast
        identically and the two paths share one finalization."""
        if self.session_robust_z:
            # Apply BEFORE c_max padding: stats cover the real C electrodes only;
            # padded channels stay exactly 0 (normalizing them would map 0 →
            # -median/σ ≠ 0 and pollute the pad).
            spec = self._apply_session_robust_z(event, spec)
        if self.c_max is not None:
            c_event = int(spec.shape[0])
            if c_event > self.c_max:
                raise ValueError(
                    f"event has {c_event} electrodes which exceeds c_max={self.c_max}"
                )
            padded = torch.zeros(
                self.c_max, int(spec.shape[1]), int(spec.shape[2]), dtype=spec.dtype,
            )
            padded[:c_event] = spec
            spec = padded
        n_time_bins = int(spec.shape[-1])
        new_frequency = float(n_time_bins) / float(duration)
        return TimedArray(
            frequency=new_frequency,
            start=start,
            duration=duration,
            data=spec.cpu().numpy(),
        )

    def _apply_session_robust_z(
        self, event, spec: torch.Tensor,
    ) -> torch.Tensor:
        """Apply this session's frozen robust-z stats to a ``(C_global, F, T)``
        clip spec. The session is keyed by ``event._splittable_event_uid()`` —
        the same key the cached raw (and therefore the fit) is keyed on. The
        stats were scattered into the global-channel index at fit time (see
        ``_fit_session_robust_z``) so they broadcast against the global-channel
        clip ``super()._get_timed_array`` produced — across sessions with
        differing electrode counts the absent rows are zero in both."""
        if not self._stats_ready:
            # ``MneRaw.prepare`` shape-probe runs before the fit; robust-z does
            # not change the output shape, so pass the raw spec through.
            return spec
        key = event._splittable_event_uid()
        normalizer = self._session_stats.get(key)
        if normalizer is None:
            raise KeyError(
                f"MultiStftView.session_robust_z is on but no fitted stats for "
                f"session {key!r}. prepare() must run (and see this session) "
                f"before clips are drawn. Have stats for: "
                f"{sorted(self._session_stats)[:8]}..."
            )
        return normalizer.transform(spec)

    def prepare(self, obj) -> None:  # type: ignore[override]
        # ORDER IS LOAD-BEARING (Gate-C showstopper, 2026-06-03). ``self._get_data``
        # is an exca-cached property; ``super().prepare`` materializes it for EVERY
        # session (to populate ``_channels`` and fire a shape-probe). The cache key
        # does NOT include ``_bads_ready``, so if that first materialization ran
        # with bads un-armed it would memoize the PRE-LOF (no-drop) raw per session
        # and every later apply/robust-z fit would read it stale — LOF detected,
        # logged, but never actually dropped. So fit + ARM the bad sets FIRST: then
        # the very first ``self._get_data`` already stamps+drops, and the cache,
        # ``_channels``, the robust-z fit, and per-clip apply are all consistent.
        # The fit itself reads a ``car=None`` SIBLING (distinct cache namespace —
        # ``car`` is in the uid), so it cannot poison self's cache. It needs only
        # ``_event_types_helper`` (set in __init__) and ``_get_data`` (works pre-
        # prepare — prepare is its own first caller), NOT ``_channels``, so it is
        # safe to run before ``super().prepare``.
        if self.lof_bad_channels:
            self._fit_session_bads(obj)
            self._bads_ready = True
        # ``super().prepare`` populates ``_channels`` (needed by the robust-z fit's
        # global-channel scatter) and fires a shape-probe clip; the probe runs with
        # ``_stats_ready`` still False so it skips robust-z safely (it DOES drop LOF
        # bads now, which is exactly what we want cached). Only after the fit do we
        # arm ``_stats_ready`` so real clips normalize.
        super().prepare(obj)
        if self.spec_cache_dir is not None:
            # The cache build drives BOTH the materialized whole-movie spec AND
            # (when on) the robust-z fit, from the SAME seam-free frames, so the
            # per-clip slice and the stats can never diverge (fork 3). Replaces
            # the seam-chunked ``_fit_session_robust_z`` on the cache path.
            self._build_spec_cache_and_fit(obj)
            self._spec_ready = True
            if self.session_robust_z:
                self._stats_ready = True
        elif self.session_robust_z:
            self._fit_session_robust_z(obj)
            self._stats_ready = True

    def _fit_session_robust_z(self, obj) -> None:
        """Fit one ``SessionRobustZNormalizer`` per session over that session's
        own full recording (B13: per-(electrode, freq, session), identical train
        and eval). Reads each session's cached preprocessed raw once, computes the
        full-recording filterbank in ``session_z_chunk_s`` slices to bound peak
        memory, concatenates along time, and fits median/MAD. Runs on the main
        process inside ``Data.build``; the fitted stats are inherited by forked
        DataLoader workers."""
        events = self._event_types_helper.extract(obj)  # type: ignore[attr-defined]
        seen: set[str] = set()
        for event in events:
            key = event._splittable_event_uid()
            if key in seen:
                continue
            seen.add(key)
            raw_ta = next(self._get_data([event]))
            waveform = torch.from_numpy(np.asarray(raw_ta.data)).float()
            sample_rate = int(float(raw_ta.frequency))
            chunk = max(
                int(round(self.session_z_chunk_s * sample_rate)), self.nperseg_low,
            )
            n_samples = int(waveform.shape[-1])
            specs: list[torch.Tensor] = []
            for c0 in range(0, n_samples, chunk):
                seg = waveform[:, c0 : c0 + chunk]
                if int(seg.shape[-1]) < self.nperseg_low:
                    continue  # too short for the largest STFT window — drop tail
                specs.append(self._spec_from_waveform(seg, sample_rate))
            if not specs:
                # whole recording shorter than one nperseg_low window
                specs = [self._spec_from_waveform(waveform, sample_rate)]
            # Free the full-session waveform (up to ~15 GB for a 2 h / 250-ch
            # recording) BEFORE the fit allocates its median sort buffer — it is
            # unused past the STFT pass. Cuts worst-case fit peak ~25 GB → ~7 GB
            # with no numeric change (the concat-then-median is identical).
            del waveform
            frames = torch.cat(specs, dim=-1)  # (C_session, F, T_session)
            del specs  # torch.cat copied the data; the per-chunk specs are dead
            normalizer = SessionRobustZNormalizer(
                sigma_floor=self.session_z_sigma_floor,
            ).fit(frames)
            # Scatter the session-indexed (C_session, F, 1) stats into the
            # GLOBAL channel index. ``_get_data`` yields the session's own
            # C_session channels in ``raw_ta.ch_names`` order, but the apply
            # path's ``super()._get_timed_array`` scatters each clip into a
            # ``(max(_channels)+1, ...)`` global array via the SAME
            # ``_get_channels(ch_names)`` map (neuro.py:453,463). Placing the
            # stats at those identical global rows makes fit and apply align by
            # construction regardless of channel_order, and lets one set of
            # stats broadcast against clips from sessions with differing
            # electrode counts. Global rows with no source channel keep σ=0 →
            # ``transform`` zeros them, matching the clip's zero pad rows.
            self._scatter_stats_to_global(normalizer, raw_ta.ch_names)
            self._session_stats[key] = normalizer
            del frames, raw_ta

    def _scatter_stats_to_global(
        self, normalizer: SessionRobustZNormalizer, ch_names: list[str],
    ) -> None:
        """In-place: replace ``normalizer.median``/``.sigma`` ``(C_session,F,1)``
        with global-channel ``(C_global,F,1)`` tensors, the session's stats
        placed at the global indices ``_get_channels(ch_names)`` assigns — the
        exact rows the apply-path scatter lands this session's clip data on."""
        channel_idx = self._get_channels(ch_names)  # type: ignore[attr-defined]
        c_global = max(self._channels.values()) + 1  # type: ignore[attr-defined]
        idx = torch.as_tensor(channel_idx, dtype=torch.long)
        session_median = normalizer.median
        session_sigma = normalizer.sigma
        assert session_median is not None and session_sigma is not None, (
            "normalizer must be .fit() before scatter"
        )
        f_bins = int(session_median.shape[1])
        global_median = torch.zeros(c_global, f_bins, 1, dtype=session_median.dtype)
        global_sigma = torch.zeros(c_global, f_bins, 1, dtype=session_sigma.dtype)
        global_median[idx] = session_median
        global_sigma[idx] = session_sigma
        normalizer.median = global_median
        normalizer.sigma = global_sigma

    # ------------------------------------------------------------------ #
    # D2: per-session MNE-LOF bad-channel detection (T1.7)                #
    # ------------------------------------------------------------------ #
    def _stamp_lof_bads(self, raw, event) -> None:
        """Set ``raw.info["bads"]`` to this session's LOF-flagged channels that
        are present in ``raw`` so the inherited ``drop_bads`` step removes them
        before shaft-CAR. No-op until the fit has armed ``_bads_ready`` (the
        prepare shape-probe runs through here first)."""
        if not (self._bads_ready and self.lof_bad_channels):
            return
        key = event._splittable_event_uid()
        bads = self._session_bads.get(key)
        if bads is None:
            raise KeyError(
                f"MultiStftView.lof_bad_channels is on but no fitted bad-channel "
                f"set for session {key!r}. prepare() must run (and see this "
                f"session) before clips are drawn. Have bads for: "
                f"{sorted(self._session_bads)[:8]}..."
            )
        # Union, not overwrite: keep any bads the upstream loader pre-marked and
        # ADD the LOF-flagged ones (never drop fewer channels than before).
        present = set(raw.ch_names)
        existing = list(raw.info["bads"])
        raw.info["bads"] = existing + [
            b for b in bads if b in present and b not in existing
        ]

    def _preprocess_raw(self, raw, event):  # type: ignore[override]
        self._stamp_lof_bads(raw, event)
        return super()._preprocess_raw(raw, event)

    def _fit_session_bads(self, obj) -> None:
        """Flag bad channels per session via MNE LOF over that session's FILTERED,
        PRE-CAR voltage. Loads each session once through a ``car=None`` sibling
        view (CAR's rank-deficiency would distort LOF's neighbor structure and a
        bad channel must not be in the CAR average — the recipe runs LOF before
        shaft-CAR). Runs on the main process inside ``Data.build``; the per-session
        bad sets are inherited by forked DataLoader workers. Drop counts are logged
        loud and (if ``lof_report_path`` is set) written to JSON — the Ben gate."""
        if not self.drop_bads:
            raise ValueError(
                "MultiStftView.lof_bad_channels=True requires drop_bads=True so the "
                "LOF-flagged channels are actually removed before shaft-CAR."
            )
        # car=None sibling for the fit load. deep=True IS LOAD-BEARING (Gate-C
        # re-audit, 2026-06-03): a SHALLOW model_copy shares the exca infra object,
        # whose ``_obj`` stays bound to self (car=shaft) — the sibling would then
        # (1) preprocess with CAR after all (wrong LOF input) and (2) materialize a
        # car=shaft NO-DROP entry under self's production cache uid before bads are
        # armed, re-seeding the very poison this whole ordering kills. deep=True
        # forks the infra so the sibling gets the distinct car=None uid (verified ==
        # an independently-built car=None view) and preprocesses pre-CAR. The copy
        # carries the private state (_event_types_helper, _session_bads); _bads_ready
        # is still False on it so its _preprocess_raw passes through and LOF sees
        # every channel — it is the detector, not a consumer.
        precar = self.model_copy(update={"car": None}, deep=True)
        # deep=True ALSO deep-copies exca's memoized uid; reset it so the sibling
        # recomputes its car=None uid instead of inheriting self's car=shaft one if
        # self.infra.uid() was computed earlier (see _reset_infra_uid_cache).
        _reset_infra_uid_cache(precar.infra)
        events = self._event_types_helper.extract(obj)  # type: ignore[attr-defined]
        report: list[dict] = []
        seen: set[str] = set()
        for event in events:
            key = event._splittable_event_uid()
            if key in seen:
                continue
            seen.add(key)
            raw_ta = next(precar._get_data([event]))
            voltage = np.asarray(raw_ta.data)
            ch_names = list(raw_ta.ch_names)
            sample_rate = float(raw_ta.frequency)
            mask = lof_bad_channel_mask(
                voltage,
                sample_rate_hz=sample_rate,
                threshold=self.lof_threshold,
                n_neighbors=self.lof_n_neighbors,
                ch_names=ch_names,
                ch_type=self.lof_ch_type,
            )
            bad_names = [ch_names[i] for i in range(len(ch_names)) if bool(mask[i])]
            self._session_bads[key] = bad_names
            try:
                subject_id = int(event._get_field_or_extra("subject_id"))
            except (KeyError, AttributeError, ValueError, TypeError) as exc:
                # Don't silently bucket under -1: a missing/garbled subject_id
                # corrupts the per-subject drop attribution the Ben gate reads.
                subject_id = -1
                logger.warning(
                    "MNE-LOF: could not resolve subject_id for session %s (%s); "
                    "its %d bad channels are attributed to subject_id=-1",
                    key, exc, len(bad_names),
                )
            report.append({
                "session": key,
                "subject_id": subject_id,
                "n_channels": len(ch_names),
                "n_bad": len(bad_names),
                "bad_channels": bad_names,
            })
            logger.warning(
                "MNE-LOF session %s (subject %s): %d/%d channels flagged bad %s",
                key, subject_id, len(bad_names), len(ch_names), bad_names,
            )
            del raw_ta, voltage
        self._log_and_write_lof_report(report)

    def _log_and_write_lof_report(self, report: list[dict]) -> None:
        total_bad = sum(r["n_bad"] for r in report)
        total_ch = sum(r["n_channels"] for r in report)
        by_subject: dict[int, list[int]] = {}
        for r in report:
            agg = by_subject.setdefault(r["subject_id"], [0, 0])
            agg[0] += r["n_bad"]
            agg[1] += r["n_channels"]
        logger.warning(
            "MNE-LOF SUMMARY (threshold=%.2f, n_neighbors=%d): %d sessions, "
            "%d/%d channels dropped (%.2f%%). Per-subject n_bad/n_ch: %s",
            self.lof_threshold, self.lof_n_neighbors, len(report),
            total_bad, total_ch, 100.0 * total_bad / max(total_ch, 1),
            {s: f"{b}/{t}" for s, (b, t) in sorted(by_subject.items())},
        )
        if self.lof_report_path:
            payload = {
                "threshold": self.lof_threshold,
                "n_neighbors": self.lof_n_neighbors,
                "total_sessions": len(report),
                "total_bad": total_bad,
                "total_channels": total_ch,
                "by_subject": {
                    str(s): {"n_bad": b, "n_channels": t}
                    for s, (b, t) in sorted(by_subject.items())
                },
                "sessions": report,
            }
            path = Path(self.lof_report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2))
            logger.warning("MNE-LOF drop-count report written -> %s", path)

    # ------------------------------------------------------------------ #
    # Whole-movie raw-|STFT| feature cache (Ben-approved 2026-06-05)      #
    # ------------------------------------------------------------------ #
    def _whole_movie_spec(
        self, waveform: torch.Tensor, sample_rate: int,
    ) -> torch.Tensor:
        """``(C_session, T)`` waveform → seam-free whole-recording raw |STFT|
        ``(C_session, F, T_total)``, byte-identical to one un-chunked
        ``_multi_stft_raw_view`` (pinned by ``test_multi_stft_raw_view_chunked_
        matches_unchunked``). This is the single source the cache slices and the
        robust-z stats are fit on, so fit-time and apply-time features agree by
        construction."""
        return _multi_stft_raw_view_chunked(
            waveform,
            sample_rate=sample_rate,
            hop_length=self.hop_length,
            nperseg_low=self.nperseg_low,
            nperseg_mid=self.nperseg_mid,
            nperseg_hi=self.nperseg_hi,
            raw_bins=self.raw_bins,
            log_eps=self.log_eps,
            apply_log=self.apply_log,
        )

    def _expected_raw_f_bins(self) -> int:
        """F (frequency bins) the ``front_end="raw"`` spec must have for the
        current ``raw_bins``: the inclusive bin count summed over the three
        windows. Used to fail loud if a cache hit's stored ``f_bins`` disagrees
        with the live config (defence in depth behind the namespace digest)."""
        return sum(ke - ks + 1 for (_, ks, ke) in self.raw_bins)

    def _spec_cache_namespace(self) -> str:
        """Cache namespace = ``infra.uid()`` PLUS a digest of the data-determining
        ClassVars. ``raw_bins`` and ``fbank_routing`` are ``ClassVar``s, not
        pydantic fields, so exca's field-hash ``infra.uid()`` cannot see them — an
        FE-RAW F-bin recost (HB02) that changes ``raw_bins`` would otherwise reuse
        a stale spec built with different bins. Fold them into the namespace so
        different bin configs land in different directories. The STFT/preproc
        pydantic fields are already covered by ``infra.uid()``."""
        sig = repr((
            tuple(self.raw_bins),
            tuple(self.fbank_routing),
            self.front_end,
            bool(self.apply_log),
        )).encode()
        digest = hashlib.sha1(sig).hexdigest()[:12]
        return f"{self.infra.uid()}-fe{digest}"  # type: ignore[attr-defined]

    @staticmethod
    def _spec_cache_stem(key: str) -> str:
        """Filesystem-safe fixed-length stem for a session ``key``. The raw key is
        ``event._splittable_event_uid()`` — for BT a 120-byte JSON full of
        ``{}":,`` metacharacters (legal on ext4 but fragile and near the 255-byte
        filename limit), and for the joint D-cohort/SWEC corpus a real file path
        containing ``/`` (which would escape the cache dir / hit a missing parent).
        Hashing sidesteps separators, metacharacters, and length in one move; the
        raw key is preserved inside the ``.json`` sidecar for provenance."""
        return hashlib.sha1(key.encode()).hexdigest()

    @staticmethod
    def _atomic_write(path: Path, write_fn) -> None:
        """Write ``path`` via a process-unique temp file + ``os.replace`` so two
        M0-sweep builders materializing the SAME session never share a temp path
        (a fixed ``.tmp`` name would let them tear each other's bytes), and a
        crash mid-write never leaves a half-file that looks complete. ``write_fn``
        receives the open binary handle."""
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "wb") as fh:
                write_fn(fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def _assert_cache_meta(
        self, key: str, meta: dict, expected_f: int, ch_names: list[str],
    ) -> None:
        """Validate a cache hit's stored geometry against the live config. The
        namespace digest already separates ``raw_bins``/STFT configs, so a hit
        with the wrong ``f_bins`` means the cache was corrupted or a hash collided
        — either way slicing it would silently desync features, so raise."""
        f_bins = int(meta.get("f_bins", -1))
        if f_bins != expected_f:
            raise ValueError(
                f"cached spec for {key!r} has f_bins={f_bins} but the live "
                f"raw_bins config expects {expected_f}. Stale/corrupt cache — "
                f"clear {self.spec_cache_dir!r} or check raw_bins."
            )
        n_channels = int(meta.get("n_channels", -1))
        if n_channels != len(ch_names):
            raise ValueError(
                f"cached spec for {key!r} has n_channels={n_channels} but its "
                f"sidecar lists {len(ch_names)} ch_names — corrupt cache."
            )

    @staticmethod
    def _assert_cache_finite(key: str, spec_f32: torch.Tensor) -> None:
        """Refuse to cache a session whose raw |STFT| is non-finite: an inf/nan
        would land on disk where the f32 recompute path stays finite — a silent
        feature divergence. The cache is fp32 (FE-RAW-1 raw |STFT| routinely
        exceeds the fp16 65504 range, e.g. ~117k on btbank1; fp32 stores it
        verbatim and makes the cached path bit-identical to the recompute path),
        so only finiteness is in question, not dynamic range."""
        if not torch.isfinite(spec_f32).all():
            raise ValueError(
                f"whole-movie spec for session {key!r} is non-finite before the "
                "cache write — refusing to poison the cache."
            )

    @staticmethod
    def _write_stats_sidecar(
        stats_path: Path, normalizer: SessionRobustZNormalizer,
    ) -> None:
        """Persist the fitted robust-z stats (per-(channel, freq) median + σ) next
        to the frames cache. These are a deterministic function of the cached
        frames, so caching them lets a warm hit skip BOTH the full-frame reload
        and the median refit — the dominant warm-prepare cost. ~200 KB/session vs
        the multi-GB frames. ``sigma_floor`` is NOT stored: it is applied at
        ``transform`` (not baked into σ), so the reader rebuilds with the live
        config's floor."""
        assert normalizer.median is not None and normalizer.sigma is not None
        median = normalizer.median.cpu().numpy()
        sigma = normalizer.sigma.cpu().numpy()
        MultiStftView._atomic_write(
            stats_path,
            lambda fh: np.savez(fh, median=median, sigma=sigma),
        )

    def _load_stats_sidecar(
        self, stats_path: Path, meta: dict,
    ) -> tp.Optional[SessionRobustZNormalizer]:
        """Rebuild a normalizer from the stats sidecar, or return None (→ caller
        refits) when the stored geometry disagrees with the frames meta — a stale
        or corrupt sidecar must never silently scale features by the wrong stats."""
        with np.load(stats_path) as z:
            median = torch.from_numpy(z["median"])
            sigma = torch.from_numpy(z["sigma"])
        exp_c = int(meta.get("n_channels", -1))
        exp_f = int(meta.get("f_bins", -1))
        if tuple(median.shape[:2]) != (exp_c, exp_f) or tuple(
            sigma.shape[:2]
        ) != (exp_c, exp_f):
            return None
        return SessionRobustZNormalizer.from_stats(
            median=median,
            sigma=sigma,
            sigma_floor=self.session_z_sigma_floor,
        )

    def _build_spec_cache_and_fit(self, obj) -> None:
        """Materialize each session's seam-free whole-recording raw |STFT| to the
        fp32 memmap cache (skipping sessions already on disk → cross-job reuse)
        and, when ``session_robust_z`` is on, fit that session's robust-z stats
        from the SAME fp32 frames the per-clip path will read. Fitting on the
        on-disk fp32 frames in BOTH the build branch and the cache-hit branch
        makes every sweep job's stats bit-identical regardless of who built the
        cache — and identical to the f32 recompute path (no fp16 quantization).
        Runs on the main process inside ``Data.build``; the index + stats are
        inherited by forked DataLoader workers."""
        events = self._event_types_helper.extract(obj)  # type: ignore[attr-defined]
        cache_root = Path(self.spec_cache_dir) / self._spec_cache_namespace()  # type: ignore[arg-type]
        cache_root.mkdir(parents=True, exist_ok=True)
        expected_f = self._expected_raw_f_bins()
        seen: set[str] = set()
        n_hit = 0
        n_build = 0
        n_stats_hit = 0
        build_secs = 0.0
        load_secs = 0.0
        fit_secs = 0.0
        for event in events:
            key = event._splittable_event_uid()
            if key in seen:
                continue
            seen.add(key)
            stem = self._spec_cache_stem(key)
            npy_path = cache_root / f"{stem}.npy"
            json_path = cache_root / f"{stem}.json"
            stats_path = cache_root / f"{stem}.stats.npz"
            hit = npy_path.exists() and json_path.exists()
            stats_cached = False
            if hit:
                n_hit += 1
                meta = json.loads(json_path.read_text())
                ch_names = list(meta["ch_names"])
                total_frames = int(meta["total_frames"])
                sample_rate = int(meta["sample_rate"])
                # Defence in depth behind the namespace digest: a hit whose stored
                # geometry disagrees with the live config is a stale/corrupt spec —
                # fail loud rather than slice mismatched features.
                self._assert_cache_meta(key, meta, expected_f, ch_names)
                # With a robust-z stats sidecar present we can skip the full-frame
                # reload entirely — the stats are all prepare() needs (per-clip
                # reads memmap-slice the .npy lazily at train time). Only reload the
                # frames when we must (re)fit: no robust-z, or no sidecar yet.
                stats_cached = self.session_robust_z and stats_path.exists()
                if self.session_robust_z and not stats_cached:
                    _tl = time.perf_counter()
                    frames = torch.from_numpy(np.load(npy_path))
                    load_secs += time.perf_counter() - _tl
                else:
                    frames = None
            else:
                n_build += 1
                _t0 = time.perf_counter()
                raw_ta = next(self._get_data([event]))
                waveform = torch.from_numpy(np.asarray(raw_ta.data)).float()
                sample_rate = int(float(raw_ta.frequency))
                ch_names = list(raw_ta.ch_names)
                spec_f32 = self._whole_movie_spec(waveform, sample_rate)
                build_secs += time.perf_counter() - _t0
                del waveform
                # Finite guard only: the fp32 cache holds the raw |STFT| exactly
                # (FE-RAW-1 magnitudes exceed the old fp16 range), but a non-finite
                # STFT would still poison the cache and diverge from a finite
                # recompute, so fail loud on the offending session.
                self._assert_cache_finite(key, spec_f32)
                # The f32 STFT lands on disk verbatim; the per-clip read upcasts
                # trivially and fits robust-z on these SAME f32 frames, so the
                # cache and recompute paths are bit-identical.
                frames = spec_f32
                total_frames = int(frames.shape[-1])
                meta = {
                    "key": key,
                    "ch_names": ch_names,
                    "total_frames": total_frames,
                    "sample_rate": sample_rate,
                    "f_bins": int(frames.shape[1]),
                    "n_channels": int(frames.shape[0]),
                    "dtype": "float32",
                }
                # Commit order: json FIRST, npy LAST, each atomic. The hit check is
                # (npy AND json), so a reader only sees a hit once npy lands — by
                # which time json is already complete. A crash between the two
                # leaves an orphan json (hit=False → recompute, self-healing).
                self._atomic_write(
                    json_path,
                    lambda fh, m=meta: fh.write(json.dumps(m).encode()),
                )
                self._atomic_write(
                    npy_path,
                    lambda fh, fr=frames: np.save(fh, fr.cpu().numpy()),
                )
                del raw_ta
            if sample_rate != int(self.sample_rate_hz):
                raise ValueError(
                    f"cached session {key!r} sample_rate {sample_rate} != "
                    f"view.sample_rate_hz {int(self.sample_rate_hz)}: the per-clip "
                    "frame grid (n_time_bins_for_duration) and the encoder RoPE "
                    "ceiling both assume a single rate. Resample upstream."
                )
            if self.session_robust_z:
                normalizer = None
                if stats_cached:
                    normalizer = self._load_stats_sidecar(stats_path, meta)
                    if normalizer is not None:
                        n_stats_hit += 1
                if normalizer is None:
                    # No usable sidecar (fresh build, legacy frames-only cache, or a
                    # geometry-mismatched sidecar): fit from frames. If the sidecar
                    # was present-but-invalid we skipped the frame load above, so
                    # pull them now before fitting.
                    if frames is None:
                        _tl = time.perf_counter()
                        frames = torch.from_numpy(np.load(npy_path))
                        load_secs += time.perf_counter() - _tl
                    _tf = time.perf_counter()
                    normalizer = SessionRobustZNormalizer(
                        sigma_floor=self.session_z_sigma_floor,
                    ).fit(frames.float())
                    fit_secs += time.perf_counter() - _tf
                    # Self-healing: write the sidecar so the NEXT prepare skips the
                    # reload + refit. Existing frames-only caches upgrade in place.
                    self._write_stats_sidecar(stats_path, normalizer)
                self._scatter_stats_to_global(normalizer, ch_names)
                self._session_stats[key] = normalizer
            self._spec_cache_index[key] = _SpecCacheEntry(
                path=str(npy_path),
                ch_names=tuple(ch_names),
                total_frames=total_frames,
                sample_rate=sample_rate,
            )
            del frames
        logger.warning(
            "spec cache (#80): %d/%d sessions hit, %d built, %d robust-z stats "
            "from sidecar (%.1fs whole-movie |STFT| build tax; %.1fs npy load + "
            "%.1fs robust-z fit on stats-cold sessions) -> %s",
            n_hit,
            n_hit + n_build,
            n_build,
            n_stats_hit,
            build_secs,
            load_secs,
            fit_secs,
            cache_root,
        )

    def _scatter_spec_to_global(
        self, spec_session: torch.Tensor, ch_names: list[str],
    ) -> torch.Tensor:
        """``(C_session, F, T)`` session-order spec → ``(C_global, F, T)`` global
        order, the session's rows placed at the SAME global indices
        ``_get_channels(ch_names)`` assigns the waveform clip in the base
        ``_get_timed_array`` (neuro.py:453). Background rows stay 0, which equals
        the recompute path's STFT-of-zero rows because ``front_end="raw"`` +
        ``apply_log=False`` (enforced by ``_validate_spec_cache_config``) make
        |STFT|(0) = 0. So scatter-then-STFT (cache) and STFT-of-scatter (recompute)
        agree row-for-row."""
        channel_idx = self._get_channels(ch_names)  # type: ignore[attr-defined]
        c_global = max(self._channels.values()) + 1  # type: ignore[attr-defined]
        f_bins = int(spec_session.shape[1])
        t_bins = int(spec_session.shape[2])
        out = torch.zeros(c_global, f_bins, t_bins, dtype=spec_session.dtype)
        idx = torch.as_tensor(channel_idx, dtype=torch.long)
        out[idx] = spec_session
        return out

    def _cached_clip(self, event, start: float, duration: float) -> TimedArray:
        """Per-clip read off the whole-movie memmap: map [start, start+duration]
        to the global hop grid, slice the session memmap, scatter to global, then
        share ``_finalize_clip`` (robust-z + c_max + wrap) with the recompute
        path. The three Ben-accepted forks live here: the slice reads real
        neighbouring frames at the edges (no reflect pad), and the start snaps to
        the global hop grid (``round(s0/hop)``, ≤ hop/2 shift)."""
        key = event._splittable_event_uid()
        entry = self._spec_cache_index.get(key)
        if entry is None:
            raise KeyError(
                f"MultiStftView.spec_cache_dir is on but no cached spec for "
                f"session {key!r}. prepare() must run (and see this session) "
                f"before clips are drawn. Have specs for: "
                f"{sorted(self._spec_cache_index)[:8]}..."
            )
        spec_mm = np.load(entry.path, mmap_mode="r")  # (C_session, F, T_total) fp32
        freq = float(entry.sample_rate)
        # Sample offset of the clip in the _get_data array (base relocates that
        # array so sample 0 == event.start; the base also adds self.offset to
        # start before windowing — replicate both). center=True frame g is
        # centered at sample g*hop → g0 = round(s0/hop) (fork 2: global grid).
        s0 = int(round((float(start) + float(self.offset) - float(event.start)) * freq))
        g0 = int(round(s0 / self.hop_length))
        n_frames = self.n_time_bins_for_duration(duration)
        total = int(entry.total_frames)
        # Clamp the window inside the recording (clips drawn within bounds keep
        # their grid position; a window past the end is pulled back, matching the
        # whole-movie spec's own boundary frames).
        g0 = max(0, min(g0, max(0, total - n_frames)))
        sliced = np.asarray(spec_mm[:, :, g0 : g0 + n_frames]).astype(np.float32)
        del spec_mm
        if sliced.shape[-1] < n_frames:
            # Recording shorter than one clip window: zero-pad the tail (the same
            # absent-frame convention as a too-short recompute window).
            pad = np.zeros(
                (sliced.shape[0], sliced.shape[1], n_frames - sliced.shape[-1]),
                dtype=np.float32,
            )
            sliced = np.concatenate([sliced, pad], axis=-1)
        spec = self._scatter_spec_to_global(
            torch.from_numpy(sliced), list(entry.ch_names),
        )
        return self._finalize_clip(event, spec, start, duration)
