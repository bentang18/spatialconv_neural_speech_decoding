"""V14 view extractors: single-STFT (legacy) + Multi-STFT (T1.5, 5/22 pivot).

T1.5 / 5/22 spec lock (project_v14_imindbench_multistft_pivot_2026_05_22.md):
v14's front-end-of-record is ``MultiStftView`` — 3 STFTs at common
hop=256 @ 2048 Hz (8 Hz frame rate, B20 v4 lock 2026-05-24;
Nperseg=1024/512/256) feeding a 30-bin ⅓-octave filterbank with per-bin
STFT routing (k0–k14 from low, k15–k21 from mid, k22–k29 from hi).
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

Per-corpus valid-bin mask: ``multi_stft_valid_bin_mask(low_hz, high_hz)`` —
boolean (F=30,) tensor flagging which filterbank bins are spanned by a
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
emits ``(C, F=30, T_bin)`` at the common hop=256 → 8 Hz frame rate (B20
v4 lock). The v14 encoder transposes to ``(C, T_bin, F_bin)`` internally
via its ``time_last_input`` flag.

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
import typing as tp

import numpy as np
import torch
from neuralset.base import TimedArray

from speech_decoding.extractors.reference import CARIeegExtractor


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

    stft_nperseg: int = 512
    stft_poverlap: float = 0.75
    stft_max_freq_hz: float = 150.0
    stft_min_freq_hz: float = 0.0
    stft_log_eps: float = 1e-6
    apply_log: bool = False
    c_max: int | None = None

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


class MultiStftView(CARIeegExtractor):
    """v14 5/22 Multi-STFT front-end on top of CARIeegExtractor's waveform
    pipeline. Replaces ``LogStftView`` as the v14 default; single-STFT lives
    on as the F-single-STFT sister cell.

    Per-event output: ``(C, F=30, T_bin)`` magnitude filterbank, time-last.
    Three internal STFTs at common hop=256 @ 2048 Hz (Nperseg=1024/512/256)
    feed a 30-bin ⅓-octave filterbank centered at ``2^(k/3)`` Hz for
    ``k ∈ [0, 30)`` (~1 Hz → ~813 Hz). Routing: k0–k14 from low, k15–k21 from
    mid, k22–k29 from hi (see ``MULTI_STFT_ROUTING``). Hop=256 yields 8 Hz
    frame rate (B20 v4 lock 2026-05-24); previous default hop=128 (true rate
    16 Hz; "14.7 Hz" was the pre-B20 mislabel) is retired.

    5/25 swap: ``apply_log`` defaults to ``False`` — iMINDBench-parity raw
    magnitude. Set ``apply_log=True`` for the F-log-amplitude sister cell
    (pre-5/25 default behavior).
    """

    sample_rate_hz: int = 2048
    # FE-01 (B20 v4 lock 2026-05-24): hop=256 @ 2048 Hz → 8 Hz frame rate.
    # Matches Whisper teacher-pool target (B06 PM lock 5/25: 50 Hz → 8 Hz),
    # avoids upsample at Phase-3 student side. Previous default hop=128 (true
    # rate 16 Hz; "14.7 Hz" was the pre-B20 mislabel) was v3-era and is retired.
    hop_length: int = 256
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
    # Pydantic doesn't allow tuple[int, ...] as a default freely — keep
    # the routing as a class-level constant pulled from the module.
    fbank_routing: tp.ClassVar[tuple[int, ...]] = MULTI_STFT_ROUTING

    def _get_timed_array(
        self, event, start: float, duration: float,
    ) -> TimedArray:
        waveform_ta = super()._get_timed_array(event, start, duration)
        waveform_t = torch.from_numpy(np.asarray(waveform_ta.data)).float()
        spec = _multi_stft_view(
            waveform_t,
            sample_rate=int(float(waveform_ta.frequency)),
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
