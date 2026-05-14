"""V14 I2L log-STFT view extractor.

Inherits :class:`speech_decoding.extractors.reference.CARIeegExtractor` (which
itself inherits NeuralSet's :class:`neuralset.extractors.neuro.IeegExtractor`),
so the v14 default preprocessing chain ``N1 × R2 × F1 × A.0`` runs in front of
the STFT:

  * ``N1 train-set z-score``  → ``scaler="StandardScaler"``
  * ``R2 shaft CAR``          → ``car="shaft"``
  * ``F1 60 Hz notch``        → ``notch_filter=60.0``
  * ``A.0 [0, 1]s anchor``    → ``start=0.0, duration=1.0`` at the segmenter

Then applies a torch-STFT byte-equivalent to the upstream Neuroprobe baseline
helper ``preprocess_stft(..., preprocess="stft_abs")`` with log compression
(``log(x + eps)``) to match Whisper-large-v3 input dynamic range for the
Stage-2 D-SigLIP cross-modal contrastive loss.

Output shape: ``(C, F_bin=38, T_bin=17)`` per 1-s @ 2048 Hz Ieeg trigger
window. Time-last per NeuralSet's :class:`~neuralset.base.TimedArray`
convention (``frequency = T_bin / duration = 17 Hz``). The v14 encoder
transposes to ``(C, T_bin, F_bin)`` internally via its ``time_last_input``
flag.

STFT params source: Neuroprobe Section D (page 18) — ``nperseg=512``,
``poverlap=0.75``, ``window=hann``, freq range ``0-150 Hz``, sample rate
``2048 Hz``. Frame count math: ``torch.stft(center=True)`` reflect-pads
``nperseg//2 = 256`` samples each side → padded length 2560 →
``1 + (2560 - 512) // 128 = 17`` frames. Freq bin math: ``rfftfreq(512,
1/2048)`` has bins at ``[0, 4, 8, ..., 1024]`` Hz; keeping ``0 ≤ f ≤ 150``
selects 38 bins.
"""

from __future__ import annotations

import numpy as np
import torch
from neuralset.base import TimedArray

from speech_decoding.extractors.reference import CARIeegExtractor


def _log_stft_view(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    nperseg: int,
    poverlap: float,
    min_freq_hz: float,
    max_freq_hz: float,
    log_eps: float,
) -> torch.Tensor:
    """Compute the upstream-byte-parity log-STFT magnitude.

    Matches Neuroprobe ``preprocess_stft(..., "stft_abs")`` then applies
    ``log(x + eps)``. Input shape ``(..., C, T_samples)``; output shape
    ``(..., C, F_bin, T_bin)``.
    """

    noverlap = int(nperseg * poverlap)
    hop_length = nperseg - noverlap
    window = torch.hann_window(nperseg, device=waveform.device)

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
    return torch.log(torch.abs(complex_stft) + log_eps)


class LogStftView(CARIeegExtractor):
    """v14 I2L log-STFT view on top of CARIeegExtractor's waveform pipeline.

    Per-event output: ``(C, F_bin, T_bin)`` log-STFT magnitude, time-last.
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
