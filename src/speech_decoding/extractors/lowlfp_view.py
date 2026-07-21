"""Raw-waveform low-field-signal (LFS) band producer for the v3 spec cache.

``LowLfpView`` bakes a 1-30 Hz raw-waveform band at 64 Hz as a ``(C, F=1, T_64)``
cache leaf, reusing ALL of :class:`MultiStftView`'s whole-movie cache-build /
prepare / robust-z / atomic-write machinery. It replaces ONLY the per-band
producer (the ``|STFT|`` call) with a SOS Butterworth bandpass + grid-sample onto
the 64 Hz frame clock — no STFT is ever computed.

Chang 2-stream frontend redesign (2026-07-21, memo
``project-chang-2stream-frontend-redesign-tasklist-2026-07-21``): the LFS stream is
``[raw LFS HP1/LP30 conv-pool 64->32]``. The 1/f mid band is redundant in the
whitened-HGA/|STFT| stream but the raw LFS keeps the oscillation, so this band is a
raw-domain producer, not a spectrogram.

KEY TRICK — construct with ``front_end="band"`` and DUMMY-BUT-VALID band params
(``band_nperseg=64, band_hop=32``, ``hop_length=32``) so the parent's
``_validate_band_config`` (64 % 32 == 0, half=32 % 32 == 0, hop_length == band_hop)
and ``_validate_spec_cache_config`` (front_end in {"raw","band"}) pass with NO edit
to the locked MultiStftView. Those dummy STFT params are IGNORED by the overridden
producer; ``band_f_lo_hz=1.0 / band_f_hi_hz=30.0`` carry the REAL passband.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import scipy.signal
import torch

from speech_decoding.extractors.view import MultiStftView


class LowLfpView(MultiStftView):
    """1-30 Hz raw-waveform LFS band at 64 Hz, baked as a ``(C, 1, T_64)`` leaf.

    Subclasses :class:`MultiStftView` and overrides ONLY the per-band producer +
    the four cache-metadata hooks below, so every session-robust-z / atomic-write /
    memmap-slice code path is inherited verbatim. Build it via
    ``LowLfpView(front_end="band", band_nperseg=64, band_hop=32, hop_length=32,
    band_f_lo_hz=1.0, band_f_hi_hz=30.0, ...)`` — the dummy nperseg/hop only pass the
    parent validators; the real work is bandpass + grid-sample in :meth:`_lfs_frames`.
    """

    def _lfs_frames(
        self, waveform: torch.Tensor, sample_rate: int,
    ) -> torch.Tensor:
        """``(C, T)`` raw waveform → ``(C, 1, n_frames)`` LFS frames at 64 Hz.

        1-30 Hz 4th-order Butterworth bandpass (SOS + zero-phase ``sosfiltfilt``,
        the same filter as ``views._low_lfp``), then grid-sampled onto the
        ``torch.stft(center=True)`` frame grid at ``hop = self.hop_length`` (=32 for
        2048 Hz raw) so frame ``i`` is centered at raw-sample ``i*hop`` — the SAME window-center
        grid an HGA STFT (N=64, hop=16) rides, one LFS frame per two HGA frames.

        Single source of truth for the fit pass (``_whole_movie_spec``) and the
        per-clip apply pass (``_spec_from_waveform``), so fit-time and apply-time
        features are identical by construction.
        """
        # Hop is the INHERITED cache clock (``self.hop_length``), NOT a fresh
        # ``sample_rate // 64`` — the parent's ``n_time_bins_for_duration`` counts
        # frames with ``hop_length``, so ``_lfs_frames`` MUST sample on the same hop
        # or the LFS leaf and the clock disagree by a frame at other sample rates.
        # The assert pins that hop to a true 64 Hz clock (2048 Hz raw -> 32).
        hop = self.hop_length
        if sample_rate != 64 * hop:
            raise ValueError(
                f"LowLfpView 64 Hz clock needs sample_rate == 64*hop_length "
                f"(got sample_rate={sample_rate}, hop_length={hop} -> "
                f"{64 * hop} Hz); reconfigure band_hop/hop_length for this corpus."
            )
        nyq = sample_rate / 2
        sos = scipy.signal.butter(
            4, [1.0 / nyq, 30.0 / nyq], btype="band", output="sos"
        )
        arr = waveform.detach().cpu().numpy().astype(np.float64)
        try:
            filtered = scipy.signal.sosfiltfilt(sos, arr, axis=-1)
        except ValueError as exc:
            # neuralset's ``prepare()`` probes the extractor geometry with a ~1 ms
            # (duration=0.001, ~2-sample) clip that is SHORTER than the zero-phase
            # filter's default padlen (~27 for order 4). That probe only needs the
            # OUTPUT SHAPE, so filter it with a clamped pad. REAL clips (>=1 s =
            # >=2048 samples) are >> padlen, so they never enter here and keep
            # scipy's default padlen ⇒ byte-identical features.
            if "padlen" not in str(exc):
                raise
            filtered = scipy.signal.sosfiltfilt(
                sos, arr, axis=-1, padlen=max(0, arr.shape[-1] - 1)
            )
        filtered = np.asarray(filtered)
        L = filtered.shape[-1]
        n_frames = 1 + L // hop
        idx = [min(k * hop, L - 1) for k in range(n_frames)]
        frames = filtered[:, idx]
        return torch.from_numpy(frames.astype(np.float32)).unsqueeze(1)

    def _whole_movie_spec(
        self, waveform: torch.Tensor, sample_rate: int,
    ) -> torch.Tensor:
        return self._lfs_frames(waveform, sample_rate)

    def _spec_from_waveform(
        self, waveform_t: torch.Tensor, sample_rate: int,
    ) -> torch.Tensor:
        return self._lfs_frames(waveform_t, sample_rate)

    def _expected_raw_f_bins(self) -> int:
        return 1

    def _spec_cache_namespace(self) -> str:
        # XOR in an "lfp" marker so a raw-waveform LFS cache can NEVER collide with
        # an |STFT| band dir that happens to share the same (dummy) band geometry.
        return super()._spec_cache_namespace() + "-lfp"

    def _spec_cache_geometry(self) -> dict:
        # Honest LFS geometry (NOT the dummy band_nperseg/band_hop): this is what
        # actually determines the cached frames, asserted-if-present on a cache hit.
        return {
            "hop_length": 32,
            "lfp_lo_hz": 1.0,
            "lfp_hi_hz": 30.0,
            "lfp_target_hz": 64,
        }

    def _winsor_band_name(self) -> tp.Optional[str]:
        # Global scalar cap for v1 (no per-band |z| tag). The raw-LFS |z|
        # distribution differs from any |STFT| band, so it must NOT reuse an STFT
        # winsor tag.
        # TODO: a raw-waveform-specific winsor tag (e.g. "v3lfs") may be added later.
        return None
