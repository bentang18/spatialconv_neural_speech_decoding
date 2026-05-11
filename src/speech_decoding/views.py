"""NeuralSet-aligned reference × view preprocessing for Stage 0 L.2 sweep.

Replaces upstream's `preprocess_data` with a strictly factored
(reference) × (view) pipeline so each L.2 cell maps to a real, distinct
operation. The previous wrapper collapsed bipolar / shaft-Lap / HG /
multi-band / wavelet onto upstream's tiny string vocabulary
(`laplacian`, `stft_abs`, `none`, `laplacian-stft_abs`) — five hours of
cluster time produced four conditions instead of nine.

Design contract:
    preprocess_views(data, electrode_labels, ref_kind, view_kind, sampling_rate,
                     upstream_helpers) -> torch.Tensor of shape (B, C', F)

`B` = batch (typically 1, the wrapper concatenates externally),
`C'` = post-reference channel count (≤ original C),
`F` = view-specific feature dim per channel.

Tier-A (this module): raw / bipolar / shaft_laplacian × raw_voltage / stft_abs
/ hg_envelope. Tier-B (CAR, multi-band, wavelet) lands when needed.

Parity contract: ref_kind="shaft_laplacian" + view_kind="stft_abs" must be
byte-equivalent to upstream `preprocess_data(..., preprocess="laplacian-stft_abs")`.
The two primitives are reused directly from upstream where they exist
(`laplacian_rereference_neural_data`, `preprocess_stft`).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from scipy import signal

from speech_decoding.extractors.reference import parse_shaft  # noqa: F401


REFERENCES: tuple[str, ...] = (
    "raw",
    "bipolar",
    "shaft_laplacian",
    "global_car",
    "shaft_car",
    "median",
)
VIEWS: tuple[str, ...] = (
    "raw_voltage",
    "stft_abs",
    "hg_envelope",
    "log_stft",
    "hg_envelope_wide",
    "low_lfp",
    "multi_band_log_power",
    "wavelet_db4",
    "instantaneous_phase",
)


def preprocess_views(
    data: torch.Tensor,
    electrode_labels: list[str],
    ref_kind: str,
    view_kind: str,
    *,
    sampling_rate: int = 2048,
    upstream_helpers: dict[str, Any],
) -> tuple[torch.Tensor, list[str]]:
    """Apply (reference, view) pair. Returns (features, post-ref electrode_labels)."""
    if ref_kind not in REFERENCES:
        raise ValueError(f"Unknown ref_kind {ref_kind!r}; expected one of {REFERENCES}")
    if view_kind not in VIEWS:
        raise ValueError(f"Unknown view_kind {view_kind!r}; expected one of {VIEWS}")

    data_ref, labels_ref = apply_reference(
        data, electrode_labels, ref_kind, upstream_helpers=upstream_helpers
    )
    feats = apply_view(
        data_ref,
        view_kind,
        sampling_rate=sampling_rate,
        upstream_helpers=upstream_helpers,
    )
    return feats, labels_ref


def apply_reference(
    data: torch.Tensor,
    electrode_labels: list[str],
    ref_kind: str,
    *,
    upstream_helpers: dict[str, Any],
) -> tuple[torch.Tensor, list[str]]:
    """Spatial filter. Input/output shape: (B, C, T) -> (B, C', T)."""
    if ref_kind == "raw":
        return data, list(electrode_labels)
    if ref_kind == "shaft_laplacian":
        return _shaft_laplacian_via_upstream(
            data, electrode_labels, upstream_helpers["laplacian"]
        )
    if ref_kind == "bipolar":
        return _adjacent_bipolar(data, electrode_labels)
    if ref_kind == "global_car":
        return _global_car(data, electrode_labels)
    if ref_kind == "shaft_car":
        return _shaft_car(data, electrode_labels)
    if ref_kind == "median":
        return _median_reference(data, electrode_labels)
    raise ValueError(f"Unknown ref_kind: {ref_kind}")


def apply_view(
    data: torch.Tensor,
    view_kind: str,
    *,
    sampling_rate: int,
    upstream_helpers: dict[str, Any],
) -> torch.Tensor:
    """Spectral / temporal view. Input shape: (B, C, T) -> (B, C, F)."""
    if view_kind == "raw_voltage":
        return data
    if view_kind == "stft_abs":
        upstream_stft: Callable[..., torch.Tensor] = upstream_helpers["stft"]
        params = upstream_helpers["stft_params"]
        return upstream_stft(
            data, sampling_rate=sampling_rate,
            preprocess="stft_abs", preprocess_parameters=params,
        )
    if view_kind == "hg_envelope":
        return _hg_envelope(data, sampling_rate, lo_hz=70.0, hi_hz=150.0)
    if view_kind == "log_stft":
        upstream_stft: Callable[..., torch.Tensor] = upstream_helpers["stft"]
        params = upstream_helpers["stft_params"]
        x = upstream_stft(
            data, sampling_rate=sampling_rate,
            preprocess="stft_abs", preprocess_parameters=params,
        )
        return torch.log(x + 1e-6)
    if view_kind == "hg_envelope_wide":
        return _hg_envelope(data, sampling_rate, lo_hz=70.0, hi_hz=200.0)
    if view_kind == "low_lfp":
        return _low_lfp(data, sampling_rate)
    if view_kind == "multi_band_log_power":
        return _multi_band_log_power(data, sampling_rate)
    if view_kind == "wavelet_db4":
        return _wavelet_db4(data, sampling_rate)
    if view_kind == "instantaneous_phase":
        return _instantaneous_phase(data, sampling_rate)
    raise ValueError(f"Unknown view_kind: {view_kind}")


def _shaft_laplacian_via_upstream(
    data: torch.Tensor,
    electrode_labels: list[str],
    upstream_laplacian: Callable[..., Any],
) -> tuple[torch.Tensor, list[str]]:
    """Use upstream's `laplacian_rereference_neural_data` for byte parity.

    Upstream signature: (electrode_data, electrode_labels, remove_non_laplacian=True)
    where electrode_data shape can be (B, C, T) or (T, C). We pass (B, C, T)
    matching upstream's preprocess_data caller.
    """
    out, labels_out, _ = upstream_laplacian(
        data, list(electrode_labels), remove_non_laplacian=False
    )
    return out, list(labels_out)


def _adjacent_bipolar(
    data: torch.Tensor, electrode_labels: list[str]
) -> tuple[torch.Tensor, list[str]]:
    """Within-shaft adjacent-pair bipolar: ch[i] - ch[i+1] for i, i+1 on same shaft.

    Output channel count = sum_over_shafts(n_shaft - 1). Order preserves original.
    """
    shafts = [parse_shaft(name) for name in electrode_labels]
    pairs: list[tuple[int, int, str]] = []
    for i in range(len(electrode_labels) - 1):
        s_i, n_i = shafts[i]
        s_j, n_j = shafts[i + 1]
        if s_i == s_j and n_i is not None and n_j is not None and n_j == n_i + 1:
            pairs.append((i, i + 1, f"{electrode_labels[i]}-{electrode_labels[i + 1]}"))
    if not pairs:
        raise ValueError(
            "No adjacent within-shaft pairs found; check electrode label ordering"
        )
    anode_idx = torch.tensor([a for a, _, _ in pairs], dtype=torch.long)
    cathode_idx = torch.tensor([c for _, c, _ in pairs], dtype=torch.long)
    out = data.index_select(dim=1, index=anode_idx) - data.index_select(
        dim=1, index=cathode_idx
    )
    new_labels = [name for _, _, name in pairs]
    return out, new_labels


def _global_car(
    data: torch.Tensor, electrode_labels: list[str]
) -> tuple[torch.Tensor, list[str]]:
    return data - data.mean(dim=1, keepdim=True), list(electrode_labels)


def _shaft_car(
    data: torch.Tensor, electrode_labels: list[str]
) -> tuple[torch.Tensor, list[str]]:
    """Subtract per-shaft mean from each channel."""
    shafts = [parse_shaft(name)[0] for name in electrode_labels]
    out = data.clone()
    for shaft in set(shafts):
        idx = torch.tensor(
            [i for i, s in enumerate(shafts) if s == shaft], dtype=torch.long
        )
        if idx.numel() == 0:
            continue
        shaft_mean = out.index_select(dim=1, index=idx).mean(dim=1, keepdim=True)
        out[:, idx, :] = out[:, idx, :] - shaft_mean
    return out, list(electrode_labels)


def _hg_envelope(
    data: torch.Tensor, sampling_rate: int, *, lo_hz: float = 70.0, hi_hz: float = 150.0
) -> torch.Tensor:
    """Bandpass + Hilbert envelope, per channel. Default 70-150 Hz HG; 70-200 for wide.

    Output shape: (B, C, T) — same length as input, raw envelope (no log,
    no downsample) so the downstream flatten doesn't change dimension.
    """
    arr = data.detach().cpu().numpy().astype(np.float64)
    nyq = 0.5 * sampling_rate
    sos = signal.butter(4, [lo_hz / nyq, hi_hz / nyq], btype="band", output="sos")
    filtered = np.asarray(signal.sosfiltfilt(sos, arr, axis=-1))
    analytic = np.asarray(signal.hilbert(filtered, axis=-1))
    env = np.abs(analytic).astype(np.float32)
    return torch.from_numpy(env)


def _median_reference(
    data: torch.Tensor, electrode_labels: list[str]
) -> tuple[torch.Tensor, list[str]]:
    """Subtract per-timepoint median across electrodes (MVPFormer 2026 ICLR)."""
    return data - data.median(dim=1, keepdim=True).values, list(electrode_labels)


def _low_lfp(data: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    """Low-LFP (< 30 Hz) bandpass, per channel. Output shape (B, C, T)."""
    arr = data.detach().cpu().numpy().astype(np.float64)
    nyq = 0.5 * sampling_rate
    sos = signal.butter(4, [1.0 / nyq, 30.0 / nyq], btype="band", output="sos")
    filtered = np.asarray(signal.sosfiltfilt(sos, arr, axis=-1)).astype(np.float32)
    return torch.from_numpy(filtered)


_MULTI_BANDS: tuple[tuple[float, float], ...] = (
    (1.0, 4.0),     # delta
    (4.0, 8.0),     # theta
    (8.0, 13.0),    # alpha
    (13.0, 30.0),   # beta
    (30.0, 70.0),   # gamma
    (70.0, 150.0),  # high-gamma
)


def _multi_band_log_power(data: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    """6-band log-power (delta, theta, alpha, beta, gamma, HG).

    Per band: 4th-order Butterworth bandpass + Hilbert envelope + log + per-channel
    mean over time. Output shape (B, C, n_bands).
    """
    arr = data.detach().cpu().numpy().astype(np.float64)
    nyq = 0.5 * sampling_rate
    band_powers: list[np.ndarray] = []
    for lo_hz, hi_hz in _MULTI_BANDS:
        sos = signal.butter(4, [lo_hz / nyq, hi_hz / nyq], btype="band", output="sos")
        filtered = np.asarray(signal.sosfiltfilt(sos, arr, axis=-1))
        analytic = np.asarray(signal.hilbert(filtered, axis=-1))
        env = np.abs(analytic)
        log_power = np.log(env + 1e-6).mean(axis=-1)
        band_powers.append(log_power)
    feats = np.stack(band_powers, axis=-1).astype(np.float32)
    return torch.from_numpy(feats)


def _wavelet_db4(data: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    """db4 multi-resolution wavelet decomposition (MVPFormer 2026 ICLR).

    Per channel: pywt.wavedec at level=6 with db4 wavelet, gives 7 coefficient
    arrays (1 approx + 6 details). For linear-readout fairness with STFT, we
    return per-band log-energy summary: log(mean(|coef|^2)). Output shape
    (B, C, n_levels+1) = (B, C, 7).
    """
    import pywt
    arr = data.detach().cpu().numpy().astype(np.float64)
    B, C, T = arr.shape
    n_levels = 6
    feats = np.zeros((B, C, n_levels + 1), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            coeffs = pywt.wavedec(arr[b, c], "db4", level=n_levels)
            for k, coef in enumerate(coeffs):
                feats[b, c, k] = np.log(np.mean(coef ** 2) + 1e-12)
    return torch.from_numpy(feats)


def _instantaneous_phase(data: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    """Theta-band (4-8 Hz) instantaneous phase, summarized per channel.

    Bandpass to theta + Hilbert -> instantaneous phase phi(t). For a linear
    readout with fixed feature-dim per channel, summarize as
    (mean cos(phi), mean sin(phi)) — these are the real and imaginary parts
    of the time-averaged complex phase, magnitude = within-window PLV proxy.
    Output shape (B, C, 2).
    """
    arr = data.detach().cpu().numpy().astype(np.float64)
    nyq = 0.5 * sampling_rate
    sos = signal.butter(4, [4.0 / nyq, 8.0 / nyq], btype="band", output="sos")
    filtered = np.asarray(signal.sosfiltfilt(sos, arr, axis=-1))
    analytic = np.asarray(signal.hilbert(filtered, axis=-1))
    phase = np.angle(analytic)
    cos_mean = np.cos(phase).mean(axis=-1)
    sin_mean = np.sin(phase).mean(axis=-1)
    feats = np.stack([cos_mean, sin_mean], axis=-1).astype(np.float32)
    return torch.from_numpy(feats)


def apply_temporal_filter_inplace(
    tensor: torch.Tensor,
    *,
    sampling_rate: int,
    notch_freqs: tuple[float, ...] = (),
    notch_q: float = 30.0,
    hpf_hz: float = 0.0,
    hpf_order: int = 4,
) -> None:
    """In-place IIR notch + Butterworth HPF on a session-length voltage tensor.

    Operates on tensors of shape (C, T) — the canonical
    `subject.neural_data_cache[trial_id]` layout. Applied to the FULL session
    voltage *before* trial windowing so 0.5/1 Hz HPF cutoffs (period 1-2 s)
    have enough samples to escape filtfilt edge artifacts that would otherwise
    dominate a 1-s windowed slice. Notch + HPF are stable IIR sections and
    apply per-channel via `sosfiltfilt` (zero-phase).

    No-op when both `notch_freqs` is empty and `hpf_hz <= 0`.
    """
    if not notch_freqs and hpf_hz <= 0:
        return
    arr = tensor.detach().cpu().numpy().astype(np.float64)
    nyq = 0.5 * sampling_rate
    for f0 in notch_freqs:
        if f0 <= 0 or f0 >= nyq:
            continue
        b, a = signal.iirnotch(f0 / nyq, notch_q)
        sos = signal.tf2sos(b, a)
        arr = np.asarray(signal.sosfiltfilt(sos, arr, axis=-1))
    if hpf_hz > 0:
        sos = signal.butter(hpf_order, hpf_hz / nyq, btype="highpass", output="sos")
        arr = np.asarray(signal.sosfiltfilt(sos, arr, axis=-1))
    tensor.copy_(torch.from_numpy(arr.astype(np.float32)))


def make_upstream_helpers(
    preprocess_stft: Callable[..., torch.Tensor],
    laplacian_rereference_neural_data: Callable[..., Any],
    stft_params: dict[str, Any],
) -> dict[str, Any]:
    """Bundle upstream callables so the wrapper can stay decoupled from imports."""
    return {
        "stft": preprocess_stft,
        "laplacian": laplacian_rereference_neural_data,
        "stft_params": stft_params,
    }
