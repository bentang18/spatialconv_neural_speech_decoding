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

import math
from typing import Any, Callable

import numpy as np
import torch
from scipy import signal

from speech_decoding.extractors.reference import parse_shaft  # noqa: F401
from speech_decoding.extractors.view import (
    _build_multi_stft_filterbank,
    _multi_stft_view,
    multi_stft_bin_centers_hz,
)


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
    "hg_envelope_70_90",
    "hg_envelope_90_120",
    "hg_envelope_120_150",
    "multi_stft",
    "raw_multi_stft",
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
    if view_kind == "hg_envelope_70_90":
        return _hg_envelope(data, sampling_rate, lo_hz=70.0, hi_hz=90.0)
    if view_kind == "hg_envelope_90_120":
        return _hg_envelope(data, sampling_rate, lo_hz=90.0, hi_hz=120.0)
    if view_kind == "hg_envelope_120_150":
        return _hg_envelope(data, sampling_rate, lo_hz=120.0, hi_hz=150.0)
    if view_kind == "low_lfp":
        return _low_lfp(data, sampling_rate)
    if view_kind == "multi_band_log_power":
        return _multi_band_log_power(data, sampling_rate)
    if view_kind == "wavelet_db4":
        return _wavelet_db4(data, sampling_rate)
    if view_kind == "instantaneous_phase":
        return _instantaneous_phase(data, sampling_rate)
    if view_kind == "multi_stft":
        params = _resolve_multi_stft_params(upstream_helpers.get("multi_stft_params"))
        return _multi_stft_band(data, sampling_rate, params)
    if view_kind == "raw_multi_stft":
        params = _resolve_multi_stft_params(upstream_helpers.get("multi_stft_params"))
        return _raw_multi_stft_bins(data, sampling_rate, params)
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
    """Subtract per-shaft mean from each channel.

    Relies on the caller passing electrode labels already stripped of BT's `*`/`#`
    markers (upstream `BrainTreebankSubject._clean_electrode_label` does this). Raw
    BT labels keep those markers, which make `parse_shaft` mis-split a contact into
    a singleton shaft whose CAR then zeroes it — the sub_2 zeroing bug. The sweep
    consumes upstream-cleaned labels, so it is immune; do not feed raw labels here.
    """
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


# Channel tiling for the in-place session filter. Pure memory knob (numerically
# inert — per-channel filtering is independent), sized so one float64 block plus
# the sosfiltfilt input+output pair stays well under a node's RAM on a ~10^7-sample
# session: 32 ch x 27.5M samp x 8 B ~ 7 GB/block, ~14 GB transient.
_FILTER_CHUNK_CH = 32


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

    Filtered in channel chunks: `sosfiltfilt` along `axis=-1` is independent per
    channel, so chunking rows is bit-identical to filtering the whole tensor at
    once, but caps peak memory at ~`_FILTER_CHUNK_CH x T x 8` bytes instead of the
    full `C x T` float64 copy (plus the sosfiltfilt input+output pair). A full
    BrainTreebank session is ~10^7 samples; the whole-tensor float64 path peaks at
    tens of GB and OOMs the cross cells (sub 2 / trial 4 = 164 ch x 27.5M samp).
    """
    if not notch_freqs and hpf_hz <= 0:
        return
    nyq = 0.5 * sampling_rate
    sos_sections: list = []  # scipy SOS arrays (untyped upstream)
    for f0 in notch_freqs:
        if f0 <= 0 or f0 >= nyq:
            continue
        b, a = signal.iirnotch(f0 / nyq, notch_q)
        sos_sections.append(signal.tf2sos(b, a))
    if hpf_hz > 0:
        sos_sections.append(
            signal.butter(hpf_order, hpf_hz / nyq, btype="highpass", output="sos")
        )
    if not sos_sections:
        return
    n_ch = tensor.shape[0]
    for start in range(0, n_ch, _FILTER_CHUNK_CH):
        sl = slice(start, min(start + _FILTER_CHUNK_CH, n_ch))
        block = tensor[sl].detach().cpu().numpy().astype(np.float64)
        for sos in sos_sections:
            block = np.asarray(signal.sosfiltfilt(sos, block, axis=-1))
        tensor[sl].copy_(torch.from_numpy(block.astype(np.float32)))


# ---------------------------------------------------------------------------
# Multi-STFT front-end sweep views (FE filterbank sweep, 2026-06-03)
#
# Two sweep view-kinds sitting on the frozen 3-STFT grid (nperseg 1024/512/256,
# hop 128 @ 2048 Hz):
#   * ``multi_stft``     — v14's constant-Q log-octave triangular filterbank.
#   * ``raw_multi_stft`` — iMINDBench raw |STFT| bins, no filterbank (the
#                          control / harness gate). Spec: build doc §2.
# Filterbank knobs (f0, cap, octave_step, half_bw) are threaded through
# ``upstream_helpers["multi_stft_params"]`` so the runner can sweep them without
# touching the live MultiStftView extractor.
# ---------------------------------------------------------------------------

_MULTI_STFT_DEFAULTS: dict[str, Any] = {
    # STFT grid (frozen, iMINDBench parity).
    "nperseg_low": 1024,
    "nperseg_mid": 512,
    "nperseg_hi": 256,
    "hop_length": 128,
    # Filterbank knobs — default = build-doc cell 1 (constant-Q matched).
    "f0_hz": 2.0,
    "cap_hz": 256.0,
    "octave_step": 0.5,
    "half_bw_octaves": 0.5,
    "log_eps": 1e-6,
    "apply_log": False,
    # raw_multi_stft band edges (Hz) per source STFT (build doc §2).
    "raw_low_band": (2.0, 40.0),
    "raw_mid_band": (20.0, 148.0),
    "raw_hi_band": (80.0, 248.0),
}

# §6 routing crossovers (iMINDBench-parity band midpoints) and 1-cycle floors.
_ROUTING_CROSSOVERS: tuple[float, float] = (32.0, 152.0)
_TIER_ONE_CYCLE_FLOOR_HZ: dict[int, float] = {0: 2.0, 1: 4.0, 2: 8.0}


def _resolve_multi_stft_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Merge caller params over the frozen defaults."""
    merged = dict(_MULTI_STFT_DEFAULTS)
    if params:
        merged.update(params)
    return merged


def _multi_stft_n_bins(f0_hz: float, cap_hz: float, octave_step: float) -> int:
    """Largest ``n`` with ``f0 * 2^((n-1)*step) <= cap`` — contiguous octave bins."""
    if cap_hz < f0_hz:
        raise ValueError(f"cap_hz {cap_hz} must be >= f0_hz {f0_hz}")
    return int(math.floor(math.log2(cap_hz / f0_hz) / octave_step + 1e-9)) + 1


def _derive_multi_stft_routing(centers: torch.Tensor) -> tuple[int, ...]:
    """Per §6: assign each bin center to a source STFT tier at the crossovers.

    center < 32 Hz → low (1024); 32 ≤ center < 152 → mid (512); ≥ 152 → hi (256).
    """
    lo_x, hi_x = _ROUTING_CROSSOVERS
    routing: list[int] = []
    for c in centers.tolist():
        if c < lo_x:
            routing.append(0)
        elif c < hi_x:
            routing.append(1)
        else:
            routing.append(2)
    return tuple(routing)


def _multi_stft_valid_mask(
    centers: torch.Tensor, routing: tuple[int, ...]
) -> torch.Tensor:
    """1-cycle floor guard (FE-MULTISTFT-1): a bin below its tier floor is dead."""
    return torch.tensor(
        [c >= _TIER_ONE_CYCLE_FLOOR_HZ[r] for c, r in zip(centers.tolist(), routing)],
        dtype=torch.bool,
    )


def _multi_stft_band(
    data: torch.Tensor, sampling_rate: int, params: dict[str, Any]
) -> torch.Tensor:
    """Constant-Q log-octave filterbank view via the live ``_multi_stft_view``.

    Input ``(..., C, T)`` → output ``(..., C, n_bins, T_bin)``. n_bins, centers
    and routing are derived from {f0, cap, step}; the live filterbank function
    is then driven with the matching contiguous-octave parameters, so this view
    is byte-identical to the v14 MultiStftView for the same knobs.
    """
    f0 = float(params["f0_hz"])
    cap = float(params["cap_hz"])
    step = float(params["octave_step"])
    half_bw = float(params["half_bw_octaves"])

    n_bins = _multi_stft_n_bins(f0, cap, step)
    centers = multi_stft_bin_centers_hz(n_bins=n_bins, f0_hz=f0, octave_step=step)
    routing = _derive_multi_stft_routing(centers)

    # 1-cycle floor guard. With f0>=2 (the grid) every bin clears its tier floor,
    # so `keep` is all-True and nothing is dropped. The branch only fires for an
    # out-of-grid f0<2; dead low bins are a contiguous prefix, so dropping them
    # and lifting f0 to the lowest survivor preserves the octave geometry that
    # `_multi_stft_view` reconstructs internally.
    keep = _multi_stft_valid_mask(centers, routing)
    if not bool(keep.all()):
        centers = centers[keep]
        routing = tuple(r for r, k in zip(routing, keep.tolist()) if k)
        n_bins = int(keep.sum().item())
        if n_bins == 0:
            raise ValueError(
                "multi_stft floor guard dropped every bin — degenerate config "
                f"(f0={f0}, cap={cap} below the 2 Hz tier-0 floor). Out of the swept grid."
            )
        f0 = float(centers[0].item())

    flat = data.reshape(-1, data.shape[-1])
    spec = _multi_stft_view(
        flat,
        sample_rate=int(sampling_rate),
        hop_length=int(params["hop_length"]),
        nperseg_low=int(params["nperseg_low"]),
        nperseg_mid=int(params["nperseg_mid"]),
        nperseg_hi=int(params["nperseg_hi"]),
        n_bins=n_bins,
        f0_hz=f0,
        octave_step=step,
        half_bw_octaves=half_bw,
        routing=routing,
        log_eps=float(params["log_eps"]),
        apply_log=bool(params["apply_log"]),
    )
    return spec.reshape(*data.shape[:-1], spec.shape[-2], spec.shape[-1])


def _raw_multi_stft_bins(
    data: torch.Tensor, sampling_rate: int, params: dict[str, Any]
) -> torch.Tensor:
    """iMINDBench raw-|STFT|-bins control view (no filterbank).

    Concatenates the raw magnitude bins of each source STFT over its band:
    low (1024) 2–40 Hz, mid (512) 20–148, hi (256) 80–248. Input ``(..., C, T)``
    → output ``(..., C, n_raw_bins, T_bin)``. STFT params match the live grid
    bit-for-bit (Hann, center=True, normalized=False).
    """
    hop = int(params["hop_length"])
    bands = (
        (int(params["nperseg_low"]), params["raw_low_band"]),
        (int(params["nperseg_mid"]), params["raw_mid_band"]),
        (int(params["nperseg_hi"]), params["raw_hi_band"]),
    )
    flat = data.reshape(-1, data.shape[-1])
    mags: list[torch.Tensor] = []
    t_bins: list[int] = []
    for nperseg, (lo_hz, hi_hz) in bands:
        win = torch.hann_window(nperseg, device=flat.device)
        wf = flat
        if wf.shape[-1] < nperseg:
            wf = torch.nn.functional.pad(wf, (0, nperseg - wf.shape[-1]))
        spec = torch.stft(
            wf,
            n_fft=nperseg,
            hop_length=hop,
            win_length=nperseg,
            window=win,
            return_complex=True,
            normalized=False,
            center=True,
        )
        mag = torch.abs(spec)  # (N, F_stft, T)
        # BUG-VIEWS-1: build freqs/keep on the data's device so the boolean mask
        # indexes `mag` without a CPU↔GPU device mismatch (mirrors _multi_stft_view).
        freqs = torch.fft.rfftfreq(nperseg, d=1.0 / sampling_rate, device=flat.device)
        keep = (freqs >= float(lo_hz)) & (freqs <= float(hi_hz))
        mags.append(mag[..., keep, :])
        t_bins.append(int(mag.shape[-1]))
    t = min(t_bins)
    banded = torch.cat([m[..., :t] for m in mags], dim=-2)  # (N, F_total, T)
    return banded.reshape(*data.shape[:-1], banded.shape[-2], banded.shape[-1])


def multi_stft_dead_bin_count(sampling_rate: int, params: dict[str, Any]) -> int:
    """Count constant-Q output bins that receive zero rfft support (FE-MULTISTFT-1).

    At f0=2 Hz the 1024-nperseg STFT has 2 Hz resolution; a narrow constant-Q
    triangle can fall entirely between two rfft bins, so its L1-normalized weight
    column is all-zero (the builder leaves it at 0 when the triangle integral is
    0). Such bins are constant-zero features — they confound the resolution axis
    (cells 4/5 gain more of them as the step narrows). This reports the count per
    cell so the analyst can read the confound off the manifest; it does NOT change
    the filterbank (the guard is a Ben/FE-MULTISTFT-1 decision).
    """
    p = _resolve_multi_stft_params(params)
    f0 = float(p["f0_hz"]); cap = float(p["cap_hz"])
    step = float(p["octave_step"]); half_bw = float(p["half_bw_octaves"])

    n_bins = _multi_stft_n_bins(f0, cap, step)
    centers = multi_stft_bin_centers_hz(n_bins=n_bins, f0_hz=f0, octave_step=step)
    routing = _derive_multi_stft_routing(centers)
    keep = _multi_stft_valid_mask(centers, routing)  # mirror _multi_stft_band's floor guard
    if not bool(keep.all()):
        centers = centers[keep]
        routing = tuple(r for r, k in zip(routing, keep.tolist()) if k)
        n_bins = int(keep.sum().item())
        if n_bins == 0:
            raise ValueError(
                "multi_stft floor guard dropped every bin — degenerate config "
                f"(f0={f0}, cap={cap} below the 2 Hz tier-0 floor). Out of the swept grid."
            )
        f0 = float(centers[0].item())

    fbank = _build_multi_stft_filterbank(
        int(sampling_rate),
        int(p["nperseg_low"]), int(p["nperseg_mid"]), int(p["nperseg_hi"]),
        n_bins, f0, step, half_bw, routing,
    )
    dead = 0
    for weights, _idx in fbank.values():
        dead += int((weights.sum(dim=0) == 0).sum())
    return dead


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
