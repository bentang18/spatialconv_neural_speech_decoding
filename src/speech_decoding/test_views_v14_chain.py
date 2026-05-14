"""End-to-end smoke test for the v14 preprocessing recipe `R2 × I2L × F1`.

Validates that the chain ``apply_temporal_filter_inplace`` (F1 60Hz notch)
→ window slice → ``preprocess_views(shaft_car, log_stft)`` (R2 + I2L)
composes cleanly with realistic sEEG-shaped synthetic input. Shape, dtype,
finiteness, and the load-bearing physics invariants (notch attenuates 60Hz,
shaftCAR zeros per-shaft mean, log-STFT preserves frequency-bin layout).

BT-scale runs live on DCC; this is the laptop-side composition guard.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from speech_decoding.views import (
    apply_temporal_filter_inplace,
    make_upstream_helpers,
    preprocess_views,
)


SR = 2048
T_SEC = 1.0
SESSION_SEC = 8.0


def _stft_params() -> dict[str, Any]:
    return {
        "stft": {
            "nperseg": 512,
            "poverlap": 0.75,
            "window": "hann",
            "max_frequency": 150,
            "min_frequency": 0,
        }
    }


def _upstream_helpers() -> dict[str, Any]:
    """Load upstream STFT + Laplacian helpers without requiring BT data on disk.

    Upstream `eval_utils` imports `neuroprobe.config`, which reads
    `ROOT_DIR_BRAINTREEBANK` at import time. The helper functions themselves
    never touch BT data — only the config module gates the import. Stub the
    env var to a path that doesn't have to exist.
    """
    import os
    import sys

    os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp/_bt_root_stub_for_tests")

    examples_dir = (
        "/Users/bentang/Documents/Code/speech/.cache/neuroprobe_upstream/examples"
    )
    pytest.importorskip("neuroprobe", reason="upstream Neuroprobe package missing")
    if not os.path.isdir(examples_dir):
        pytest.skip(f"upstream examples dir missing: {examples_dir}")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    from eval_utils import (  # type: ignore[import-not-found]  # noqa: PLC0415
        laplacian_rereference_neural_data,
        preprocess_stft,
    )

    return make_upstream_helpers(
        preprocess_stft=preprocess_stft,
        laplacian_rereference_neural_data=laplacian_rereference_neural_data,
        stft_params=_stft_params(),
    )


def _synthetic_session(
    n_shafts: int = 4, contacts_per_shaft: int = 6, seconds: float = SESSION_SEC
) -> tuple[torch.Tensor, list[str]]:
    """Two-shaft sEEG-shaped session with 1/f noise + 60Hz line + per-shaft drift."""
    rng = np.random.default_rng(0)
    n_channels = n_shafts * contacts_per_shaft
    n_samples = int(seconds * SR)
    t = np.arange(n_samples) / SR

    pink = rng.standard_normal((n_channels, n_samples)).astype(np.float32) * 30.0
    line_60 = 12.0 * np.sin(2 * np.pi * 60.0 * t).astype(np.float32)
    drift = np.zeros((n_channels, n_samples), dtype=np.float32)
    labels: list[str] = []
    for shaft_idx in range(n_shafts):
        shaft_name = f"OF{chr(ord('a') + shaft_idx)}"
        drift_amplitude = 5.0 * (shaft_idx + 1)
        for contact_idx in range(1, contacts_per_shaft + 1):
            ch = shaft_idx * contacts_per_shaft + (contact_idx - 1)
            drift[ch] = drift_amplitude * np.sin(2 * np.pi * 0.7 * t).astype(np.float32)
            labels.append(f"{shaft_name}{contact_idx}")

    voltage = pink + line_60[None, :] + drift
    return torch.from_numpy(voltage.astype(np.float32)), labels


def _band_power(arr: np.ndarray, sr: int, lo: float, hi: float) -> float:
    fft = np.fft.rfft(arr, axis=-1)
    freqs = np.fft.rfftfreq(arr.shape[-1], d=1.0 / sr)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.mean(np.abs(fft[..., mask]) ** 2))


def test_v14_chain_shape_dtype_finite() -> None:
    voltage, labels = _synthetic_session()

    apply_temporal_filter_inplace(
        voltage, sampling_rate=SR, notch_freqs=(60.0, 120.0, 180.0), hpf_hz=0.0
    )
    assert voltage.dtype == torch.float32
    assert torch.isfinite(voltage).all()

    window = voltage[:, : int(T_SEC * SR)].unsqueeze(0)
    feats, post_labels = preprocess_views(
        window,
        labels,
        ref_kind="shaft_car",
        view_kind="log_stft",
        sampling_rate=SR,
        upstream_helpers=_upstream_helpers(),
    )

    assert feats.dtype == torch.float32
    assert feats.ndim == 4, f"expected (B, C, T_bins, F_bins), got {tuple(feats.shape)}"
    assert feats.shape[0] == 1
    assert feats.shape[1] == voltage.shape[0]
    assert feats.shape[1] == len(post_labels)
    assert torch.isfinite(feats).all(), "log-STFT produced NaN/inf"


def test_v14_chain_notch_attenuates_60hz() -> None:
    voltage, _ = _synthetic_session()
    before = voltage.clone()

    apply_temporal_filter_inplace(
        voltage, sampling_rate=SR, notch_freqs=(60.0, 120.0, 180.0), hpf_hz=0.0
    )

    p_before = _band_power(before.numpy(), SR, 59.0, 61.0)
    p_after = _band_power(voltage.numpy(), SR, 59.0, 61.0)
    assert p_after < 0.05 * p_before, (
        f"60Hz notch did not attenuate (before={p_before:.2e}, after={p_after:.2e})"
    )


def test_v14_chain_shaft_car_zeros_per_shaft_mean() -> None:
    voltage, labels = _synthetic_session()
    apply_temporal_filter_inplace(
        voltage, sampling_rate=SR, notch_freqs=(60.0, 120.0, 180.0), hpf_hz=0.0
    )
    window = voltage[:, : int(T_SEC * SR)].unsqueeze(0)

    from speech_decoding.views import apply_reference  # noqa: PLC0415

    ref_out, _ = apply_reference(
        window, labels, "shaft_car", upstream_helpers=_upstream_helpers()
    )

    shafts: dict[str, list[int]] = {}
    for idx, name in enumerate(labels):
        shaft = name.rstrip("0123456789")
        shafts.setdefault(shaft, []).append(idx)
    for shaft, idxs in shafts.items():
        shaft_mean = ref_out[0, idxs, :].mean(dim=0)
        np.testing.assert_allclose(
            shaft_mean.numpy(), 0.0, atol=1e-5,
            err_msg=f"shaft {shaft!r} mean not zeroed",
        )
