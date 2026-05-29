"""Whisper-large-v2 distillation contract + clock-consistency assertion.

Provenance: v14 B05+B06 lock (5/25 PM). Whisper variant verified by inspecting
transformers WhisperConfig for openai/whisper-large-v2:
    n_mels=80, d_model=1280, encoder_layers=32, hop=160 samples @ 16 kHz.
Large-v3 has n_mels=128 → incompatible with v14 src whisper_adapter.py (in_dim=1280
+ n_mels=80 assumption). Goldstein-2025 L8 semantics also match large-v2.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

WHISPER_CONTRACT = {
    "variant": "openai/whisper-large-v2",
    "n_mels": 80,
    "sr": 16000,
    "hop": 160,
    "n_fft": 400,
    "d_model": 1280,
    "encoder_layers": 32,
    "l8_layer_index": 8,
    "l8_native_rate_hz": 50,
}

TEACHER_RATE_HZ = 50
STUDENT_RATE_HZ = 8
POOL_FACTOR = TEACHER_RATE_HZ / STUDENT_RATE_HZ
POOL_FWHM_MS = 250.0
NEURAL_SAMPLE_RATE_HZ = 2048


def triangular_pool_kernel(
    fwhm_ms: float = POOL_FWHM_MS, teacher_rate_hz: int = TEACHER_RATE_HZ
) -> np.ndarray:
    """Parameterless sum-to-1 triangular kernel for teacher-side pool, zero-padded edges.

    FWHM_samples = fwhm_ms / 1000 * teacher_rate_hz. Kernel half-width = FWHM (so total
    length = 2*FWHM + 1 samples). Triangular shape rises linearly to a centered apex.
    """
    fwhm_samples = fwhm_ms / 1000.0 * teacher_rate_hz
    half = int(np.ceil(fwhm_samples))
    x = np.arange(-half, half + 1)
    raw = np.maximum(0.0, 1.0 - np.abs(x) / fwhm_samples)
    return raw / raw.sum()


def assert_clock_consistency(
    timings_df: pd.DataFrame,
    words_df: pd.DataFrame,
    neural_n_samples: int,
    sample_rate: int = NEURAL_SAMPLE_RATE_HZ,
) -> None:
    """One-line sanity assertion: subject_timings.index, words_df.est_idx, neural sample
    stream all on the same sample rate + same epoch.

    Raises AssertionError on mismatch.
    """
    assert sample_rate == NEURAL_SAMPLE_RATE_HZ, (
        f"sample_rate {sample_rate} != contract {NEURAL_SAMPLE_RATE_HZ}"
    )
    max_trigger_idx = int(timings_df["index"].max())
    assert max_trigger_idx < neural_n_samples, (
        f"timings.index max {max_trigger_idx} >= neural_n_samples {neural_n_samples}"
    )
    max_word_idx = int(words_df["est_idx"].max())
    assert max_word_idx < neural_n_samples, (
        f"words_df.est_idx max {max_word_idx} >= neural_n_samples {neural_n_samples}"
    )
