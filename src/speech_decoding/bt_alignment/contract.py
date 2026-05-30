"""Whisper-large-v3 distillation contract + clock-consistency assertion.

Provenance: v14 B05+B06 lock (5/25 PM) + v3 upgrade 2026-05-28 per memo
``project_v14_whisper_teacher_v3_upgrade_2026_05_28.md``. Whisper variant
verified by inspecting transformers WhisperConfig for openai/whisper-large-v3:
    n_mels=128, d_model=1280, encoder_layers=32, hop=160 samples @ 16 kHz.
Encoder topology is identical to large-v2 (32 layers, d=1280, layer-8 native
50 Hz); v3 differs only in the mel front-end (80→128 bins) which lives
UPSTREAM of the encoder. Cache writer + whisper_adapter consume layer-8
hidden state (B, T, 1280), so the mel-bin change is invisible to the
distillation contract; only the source model string and n_mels need to
flip. Goldstein-2025 L8 semantics still apply (v3 was a drop-in noisy-speech
robustness upgrade, −10–20% WER on noisy benchmarks per Whisper v3
release notes).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

WHISPER_CONTRACT = {
    "variant": "openai/whisper-large-v3",
    "n_mels": 128,
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
POOL_BASE_MS = 250.0  # triangle BASE (full w>0 support), not the half-max FWHM (=125 ms)
NEURAL_SAMPLE_RATE_HZ = 2048


def triangular_pool_kernel(
    base_ms: float = POOL_BASE_MS, teacher_rate_hz: int = TEACHER_RATE_HZ
) -> np.ndarray:
    """Parameterless sum-to-1 triangular kernel for teacher-side pool, zero-padded edges.

    Reconciled to the canonical live kernel A in
    ``extractors.whisper_teacher_pool``: ``base_ms`` is the triangle BASE (full
    w>0 support), so the half-base = ``base_ms/1000 * teacher_rate_hz / 2`` acts
    as the linear-decay half-width = bucket stride. At 50→8 Hz this gives a
    250 ms base (half-base 6.25 frames = stride; true half-max FWHM 125 ms), ~13
    nonzero samples, length 15 — matching kernel A, not the prior 2×-wide form.
    Triangular shape rises linearly to a centered apex.
    """
    half_base_samples = base_ms / 1000.0 * teacher_rate_hz / 2.0
    half = int(np.ceil(half_base_samples))
    x = np.arange(-half, half + 1)
    raw = np.maximum(0.0, 1.0 - np.abs(x) / half_base_samples)
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
