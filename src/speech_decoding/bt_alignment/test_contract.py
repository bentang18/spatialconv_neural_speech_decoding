"""Tests for Whisper-large-v3 distillation contract (v3 upgrade 2026-05-28)."""
from speech_decoding.bt_alignment.contract import (
    WHISPER_CONTRACT,
    STUDENT_RATE_HZ,
    TEACHER_RATE_HZ,
    POOL_FACTOR,
    triangular_pool_kernel,
    assert_clock_consistency,
)


def test_whisper_contract_large_v3():
    c = WHISPER_CONTRACT
    assert c["variant"] == "openai/whisper-large-v3"
    assert c["n_mels"] == 128
    assert c["sr"] == 16000
    assert c["hop"] == 160
    assert c["n_fft"] == 400
    assert c["d_model"] == 1280
    assert c["encoder_layers"] == 32
    assert c["native_rate_hz"] == 50
    # Teacher target = plain mean over all 32 encoder layers (all-layer-mean
    # supersedes the single-layer-8 default; ceiling probe 2026-05-29).
    assert c["layer_merge"] == "mean_all"
    assert c["merged_layers"] == 32


def test_whisper_contract_project_up_b33():
    """B33 (2026-05-30): student projects 256→1280, no teacher-side adapter,
    mandatory per-channel z-score, MLP student head. Channel-stats scope flipped
    train-only → FULL-CORPUS 2026-06-04 (project_v14_b33_channel_stats_full_corpus
    — transductive norm of a frozen-model target, not a leak; R-train-only-stats
    is the purist sister)."""
    c = WHISPER_CONTRACT
    assert c["project_direction"] == "up"
    assert c["target_standardization"] == "per_channel_zscore_full_corpus"
    assert c["teacher_down_adapter"] is False
    assert c["student_head"] == "mlp_256_1280"


def test_pool_factor():
    assert TEACHER_RATE_HZ == 50
    assert STUDENT_RATE_HZ == 8
    assert POOL_FACTOR == TEACHER_RATE_HZ / STUDENT_RATE_HZ == 6.25


def test_triangular_kernel_base_250ms_sum_to_one():
    # Triangle base 250 ms @ 50 Hz: half-base = stride = 6.25 frames
    # (true half-max FWHM 125 ms). Reconciled to live kernel A.
    k = triangular_pool_kernel(base_ms=250.0, teacher_rate_hz=50)
    assert k.ndim == 1
    assert abs(k.sum() - 1.0) < 1e-9
    assert k[k.shape[0] // 2] == k.max()
    import math
    half_base_samples = 250.0 / 1000.0 * 50 / 2.0  # 6.25 = stride
    expected_len = 2 * math.ceil(half_base_samples) + 1
    assert k.shape[0] == expected_len
    # ~13 nonzero samples (base 250 ms with half-base = stride).
    assert 11 <= int((k > 0).sum()) <= 14


def test_clock_consistency_pass():
    import pandas as pd
    timings = pd.DataFrame({"index": [0, 100, 200], "movie_time": [0.0, 0.049, 0.097]})
    words = pd.DataFrame({"est_idx": [50, 150, 250]})
    assert_clock_consistency(timings, words, neural_n_samples=1000, sample_rate=2048)


def test_clock_consistency_rate_mismatch_raises():
    import pandas as pd, pytest
    timings = pd.DataFrame({"index": [0, 100, 200], "movie_time": [0.0, 0.049, 0.097]})
    words = pd.DataFrame({"est_idx": [50, 150, 250]})
    with pytest.raises(AssertionError):
        assert_clock_consistency(timings, words, neural_n_samples=1000, sample_rate=1024)


def test_clock_consistency_est_idx_overflow_raises():
    import pandas as pd, pytest
    timings = pd.DataFrame({"index": [0, 100, 200], "movie_time": [0.0, 0.049, 0.097]})
    words = pd.DataFrame({"est_idx": [50, 150, 2000]})
    with pytest.raises(AssertionError):
        assert_clock_consistency(timings, words, neural_n_samples=1000, sample_rate=2048)
