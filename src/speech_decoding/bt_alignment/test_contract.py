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
    assert c["l8_native_rate_hz"] == 50


def test_pool_factor():
    assert TEACHER_RATE_HZ == 50
    assert STUDENT_RATE_HZ == 8
    assert POOL_FACTOR == TEACHER_RATE_HZ / STUDENT_RATE_HZ == 6.25


def test_triangular_kernel_fwhm_250ms_sum_to_one():
    k = triangular_pool_kernel(fwhm_ms=250.0, teacher_rate_hz=50)
    assert k.ndim == 1
    assert abs(k.sum() - 1.0) < 1e-9
    assert k[k.shape[0] // 2] == k.max()
    import math
    fwhm_samples = 250.0 / 1000.0 * 50
    expected_len = 2 * math.ceil(fwhm_samples) + 1
    assert k.shape[0] == expected_len


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
