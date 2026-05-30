"""Tests for forced-aligner drift check.

Uses a synthetic pipeline_fn so the test suite does not require Whisper to be loaded
into memory. A separate dispatch script does the real run on local CPU.
"""
import numpy as np
import pandas as pd

from speech_decoding.bt_alignment.forced_align import (
    align_anchors, summarize, pick_anchor_words,
    normalize_word, find_word_in_chunks, extract_window,
    DRIFT_GATE_MS, DEFAULT_WINDOW_S,
)


def test_normalize_word_strips_punct_lowers():
    assert normalize_word("Hello,") == "hello"
    assert normalize_word("  WORLD!  ") == "world"
    assert normalize_word("Don't") == "dont"


def test_pick_anchor_words_filters_short_words():
    df = pd.DataFrame({
        "text": ["a", "to", "hello", "world", "the", "foundation"],
        "start": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    picks = pick_anchor_words(df, n=10)
    assert all(len(normalize_word(t)) >= 3 for t in picks["text"])


def test_pick_anchor_words_dedupes_normalized():
    df = pd.DataFrame({
        "text": ["Hello", "hello!", "world", "World."],
        "start": [1.0, 2.0, 3.0, 4.0],
    })
    picks = pick_anchor_words(df, n=10)
    norms = set(picks["text"].map(normalize_word))
    assert len(norms) == 2


def test_extract_window_center_and_offset():
    sr = 16000
    wav = np.arange(sr * 10, dtype=np.float32)
    w, off = extract_window(wav, sr, 5.0, 4.0)
    # center 5s, half window 2s -> 3 to 7s -> 64000 samples
    assert len(w) == 4 * sr
    assert off == 2.0  # 5.0 - 3.0


def test_extract_window_clamps_at_start():
    sr = 16000
    wav = np.arange(sr * 10, dtype=np.float32)
    w, off = extract_window(wav, sr, 0.5, 4.0)
    # request -1.5 to 2.5 -> clamped to 0..2.5
    assert off == 0.5


def test_find_word_in_chunks_exact_match():
    chunks = [
        {"text": " hello", "timestamp": (0.4, 0.7)},
        {"text": " world", "timestamp": (0.7, 1.0)},
    ]
    r = find_word_in_chunks(chunks, "world")
    assert r is not None
    assert r["timestamp"] == (0.7, 1.0)


def test_find_word_in_chunks_no_match_returns_none():
    chunks = [{"text": " hello", "timestamp": (0.0, 0.5)}]
    assert find_word_in_chunks(chunks, "world") is None


def test_align_anchors_with_perfect_pipeline_zero_drift():
    """A pipeline that always returns the target word at the center of the window
    should produce ~0 ms drift."""
    sr = 16000
    wav = np.zeros(sr * 60, dtype=np.float32)
    anchors = pd.DataFrame({
        "text": ["hello", "world", "foobar"],
        "start": [10.0, 20.0, 30.0],
    })
    window_s = 4.0
    half = window_s / 2.0

    def perfect_pipeline(w, sample_rate):
        # Always emit the target word at center of window
        return {"chunks": [
            {"text": " hello", "timestamp": (half, half + 0.4)},
            {"text": " world", "timestamp": (half, half + 0.4)},
            {"text": " foobar", "timestamp": (half, half + 0.4)},
        ]}

    out = align_anchors(anchors, wav, sr, perfect_pipeline, window_s=window_s)
    assert len(out) == 3
    for a in out:
        assert a.drift_ms is not None
        assert abs(a.drift_ms) < 1.0


def test_align_anchors_skewed_pipeline_caught():
    """A pipeline that consistently emits the word 100 ms late should report ~+100 ms drift."""
    sr = 16000
    wav = np.zeros(sr * 60, dtype=np.float32)
    anchors = pd.DataFrame({"text": ["hello"], "start": [10.0]})
    window_s = 4.0
    half = window_s / 2.0

    def late_pipeline(w, sample_rate):
        return {"chunks": [{"text": " hello", "timestamp": (half + 0.1, half + 0.4)}]}

    out = align_anchors(anchors, wav, sr, late_pipeline, window_s=window_s)
    assert out[0].drift_ms is not None
    assert 95 < out[0].drift_ms < 105


def test_summarize_pass_under_gate():
    anchors = [
        type("A", (), {
            "bt_text": "x", "bt_start_s": 1.0, "whisper_text": "x",
            "whisper_start_s": 1.05, "drift_ms": 50.0,
            "to_dict": lambda self: {},
        })() for _ in range(10)
    ]
    r = summarize("megamind", anchors)
    assert r.pass_gate is True
    assert r.median_drift_ms == 50.0


def test_summarize_fails_when_drift_exceeds_gate():
    anchors = [
        type("A", (), {
            "bt_text": "x", "bt_start_s": 1.0, "whisper_text": "x",
            "whisper_start_s": 1.5, "drift_ms": 500.0,
            "to_dict": lambda self: {},
        })() for _ in range(10)
    ]
    r = summarize("megamind", anchors)
    assert r.pass_gate is False


def test_drift_gate_constant():
    assert DRIFT_GATE_MS == 200.0


def test_default_window_constant():
    assert DEFAULT_WINDOW_S == 6.0
