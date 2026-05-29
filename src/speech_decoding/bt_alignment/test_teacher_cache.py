"""Tests for Whisper teacher-feature cache writer.

Uses whisper-tiny (4 layers, 384-d) for the CPU smoke. The same interface drives
the DCC dispatch with whisper-large-v3 (32 layers, 1280-d) at L8.
"""
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from speech_decoding.bt_alignment.teacher_cache import (
    WhisperFeatureExtractor, write_clip_cache,
    DEFAULT_TEACHER_LAYER, DEFAULT_TEACHER_HZ, WHISPER_SR,
)


def test_default_constants():
    assert DEFAULT_TEACHER_LAYER == 8
    assert DEFAULT_TEACHER_HZ == 50
    assert WHISPER_SR == 16000


@pytest.fixture(scope="module")
def whisper_tiny():
    """Whisper-tiny: 4 encoder layers, 384-d. Layer index 3 is the deepest."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model.eval()
    return model, proc


def test_extractor_raises_if_layer_too_deep(whisper_tiny):
    model, proc = whisper_tiny
    with pytest.raises(ValueError, match="encoder depth"):
        WhisperFeatureExtractor(model, proc, layer=8)


def test_extract_returns_correct_shape_and_dtype(whisper_tiny):
    model, proc = whisper_tiny
    fe = WhisperFeatureExtractor(model, proc, layer=3)
    sr = WHISPER_SR
    wav = np.zeros(int(sr * 30.0), dtype=np.float32)
    try:
        feat = fe.extract(wav, sample_rate=sr)
        # Whisper-tiny encoder produces (T=1500, 384) at 50 Hz for 30 s input
        assert feat.ndim == 2
        assert feat.shape[0] == 1500  # 30 s × 50 Hz
        assert feat.shape[1] == 384   # whisper-tiny hidden dim
        assert feat.dtype == torch.float16
    finally:
        fe.close()


def test_extract_rejects_wrong_sample_rate(whisper_tiny):
    model, proc = whisper_tiny
    fe = WhisperFeatureExtractor(model, proc, layer=3)
    try:
        with pytest.raises(ValueError, match="16000"):
            fe.extract(np.zeros(1000, dtype=np.float32), sample_rate=22050)
    finally:
        fe.close()


def test_write_clip_cache_round_trip(whisper_tiny):
    model, proc = whisper_tiny
    fe = WhisperFeatureExtractor(model, proc, layer=3)
    sr = WHISPER_SR
    wav = (np.random.RandomState(0).randn(int(sr * 5.0)) * 0.1).astype(np.float32)
    try:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            entry = write_clip_cache(
                fe, wav, sample_rate=sr,
                clip_id="anchor_0042", film="megamind",
                t0_movie_s=42.5, out_dir=out_dir,
            )
            assert entry.n_frames > 0
            assert entry.d_model == 384
            saved_path = Path(entry.out_path)
            assert saved_path.exists()
            payload = torch.load(saved_path, weights_only=False)
            assert payload["rate_hz"] == 50
            assert payload["t0_movie_s"] == 42.5
            assert payload["layer"] == 3
            assert payload["features"].shape == (entry.n_frames, 384)
    finally:
        fe.close()
