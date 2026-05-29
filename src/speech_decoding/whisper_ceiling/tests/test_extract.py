"""Tests for whisper_ceiling.extract."""
import numpy as np
import pandas as pd
import pytest

from speech_decoding.whisper_ceiling.extract import (
    DEFAULT_LAYERS,
    DEFAULT_TEACHER_HZ,
    WHISPER_CHUNK_S,
    WHISPER_SR,
    WhisperMultiLayerExtractor,
    extract_trial_features,
    plan_chunks,
    word_window_to_frames,
)


def test_default_constants():
    assert WHISPER_SR == 16_000
    assert WHISPER_CHUNK_S == 30.0
    assert DEFAULT_TEACHER_HZ == 50
    assert 8 in DEFAULT_LAYERS  # the v14 distillation target layer must be in the panel


def test_word_window_to_frames_centered():
    # Word at t=5s in absolute movie time, chunk starts at t=0
    # window = [5.0, 6.0] post onset (Neuroprobe default)
    # frames = [250, 300]
    out = word_window_to_frames(
        word_start_s=5.0, chunk_start_s=0.0,
        teacher_hz=50, before_s=0.0, after_s=1.0, chunk_n_frames=1500,
    )
    assert out == (250, 300)


def test_word_window_rejects_out_of_chunk_word():
    out = word_window_to_frames(
        word_start_s=29.7, chunk_start_s=0.0,
        teacher_hz=50, before_s=0.0, after_s=1.0, chunk_n_frames=1500,
    )
    assert out is None  # window ends at 30.7s, past chunk end


def test_plan_chunks_drops_pre_chunk_window_words():
    """A word at t=0.1 with before_s=0.5 has window starting at -0.4 -> drop."""
    words_df = pd.DataFrame({"start": [0.1, 5.0, 7.0]})
    chunks, plan = plan_chunks(
        words_df=words_df, total_wav_seconds=60.0,
        chunk_seconds=30.0, before_s=0.5, after_s=1.0, teacher_hz=50,
    )
    # Word 0 dropped (before_s pushes start of window negative).
    # Word 1 (5.0s) and Word 2 (7.0s) both kept.
    word_indices = [r.word_index for r in plan]
    assert 0 not in word_indices
    assert {1, 2} <= set(word_indices)


def test_plan_chunks_progresses_through_long_movie():
    """Synthetic 90-s movie with 1 word per second -> ~3 chunks."""
    words_df = pd.DataFrame({"start": list(range(2, 89))})  # 87 words
    chunks, plan = plan_chunks(
        words_df=words_df, total_wav_seconds=90.0,
        chunk_seconds=30.0, before_s=0.0, after_s=1.0, teacher_hz=50,
    )
    assert len(chunks) >= 2
    # Every word in the plan must map into a valid chunk.
    for r in plan:
        cs = chunks[r.chunk_index]
        assert r.movie_start_s >= cs
        assert r.movie_start_s + 1.0 <= cs + 30.0


# ---- Whisper-tiny smoke (uses real model) --------------------------------

@pytest.fixture(scope="module")
def whisper_tiny():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model.eval()
    return model, proc


def test_multi_layer_extractor_rejects_deep_layer(whisper_tiny):
    model, proc = whisper_tiny
    with pytest.raises(ValueError, match="encoder depth"):
        WhisperMultiLayerExtractor(model, proc, layers=(0, 4))


def test_multi_layer_extractor_shapes(whisper_tiny):
    model, proc = whisper_tiny
    fe = WhisperMultiLayerExtractor(model, proc, layers=(1, 3))
    try:
        wav = np.zeros(int(WHISPER_SR * WHISPER_CHUNK_S), dtype=np.float32)
        out = fe.forward_chunk(wav)
        assert set(out.keys()) == {1, 3}
        assert out[1].shape == (1500, 384)
        assert out[3].shape == (1500, 384)
        assert out[1].dtype == np.float16
    finally:
        fe.close()


def test_forward_chunk_rejects_wrong_length(whisper_tiny):
    model, proc = whisper_tiny
    fe = WhisperMultiLayerExtractor(model, proc, layers=(1,))
    try:
        with pytest.raises(ValueError, match="30 s"):
            fe.forward_chunk(np.zeros(1234, dtype=np.float32))
    finally:
        fe.close()


def test_extract_trial_features_returns_aligned_arrays(whisper_tiny):
    """End-to-end: 60-s synthetic audio + 3 word anchors -> 3 (1, 384) rows per layer."""
    model, proc = whisper_tiny
    fe = WhisperMultiLayerExtractor(model, proc, layers=(1, 3))
    try:
        rng = np.random.default_rng(0)
        wav = (rng.standard_normal(int(WHISPER_SR * 60)).astype(np.float32) * 0.05)
        words_df = pd.DataFrame({"start": [5.0, 10.0, 20.0]})
        payload = extract_trial_features(
            extractor=fe, wav=wav, sample_rate=WHISPER_SR,
            words_df=words_df, before_s=0.0, after_s=1.0,
        )
        for L in (1, 3):
            assert payload[f"layer_{L}"].shape == (3, 384)
            assert payload[f"layer_{L}"].dtype == np.float16
        assert payload["word_index"].tolist() == [0, 1, 2]
        assert payload["movie_start_s"].tolist() == [5.0, 10.0, 20.0]
        assert payload["layers"].tolist() == [1, 3]
    finally:
        fe.close()
