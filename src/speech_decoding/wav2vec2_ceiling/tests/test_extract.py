"""Tests for wav2vec2_ceiling.extract.

The real-model smoke uses a tiny wav2vec2 built from config with the SAME conv
stride stack as xlsr-53 (cumulative stride 320 @ 16 kHz -> 50 Hz), so no network
download is needed and the 50-Hz frame math is exercised end-to-end.
"""
import numpy as np
import pandas as pd
import pytest

from speech_decoding.wav2vec2_ceiling.extract import (
    W2V_DEFAULT_MODEL,
    W2V_SR,
    Wav2Vec2MultiLayerExtractor,
    extract_trial_features_w2v,
)


def test_default_constants():
    assert W2V_SR == 16_000
    assert W2V_DEFAULT_MODEL == "facebook/wav2vec2-large-xlsr-53"


@pytest.fixture(scope="module")
def w2v_tiny():
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

    cfg = Wav2Vec2Config(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
        # xlsr-53's conv stack: cumulative stride 5*2*2*2*2*2*2 = 320 -> 50 Hz
        conv_dim=(32,) * 7,
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        feat_extract_norm="layer",
        do_stable_layer_norm=True,
    )
    model = Wav2Vec2Model(cfg).eval()
    proc = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, do_normalize=True,
        return_attention_mask=True,
    )
    return model, proc


def test_captures_all_hidden_states(w2v_tiny):
    model, proc = w2v_tiny
    ext = Wav2Vec2MultiLayerExtractor(model, proc)
    # num_hidden_layers + 1 = 5 hidden states (layer_0 = feature projection).
    assert ext.layers == (0, 1, 2, 3, 4)


def test_multi_layer_extractor_rejects_deep_layer(w2v_tiny):
    model, proc = w2v_tiny
    with pytest.raises(ValueError, match="hidden states"):
        Wav2Vec2MultiLayerExtractor(model, proc, layers=(0, 5))


def test_forward_chunk_is_50hz(w2v_tiny):
    model, proc = w2v_tiny
    ext = Wav2Vec2MultiLayerExtractor(model, proc, layers=(0, 2))
    chunk = np.zeros(W2V_SR * 30, dtype=np.float32)
    out = ext.forward_chunk(chunk)
    assert set(out.keys()) == {0, 2}
    T = out[0].shape[0]
    # 30 s @ 50 Hz ~= 1500 frames; conv receptive field shaves a few -> ~1499.
    assert 1495 <= T <= 1500
    assert out[0].shape[1] == 32
    assert out[0].dtype == np.float16


def test_extract_trial_features_aligned(w2v_tiny):
    """End-to-end: 90-s synthetic audio + word anchors -> aligned per-layer rows."""
    model, proc = w2v_tiny
    ext = Wav2Vec2MultiLayerExtractor(model, proc, layers=(0, 3))
    rng = np.random.default_rng(0)
    wav = (rng.standard_normal(W2V_SR * 90).astype(np.float32) * 0.05)
    words_df = pd.DataFrame({"start": [5.0, 10.0, 35.0, 62.0]})
    payload = extract_trial_features_w2v(
        extractor=ext, wav=wav, sample_rate=W2V_SR,
        words_df=words_df, before_s=0.0, after_s=1.0,
    )
    n = len(payload["word_index"])
    assert n >= 3  # all in-chunk words kept
    for L in (0, 3):
        assert payload[f"layer_{L}"].shape == (n, 32)
        assert payload[f"layer_{L}"].dtype == np.float16
    assert payload["layers"].tolist() == [0, 3]
    # movie_start_s aligns row-for-row with kept word_index
    assert len(payload["movie_start_s"]) == n


def test_boundary_word_clamps_to_actual_T(w2v_tiny):
    """A word whose nominal frame_hi=1500 must still be kept (clamped to T~1499),
    preserving word-set parity with the Whisper NPZs."""
    model, proc = w2v_tiny
    ext = Wav2Vec2MultiLayerExtractor(model, proc, layers=(0,))
    rng = np.random.default_rng(1)
    wav = (rng.standard_normal(W2V_SR * 60).astype(np.float32) * 0.05)
    # Word at 28.99 s: window [28.99, 29.99] -> frames [1449, 1500] in a 1500-grid;
    # wav2vec2 chunk has ~1499 frames so frame_hi clamps to T but the word stays.
    words_df = pd.DataFrame({"start": [5.0, 28.99]})
    payload = extract_trial_features_w2v(
        extractor=ext, wav=wav, sample_rate=W2V_SR,
        words_df=words_df, before_s=0.0, after_s=1.0,
    )
    assert 1 in payload["word_index"].tolist()  # boundary word not dropped


def test_rejects_wrong_sample_rate(w2v_tiny):
    model, proc = w2v_tiny
    ext = Wav2Vec2MultiLayerExtractor(model, proc, layers=(0,))
    with pytest.raises(ValueError, match="16000 Hz"):
        extract_trial_features_w2v(
            extractor=ext, wav=np.zeros(8000, dtype=np.float32),
            sample_rate=8000, words_df=pd.DataFrame({"start": [1.0]}),
        )
