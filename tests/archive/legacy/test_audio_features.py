"""Tests for audio feature extraction and speech masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from speech_decoding.data.audio_features import (
    build_segment_masks,
    build_speech_mask,
    extract_audio_segment,
    extract_hubert_embeddings,
    load_phoneme_timing,
    resample_to_backbone_frames,
)


class TestSpeechMask:
    def test_mask_shape_and_dtype(self):
        mask = build_speech_mask([(0.0, 0.5)], n_frames=50, frame_dur=0.05, window_start=-1.0)
        assert mask.shape == (50,)
        assert mask.dtype == np.float32

    def test_mask_marks_speech_only(self):
        mask = build_speech_mask([(0.0, 0.5)], n_frames=50, frame_dur=0.05, window_start=-1.0)
        assert mask[:20].sum() == 0.0
        assert mask[20:30].sum() == 10.0
        assert mask[30:].sum() == 0.0

    def test_empty_mask(self):
        mask = build_speech_mask([], n_frames=50, frame_dur=0.05, window_start=-1.0)
        assert mask.sum() == 0.0

    def test_segment_masks(self):
        masks = build_segment_masks([(0.0, 0.2), (0.2, 0.4)], n_frames=50, frame_dur=0.05, window_start=-1.0)
        assert masks.shape == (2, 50)
        assert masks[0, 20:24].sum() == 4.0
        assert masks[1, 24:28].sum() == 4.0


class TestAudioSegmentation:
    def test_extract_audio_segment_with_padding(self):
        audio = np.arange(10, dtype=np.float32)
        segment = extract_audio_segment(audio, sr=2, center_time=0.5, pre_s=1.0, post_s=1.5)
        assert segment.shape == (5,)
        assert np.allclose(segment[:1], 0.0)

    def test_resample_to_backbone_frames(self):
        x = np.random.randn(125, 8).astype(np.float32)
        y = resample_to_backbone_frames(x, n_frames=50)
        assert y.shape == (50, 8)
        assert y.dtype == np.float32


class TestPhonemeTiming:
    @pytest.mark.slow
    def test_load_real_phoneme_timing_s14(self):
        bids_root = Path(
            "BIDS_1.0_Phoneme_Sequence_uECoG/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS"
        )
        if not bids_root.exists():
            pytest.skip("PS BIDS data not available")
        timing = load_phoneme_timing("S14", bids_root)
        assert len(timing) > 0
        assert len(timing[0].phoneme_intervals) == 3


class _DummyProcessor:
    def __call__(self, audio, sampling_rate, return_tensors, padding):
        x = np.asarray(audio, dtype=np.float32)
        return {"input_values": np.expand_dims(x, 0)}


class _DummyModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_values, output_hidden_states=True):
        assert output_hidden_states
        batch = input_values.shape[0]
        hidden = np.ones((batch, 125, 768), dtype=np.float32)
        tensors = [None] * 13
        tensors[6] = __import__("torch").from_numpy(hidden)
        return type("DummyOutput", (), {"hidden_states": tensors})()


class TestHuBERTExtraction:
    def test_extract_hubert_embeddings_with_mock_model(self):
        audio = np.random.randn(16000 * 2).astype(np.float32)
        emb = extract_hubert_embeddings(
            audio,
            sr=16000,
            processor=_DummyProcessor(),
            model=_DummyModel(),
        )
        assert emb.shape == (125, 768)
        assert emb.dtype == np.float32
