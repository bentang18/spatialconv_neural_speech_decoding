"""Whisper-L8 teacher-feature cache writer (v14 P3 distillation).

For each (film, clip-window) anchor pair, extract the encoder's intermediate
representation at layer 8 of the Whisper-large-v3 encoder (B06 lock; v2→v3
upgrade 2026-05-28 per memo
``project_v14_whisper_teacher_v3_upgrade_2026_05_28.md``). Same encoder
topology as v2 (32 layers, d=1280, native 50 Hz); v3 differs only in mel
front-end (80→128 bins) and produces −10–20% WER on noisy speech. The
layer-8 features are 1280-d at 50 Hz native (1 step per 20 ms WAV @ 16
kHz / 160-hop / 2× downsample inside the encoder).

Cache schema (per clip):
    out_dir / <film> / <clip_id>.pt   ->  dict({
        "features": Tensor of shape (T, 1280) float16,
        "rate_hz": 50,
        "t0_movie_s": float,  # absolute movie clock when feature[0] applies
        "model": "openai/whisper-large-v3",
        "layer": 8,
    })

Implementation notes:
- Uses a forward hook on encoder.layers[L] to capture the output of that block.
- For unit-testing on a CPU laptop, swap whisper-large-v3 (1280-d) for whisper-tiny
  (384-d, 4 layers). The cache writer is the same; only the model and the layer
  index differ.
- This module DOES NOT load a model at import; the caller passes in a loaded
  Whisper. The DCC dispatch script wires that up.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

DEFAULT_TEACHER_LAYER = 8
DEFAULT_TEACHER_HZ = 50
WHISPER_SR = 16000
WHISPER_HOP = 160  # 20 ms in mel-input space


@dataclass
class TeacherCacheEntry:
    clip_id: str
    film: str
    t0_movie_s: float
    rate_hz: int
    n_frames: int
    d_model: int
    out_path: str


class WhisperFeatureExtractor:
    """Wraps a transformers Whisper model and a forward hook to grab encoder
    layer K's output. Designed for batch-of-1 inference (one clip at a time)."""

    def __init__(self, model, processor, layer: int = DEFAULT_TEACHER_LAYER):
        self.model = model
        self.processor = processor
        self.layer = layer
        self._hook_output = None
        encoder_layers = model.model.encoder.layers
        if layer >= len(encoder_layers):
            raise ValueError(
                f"layer {layer} >= encoder depth {len(encoder_layers)} "
                f"(use whisper-large-v3 for L8)"
            )
        # transformer EncoderLayer.forward returns a tuple; output[0] is hidden_states
        self._handle = encoder_layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        # output is (hidden_states,) or a single tensor
        self._hook_output = output[0] if isinstance(output, tuple) else output

    def extract(self, wav: np.ndarray, sample_rate: int = WHISPER_SR):
        if sample_rate != WHISPER_SR:
            raise ValueError(f"Whisper expects {WHISPER_SR} Hz, got {sample_rate}")
        import torch
        inputs = self.processor(
            wav, sampling_rate=sample_rate, return_tensors="pt"
        )
        with torch.no_grad():
            self.model.model.encoder(inputs.input_features)
        feat = self._hook_output
        # feat shape: (1, T, d_model)
        if feat is None:
            raise RuntimeError("Hook captured no output")
        return feat.squeeze(0).cpu().to(torch.float16)

    def close(self):
        self._handle.remove()


def write_clip_cache(
    feature_extractor: WhisperFeatureExtractor,
    wav: np.ndarray,
    sample_rate: int,
    clip_id: str,
    film: str,
    t0_movie_s: float,
    out_dir: Path,
    rate_hz: int = DEFAULT_TEACHER_HZ,
) -> TeacherCacheEntry:
    """Extract L8 features for one clip and save to out_dir/<film>/<clip_id>.pt."""
    import torch
    feat = feature_extractor.extract(wav, sample_rate)
    film_dir = out_dir / film
    film_dir.mkdir(parents=True, exist_ok=True)
    out_path = film_dir / f"{clip_id}.pt"
    torch.save({
        "features": feat,
        "rate_hz": rate_hz,
        "t0_movie_s": float(t0_movie_s),
        "model": "openai/whisper-large-v3",
        "layer": feature_extractor.layer,
    }, out_path)
    return TeacherCacheEntry(
        clip_id=clip_id, film=film, t0_movie_s=float(t0_movie_s),
        rate_hz=rate_hz, n_frames=int(feat.shape[0]), d_model=int(feat.shape[1]),
        out_path=str(out_path),
    )
