"""wav2vec2-large-xlsr-53 multi-layer feature extraction at per-word windows.

Mirror of ``whisper_ceiling.extract`` but for wav2vec2. The DCC-side pipeline
per (subject, trial):
    1. Load WAV (16 kHz mono).
    2. Load words_df + features.csv labels  -- reused from whisper_ceiling.
    3. Plan non-overlapping 30-s chunks on the same grid Whisper uses, so the
       kept word set is IDENTICAL to the Whisper NPZs (clean head-to-head).
    4. Forward each 30-s chunk through the wav2vec2 encoder ONCE with
       output_hidden_states=True; capture all 25 hidden states
       (1 feature-projection + 24 transformer layers), each (T, 1024).
    5. For each word whose [start, start+1 s] window falls inside the chunk,
       mean-pool the ~50 frames at 50 Hz to a single (1024,) vector per layer.
    6. Dump NPZ in the EXACT same key layout the probe expects:
          {layer_0: (W, 1024), ..., layer_24: (W, 1024),
           word_index, movie_start_s, layers, label_columns, label_<col>}

Layer convention: ``layer_i`` == ``hidden_states[i]`` == output of transformer
layer ``i`` (with layer_0 = feature-projection output, pre-transformer). This
matches Evanson's "layer 19" indexing directly.

wav2vec2 runs at 50 Hz (conv stride 320 @ 16 kHz) -- same frame rate as
Whisper-large-v3 -- so the reused frame-window math (teacher_hz=50) is correct.
A 30-s chunk yields ~1499 frames (not exactly 1500) because of the conv
receptive field, so frame_hi is clamped to the actual T; boundary words lose at
most one frame, and the kept word set stays identical to Whisper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from speech_decoding.whisper_ceiling.extract import (
    DEFAULT_TEACHER_HZ,
    DEFAULT_WINDOW_AFTER_S,
    DEFAULT_WINDOW_BEFORE_S,
    WHISPER_CHUNK_S,
    plan_chunks,
)

# wav2vec2 native rate: 16 kHz input, 50 Hz encoder frames (conv stride 320).
W2V_SR = 16_000
W2V_DEFAULT_MODEL = "facebook/wav2vec2-large-xlsr-53"


class Wav2Vec2MultiLayerExtractor:
    """Wraps a transformers Wav2Vec2Model, capturing every hidden state.

    forward_chunk(chunk) returns {layer_idx -> (T, d)} for layer_idx in
    range(num_hidden_states), where hidden_states[i] is captured as layer i.
    """

    def __init__(self, model, processor, layers: tuple[int, ...] | None = None):
        self.model = model
        self.processor = processor
        # num_hidden_states = 1 (feature projection) + num_hidden_layers.
        n_states = model.config.num_hidden_layers + 1
        self.layers = tuple(layers) if layers is not None else tuple(range(n_states))
        for L in self.layers:
            if L >= n_states:
                raise ValueError(
                    f"layer {L} >= {n_states} hidden states; "
                    f"{model.config._name_or_path} has "
                    f"{model.config.num_hidden_layers} transformer layers "
                    f"(hidden_states 0..{n_states - 1})"
                )

    def forward_chunk(self, wav_chunk: np.ndarray) -> dict:
        """Run one chunk through the encoder. Returns dict[layer -> (T, d)] fp16.

        Unlike Whisper, wav2vec2 takes raw waveform (no fixed 30-s mel pad); the
        HF processor applies the model's own zero-mean/unit-var normalization.
        """
        import torch  # type: ignore[import-not-found]

        self._captures: dict = {}
        inputs = self.processor(
            wav_chunk,
            sampling_rate=W2V_SR,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        input_values = inputs.input_values.to(device=device, dtype=dtype)
        attention_mask = getattr(inputs, "attention_mask", None)
        kw = {}
        if attention_mask is not None:
            kw["attention_mask"] = attention_mask.to(device=device)
        with torch.no_grad():
            out = self.model(input_values, output_hidden_states=True, **kw)
        hidden = out.hidden_states  # tuple len = num_hidden_layers + 1
        result = {}
        for L in self.layers:
            feat = hidden[L].squeeze(0).detach().cpu().to(torch.float16).numpy()
            result[L] = feat
        return result

    def close(self):
        # No forward hooks to remove (uses output_hidden_states); kept for API
        # parity with WhisperMultiLayerExtractor.
        pass


def extract_trial_features_w2v(
    extractor: Wav2Vec2MultiLayerExtractor,
    wav: np.ndarray,
    sample_rate: int,
    words_df: pd.DataFrame,
    chunk_seconds: float = WHISPER_CHUNK_S,
    before_s: float = DEFAULT_WINDOW_BEFORE_S,
    after_s: float = DEFAULT_WINDOW_AFTER_S,
    teacher_hz: int = DEFAULT_TEACHER_HZ,
    progress_cb=None,
) -> dict:
    """Extract per-word mean-pooled wav2vec2 features at every hidden state.

    Reuses ``plan_chunks`` (chunk_n_frames=1500, Whisper's grid) to decide which
    words are kept, then clamps each frame slice to the actual per-chunk frame
    count T. This guarantees the kept word_index set matches the Whisper NPZs.

    Returns dict with keys:
        layer_<L>: (W, d) fp16, one row per accepted word
        word_index: (W,) int64
        movie_start_s: (W,) float32
        layers: (n,) int32
    """
    if sample_rate != W2V_SR:
        raise ValueError(f"need {W2V_SR} Hz audio, got {sample_rate}")
    total_seconds = len(wav) / sample_rate
    chunk_starts, plan = plan_chunks(
        words_df=words_df,
        total_wav_seconds=total_seconds,
        chunk_seconds=chunk_seconds,
        before_s=before_s,
        after_s=after_s,
        teacher_hz=teacher_hz,
    )

    chunk_n_samples = int(chunk_seconds * sample_rate)
    rows_by_chunk: dict[int, list] = {}
    for r in plan:
        rows_by_chunk.setdefault(r.chunk_index, []).append(r)

    out_per_layer: dict[int, list[np.ndarray]] = {L: [] for L in extractor.layers}
    out_word_idx: list[int] = []
    out_start: list[float] = []

    for ci, cstart in enumerate(chunk_starts):
        s0 = int(round(cstart * sample_rate))
        s1 = s0 + chunk_n_samples
        if s1 > len(wav):
            break
        chunk = wav[s0:s1].astype(np.float32)
        feats = extractor.forward_chunk(chunk)
        any_layer = extractor.layers[0]
        T = feats[any_layer].shape[0]
        for r in rows_by_chunk.get(ci, []):
            lo = r.frame_lo
            hi = min(r.frame_hi, T)
            if hi <= lo:
                continue
            for L in extractor.layers:
                pooled = feats[L][lo:hi].mean(axis=0).astype(np.float16)
                out_per_layer[L].append(pooled)
            out_word_idx.append(r.word_index)
            out_start.append(r.movie_start_s)
        if progress_cb is not None:
            progress_cb(ci, len(chunk_starts))

    payload = {
        f"layer_{L}": (
            np.stack(out_per_layer[L], axis=0)
            if out_per_layer[L]
            else np.zeros((0, 1), dtype=np.float16)
        )
        for L in extractor.layers
    }
    payload["word_index"] = np.array(out_word_idx, dtype=np.int64)
    payload["movie_start_s"] = np.array(out_start, dtype=np.float32)
    payload["layers"] = np.array(extractor.layers, dtype=np.int32)
    return payload
