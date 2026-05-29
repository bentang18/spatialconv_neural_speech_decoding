"""Whisper-large-v3 multi-layer feature extraction at per-word windows.

The DCC-side pipeline per (subject, trial):
    1. Load WAV (16 kHz mono).
    2. Load words_df + features.csv → per-word (start, end, labels) DataFrame.
    3. Split WAV into 30-s chunks aligned to a 30-s grid.
    4. Forward each chunk through Whisper encoder ONCE; hook layers
       {4, 8, 12, 16, 20, 24, 28, 31} (0-indexed) and capture (T=1500, 1280).
    5. For each word whose [start, start+1s] window falls inside the chunk,
       mean-pool the 50 frames at 50 Hz to a single (1280,) vector per layer.
    6. Dump NPZ:
          {layer_4: (W, 1280), ..., layer_31: (W, 1280),
           starts: (W,), word_index: (W,), labels: dict[task->(W,)]}

Why per-chunk batch instead of per-word: Whisper-large-v3 forces 30-s input
through its mel pad, so per-word forwards waste 90% of the encoder pass.
Per-chunk processes ~25 words at once for the same wall time.

Pure mechanics: no model loading at import. The DCC entry-point script
constructs the WhisperMultiLayerExtractor with whisper-large-v3; tests use
whisper-tiny.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Whisper-large-v3 native rate: 50 Hz encoder, 30-s chunks → 1500 frames each
WHISPER_SR = 16_000
WHISPER_CHUNK_S = 30.0
DEFAULT_TEACHER_HZ = 50

# Neuroprobe brain window: [0, +1 s] post word onset
DEFAULT_WINDOW_BEFORE_S = 0.0
DEFAULT_WINDOW_AFTER_S = 1.0

# Layers to hook. 0-indexed; whisper-large-v3 has 32 encoder layers.
# Default = ALL 32 layers, so the downstream probe can sweep the per-layer +
# merge-strategy axes without re-running encoder forward. Per-trial cache size
# scales linearly (~20 GB total at 32 layers; ~5 GB at 8). Encoder forward
# wall-time is unchanged regardless of hook count.
DEFAULT_LAYERS = tuple(range(32))


def word_window_to_frames(
    word_start_s: float,
    chunk_start_s: float,
    teacher_hz: int = DEFAULT_TEACHER_HZ,
    before_s: float = DEFAULT_WINDOW_BEFORE_S,
    after_s: float = DEFAULT_WINDOW_AFTER_S,
    chunk_n_frames: int = 1500,
) -> tuple[int, int] | None:
    """Map word_start_s in absolute movie time to frame range inside its chunk.

    Returns (frame_lo, frame_hi) in the chunk's local frame coords, or None if
    the [word_start - before, word_start + after] window doesn't fully fit in
    the chunk.
    """
    win_lo_s = word_start_s - before_s - chunk_start_s
    win_hi_s = word_start_s + after_s - chunk_start_s
    frame_lo = int(round(win_lo_s * teacher_hz))
    frame_hi = int(round(win_hi_s * teacher_hz))
    if frame_lo < 0 or frame_hi > chunk_n_frames:
        return None
    if frame_hi <= frame_lo:
        return None
    return frame_lo, frame_hi


@dataclass
class WordFeatureRow:
    word_index: int                # row index in words_df
    movie_start_s: float
    chunk_index: int
    frame_lo: int
    frame_hi: int


class WhisperMultiLayerExtractor:
    """Wraps a transformers Whisper model with multi-layer forward hooks."""

    def __init__(self, model, processor, layers: tuple[int, ...] = DEFAULT_LAYERS):
        self.model = model
        self.processor = processor
        self.layers = tuple(layers)
        encoder_layers = model.model.encoder.layers
        for L in self.layers:
            if L >= len(encoder_layers):
                raise ValueError(
                    f"layer {L} >= encoder depth {len(encoder_layers)}; "
                    f"whisper-large-v3 has 32 encoder layers (0..31)"
                )
        self._captures: dict = {}
        self._handles = []
        for L in self.layers:
            handle = encoder_layers[L].register_forward_hook(self._make_hook(L))
            self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            self._captures[layer_idx] = (
                output[0] if isinstance(output, tuple) else output
            )
        return hook

    def forward_chunk(self, wav_chunk_30s: np.ndarray) -> dict:
        """Run one 30-s chunk through the encoder. Returns dict[layer -> (T, d)]."""
        import torch  # type: ignore[import-not-found]
        expected = int(WHISPER_SR * WHISPER_CHUNK_S)
        if wav_chunk_30s.shape[0] != expected:
            raise ValueError(
                f"chunk must be {expected} samples "
                f"(30 s @ {WHISPER_SR} Hz); got {wav_chunk_30s.shape[0]}"
            )
        self._captures.clear()
        inputs = self.processor(
            wav_chunk_30s, sampling_rate=WHISPER_SR, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        with torch.no_grad():
            self.model.model.encoder(inputs.input_features.to(device))
        out = {}
        for L in self.layers:
            tensor = self._captures[L]
            feat = tensor.squeeze(0).detach().cpu().to(torch.float16).numpy()
            out[L] = feat
        return out

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def plan_chunks(
    words_df: pd.DataFrame,
    total_wav_seconds: float,
    chunk_seconds: float = WHISPER_CHUNK_S,
    chunk_n_frames: int = 1500,
    before_s: float = DEFAULT_WINDOW_BEFORE_S,
    after_s: float = DEFAULT_WINDOW_AFTER_S,
    teacher_hz: int = DEFAULT_TEACHER_HZ,
) -> tuple[list[float], list[WordFeatureRow]]:
    """Plan which 30-s chunks to forward, and which words live in which chunk.

    Strategy: greedy, non-overlapping chunks starting at the floor-30 of the
    earliest unprocessed word. Drops any word whose [start-before, start+after]
    window straddles a chunk boundary (loss is <0.1% in practice given the
    chunk_s/window_s ratio).
    """
    rows: list[WordFeatureRow] = []
    chunk_starts: list[float] = []

    word_idx = 0
    n_words = len(words_df)
    word_starts = words_df["start"].to_numpy(dtype=float)

    while word_idx < n_words:
        wstart = word_starts[word_idx]
        if wstart - before_s < 0:
            word_idx += 1
            continue
        chunk_start = max(0.0, wstart - before_s)
        if chunk_start + chunk_seconds > total_wav_seconds:
            break
        ci = len(chunk_starts)
        chunk_starts.append(chunk_start)
        chunk_max = chunk_start + chunk_seconds
        while word_idx < n_words:
            ws = word_starts[word_idx]
            if ws + after_s > chunk_max:
                break
            window = word_window_to_frames(
                word_start_s=ws,
                chunk_start_s=chunk_start,
                teacher_hz=teacher_hz,
                before_s=before_s,
                after_s=after_s,
                chunk_n_frames=chunk_n_frames,
            )
            if window is not None:
                wi_value = words_df.index[word_idx]
                rows.append(
                    WordFeatureRow(
                        word_index=int(wi_value),  # type: ignore[arg-type]
                        movie_start_s=float(ws),
                        chunk_index=ci,
                        frame_lo=window[0],
                        frame_hi=window[1],
                    )
                )
            word_idx += 1

    return chunk_starts, rows


def load_trial_labels(
    subject_id: int,
    trial_id: int,
    movie_name: str,
    words_df_dir: Path,
    transcripts_root: Path,
    pitch_volume_dir: Path | None = None,
) -> pd.DataFrame:
    """Load and merge per-word labels for one (subject, trial).

    Returns a DataFrame with words_df.csv columns + all transcripts features.csv
    columns merged on original_index. If pitch_volume_dir is provided, the
    enhanced pitch/volume features are also merged keyed by 5-decimal-rounded
    start time.
    """
    words_path = words_df_dir / f"subject{subject_id}_trial{trial_id}_words_df.csv"
    words_df = pd.read_csv(words_path)

    features_path = transcripts_root / movie_name / "features.csv"
    features_df = pd.read_csv(features_path).set_index("Unnamed: 0")
    new_cols = [c for c in features_df.columns if c not in words_df.columns]
    for col in new_cols:
        col_series = features_df[col]  # type: ignore[assignment]
        words_df[col] = words_df["original_index"].map(col_series)  # type: ignore[arg-type]

    if pitch_volume_dir is not None:
        pv_path = pitch_volume_dir / f"{movie_name}_pitch_volume_features.json"
        if pv_path.exists():
            with open(pv_path) as fh:
                raw = json.load(fh)
            normalized = {f"{float(k):.5f}": v for k, v in raw.items()}
            pv_keys = next(iter(normalized.values())).keys() if normalized else []
            for k in pv_keys:
                if k in words_df.columns:
                    continue
                vals = []
                for s in words_df["start"]:
                    key = f"{float(s):.5f}"
                    vals.append(
                        normalized[key][k] if key in normalized else np.nan
                    )
                words_df[k] = vals
    return words_df


def extract_trial_features(
    extractor: WhisperMultiLayerExtractor,
    wav: np.ndarray,
    sample_rate: int,
    words_df: pd.DataFrame,
    chunk_seconds: float = WHISPER_CHUNK_S,
    before_s: float = DEFAULT_WINDOW_BEFORE_S,
    after_s: float = DEFAULT_WINDOW_AFTER_S,
    teacher_hz: int = DEFAULT_TEACHER_HZ,
    progress_cb=None,
) -> dict:
    """Extract per-word mean-pooled features at every hooked layer.

    Returns dict with keys:
        layer_<L>: (W, d_model) fp16 array, one row per accepted word
        word_index: (W,) int array
        movie_start_s: (W,) float32 array
    """
    if sample_rate != WHISPER_SR:
        raise ValueError(f"need {WHISPER_SR} Hz audio, got {sample_rate}")
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
    rows_by_chunk: dict[int, list[WordFeatureRow]] = {}
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
        for r in rows_by_chunk.get(ci, []):
            for L in extractor.layers:
                pooled = feats[L][r.frame_lo:r.frame_hi].mean(axis=0).astype(np.float16)
                out_per_layer[L].append(pooled)
            out_word_idx.append(r.word_index)
            out_start.append(r.movie_start_s)
        if progress_cb is not None:
            progress_cb(ci, len(chunk_starts))

    payload = {
        f"layer_{L}": np.stack(out_per_layer[L], axis=0) if out_per_layer[L] else np.zeros((0, 1), dtype=np.float16)
        for L in extractor.layers
    }
    payload["word_index"] = np.array(out_word_idx, dtype=np.int64)
    payload["movie_start_s"] = np.array(out_start, dtype=np.float32)
    payload["layers"] = np.array(extractor.layers, dtype=np.int32)
    return payload
