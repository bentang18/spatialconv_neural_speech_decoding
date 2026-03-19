"""Audio feature extraction and speech-mask construction.

This module supports the speech-embedding regression experiments by:
1. Loading per-trial phoneme timing metadata.
2. Extracting fixed audio windows around response onset.
3. Computing HuBERT or mel-spectrogram targets on a canonical 50-frame grid.
4. Building binary speech masks aligned to that same frame grid.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample, resample_poly, stft

from speech_decoding.data.bids_dataset import _find_fif_path

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_WINDOW = (-1.0, 1.5)
DEFAULT_N_FRAMES = 50
DEFAULT_FRAME_DUR = 0.05
DEFAULT_HUBERT_LAYER = 6


@dataclass(frozen=True)
class PhonemeTimingInfo:
    """Per-trial phoneme timing aligned to response onset."""

    trial: int
    response_onset: float
    response_offset: float
    phoneme_intervals: tuple[tuple[float, float], ...]
    phonemes: tuple[str, ...]
    syllable: str


def _phoneme_csv_path(bids_root: str | Path, subject: str) -> Path:
    bids_root = Path(bids_root)
    path = (
        bids_root
        / "derivatives"
        / "phoneme"
        / f"sub-{subject}"
        / "event"
        / f"sub-{subject}_task-phoneme_acq-01_run-01_desc-production_events.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Phoneme timing CSV not found: {path}")
    return path


def _audio_wav_path(
    bids_root: str | Path,
    subject: str,
    desc: str = "production",
) -> Path:
    bids_root = Path(bids_root)
    path = (
        bids_root
        / "derivatives"
        / "audio"
        / f"sub-{subject}"
        / "microphone"
        / f"sub-{subject}_task-phoneme_acq-01_run-01_desc-{desc}_microphone.wav"
    )
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path


def load_phoneme_timing(
    subject: str,
    bids_root: str | Path,
) -> list[PhonemeTimingInfo]:
    """Load per-trial phoneme timings from the production events CSV."""
    rows_by_trial: dict[int, list[dict[str, str]]] = {}
    with _phoneme_csv_path(bids_root, subject).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial = int(row["trial"])
            rows_by_trial.setdefault(trial, []).append(row)

    timing: list[PhonemeTimingInfo] = []
    for trial in sorted(rows_by_trial):
        rows = sorted(rows_by_trial[trial], key=lambda r: int(r["phoneme_idx"]))
        response_onset = float(rows[0]["response_onset"])
        response_offset = float(rows[0]["response_offset"])
        intervals = []
        phonemes = []
        for row in rows:
            onset = float(row["onset"]) - response_onset
            offset = float(row["offset"]) - response_onset
            intervals.append((onset, offset))
            phonemes.append(row["phoneme"])
        timing.append(
            PhonemeTimingInfo(
                trial=trial,
                response_onset=response_onset,
                response_offset=response_offset,
                phoneme_intervals=tuple(intervals),
                phonemes=tuple(phonemes),
                syllable=rows[0]["syllable"],
            )
        )
    return timing


def build_speech_mask(
    intervals: Iterable[tuple[float, float]],
    n_frames: int = DEFAULT_N_FRAMES,
    frame_dur: float = DEFAULT_FRAME_DUR,
    window_start: float = DEFAULT_AUDIO_WINDOW[0],
) -> np.ndarray:
    """Build a binary speech mask on the canonical frame grid."""
    mask = np.zeros(n_frames, dtype=np.float32)
    for start_s, end_s in intervals:
        if end_s <= start_s:
            continue
        start_frame = int(np.floor((start_s - window_start) / frame_dur))
        end_frame = int(np.ceil((end_s - window_start) / frame_dur))
        start_frame = max(start_frame, 0)
        end_frame = min(end_frame, n_frames)
        if end_frame > start_frame:
            mask[start_frame:end_frame] = 1.0
    return mask


def build_segment_masks(
    intervals: Iterable[tuple[float, float]],
    n_frames: int = DEFAULT_N_FRAMES,
    frame_dur: float = DEFAULT_FRAME_DUR,
    window_start: float = DEFAULT_AUDIO_WINDOW[0],
) -> np.ndarray:
    """Build one binary frame mask per interval on the canonical grid."""
    intervals = list(intervals)
    masks = np.zeros((len(intervals), n_frames), dtype=np.float32)
    for idx, (start_s, end_s) in enumerate(intervals):
        if end_s <= start_s:
            continue
        start_frame = int(np.floor((start_s - window_start) / frame_dur))
        end_frame = int(np.ceil((end_s - window_start) / frame_dur))
        start_frame = max(start_frame, 0)
        end_frame = min(end_frame, n_frames)
        if end_frame > start_frame:
            masks[idx, start_frame:end_frame] = 1.0
    return masks


def load_epoch_response_times(
    subject: str,
    bids_root: str | Path,
    task: str = "PhonemeSequence",
    n_phons: int = 3,
    desc: str = "productionZscore",
) -> np.ndarray:
    """Load response-onset times from the MNE epochs event sample numbers.

    The event samples are stored on the original 2 kHz acquisition timeline.
    """
    fif_path = _find_fif_path(Path(bids_root), subject, task, desc)
    epochs = mne.read_epochs(str(fif_path), preload=False, verbose=False)
    return epochs.events[0::n_phons, 0].astype(np.float64) / 2000.0


def load_audio(audio_path: str | Path) -> tuple[np.ndarray, int]:
    """Load a mono WAV file and return float32 samples in [-1, 1]."""
    sr, data = wavfile.read(str(audio_path))
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        scale = float(np.iinfo(data.dtype).max)
        audio = data.astype(np.float32) / max(scale, 1.0)
    else:
        audio = data.astype(np.float32)
    return audio, int(sr)


def load_patient_audio(
    subject: str,
    bids_root: str | Path,
    desc: str = "production",
) -> tuple[np.ndarray, int]:
    """Load one patient's production audio track."""
    return load_audio(_audio_wav_path(bids_root, subject, desc=desc))


def extract_audio_segment(
    audio: np.ndarray,
    sr: int,
    center_time: float,
    pre_s: float = 1.0,
    post_s: float = 1.5,
) -> np.ndarray:
    """Extract a fixed window around center_time, zero-padding at boundaries."""
    start_idx = int(round((center_time - pre_s) * sr))
    end_idx = int(round((center_time + post_s) * sr))
    target_len = int(round((pre_s + post_s) * sr))

    seg = np.zeros(target_len, dtype=np.float32)
    src_start = max(start_idx, 0)
    src_end = min(end_idx, len(audio))
    dst_start = max(-start_idx, 0)
    dst_end = dst_start + max(src_end - src_start, 0)
    if src_end > src_start:
        seg[dst_start:dst_end] = audio[src_start:src_end]
    return seg


def _ensure_16khz(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if sr == 16000:
        return audio.astype(np.float32), sr
    audio_16k = resample_poly(audio.astype(np.float32), 16000, sr).astype(np.float32)
    return audio_16k, 16000


@lru_cache(maxsize=1)
def _get_hubert(model_name: str = "facebook/hubert-base-ls960"):
    """Load HuBERT processor and model lazily."""
    try:
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required for HuBERT feature extraction."
        ) from exc

    logger.info("Loading HuBERT model: %s", model_name)
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def extract_hubert_embeddings(
    audio_segment: np.ndarray,
    sr: int,
    layer: int = DEFAULT_HUBERT_LAYER,
    device: str = "cpu",
    processor=None,
    model=None,
) -> np.ndarray:
    """Extract HuBERT hidden states for a single audio segment."""
    audio_16k, sr = _ensure_16khz(audio_segment, sr)
    if processor is None or model is None:
        processor, model = _get_hubert()
    model = model.to(device)
    inputs = processor(
        audio_16k,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    tensor_inputs = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        tensor_inputs[key] = value.to(device)
    with torch.no_grad():
        out = model(**tensor_inputs, output_hidden_states=True)
    hidden = out.hidden_states[layer][0].detach().cpu().numpy().astype(np.float32)
    return hidden


def resample_to_backbone_frames(
    features: np.ndarray,
    n_frames: int = DEFAULT_N_FRAMES,
) -> np.ndarray:
    """Resample framewise features to the backbone's canonical frame count."""
    if features.shape[0] == n_frames:
        return features.astype(np.float32, copy=False)
    return resample(features, n_frames, axis=0).astype(np.float32)


def _hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(freq_mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (freq_mel / 2595.0) - 1.0)


def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    fmax = fmax if fmax is not None else sr / 2.0
    mel_points = np.linspace(_hz_to_mel(np.array([fmin]))[0], _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


def extract_mel_spectrogram(
    audio_segment: np.ndarray,
    sr: int,
    n_mels: int = 40,
    n_frames: int = DEFAULT_N_FRAMES,
) -> np.ndarray:
    """Extract a log-mel spectrogram and resample it to the canonical frame count."""
    audio_16k, sr = _ensure_16khz(audio_segment, sr)
    n_fft = 512
    _, _, zxx = stft(
        audio_16k,
        fs=sr,
        nperseg=400,
        noverlap=240,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx) ** 2
    mel_fb = _mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel = mel_fb @ power
    mel = np.log10(np.maximum(mel, 1e-10)).T.astype(np.float32)
    return resample_to_backbone_frames(mel, n_frames=n_frames)
