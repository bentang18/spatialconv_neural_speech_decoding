"""Whisper teacher-ceiling probe (L.7 prerun preflight).

For each Neuroprobe Lite task: extract Whisper-large-v3 encoder features at
multiple layers for each word window, then probe with LogReg on the same
binary-quartile contrast Neuroprobe uses for its brain-side gates.

Validates that the v14 P3 distillation target (Whisper-L8) actually carries the
task signal: if a frozen LogReg on Whisper-L8 features can't beat iEEG SOTA,
distillation cannot help. The deeper layers + concat sister probe also picks the
best teacher layer.
"""
from speech_decoding.whisper_ceiling.extract import (
    DEFAULT_LAYERS,
    DEFAULT_WINDOW_BEFORE_S,
    DEFAULT_WINDOW_AFTER_S,
    DEFAULT_TEACHER_HZ,
    WHISPER_CHUNK_S,
    WHISPER_SR,
    WhisperMultiLayerExtractor,
    extract_trial_features,
    word_window_to_frames,
)
from speech_decoding.whisper_ceiling.probe import (
    DEFAULT_TASKS,
    binary_labels_from_continuous,
    fit_probe_within_trial,
    fit_probe_cross_session,
    fit_probe_cross_subject,
    late_fusion_within,
    topk_layers_by_within_auroc,
)

__all__ = [
    "DEFAULT_LAYERS",
    "DEFAULT_WINDOW_BEFORE_S",
    "DEFAULT_WINDOW_AFTER_S",
    "DEFAULT_TEACHER_HZ",
    "WHISPER_CHUNK_S",
    "WHISPER_SR",
    "WhisperMultiLayerExtractor",
    "extract_trial_features",
    "word_window_to_frames",
    "DEFAULT_TASKS",
    "binary_labels_from_continuous",
    "fit_probe_within_trial",
    "fit_probe_cross_session",
    "fit_probe_cross_subject",
    "late_fusion_within",
    "topk_layers_by_within_auroc",
]
