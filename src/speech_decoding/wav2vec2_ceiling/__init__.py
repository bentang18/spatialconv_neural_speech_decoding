"""wav2vec2 teacher-ceiling probe — head-to-head against Whisper.

Same teacher-ceiling question as ``whisper_ceiling`` (can a frozen LogReg on
the audio-FM features beat iEEG SOTA on Neuroprobe Lite tasks?), but with
wav2vec2-large-xlsr-53 as the audio teacher instead of Whisper-large-v3.

Evanson 2025 ("From Minutes to Days", arXiv 2512.15830) aligns brain to
wav2vec2-large-xlsr-53 layer 19 — so this measures whether *their* teacher
carries more decodable task signal than ours.

The probe itself (``whisper_ceiling.probe``) is model-agnostic and reused
verbatim: it sweeps whatever ``layer_<L>`` keys the NPZ carries. Only the
forward+pool is wav2vec2-specific. Label-loading and 30-s chunk planning are
reused from ``whisper_ceiling.extract`` so the kept word set + labels are
*identical* to the Whisper NPZs — a clean same-words head-to-head.
"""
from speech_decoding.wav2vec2_ceiling.extract import (
    W2V_SR,
    W2V_DEFAULT_MODEL,
    Wav2Vec2MultiLayerExtractor,
    extract_trial_features_w2v,
)

__all__ = [
    "W2V_SR",
    "W2V_DEFAULT_MODEL",
    "Wav2Vec2MultiLayerExtractor",
    "extract_trial_features_w2v",
]
