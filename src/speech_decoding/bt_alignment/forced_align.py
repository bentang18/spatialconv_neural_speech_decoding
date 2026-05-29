"""Forced-alignment drift check.

Picks N anchor words from BT features.csv at high_real triggers, extracts a window
of WAV around each, runs Whisper with word_timestamps=True, and compares Whisper's
estimated word start to BT's claimed start. The drift distribution tells us whether
BT's per-word timestamps are usable as L.7 distillation anchors.

Implementation choice: openai/whisper-tiny via transformers (39M params, CPU-fast).
This is for *gate testing only*; the actual L8 teacher cache (#25) uses whisper-large-v3.

Pass criterion: median |drift| < DRIFT_GATE_MS over the anchor batch.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import re
import numpy as np
import pandas as pd

DRIFT_GATE_MS = 200.0
DEFAULT_WINDOW_S = 6.0
DEFAULT_N_ANCHORS = 20


@dataclass
class AnchorAlignment:
    bt_text: str
    bt_start_s: float
    whisper_text: str | None
    whisper_start_s: float | None
    drift_ms: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ForcedAlignResult:
    film: str
    n_anchors: int
    n_aligned: int
    median_drift_ms: float
    mean_drift_ms: float
    p95_drift_ms: float
    pass_gate: bool
    anchors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


_NON_WORD = re.compile(r"[^a-z]+")


def normalize_word(s: str) -> str:
    return _NON_WORD.sub("", s.lower().strip())


def pick_anchor_words(
    features_df: pd.DataFrame, n: int = DEFAULT_N_ANCHORS,
) -> pd.DataFrame:
    """Pick n high-quality anchor words: text only (no numbers/symbols), spaced
    across the movie clock, with a minimum word-length to reduce Whisper miss.
    """
    df = features_df[features_df["text"].notna() & features_df["start"].notna()].copy()
    df["_norm"] = df["text"].astype(str).map(normalize_word)
    df = df[df["_norm"].str.len() >= 3]  # min 3 chars
    df = df.drop_duplicates("_norm")
    if len(df) == 0:
        return df.head(0)
    df = df.sort_values("start").reset_index(drop=True)
    step = max(1, len(df) // n)
    return df.iloc[::step].iloc[:n].reset_index(drop=True)


def extract_window(
    wav: np.ndarray, sample_rate: int, t_s: float, window_s: float
) -> tuple[np.ndarray, float]:
    """Window of `window_s` centered on t. Returns (wav_window, t_offset_in_window)."""
    half = window_s / 2.0
    lo = max(0.0, t_s - half)
    hi = min(len(wav) / sample_rate, t_s + half)
    lo_i = int(round(lo * sample_rate))
    hi_i = int(round(hi * sample_rate))
    return wav[lo_i:hi_i], t_s - lo


def find_word_in_chunks(chunks: list, target_norm: str) -> dict | None:
    """Scan Whisper chunks for a matching word; chunks: [{'text': str, 'timestamp': (s, e)}]."""
    for c in chunks:
        if c.get("text") is None:
            continue
        n = normalize_word(c["text"])
        if n == target_norm or (len(target_norm) >= 4 and target_norm in n):
            return c
    return None


def align_anchors(
    anchors: pd.DataFrame,
    wav: np.ndarray,
    sample_rate: int,
    pipeline_fn,
    window_s: float = DEFAULT_WINDOW_S,
) -> list[AnchorAlignment]:
    """Run a Whisper word-timestamp pipeline (callable) on each anchor window.

    pipeline_fn signature: (np.ndarray, sample_rate) -> {"chunks": [...]}.
    Designed so unit tests can pass a fake pipeline.
    """
    out: list[AnchorAlignment] = []
    for _, row in anchors.iterrows():
        bt_text = str(row["text"])
        bt_start = float(row["start"])
        target_norm = normalize_word(bt_text)
        window_wav, offset_in_window = extract_window(wav, sample_rate, bt_start, window_s)
        if len(window_wav) < int(0.5 * sample_rate):
            out.append(AnchorAlignment(bt_text, bt_start, None, None, None))
            continue
        result = pipeline_fn(window_wav, sample_rate)
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        match = find_word_in_chunks(chunks, target_norm)
        if match is None or match.get("timestamp") is None or match["timestamp"][0] is None:
            out.append(AnchorAlignment(bt_text, bt_start, None, None, None))
            continue
        whisper_t_in_window = float(match["timestamp"][0])
        whisper_t_global = bt_start - offset_in_window + whisper_t_in_window
        drift_ms = (whisper_t_global - bt_start) * 1000.0
        out.append(AnchorAlignment(bt_text, bt_start, match.get("text"), whisper_t_global, drift_ms))
    return out


def summarize(film: str, alignments: list[AnchorAlignment],
              gate_ms: float = DRIFT_GATE_MS) -> ForcedAlignResult:
    drifts = np.array([a.drift_ms for a in alignments if a.drift_ms is not None])
    n_aligned = len(drifts)
    if n_aligned == 0:
        return ForcedAlignResult(film, len(alignments), 0,
                                 float("nan"), float("nan"), float("nan"),
                                 False, [a.to_dict() for a in alignments])
    abs_d = np.abs(drifts)
    return ForcedAlignResult(
        film=film,
        n_anchors=len(alignments),
        n_aligned=n_aligned,
        median_drift_ms=float(np.median(abs_d)),
        mean_drift_ms=float(abs_d.mean()),
        p95_drift_ms=float(np.percentile(abs_d, 95)),
        pass_gate=bool(np.median(abs_d) < gate_ms),
        anchors=[a.to_dict() for a in alignments],
    )
