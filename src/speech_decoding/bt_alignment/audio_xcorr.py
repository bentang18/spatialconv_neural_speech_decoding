"""Per-trial audio cross-correlation + drift-slope estimate.

Compares the rip's RMS envelope to BT's per-word RMS feature at anchor times. A
zero-lag peak with a flat residual implies no drift; a sloped residual implies
the BT clock and rip clock tick at different rates.

Why this matters for L.7:
- We get Whisper L8 features by running Whisper on the rip's audio.
- BT gives us `subject_timings.csv` mapping neural-sample-idx → movie_time (s).
- The student-teacher pairing assumes movie_time on BT's side = movie_time on
  the rip's side. A drift of even 100 ms over 2 h would mean L8 features pulled
  for late-movie clips correspond to dialog ~1 word off from the BT word window.

Anchor strategy:
- Pick high_real-triggered words spread evenly across the trial's movie-clock.
- Compute RMS of the rip at [t - 0.25, t + 0.25] (0.5 s window).
- BT exposes `rms` per word in features.csv (scalar).
- Linear regression of (rip_rms_log - bt_rms_log) vs movie_time gives the slope;
  Pearson r at zero lag gives the correlation.

Implementation note:
- `compute_rms_envelope_at_anchors` operates on a pre-extracted WAV; it does NOT
  invoke ffmpeg internally. The dispatch script extracts WAV once per rip.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_ANCHOR_WINDOW_S = 0.5
DEFAULT_N_ANCHORS = 50
DRIFT_GATE_MS_PER_MIN = 50.0  # screening; real check is forced-aligner (#20)
MIN_CORRELATION = 0.5


@dataclass
class XCorrResult:
    film: str
    n_anchors: int
    pearson_r: float
    drift_slope_ms_per_min: float
    intercept_s: float
    residual_std_s: float
    pass_correlation: bool
    pass_drift: bool

    def to_dict(self) -> dict:
        return asdict(self)


def compute_rms_envelope_at_anchors(
    wav: np.ndarray,
    sample_rate: int,
    anchor_times_s: np.ndarray,
    window_s: float = DEFAULT_ANCHOR_WINDOW_S,
) -> np.ndarray:
    """RMS of `wav` in [t - window/2, t + window/2] for each anchor in anchor_times_s.

    Returns NaN for anchors that fall outside the wav.
    """
    half = window_s / 2.0
    out = np.full(len(anchor_times_s), np.nan, dtype=np.float64)
    n = len(wav)
    for i, t in enumerate(anchor_times_s):
        lo = int(max(0, round((t - half) * sample_rate)))
        hi = int(min(n, round((t + half) * sample_rate)))
        if hi - lo < int(0.1 * sample_rate):  # too short
            continue
        seg = wav[lo:hi].astype(np.float64)
        out[i] = float(np.sqrt(np.mean(seg * seg)))
    return out


def pick_anchors(words_df: pd.DataFrame, n: int = DEFAULT_N_ANCHORS,
                 time_col: str = "start") -> pd.DataFrame:
    """Pick n evenly-spaced anchor words across the movie clock.

    Filters to words with a numeric rms column and finite start time.
    """
    valid = words_df[
        words_df["rms"].notna() & words_df[time_col].notna()
    ].sort_values(time_col).reset_index(drop=True)
    if len(valid) == 0:
        return valid
    step = max(1, len(valid) // n)
    sampled = valid.iloc[::step].iloc[:n].reset_index(drop=True)
    return sampled


def estimate_drift(
    anchor_times_s: np.ndarray,
    rip_rms: np.ndarray,
    bt_rms: np.ndarray,
    film: str = "",
    gate_ms_per_min: float = DRIFT_GATE_MS_PER_MIN,
    min_correlation: float = MIN_CORRELATION,
) -> XCorrResult:
    """Fit log(rip_rms) = a * t + b * log(bt_rms) + c, report drift in time units.

    A simpler proxy: regress (log_rip - log_bt) on t. The slope is unitless per-second;
    we convert it to ms/min as the rate at which residual time-drift accumulates.
    """
    mask = (
        np.isfinite(anchor_times_s)
        & np.isfinite(rip_rms)
        & np.isfinite(bt_rms)
        & (rip_rms > 0)
        & (bt_rms > 0)
    )
    t = anchor_times_s[mask]
    a = np.log(rip_rms[mask])
    b = np.log(bt_rms[mask])

    n = int(mask.sum())
    if n < 5:
        return XCorrResult(
            film=film, n_anchors=n, pearson_r=float("nan"),
            drift_slope_ms_per_min=float("nan"), intercept_s=float("nan"),
            residual_std_s=float("nan"), pass_correlation=False, pass_drift=False,
        )

    r = float(np.corrcoef(a, b)[0, 1])
    diff = a - b
    A = np.vstack([t, np.ones_like(t)]).T
    slope_per_s, intercept = np.linalg.lstsq(A, diff, rcond=None)[0]
    residual_std = float(np.std(diff - (slope_per_s * t + intercept)))

    drift_ms_per_min = float(slope_per_s) * 60.0 * 1000.0

    return XCorrResult(
        film=film,
        n_anchors=n,
        pearson_r=r,
        drift_slope_ms_per_min=drift_ms_per_min,
        intercept_s=float(intercept),
        residual_std_s=residual_std,
        pass_correlation=r >= min_correlation,
        pass_drift=abs(drift_ms_per_min) < gate_ms_per_min,
    )


def load_features_rms(features_csv: Path) -> pd.DataFrame:
    """Load (start, end, rms) columns from a BT features.csv. Memory-cheap."""
    return pd.read_csv(features_csv, usecols=["start", "end", "rms"])
