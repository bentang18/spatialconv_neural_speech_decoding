"""Per-trial pause + trigger-quality audit for BT subject_timings.

Runs BEFORE forced-aligner step (task #20) so the aligner consumes per-segment windows
that don't span pause boundaries. BT's estimate_sample_index linearly interpolates
across pauses → words near a pause edge get est_idx off by up to ~30 min wall-clock
(verified on sub_2 trial006: 13 pauses, max sample-jump 3,656,096 samples).

Drop policy:
- Words within ±5 s of any pause boundary in movie_time
- Words whose nearest trigger has trig_type ∈ {high_ambig, low_ambig, est, manual}
- Words whose movie_time > last_movie_time (recording terminated before stimulus ended)

trig_type taxonomy is 6-wide (verified empirically): high_real, low_real, high_ambig,
low_ambig, manual, est. The "reliable" subset is {high_real, low_real}; everything else
is droppable.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

DROPPABLE_TRIG_TYPES = frozenset({"high_ambig", "low_ambig", "est", "manual"})
RELIABLE_TRIG_TYPES = frozenset({"high_real", "low_real"})

NEURAL_SAMPLE_RATE_HZ = 2048
DEFAULT_PAUSE_DROP_WINDOW_S = 5.0


@dataclass
class TimingsAuditResult:
    csv_path: str
    n_rows: int
    n_pauses: int
    pause_locations_movie_time: list
    max_pause_diff_samples: int
    max_pause_diff_minutes: float
    last_movie_time_s: float
    trig_type_counts: dict
    n_high_ambig: int
    n_droppable_total: int
    has_trig_type: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def audit_trial_timings(csv_path: Path) -> TimingsAuditResult:
    df = pd.read_csv(csv_path)
    pauses = df[df["type"] == "pause"]
    n_pauses = len(pauses)
    pause_locs = pauses["movie_time"].tolist()
    pause_diffs = pauses["diff"].tolist() if n_pauses else []
    max_diff = int(max(pause_diffs)) if pause_diffs else 0
    max_diff_min = max_diff / NEURAL_SAMPLE_RATE_HZ / 60.0

    end_mask = df["type"] == "end"
    if end_mask.any():
        last_mt = float(df.loc[end_mask, "movie_time"].to_numpy()[-1])
    else:
        last_mt = float(df["movie_time"].max())

    has_trig = "trig_type" in df.columns
    if has_trig:
        counts = df["trig_type"].value_counts().to_dict()
        counts = {str(k): int(v) for k, v in counts.items()}
        n_high_ambig = int(counts.get("high_ambig", 0))
        n_droppable_total = sum(counts.get(t, 0) for t in DROPPABLE_TRIG_TYPES)
    else:
        counts = {}
        n_high_ambig = 0
        n_droppable_total = 0

    return TimingsAuditResult(
        csv_path=str(csv_path),
        n_rows=len(df),
        n_pauses=n_pauses,
        pause_locations_movie_time=pause_locs,
        max_pause_diff_samples=max_diff,
        max_pause_diff_minutes=max_diff_min,
        last_movie_time_s=last_mt,
        trig_type_counts=counts,
        n_high_ambig=n_high_ambig,
        n_droppable_total=n_droppable_total,
        has_trig_type=has_trig,
    )


def flag_words_near_pauses(
    words_df: pd.DataFrame,
    pause_movie_times: list,
    window_s: float = DEFAULT_PAUSE_DROP_WINDOW_S,
    time_col: str = "start",
) -> pd.Series:
    """Boolean mask: True = word within ±window_s of any pause boundary in movie_time."""
    if not pause_movie_times:
        return pd.Series([False] * len(words_df), index=words_df.index)
    t = words_df[time_col].to_numpy()
    flag = pd.Series([False] * len(words_df), index=words_df.index)
    for p in pause_movie_times:
        flag = flag | ((t >= p - window_s) & (t <= p + window_s))
    return flag
