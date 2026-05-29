"""Cut-version + trial-coverage check.

Two gates:
1. **Cut identity**: rip duration (min) ≈ BT Table 2 expected runtime within ±2 min.
   A different cut (theatrical vs extended) produces a different frame-to-time map,
   so forced-aligner anchors would land on the wrong frames.

2. **Trial coverage**: per-(subject, trial) coverage_pct = last_movie_time_s / film_duration_s.
   BT subjects do not always view the full film (sessions may terminate early). This
   measures how much of the movie clock the trial actually consumed.

Sources of truth:
- (subject, trial) → film slug: upstream neuroprobe `BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING`.
- Expected runtimes per film: BT preprint Table 2 (§A.3).
- Rip durations: `reports/bt_local_audit_2026_05_28/av_quality.json` (ffprobe).
- Trial last_movie_time_s: TimingsAuditResult from `pause_audit.audit_trial_timings`.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json

CUT_TOLERANCE_MIN = 2.0

# BT preprint §A.3 Table 2 — minutes, ground-truth expected runtimes.
BT_EXPECTED_RUNTIME_MIN = {
    "ant-man": 117.1,
    "aquaman": 143.4,
    "avengers-infinity-war": 149.4,
    "black-panther": 134.6,
    "cars-2": 106.3,
    "coraline": 100.6,
    "fantastic-mr-fox": 86.8,
    "guardians-of-the-galaxy": 120.9,
    "guardians-of-the-galaxy-2": 135.8,
    "incredibles": 115.4,
    "lotr-1": 228.3,
    "lotr-2": 235.5,
    "megamind": 95.6,
    "shrek-the-third": 92.8,
    "spider-man-far-from-home": 129.4,
    "spider-man-3-homecoming": 133.5,
    "the-martian": 151.4,
    "thor-ragnarok": 130.5,
    "toy-story": 81.1,
    "venom": 112.1,
    # sesame-street-episode-3990: BT raw, not in our movies/ rip set
}

# Upstream neuroprobe c7b955b (subject_id, trial_id) -> slug
SUBJECT_TRIAL_MOVIE = {
    (1, 0): "fantastic-mr-fox",
    (1, 1): "the-martian",
    (1, 2): "thor-ragnarok",
    (2, 0): "venom",
    (2, 1): "spider-man-3-homecoming",
    (2, 2): "guardians-of-the-galaxy",
    (2, 3): "guardians-of-the-galaxy-2",
    (2, 4): "avengers-infinity-war",
    (2, 5): "black-panther",
    (2, 6): "aquaman",
    (3, 0): "cars-2",
    (3, 1): "lotr-1",
    (3, 2): "lotr-2",
    (4, 0): "shrek-the-third",
    (4, 1): "megamind",
    (4, 2): "incredibles",
    (5, 0): "fantastic-mr-fox",
    (6, 0): "megamind",
    (6, 1): "toy-story",
    (6, 4): "coraline",
    (7, 0): "cars-2",
    (7, 1): "megamind",
    (8, 0): "sesame-street-episode-3990",
    (9, 0): "ant-man",
    (10, 0): "cars-2",
    (10, 1): "spider-man-far-from-home",
}


@dataclass
class CutCheckResult:
    slug: str
    rip_duration_min: float
    expected_min: float
    delta_min: float
    cut_match: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CoverageResult:
    subject_id: int
    trial_id: int
    slug: str
    last_movie_time_s: float
    film_duration_s: float
    coverage_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


def check_cut(slug: str, rip_duration_min: float, tolerance_min: float = CUT_TOLERANCE_MIN) -> CutCheckResult:
    """Compare a rip's duration to BT Table 2 expected. Raises if slug not in table."""
    if slug not in BT_EXPECTED_RUNTIME_MIN:
        raise KeyError(f"slug '{slug}' not in BT Table 2")
    expected = BT_EXPECTED_RUNTIME_MIN[slug]
    delta = rip_duration_min - expected
    return CutCheckResult(
        slug=slug,
        rip_duration_min=float(rip_duration_min),
        expected_min=float(expected),
        delta_min=float(delta),
        cut_match=abs(delta) <= tolerance_min,
    )


def check_coverage(subject_id: int, trial_id: int, last_movie_time_s: float, film_duration_s: float) -> CoverageResult:
    """Compute per-trial movie-clock coverage."""
    key = (int(subject_id), int(trial_id))
    if key not in SUBJECT_TRIAL_MOVIE:
        raise KeyError(f"(subject={subject_id}, trial={trial_id}) not in upstream neuroprobe mapping")
    if film_duration_s <= 0:
        raise ValueError("film_duration_s must be > 0")
    return CoverageResult(
        subject_id=key[0],
        trial_id=key[1],
        slug=SUBJECT_TRIAL_MOVIE[key],
        last_movie_time_s=float(last_movie_time_s),
        film_duration_s=float(film_duration_s),
        coverage_pct=100.0 * float(last_movie_time_s) / float(film_duration_s),
    )


def load_rip_durations(av_quality_json: Path) -> dict[str, float]:
    """Load slug -> rip duration (min) from prior ffprobe audit JSON."""
    data = json.loads(Path(av_quality_json).read_text())
    out = {}
    for row in data:
        if "dur_min" in row and "slug" in row:
            out[row["slug"]] = float(row["dur_min"])
    return out
