"""Per-film alignment-readiness gate.

For every film with a BT reference + a local rip + a calibrated threshold pair,
runs:
  1. Cut-version check (rip duration vs BT Table 2, ±2 min).
  2. A/V PTS drift-slope screening (<50 ms/min).
  3. Frame-verify on all BT reference frames with per-film thresholds.
Aggregates pass_film = cut_match AND drift_pass AND frame_pass_rate ≥ MIN_FRAME_PASS_RATE.

Emits one JSON per film and a summary table.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from .cut_coverage import check_cut, load_rip_durations
from .pts_drift import check_drift
from .verify_frame import verify_frame

MIN_FRAME_PASS_RATE = 0.7  # 7/10 BT reference frames must pass per-film thresholds


@dataclass
class GateResult:
    slug: str
    rip_video: str
    cut_match: bool
    cut_delta_min: float
    drift_pass: bool
    drift_slope_ms_per_min: float
    n_frames: int
    n_frame_pass: int
    frame_pass_rate: float
    pass_film: bool
    per_frame: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def run_film_gate(
    slug: str,
    rip_video: Path,
    bt_frames_dir: Path,
    phash_thresh: int,
    dhash_thresh: int,
    av_quality_json: Path,
    min_frame_pass_rate: float = MIN_FRAME_PASS_RATE,
) -> GateResult:
    rips = load_rip_durations(av_quality_json)
    cut = check_cut(slug, rips[slug])
    drift = check_drift(rip_video)

    frames = []
    n_pass = 0
    pngs = sorted(bt_frames_dir.glob("frame_*.png"),
                  key=lambda p: int(p.stem.split("_")[1]))
    for png in pngs:
        try:
            r = verify_frame(png, rip_video,
                             phash_thresh=phash_thresh,
                             dhash_thresh=dhash_thresh)
            frames.append(r.to_dict())
            if r.pass_frame:
                n_pass += 1
        except Exception as e:
            frames.append({"bt_frame_path": str(png), "error": str(e)})

    n = len(frames)
    pass_rate = n_pass / n if n else 0.0
    frame_pass = pass_rate >= min_frame_pass_rate

    return GateResult(
        slug=slug,
        rip_video=str(rip_video),
        cut_match=cut.cut_match,
        cut_delta_min=cut.delta_min,
        drift_pass=drift.pass_gate,
        drift_slope_ms_per_min=drift.drift_slope_ms_per_min or 0.0,
        n_frames=n,
        n_frame_pass=n_pass,
        frame_pass_rate=pass_rate,
        pass_film=bool(cut.cut_match and drift.pass_gate and frame_pass),
        per_frame=frames,
    )
