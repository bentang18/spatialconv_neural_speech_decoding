"""A/V PTS drift-slope screening check.

This is a CHEAP screening pass that compares the container-level reported audio
duration to the video duration and flags grossly broken rips (>50 ms/min drift).
The real frame-accurate alignment check happens in the forced-aligner step (#20).

Why the gate is 50 ms/min, not 1 ms/min:
- ffprobe stream durations come from container headers and are quantized to
  packet boundaries (so ~250–700 ms total mismatch over a 2 h film is normal,
  i.e. 2–6 ms/min, even for a perfectly-aligned rip).
- A genuinely drifting rip would show >> 50 ms/min (10+ s of drift over a 2 h
  film), which would be visible to the forced aligner anyway.
- Container fixed offsets are handled by the forced aligner's per-word anchors.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import subprocess

DRIFT_SLOPE_GATE_MS_PER_MIN = 50.0


@dataclass
class DriftResult:
    video_path: str
    audio_duration_s: float | None
    video_duration_s: float | None
    container_duration_s: float | None
    delta_total_ms: float | None
    drift_slope_ms_per_min: float | None
    pass_gate: bool
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def ffprobe_streams(video_path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(video_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()}")
    return json.loads(out.stdout)


def _stream_duration(stream: dict, container_duration: float | None) -> float | None:
    """Get a stream's duration in seconds, falling back to container duration."""
    d = stream.get("duration")
    if d is not None:
        try:
            return float(d)
        except (TypeError, ValueError):
            pass
    tags = stream.get("tags") or {}
    for key in ("DURATION", "duration"):
        if key in tags:
            v = tags[key]
            try:
                # H:MM:SS.s format
                if isinstance(v, str) and ":" in v:
                    parts = v.split(":")
                    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
                    return h * 3600 + m * 60 + s
                return float(v)
            except (TypeError, ValueError):
                pass
    return container_duration


def check_drift(video_path: Path, gate_ms_per_min: float = DRIFT_SLOPE_GATE_MS_PER_MIN) -> DriftResult:
    info = ffprobe_streams(video_path)
    streams = info.get("streams", [])
    fmt = info.get("format", {})
    container_dur = float(fmt["duration"]) if "duration" in fmt else None

    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video = next((s for s in streams if s.get("codec_type") == "video"), None)

    if audio is None or video is None:
        return DriftResult(
            video_path=str(video_path),
            audio_duration_s=None,
            video_duration_s=None,
            container_duration_s=container_dur,
            delta_total_ms=None,
            drift_slope_ms_per_min=None,
            pass_gate=False,
            note="missing audio or video stream",
        )

    a_dur = _stream_duration(audio, container_dur)
    v_dur = _stream_duration(video, container_dur)

    if a_dur is None or v_dur is None:
        return DriftResult(
            video_path=str(video_path),
            audio_duration_s=a_dur,
            video_duration_s=v_dur,
            container_duration_s=container_dur,
            delta_total_ms=None,
            drift_slope_ms_per_min=None,
            pass_gate=False,
            note="ffprobe did not expose per-stream duration",
        )

    delta_ms = (a_dur - v_dur) * 1000.0
    base = min(a_dur, v_dur)
    slope = delta_ms / (base / 60.0) if base > 0 else float("inf")

    return DriftResult(
        video_path=str(video_path),
        audio_duration_s=float(a_dur),
        video_duration_s=float(v_dur),
        container_duration_s=container_dur,
        delta_total_ms=float(delta_ms),
        drift_slope_ms_per_min=float(slope),
        pass_gate=abs(slope) < gate_ms_per_min,
    )
