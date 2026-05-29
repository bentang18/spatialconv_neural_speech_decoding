"""Tests for A/V PTS drift-slope check."""
import json
from pathlib import Path
import pytest

from speech_decoding.bt_alignment.pts_drift import (
    check_drift, DRIFT_SLOPE_GATE_MS_PER_MIN, _stream_duration,
)

# Real rip from local cache
SAMPLE_RIP = Path(
    "/Users/bentang/Documents/Code/speech/movies/Ant-Man 2015 1080p BluRay x264 DTS-JYK/"
    "Ant-Man 2015 1080p BluRay x264 DTS-JYK.mkv"
)


def test_drift_gate_constant():
    # Coarse screening gate; the real check happens at forced-aligner stage.
    assert DRIFT_SLOPE_GATE_MS_PER_MIN == 50.0


def test_stream_duration_direct_field():
    s = {"duration": "117.123"}
    assert _stream_duration(s, container_duration=None) == pytest.approx(117.123)


def test_stream_duration_falls_back_to_tag_hms():
    s = {"tags": {"DURATION": "01:57:08.00"}}
    assert _stream_duration(s, None) == pytest.approx(7028.0, rel=1e-4)


def test_stream_duration_falls_back_to_container():
    s = {}
    assert _stream_duration(s, container_duration=100.0) == 100.0


def test_drift_on_real_rip_passes_gate():
    if not SAMPLE_RIP.exists():
        pytest.skip(f"sample rip missing: {SAMPLE_RIP}")
    r = check_drift(SAMPLE_RIP)
    assert r.video_duration_s is not None
    assert r.audio_duration_s is not None
    # Real rip should be well within drift tolerance
    assert r.pass_gate, f"unexpected drift {r.drift_slope_ms_per_min:.2f} ms/min"


def test_drift_result_is_json_serializable():
    if not SAMPLE_RIP.exists():
        pytest.skip(f"sample rip missing: {SAMPLE_RIP}")
    r = check_drift(SAMPLE_RIP)
    json.dumps(r.to_dict())
