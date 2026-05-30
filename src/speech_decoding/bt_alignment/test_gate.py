"""Tests for the per-film alignment gate.

Validates the gate runs end-to-end on one real film (megamind = small AV1 rip),
producing all sub-checks and a final pass_film verdict consistent with hand-computed values.
"""
from pathlib import Path
import json
import pytest

from speech_decoding.bt_alignment.gate import run_film_gate, MIN_FRAME_PASS_RATE


MEGAMIND_RIP = Path(
    "/Users/bentang/Documents/Code/speech/movies/"
    "Megamind (2010) 1080p BluRay AV1 Opus MULTi4 [RAV1NE]/"
    "Megamind.2010.1080p.BluRay.AV1.Opus.MULTi4-RAV1NE.mkv"
)
MEGAMIND_FRAMES = Path("/Users/bentang/Documents/Code/speech/.cache/braintreebank/movie_frames/megamind")
AV_JSON = Path("/Users/bentang/Documents/Code/speech/reports/bt_local_audit_2026_05_28/av_quality.json")


def test_gate_constant():
    assert MIN_FRAME_PASS_RATE == 0.7


def test_run_film_gate_megamind_passes_all_checks():
    if not (MEGAMIND_RIP.exists() and MEGAMIND_FRAMES.exists() and AV_JSON.exists()):
        pytest.skip("required local artifacts not present")
    r = run_film_gate(
        slug="megamind",
        rip_video=MEGAMIND_RIP,
        bt_frames_dir=MEGAMIND_FRAMES,
        phash_thresh=8,
        dhash_thresh=6,
        av_quality_json=AV_JSON,
    )
    assert r.cut_match is True
    assert r.drift_pass is True
    assert r.n_frames > 0
    assert r.frame_pass_rate >= 0.7
    assert r.pass_film is True


def test_result_jsonable():
    if not (MEGAMIND_RIP.exists() and MEGAMIND_FRAMES.exists() and AV_JSON.exists()):
        pytest.skip("required local artifacts not present")
    r = run_film_gate(
        "megamind", MEGAMIND_RIP, MEGAMIND_FRAMES, 8, 6, AV_JSON,
    )
    json.dumps(r.to_dict())
