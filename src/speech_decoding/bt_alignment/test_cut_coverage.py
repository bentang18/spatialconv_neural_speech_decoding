"""Tests for cut-version + trial-coverage check.

Reference values:
- lotr-1 expected = 228.3 min per BT Table 2.
- sub_3 trial001 = lotr-1, last_movie_time_s = 8089.5s = 134.83 min → coverage ≈ 59%.
- sub_2 trial006 = aquaman, last_movie_time_s = 7900.5s = 131.67 min vs Table 2 = 143.4 → 91.8%.
"""
import json
from pathlib import Path
import pytest

from speech_decoding.bt_alignment.cut_coverage import (
    BT_EXPECTED_RUNTIME_MIN,
    SUBJECT_TRIAL_MOVIE,
    CUT_TOLERANCE_MIN,
    check_cut,
    check_coverage,
    load_rip_durations,
)


def test_bt_table_has_20_films():
    # 20 movies in our rip set + sesame-street excluded
    assert len(BT_EXPECTED_RUNTIME_MIN) == 20
    assert "lotr-1" in BT_EXPECTED_RUNTIME_MIN
    assert BT_EXPECTED_RUNTIME_MIN["lotr-1"] == 228.3


def test_subject_trial_mapping_has_26_entries():
    assert len(SUBJECT_TRIAL_MOVIE) == 26
    assert SUBJECT_TRIAL_MOVIE[(3, 1)] == "lotr-1"
    assert SUBJECT_TRIAL_MOVIE[(2, 6)] == "aquaman"
    assert SUBJECT_TRIAL_MOVIE[(8, 0)] == "sesame-street-episode-3990"


def test_cut_check_match_within_tolerance():
    # Theatrical-cut ant-man 117.1 vs rip 117.5 -> 0.4 min delta -> match
    r = check_cut("ant-man", rip_duration_min=117.5)
    assert r.cut_match is True
    assert abs(r.delta_min - 0.4) < 1e-6


def test_cut_check_fail_outside_tolerance():
    # rip 130 min vs expected 117.1 -> +12.9 -> FAIL
    r = check_cut("ant-man", rip_duration_min=130.0)
    assert r.cut_match is False
    assert r.delta_min > CUT_TOLERANCE_MIN


def test_cut_check_unknown_slug_raises():
    with pytest.raises(KeyError):
        check_cut("not-a-real-movie", rip_duration_min=100.0)


def test_coverage_sub3_trial001_lotr1():
    # Reference values pulled from pause_audit run on real BT subject_timings
    r = check_coverage(3, 1, last_movie_time_s=8089.5, film_duration_s=228.3 * 60.0)
    assert r.slug == "lotr-1"
    assert 58.0 < r.coverage_pct < 60.0


def test_coverage_invalid_film_duration_raises():
    with pytest.raises(ValueError):
        check_coverage(3, 1, last_movie_time_s=8089.5, film_duration_s=0.0)


def test_coverage_unknown_subject_trial_raises():
    with pytest.raises(KeyError):
        check_coverage(99, 99, last_movie_time_s=100.0, film_duration_s=1000.0)


def test_load_rip_durations_from_av_audit():
    # Should read prior session's audit JSON without error
    p = Path("/Users/bentang/Documents/Code/speech/reports/bt_local_audit_2026_05_28/av_quality.json")
    if not p.exists():
        pytest.skip("av_quality.json not present")
    rips = load_rip_durations(p)
    # At least 19 of the 20 rips should be parsed (the-martian rerip may flag)
    assert len(rips) >= 19
    for slug in ("ant-man", "lotr-1", "venom"):
        assert slug in rips
        assert rips[slug] > 50  # all films > 50 min


def test_result_jsonable():
    cut = check_cut("ant-man", 117.5)
    cov = check_coverage(3, 1, 8089.5, 228.3 * 60.0)
    json.dumps(cut.to_dict())
    json.dumps(cov.to_dict())
