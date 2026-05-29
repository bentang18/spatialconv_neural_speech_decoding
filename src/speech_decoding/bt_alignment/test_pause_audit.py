"""Tests for per-trial pause + trigger-quality audit.

Reference numbers verified empirically on local subject_timings CSVs:
- sub_3 trial001 (lotr-1): 5 pauses, max diff 1,798,995 samples (~14.6 min)
- sub_2 trial006: 13 pauses, max diff 3,656,096 samples (~29.8 min)
- sub_4 trial000: 5 pauses, max ~9.2 min
"""
from pathlib import Path
from speech_decoding.bt_alignment.pause_audit import (
    audit_trial_timings,
    TimingsAuditResult,
    DROPPABLE_TRIG_TYPES,
)

TIMINGS_DIR = Path("/Users/bentang/Documents/Code/speech/.cache/braintreebank/subject_timings")


def test_droppable_trig_types_constants():
    # Audit-#1/#2: high_real and low_real are reliable; everything else gets dropped
    assert "high_real" not in DROPPABLE_TRIG_TYPES
    assert "low_real" not in DROPPABLE_TRIG_TYPES
    assert "high_ambig" in DROPPABLE_TRIG_TYPES
    assert "low_ambig" in DROPPABLE_TRIG_TYPES
    assert "est" in DROPPABLE_TRIG_TYPES
    assert "manual" in DROPPABLE_TRIG_TYPES


def test_audit_sub3_trial001_lotr1():
    r = audit_trial_timings(TIMINGS_DIR / "sub_3_trial001_timings.csv")
    assert isinstance(r, TimingsAuditResult)
    assert r.n_pauses == 5
    # max sample-jump verified empirically = 1,798,995
    assert r.max_pause_diff_samples == 1798995
    assert abs(r.max_pause_diff_minutes - 14.63) < 0.05
    # last_movie_time empirically 8089.49s per sub_2 trial006 reference; lotr-1 is much shorter actual
    assert r.last_movie_time_s > 0
    # trig_type distribution should sum to 80840 (verified earlier)
    assert sum(r.trig_type_counts.values()) == 80840
    assert r.trig_type_counts["high_real"] == 79181
    assert r.trig_type_counts["high_ambig"] == 1649


def test_audit_sub2_trial006_max_gap():
    r = audit_trial_timings(TIMINGS_DIR / "sub_2_trial006_timings.csv")
    assert r.n_pauses == 13
    assert r.max_pause_diff_samples == 3656096
    # ~29.75 min (verified)
    assert 29.7 < r.max_pause_diff_minutes < 29.8
    # low_real dominates on sub_2 — verified
    assert r.trig_type_counts["low_real"] > r.trig_type_counts["high_real"]


def test_audit_no_pause_trial_graceful():
    # If a trial has 0 pauses, audit should still emit a clean result, not crash
    import pandas as pd, tempfile
    df = pd.DataFrame({
        "type": ["beginning"] + ["trigger"] * 100 + ["end"],
        "movie_time": [0.0] + [i * 0.083 for i in range(100)] + [8.3],
        "start_time": [1000.0] + [1000.0 + i * 0.083 for i in range(100)] + [1008.3],
        "end_time": [1000.001] * 102,
        "trig_type": ["manual"] + ["high_real"] * 100 + ["manual"],
        "index": list(range(102 * 200))[:102],
        "diff": [0] + [200] * 101,
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
        r = audit_trial_timings(Path(f.name))
    assert r.n_pauses == 0
    assert r.max_pause_diff_samples == 0
    assert r.pause_locations_movie_time == []


def test_drop_words_near_pause_boundary():
    """Words within ±5s of a pause edge should be flagged DROP."""
    from speech_decoding.bt_alignment.pause_audit import flag_words_near_pauses
    import pandas as pd
    r = audit_trial_timings(TIMINGS_DIR / "sub_3_trial001_timings.csv")
    # First pause at movie_time=1394.59s; words at 1390, 1392 (within ±5s) should drop
    words = pd.DataFrame({"start": [1.0, 100.0, 1390.0, 1392.0, 1400.0, 5000.0]})
    drops = flag_words_near_pauses(words, r.pause_locations_movie_time, window_s=5.0)
    assert drops.dtype == bool
    assert not drops.iloc[0]  # 1.0s far from pauses
    assert not drops.iloc[1]  # 100.0s far
    assert drops.iloc[2]   # 1390s within 5s of 1394.59
    assert drops.iloc[3]   # 1392s within 5s of 1394.59
    assert not drops.iloc[4]  # 1400s is 5.41s after pause → outside ±5


def test_summary_dict_serializable():
    """Result must be JSON-serializable for per-trial JSON emit."""
    import json
    r = audit_trial_timings(TIMINGS_DIR / "sub_3_trial001_timings.csv")
    s = r.to_dict()
    json.dumps(s)
