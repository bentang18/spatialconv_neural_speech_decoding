"""LG14 — fail-loud guards on the unguarded reader surfaces (P7 threat model).

Two reader sites turned a content-corrupt (not missing) input file into
plausible-but-wrong model input with no crash — the S9 class:

  * the trigger-track clock map (``_load_neural_to_movie_map``) never checked the
    ``movie_time`` axis for finiteness / gross non-monotonicity / truncation, so a
    corrupt CSV yields a wrong-but-finite ``movie_onset_s`` that silently
    mis-aligns every P3 Whisper-teacher target;
  * the continuous-label vector (``_continuous_labels``) took NaN straight from the
    ``original_index``-keyed ``.map(features.csv)`` join (or a corrupt
    pitch/volume value), and NaN then drops out of every percentile bucket,
    silently mislabeling clips.

These tests are POSITIVE controls (inject the corruption, assert the guard FIRES)
plus a real-data NEGATIVE control (every vendored trigger track must PASS — the
guard's jitter tolerance was calibrated so it cannot false-fire). See ledger LG14.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from speech_decoding.studies.braintreebank.labels import _assert_finite_labels
from speech_decoding.studies.braintreebank.word_events import (
    _MAX_TRIGGER_BACKSTEP_S,
    _assert_trigger_track_sane,
    _load_neural_to_movie_map,
)

_BT_ROOT = Path(".cache/braintreebank")
_TRIGGER_DIR = _BT_ROOT / "subject_timings"


def _good_track(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    xp = np.arange(n, dtype=float) * 100.0  # strictly increasing index
    yp = np.arange(n, dtype=float) * 0.085  # ~85 ms trigger spacing, monotone
    return xp, yp


# --- negative control: real vendored tracks must all PASS -----------------------
def test_all_vendored_trigger_tracks_pass_guard() -> None:
    """Every vendored trigger track loads without tripping the guard.

    This is the false-positive proof: the ~57 ms real jitter (18/26 tracks step
    movie_time backward by < 1 frame) must NOT fire the seconds-scale guard.
    """
    if not _TRIGGER_DIR.is_dir():
        pytest.skip("vendored BT subject_timings absent (CI)")
    tracks = sorted(_TRIGGER_DIR.glob("sub_*_trial*_timings.csv"))
    if not tracks:
        pytest.skip("no vendored trigger tracks")
    for track in tracks:
        m = re.search(r"sub_(\d+)_trial(\d+)_timings", track.name)
        assert m
        sub, trial = int(m.group(1)), int(m.group(2))
        result = _load_neural_to_movie_map(sub, trial, _BT_ROOT)  # raises on failure
        assert result is not None
        xp, yp = result
        assert len(xp) == len(yp) > 100


def test_jittery_but_valid_track_passes() -> None:
    """A sub-second backward jitter step (real trigger noise) must not fire."""
    xp, yp = _good_track()
    yp[200] = yp[199] - 0.057  # the worst real backstep measured across 26 tracks
    yp[201] = yp[199] + 0.085  # recover
    _assert_trigger_track_sane(xp, yp, Path("synthetic"))  # no raise


# --- positive controls: each corruption class must FIRE -------------------------
def test_guard_fires_on_truncated_track() -> None:
    xp, yp = _good_track(n=50)  # < _MIN_TRIGGER_ROWS
    with pytest.raises(ValueError, match="rows"):
        _assert_trigger_track_sane(xp, yp, Path("synthetic"))


def test_guard_fires_on_infinite_knot() -> None:
    xp, yp = _good_track()
    yp[300] = np.inf  # survives dropna upstream
    with pytest.raises(ValueError, match="non-finite"):
        _assert_trigger_track_sane(xp, yp, Path("synthetic"))


def test_guard_fires_on_non_strict_index() -> None:
    xp, yp = _good_track()
    xp[300] = xp[299]  # dedup contract broke -> tie
    with pytest.raises(ValueError, match="strictly increasing"):
        _assert_trigger_track_sane(xp, yp, Path("synthetic"))


def test_guard_fires_on_gross_non_monotone_movie_time() -> None:
    xp, yp = _good_track()
    yp[300] = yp[299] - (_MAX_TRIGGER_BACKSTEP_S + 5.0)  # seconds-scale backward jump
    with pytest.raises(ValueError, match="backward"):
        _assert_trigger_track_sane(xp, yp, Path("synthetic"))


def test_guard_fires_on_collapsed_span() -> None:
    xp = np.arange(500, dtype=float) * 100.0
    yp = np.linspace(0.0, 5.0, 500)  # only 5 s of movie -> truncated clock
    with pytest.raises(ValueError, match="spans only"):
        _assert_trigger_track_sane(xp, yp, Path("synthetic"))


# --- continuous-label finiteness guard (features.csv .map + pitch/volume) --------
def test_finite_labels_pass_through() -> None:
    arr = np.array([0.1, 2.0, -3.5, 4.0])
    out = _assert_finite_labels(arr, "rms", "features.csv")
    assert np.array_equal(out, arr)


def test_label_guard_fires_on_nan() -> None:
    """The features.csv .map() silent-NaN class: an unmatched original_index."""
    arr = np.array([0.1, np.nan, 2.0])
    with pytest.raises(ValueError, match="non-finite"):
        _assert_finite_labels(arr, "rms", "features.csv")


def test_label_guard_fires_on_inf() -> None:
    """A corrupt pitch/volume value class."""
    arr = np.array([0.1, np.inf, 2.0])
    with pytest.raises(ValueError, match="non-finite"):
        _assert_finite_labels(arr, "pitch", "pitch_volume_features")


# --- neural-axis est_idx finiteness (LG14h) -------------------------------------
def test_word_event_rows_fires_on_nan_est_idx() -> None:
    """A NaN est_idx in words_df must fail loud, not silently mis-window the clip."""
    import pandas as pd

    from speech_decoding.studies.braintreebank.word_events import _word_event_rows

    n_words, n_nonverbal = 6, 2
    est = [float(round((i + 200.0) * 2048.0)) for i in range(n_words)]
    est[3] = float("nan")  # corrupt one neural-clock onset
    words = pd.DataFrame(
        {
            "start": [float(i) for i in range(n_words)],
            "end": [float(i) + 1.0 for i in range(n_words)],
            "est_idx": est,
            "original_index": list(range(n_words)),
            "full_word": list("ABCDEF"),
        }
    )
    nv_starts = [0.5, 4.5]
    nonverbal = pd.DataFrame(
        {
            "start": nv_starts,
            "end": [s + 1.0 for s in nv_starts],
            "est_idx": [round((s + 200.0) * 2048.0) for s in nv_starts],
        }
    )
    with pytest.raises(ValueError, match="non-finite est_idx"):
        _word_event_rows(
            subject_id=2, trial_id=4, timeline="tl", words_df=words,
            nonverbal_df=nonverbal, tasks=("speech",), binary_tasks=True,
            lite=False, nano=False, random_seed=42, duration=1.0, balance=False,
        )
    assert n_nonverbal == 2  # sanity on the fixture
