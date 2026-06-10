"""LG1 guard — measure each BT subject's TRUE native sample rate from its trigger
track and assert it matches the single-source ``BT_SUBJECT_NATIVE_RATE_HZ`` registry.

This is the structural defense for the S9-class bug: a subject whose distributed h5
ticks at a rate other than the assumed 2048 (S9 is natively 1024). The pipeline reads
no on-disk rate and trusts ``neuroprobe.config.SAMPLING_RATE = 2048`` everywhere, so a
non-2048 subject is silently 2× time-stretched with a 2×-mislabeled STFT frequency
axis — plausible-looking, never crashing. A code-path audit cannot catch this (it
shares the 2048 assumption); only MEASURING the bytes does. So we measure.

The rate is recovered from the BT trigger track (``subject_timings/*.csv``): the
``index`` column is the literal h5 sample index, ``movie_time`` the movie clock. The
median *local* slope ``d(index)/d(movie_time)`` over non-pause segments == the native
sample rate (recording pauses only inflate the slope, so the median is robust).

Skips cleanly when the vendored trigger tracks are absent (CI). See
``reports/bt_alignment/data_pipeline_bug_ledger.md`` LG1.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from speech_decoding.studies.braintreebank.manifest import (
    bt_subject_native_rate_hz,
)

_TRIGGER_DIR = Path(".cache/braintreebank/subject_timings")
_REL_TOL = 0.06  # 6% — median-local-slope removes pause inflation; real grids are exact


def _measure_native_rate_hz(timings_csv: Path) -> float:
    """Median local ``d(index)/d(movie_time)`` over non-pause segments == native rate."""
    import pandas as pd

    df = (
        pd.read_csv(timings_csv)[["index", "movie_time"]]
        .dropna()
        .sort_values(by="index")
        .drop_duplicates(subset="index", keep="last")
    )
    idx = df["index"].to_numpy(dtype=float)
    movie = df["movie_time"].to_numpy(dtype=float)
    d_idx = np.diff(idx)
    d_movie = np.diff(movie)
    playing = d_movie > 0.01  # drop pause rows where movie_time barely advances
    if not playing.any():
        return float("nan")
    return float(np.median(d_idx[playing] / d_movie[playing]))


def _trigger_tracks() -> list[Path]:
    if not _TRIGGER_DIR.is_dir():
        return []
    return sorted(_TRIGGER_DIR.glob("sub_*_trial*_timings.csv"))


@pytest.mark.must_pass_before_dispatch
def test_bt_subject_native_rate_matches_registry() -> None:
    """Every vendored subject's measured native rate must equal the registry value.

    Green today (registry says S9=1024, rest 2048). Fails loud if a future-added
    subject's recording is non-2048 and the registry was not updated — i.e. it makes
    the S9-class bug impossible to add silently. Updating the registry is only valid
    after confirming the true rate on DCC AND ensuring the loader handles it.
    """
    tracks = _trigger_tracks()
    if not tracks:
        pytest.skip("vendored BT subject_timings absent (CI)")

    failures: list[str] = []
    measured_by_subject: dict[int, float] = {}
    for track in tracks:
        m = re.search(r"sub_(\d+)_trial", track.name)
        if m is None:
            continue
        sub = int(m.group(1))
        rate = _measure_native_rate_hz(track)
        # one trial per subject is enough to pin the recording rate; keep the first
        measured_by_subject.setdefault(sub, rate)
        expected = bt_subject_native_rate_hz(sub)
        if not np.isfinite(rate) or abs(rate - expected) > _REL_TOL * expected:
            failures.append(
                f"{track.name}: measured {rate:.1f} Hz vs registry {expected} Hz"
            )

    assert not failures, (
        "BT trigger-track native sample rate disagrees with "
        "BT_SUBJECT_NATIVE_RATE_HZ:\n  " + "\n  ".join(failures) + "\n"
        "A subject whose true rate != the registry is the S9-class bug (ledger LG1): "
        "the pipeline assumes 2048 Hz and will silently 2×-time-stretch + mislabel its "
        "STFT frequency axis. Update the registry ONLY after confirming the rate on DCC "
        "and ensuring the loader resamples/excludes the subject."
    )


@pytest.mark.must_pass_before_dispatch
def test_s9_native_rate_is_measured_half() -> None:
    """Pin the known S9 anomaly explicitly so a fixture/registry edit that erases it
    fails loud (meta-test for the guard above)."""
    tracks = _trigger_tracks()
    s9 = [t for t in tracks if re.match(r"sub_9_trial", t.name)]
    if not s9:
        pytest.skip("vendored S9 trigger track absent (CI)")
    rate = _measure_native_rate_hz(s9[0])
    assert 900.0 < rate < 1150.0, (
        f"S9 trigger track measured {rate:.1f} Hz; expected ~1024 (ledger LG1). "
        "If this changed, the vendored fixture or the dataset itself changed — verify."
    )
    assert bt_subject_native_rate_hz(9) == 1024
