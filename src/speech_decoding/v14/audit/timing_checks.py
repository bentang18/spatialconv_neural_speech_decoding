"""Soft cross-checks between phoneme-level `.fif` pos-0 events and the BIDS
events TSV.

The `.fif` is authoritative for the Phase-1 loader — pos-0 raw samples come
from `epochs.events[:, 0]` directly. The events TSV is a separate
derivative that can drift independently (e.g. S26 missing one row, S14
wholesale regeneration). Both checks below are SOFT: a mismatch flags the
events TSV as out-of-sync but the loader can still use the `.fif`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from speech_decoding.v14.audit.io import ECOG_RAW_SFREQ
from speech_decoding.v14.audit.schema import Check, make_check

SAMPLE_TOL = 12  # raw samples at 2000 Hz = 6 ms. Covers MFA-precision jitter
# in the events TSV `sample` column on patients without an `eventsOLD`
# backup. Large divergences (S14 events.tsv 235k off, S26 events.tsv drift
# up to 31k) still get surfaced.
ONSET_TOL_S = 0.006  # matches SAMPLE_TOL in seconds


def _pos0_samples(epochs) -> np.ndarray:
    return epochs.events[0::3, 0]


def check_fif_samples_match_events_tsv(
    epochs, authoritative_resp: pd.DataFrame
) -> Check:
    """Soft: pos-0 raw-clock sample index equals events TSV response
    `sample` for the same trial (within SAMPLE_TOL)."""
    pos0 = _pos0_samples(epochs)
    n = min(len(pos0), len(authoritative_resp))
    if n == 0:
        return make_check("fif_samples_match_events_tsv", "soft", False, "no epochs")
    old_samples = authoritative_resp["sample"].to_numpy()[:n]
    diffs = np.abs(pos0[:n].astype(np.int64) - old_samples.astype(np.int64))
    n_outside = int(np.sum(diffs > SAMPLE_TOL))
    ok = n_outside == 0 and len(pos0) == len(authoritative_resp)
    return make_check(
        "fif_samples_match_events_tsv",
        "soft",
        ok,
        f"max |Δ|={int(diffs.max())} samples, median={float(np.median(diffs)):.1f}, "
        f"n_outside_tol({SAMPLE_TOL})={n_outside} "
        f"({len(pos0)} pos-0 epochs vs {len(authoritative_resp)} responses)",
    )


def check_fif_t0_matches_events_tsv_onset(
    epochs, authoritative_resp: pd.DataFrame
) -> Check:
    """Soft: pos-0 epoch t=0 (raw-sample time in seconds) equals events
    TSV response `onset` within ONSET_TOL_S."""
    pos0 = _pos0_samples(epochs)
    n = min(len(pos0), len(authoritative_resp))
    if n == 0:
        return make_check("fif_t0_matches_events_tsv_onset", "soft", False, "no epochs")
    fif_t0_s = pos0[:n] / ECOG_RAW_SFREQ
    onsets = authoritative_resp["onset"].to_numpy()[:n]
    diffs = np.abs(fif_t0_s - onsets)
    n_outside = int(np.sum(diffs > ONSET_TOL_S))
    ok = n_outside == 0
    return make_check(
        "fif_t0_matches_events_tsv_onset",
        "soft",
        ok,
        f"max |Δ|={diffs.max() * 1000:.2f} ms, median={np.median(diffs) * 1000:.2f} ms, "
        f"n_outside_tol({ONSET_TOL_S * 1000:.0f}ms)={n_outside}",
    )
