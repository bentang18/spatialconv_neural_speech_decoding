"""Per-trial sample-index alignment between phoneme-level `.fif` and `eventsOLD.tsv`.

Phoneme-level `.fif` holds one epoch per phoneme. Position-0 epochs
(`epochs.events[0::3]`) lock to the first phoneme of each trial; for S14 this
is bit-identical to trial-level response onset at raw-sample resolution
(`phon.events[0::3, 0] == trial.events[:, 0]` across all 153 trials).

Covers:
- #6: `.fif` pos-0 raw-sample index matches `eventsOLD.tsv` response `sample`
- #7: `t=0` of each pos-0 epoch equals `eventsOLD.tsv` response `onset`
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from speech_decoding.v14.audit.io import ECOG_RAW_SFREQ
from speech_decoding.v14.audit.schema import Check, make_check

SAMPLE_TOL = 12  # raw samples at 2000 Hz = 6 ms. Allows for MFA-precision
# jitter in the `events.tsv` sample column on patients that only have the
# MFA-derived file (no `eventsOLD` backup) plus floating-point residue
# between the two `.fif` time grids. S14's events.tsv divergence (235k
# samples off) and S26's broken events.tsv (31k samples off) still get
# caught by orders of magnitude.
ONSET_TOL_S = 0.006  # 6 ms — matches SAMPLE_TOL


def _pos0_samples(epochs) -> np.ndarray:
    return epochs.events[0::3, 0]


def check_fif_samples_match_authoritative(
    epochs, authoritative_resp: pd.DataFrame
) -> Check:
    """Predicate #6: for each pos-0 epoch, its raw-clock sample index equals
    `eventsOLD.tsv` response `sample` for the same trial (within SAMPLE_TOL)."""
    pos0 = _pos0_samples(epochs)
    n = min(len(pos0), len(authoritative_resp))
    if n == 0:
        return make_check("fif_samples_match_authoritative", "must", False, "no epochs")
    old_samples = authoritative_resp["sample"].to_numpy()[:n]
    diffs = np.abs(pos0[:n].astype(np.int64) - old_samples.astype(np.int64))
    n_outside = int(np.sum(diffs > SAMPLE_TOL))
    ok = n_outside == 0 and len(pos0) == len(authoritative_resp)
    return make_check(
        "fif_samples_match_authoritative",
        "must",
        ok,
        f"max |Δ|={int(diffs.max())} samples, median={float(np.median(diffs)):.1f}, "
        f"n_outside_tol({SAMPLE_TOL})={n_outside} "
        f"({len(pos0)} pos-0 epochs vs {len(authoritative_resp)} responses)",
    )


def check_epoch_t0_equals_response_onset(
    epochs, authoritative_resp: pd.DataFrame
) -> Check:
    """Predicate #7: pos-0 epoch t=0 (raw-sample time in seconds) equals
    `eventsOLD.tsv` response `onset` within ONSET_TOL_S."""
    pos0 = _pos0_samples(epochs)
    n = min(len(pos0), len(authoritative_resp))
    if n == 0:
        return make_check("epoch_t0_equals_response_onset", "must", False, "no epochs")
    fif_t0_s = pos0[:n] / ECOG_RAW_SFREQ
    onsets = authoritative_resp["onset"].to_numpy()[:n]
    diffs = np.abs(fif_t0_s - onsets)
    n_outside = int(np.sum(diffs > ONSET_TOL_S))
    ok = n_outside == 0
    return make_check(
        "epoch_t0_equals_response_onset",
        "must",
        ok,
        f"max |Δ|={diffs.max() * 1000:.2f} ms, median={np.median(diffs) * 1000:.2f} ms, "
        f"n_outside_tol({ONSET_TOL_S * 1000:.0f}ms)={n_outside}",
    )
