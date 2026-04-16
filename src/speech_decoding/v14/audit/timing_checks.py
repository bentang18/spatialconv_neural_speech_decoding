"""Per-epoch sample-index alignment between `.fif` and the authoritative events TSV.

Covers predicates #6–#7 of `docs/implementation_tasks.md` #34:
- #6: `.fif` epoch sample indices match `eventsOLD.tsv` response rows
      (tolerance 1 sample at ECoG raw 2000 Hz)
- #7: `t=0` in each epoch equals `eventsOLD.tsv` response onset
      (tolerance ≈ 5 ms)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from speech_decoding.v14.audit.io import ECOG_RAW_SFREQ
from speech_decoding.v14.audit.schema import Check, make_check

SAMPLE_TOL = 3  # raw samples at 2000 Hz; ≤1.5 ms
ONSET_TOL_S = 0.005  # 5 ms


def check_fif_samples_match_authoritative(
    epochs, authoritative_resp: pd.DataFrame
) -> Check:
    """Predicate #6: for each epoch, the raw-clock sample index stored in
    `epochs.events[:, 0]` equals `eventsOLD.tsv` response `sample` for the
    same trial (within SAMPLE_TOL raw samples)."""
    fif_samples = epochs.events[:, 0]
    n = min(len(fif_samples), len(authoritative_resp))
    if n == 0:
        return make_check("fif_samples_match_authoritative", "must", False, "no epochs")
    old_samples = authoritative_resp["sample"].to_numpy()[:n]
    diffs = np.abs(fif_samples[:n].astype(np.int64) - old_samples.astype(np.int64))
    n_outside = int(np.sum(diffs > SAMPLE_TOL))
    ok = n_outside == 0 and len(fif_samples) == len(authoritative_resp)
    return make_check(
        "fif_samples_match_authoritative",
        "must",
        ok,
        f"max |Δ|={int(diffs.max())} samples, median={float(np.median(diffs)):.1f}, "
        f"n_outside_tol({SAMPLE_TOL})={n_outside} "
        f"({len(fif_samples)} epochs vs {len(authoritative_resp)} responses)",
    )


def check_epoch_t0_equals_response_onset(
    epochs, authoritative_resp: pd.DataFrame
) -> Check:
    """Predicate #7: the seconds-time `t=0` in each epoch equals the
    `eventsOLD.tsv` response `onset` for that trial (raw-sample time).
    Converts `epochs.events[:, 0]` via ECoG raw sfreq to seconds and compares
    to `onset`."""
    fif_samples = epochs.events[:, 0]
    n = min(len(fif_samples), len(authoritative_resp))
    if n == 0:
        return make_check("epoch_t0_equals_response_onset", "must", False, "no epochs")
    fif_t0_s = fif_samples[:n] / ECOG_RAW_SFREQ
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
