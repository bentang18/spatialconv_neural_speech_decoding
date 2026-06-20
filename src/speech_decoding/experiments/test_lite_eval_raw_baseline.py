"""Tests for the Neuroprobe-Lite raw logistic baseline's pure eval-mode loops.

Only the three eval-mode loops are laptop-testable (the per-cell materializer needs
BT voltage + the lite spec cache). They run on synthetic per-cell ``SubjectProbeData``
shaped like the real lite cells: this checks the emitted keys/ranges, determinism, the
CrossSession per-electrode mask intersection (guard-1 STATIC is per-session), and that
CrossSubject excludes same-subject test cells (it must stay cross-subject).
"""

from __future__ import annotations

import numpy as np
import torch

from speech_decoding.experiments.lite_eval_raw_baseline import (
    n_parcels_of,
    run_cross_session,
    run_cross_subject,
    run_lite_eval_raw_baseline,
    run_within_session,
)
from speech_decoding.experiments.online_probe import SubjectProbeData
from speech_decoding.models.v14_converged import BANDS


def _cell(subject_id: int, trial_id: int, n: int, c: int, n_parcels: int, *, seed: int):
    g = torch.Generator().manual_seed(seed)
    half = np.array([1.0, -1.0] * (n // 2), dtype=np.float64)[:n]
    labels = {"delta_volume": half.copy()}
    slow = torch.randn(n, c, 2, BANDS[0].n_freq_bins, BANDS[0].n_time_frames, generator=g)
    slow[:, :, 0] += torch.tensor(half, dtype=torch.float32)[:, None, None, None]
    beta = torch.randn(n, c, BANDS[1].n_freq_bins, BANDS[1].n_time_frames, generator=g).abs()
    hg = torch.randn(n, c, BANDS[2].n_freq_bins, BANDS[2].n_time_frames, generator=g).abs()
    ppe = torch.arange(c) % n_parcels
    mask = torch.ones(c, dtype=torch.bool)
    return SubjectProbeData(
        subject_id=subject_id, slow=slow, beta=beta, hg=hg,
        parcel_per_electrode=ppe, electrode_mask=mask,
        labels=labels, sessions=[(subject_id, trial_id)],
    )


def _cells():
    # Two subjects, two trials each (mirrors BT_LITE pairing); plus the (2,4) anchor.
    np_ = 4
    return {
        (2, 4): _cell(2, 4, 8, 6, np_, seed=0),
        (2, 0): _cell(2, 0, 8, 6, np_, seed=1),
        (3, 0): _cell(3, 0, 8, 5, np_, seed=2),
        (3, 1): _cell(3, 1, 8, 5, np_, seed=3),
    }


def test_within_session_keys_and_ranges():
    cells = _cells()
    summary, per_cell = run_within_session(cells, tasks=["delta_volume"])
    assert set(summary) == {"val_lite/raw/within_session/delta_volume"}
    v = summary["val_lite/raw/within_session/delta_volume"]
    assert np.isnan(v) or 0.0 <= v <= 1.0
    # one entry per cell
    assert set(per_cell) == {(s, t, "delta_volume") for (s, t) in cells}


def test_cross_session_both_directions():
    cells = _cells()
    summary, per_cell = run_cross_session(cells, tasks=["delta_volume"])
    assert set(summary) == {"val_lite/raw/cross_session/delta_volume"}
    # subject 2: (4->0) and (0->4); subject 3: (0->1) and (1->0) => 4 directed pairs
    assert set(per_cell) == {
        (2, 4, 0, "delta_volume"), (2, 0, 4, "delta_volume"),
        (3, 0, 1, "delta_volume"), (3, 1, 0, "delta_volume"),
    }


def test_cross_session_mask_intersection_aligns_dims():
    # Different per-session bad electrodes (guard-1 is per-session): the per-electrode
    # CS fit must use mask_a & mask_b so the two feature matrices share columns. If it
    # used each cell's own mask the dims would disagree and sklearn would raise.
    cells = _cells()
    a = cells[(2, 4)]
    b = cells[(2, 0)]
    a.electrode_mask[1] = False   # drop a different electrode in each trial
    b.electrode_mask[4] = False
    summary, _ = run_cross_session(
        {(2, 4): a, (2, 0): b}, tasks=["delta_volume"]
    )
    v = summary["val_lite/raw/cross_session/delta_volume"]
    assert np.isnan(v) or 0.0 <= v <= 1.0   # ran without a shape mismatch


def test_cross_subject_excludes_same_subject():
    cells = _cells()
    summary, per_cell = run_cross_subject(
        cells, tasks=["delta_volume"], n_parcels=n_parcels_of(cells), train_cell=(2, 4)
    )
    assert set(summary) == {"val_lite/raw/cross_subject/delta_volume"}
    # train cell is (2,*); only subject-3 cells may appear as test
    assert set(per_cell) == {(3, 0, "delta_volume"), (3, 1, "delta_volume")}


def test_full_baseline_keys_and_determinism():
    cells = _cells()
    a = run_lite_eval_raw_baseline(cells, tasks=["delta_volume"])
    b = run_lite_eval_raw_baseline(cells, tasks=["delta_volume"])
    assert set(a["metrics"]) == {
        "val_lite/raw/within_session/delta_volume",
        "val_lite/raw/cross_session/delta_volume",
        "val_lite/raw/cross_subject/delta_volume",
    }
    assert a["metrics"] == b["metrics"]
    assert a["n_cells"] == 4 and a["train_cell"] == [2, 4]
    assert set(a["per_cell"]) == {"within_session", "cross_session", "cross_subject"}
