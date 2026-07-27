"""End-to-end on synthetic reduced sessions, where the right answer is known by construction.

The figures are the deliverable, so the failure that matters is a pipeline that draws a
confident picture of nothing. Each test plants a known structure (identical subjects,
independent subjects, a known ceiling) and checks the number that comes back.
"""
from __future__ import annotations

import numpy as np

from scripts.neuroprobe.viz_common import load_all
from scripts.neuroprobe.viz_figures import _corr, figure_a, figure_b, quantify

TASK = "onset"
BL = (4, 16, 32)
T, C = 32, 6
TEMPORAL = 30      # ctx-lh-superiortemporal -> temporal
FRONTAL = 22       # a frontal DKT id


def _write(dirpath, subject_id, trial_id, payload_by_parcel, *, halves=None, tap="enc12"):
    """payload_by_parcel: {parcel_id: (counts, (T, C) array)}."""
    parcels = sorted(payload_by_parcel)
    counts = np.array([payload_by_parcel[p][0] for p in parcels], dtype=np.int64)
    grand = np.stack([payload_by_parcel[p][1] for p in parcels]).astype(np.float32)
    out = {
        "subject_id": np.int64(subject_id), "trial_id": np.int64(trial_id),
        "present_parcels": np.asarray(parcels, dtype=np.int64), "parcel_counts": counts,
        "band_lengths": np.asarray(BL, dtype=np.int64), "n_windows": np.int64(100),
        f"{tap}/shape": np.asarray([len(parcels), T, C], dtype=np.int64),
        f"{tap}/col_sum": np.zeros((len(parcels), T * C), dtype=np.float32),
        f"{tap}/col_sq": np.ones((len(parcels), T * C), dtype=np.float32),
    }
    for cls in (0, 1):
        sign = 1.0 if cls == 1 else -1.0
        out[f"{tap}/{TASK}/c{cls}/all"] = (sign * grand).astype(np.float32)
        h = halves if halves is not None else (grand, grand)
        out[f"{tap}/{TASK}/c{cls}/h0"] = (sign * h[0]).astype(np.float32)
        out[f"{tap}/{TASK}/c{cls}/h1"] = (sign * h[1]).astype(np.float32)
        for name in ("all", "h0", "h1"):
            out[f"n/{TASK}/c{cls}/{name}"] = np.int64(50)
        out[f"count/{TASK}/c{cls}"] = np.int64(50)
    np.savez_compressed(str(dirpath / f"red_s{subject_id}_t{trial_id}_hga.npz"), **out)


def _pattern(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, C))


def test_identical_subjects_score_a_cross_subject_r_of_one(tmp_path) -> None:
    shared = _pattern(0)
    for s in (1, 2, 3):
        _write(tmp_path, s, 0, {TEMPORAL: (10, shared)})
    sessions = load_all(str(tmp_path))
    q = quantify(sessions, ["temporal"], "enc12", TASK, 1)
    assert q["n_sessions"] == 3 and q["n_cross_pairs"] == 3
    assert q["cross_subject_r"] > 0.999
    assert q["normalized"] > 0.999


def test_independent_subjects_score_near_zero(tmp_path) -> None:
    for s in (1, 2, 3, 4, 7, 10):
        _write(tmp_path, s, 0, {TEMPORAL: (10, _pattern(s))})
    sessions = load_all(str(tmp_path))
    q = quantify(sessions, ["temporal"], "enc12", TASK, 1)
    assert abs(q["cross_subject_r"]) < 0.35, q["cross_subject_r"]


def test_a_noisy_ceiling_normalizes_the_cross_subject_score_upward(tmp_path) -> None:
    """Half the subjects' signal is shared; the ceiling must scale the raw r, not be ignored."""
    shared = _pattern(0)
    rng = np.random.default_rng(99)
    for s in (1, 2, 3, 4):
        noise = rng.normal(size=(T, C))
        grand = shared + noise
        # halves disagree by construction -> ceiling well below 1
        _write(tmp_path, s, 0, {TEMPORAL: (10, grand)},
               halves=(grand + rng.normal(size=(T, C)), grand + rng.normal(size=(T, C))))
    sessions = load_all(str(tmp_path))
    q = quantify(sessions, ["temporal"], "enc12", TASK, 1)
    assert 0.0 < q["split_half_ceiling_r"] < 1.0
    assert q["normalized"] > q["cross_subject_r"], "normalizing by a <1 ceiling must raise r"


def test_within_subject_pairs_are_reported_separately_from_cross_subject(tmp_path) -> None:
    shared = _pattern(0)
    _write(tmp_path, 1, 0, {TEMPORAL: (10, shared)})
    _write(tmp_path, 1, 1, {TEMPORAL: (10, shared)})       # same subject, second trial
    _write(tmp_path, 2, 0, {TEMPORAL: (10, _pattern(5))})
    sessions = load_all(str(tmp_path))
    q = quantify(sessions, ["temporal"], "enc12", TASK, 1)
    assert q["n_cross_pairs"] == 2, "S1T0-S1T1 must NOT count as a cross-subject pair"
    assert q["within_subject_diff_session_r"] > 0.999


def test_figures_render_and_report_explained_variance(tmp_path) -> None:
    for s in (1, 2, 3, 4):
        _write(tmp_path, s, 0, {TEMPORAL: (10, _pattern(s)), FRONTAL: (5, _pattern(s + 50))})
    sessions = load_all(str(tmp_path))
    a = figure_a(sessions, ["temporal"], "enc12", TASK, str(tmp_path / "a.png"))
    b = figure_b(sessions, "enc12", TASK, 1, str(tmp_path / "b.png"))
    assert (tmp_path / "a.png").stat().st_size > 5000
    assert (tmp_path / "b.png").stat().st_size > 5000
    assert len(a["evr_raw"]) == 3 and len(a["evr_centered"]) == 3
    assert b["n_panels"] == 4
    assert 0 < sum(b["evr"]) <= 1.0 + 1e-9


def test_corr_is_scale_and_offset_invariant() -> None:
    x = np.random.default_rng(1).normal(size=64)
    assert _corr(x, 3 * x + 7) > 0.999
    assert _corr(x, -x) < -0.999
