"""End-to-end on synthetic reduced sessions, where the right answer is known by construction.

The figures are the deliverable, so the failure that matters is a pipeline that draws a
confident picture of nothing. Each test plants a known structure (identical subjects,
independent subjects, a known ceiling) and checks the number that comes back.
"""
from __future__ import annotations

import numpy as np

from scripts.neuroprobe.viz_common import load_all
from scripts.neuroprobe.viz_figures import (
    CONTRAST, _corr, figure_a, figure_b, figure_tasks, identity_content, quantify,
    retrieval, unit_scale,
)

TASK = "onset"
BL = (4, 16, 32)
T, C = 32, 6
TEMPORAL = 30      # ctx-lh-superiortemporal -> temporal
FRONTAL = 22       # a frontal DKT id


def _write(dirpath, subject_id, trial_id, payload_by_parcel, *, halves=None, tap="enc12",
           offset=None):
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
        # NOT -1.0: an exact negation makes every session mean identically zero, which
        # would erase the between-session variance the identity split is meant to find.
        sign = 1.0 if cls == 1 else -0.5
        # identity offset is added AFTER the class sign: a session constant that does not
        # flip with condition, which is what "identity" means here.
        off = 0.0 if offset is None else offset
        out[f"{tap}/{TASK}/c{cls}/all"] = (sign * grand + off).astype(np.float32)
        h = halves if halves is not None else (grand, grand)
        out[f"{tap}/{TASK}/c{cls}/h0"] = (sign * h[0] + off).astype(np.float32)
        out[f"{tap}/{TASK}/c{cls}/h1"] = (sign * h[1] + off).astype(np.float32)
        for name in ("all", "h0", "h1"):
            out[f"n/{TASK}/c{cls}/{name}"] = np.int64(50)
        out[f"count/{TASK}/c{cls}"] = np.int64(50)
    np.savez_compressed(str(dirpath / f"red_s{subject_id}_t{trial_id}_hga.npz"), **out)


def _write_classes(dirpath, subject_id, trial_id, parcel, counts, per_class, *, tap="enc12"):
    """Write one session with EXPLICIT per-class (T, C) means. halves == the class mean."""
    out = {
        "subject_id": np.int64(subject_id), "trial_id": np.int64(trial_id),
        "present_parcels": np.asarray([parcel], dtype=np.int64),
        "parcel_counts": np.asarray([counts], dtype=np.int64),
        "band_lengths": np.asarray(BL, dtype=np.int64), "n_windows": np.int64(100),
        f"{tap}/shape": np.asarray([1, T, C], dtype=np.int64),
        f"{tap}/col_sum": np.zeros((1, T * C), dtype=np.float32),
        f"{tap}/col_sq": np.ones((1, T * C), dtype=np.float32),
    }
    for cls in (0, 1):
        m = per_class[cls][None].astype(np.float32)          # (1, T, C)
        for name in ("all", "h0", "h1"):
            out[f"{tap}/{TASK}/c{cls}/{name}"] = m
            out[f"n/{TASK}/c{cls}/{name}"] = np.int64(50)
        out[f"count/{TASK}/c{cls}"] = np.int64(50)
    np.savez_compressed(str(dirpath / f"red_s{subject_id}_t{trial_id}_hga.npz"), **out)


def test_contrast_ignores_a_shared_condition_independent_profile(tmp_path) -> None:
    """The bug this metric exists to catch.

    A huge response profile shared across subjects but IDENTICAL for both classes carries
    single-class correlations to ~1 for every task, decodable or not. The class contrast
    must see through it: here the class difference is independent per subject, so the
    honest answer is ~0.
    """
    shared_profile = 20.0 * _pattern(0)
    rng = np.random.default_rng(3)
    for s in (1, 2, 3, 4, 7, 10):
        diff = rng.normal(size=(T, C))                       # subject-specific, unshared
        _write_classes(tmp_path, s, 0, TEMPORAL, 10,
                       {0: shared_profile - 0.5 * diff, 1: shared_profile + 0.5 * diff})
    sessions = load_all(str(tmp_path))
    single = quantify(sessions, ["temporal"], "enc12", TASK, 1)
    diff_q = quantify(sessions, ["temporal"], "enc12", TASK, CONTRAST)
    assert single["cross_subject_r"] > 0.9, "fixture must reproduce the inflated single-class r"
    assert abs(diff_q["cross_subject_r"]) < 0.35, diff_q["cross_subject_r"]


def test_contrast_recovers_a_genuinely_shared_class_difference(tmp_path) -> None:
    shared_profile = 20.0 * _pattern(0)
    shared_diff = _pattern(1)
    for s in (1, 2, 3, 4):
        _write_classes(tmp_path, s, 0, TEMPORAL, 10,
                       {0: shared_profile - 0.5 * shared_diff,
                        1: shared_profile + 0.5 * shared_diff})
    sessions = load_all(str(tmp_path))
    q = quantify(sessions, ["temporal"], "enc12", TASK, CONTRAST)
    assert q["cross_subject_r"] > 0.999, q["cross_subject_r"]


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
    assert q["ceiling_usable"]
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


def test_identity_content_finds_a_planted_identity_offset(tmp_path) -> None:
    """Shared content plus a big per-subject offset: identity must dominate the variance,
    and removing it must RAISE cross-subject alignment rather than destroy it."""
    shared = _pattern(0)
    rng = np.random.default_rng(7)
    for s in (1, 2, 3, 4):
        offset = 20.0 * rng.normal(size=(1, C))       # constant per subject, huge
        _write(tmp_path, s, 0, {TEMPORAL: (10, shared)}, offset=offset)
    sessions = load_all(str(tmp_path))
    ic = identity_content(sessions, ["temporal"], "enc12", TASK)
    assert ic["identity_var_frac"] > 0.8, ic["identity_var_frac"]
    assert ic["cross_subject_r_identity_removed"] >= ic["cross_subject_r"] - 1e-6


def test_identity_content_reports_a_small_identity_share_when_there_is_none(tmp_path) -> None:
    shared = _pattern(0)
    for s in (1, 2, 3, 4):
        _write(tmp_path, s, 0, {TEMPORAL: (10, shared)})   # identical, no offsets
    sessions = load_all(str(tmp_path))
    ic = identity_content(sessions, ["temporal"], "enc12", TASK)
    assert ic["identity_var_frac"] < 0.02, ic["identity_var_frac"]
    assert ic["identity_rank"] == 0, "no session offsets -> no identity directions"
    # the two shares are a decomposition, so they must add to one
    assert abs(ic["identity_var_frac"] + ic["within_session_var_frac"] - 1.0) < 1e-6


def test_a_degenerate_ceiling_suppresses_the_normalized_score(tmp_path) -> None:
    """No within-subject reliability -> no denominator. Must be nan, not a big number."""
    rng = np.random.default_rng(11)
    for s in (1, 2, 3, 4):
        _write_classes(tmp_path, s, 0, TEMPORAL, 10,
                       {0: rng.normal(size=(T, C)), 1: rng.normal(size=(T, C))})
        # overwrite the halves with independent noise: the subject disagrees with itself
        f = tmp_path / f"red_s{s}_t0_hga.npz"
        d = dict(np.load(f))
        for cls in (0, 1):
            for name in ("h0", "h1"):
                d[f"enc12/{TASK}/c{cls}/{name}"] = rng.normal(size=(1, T, C)).astype(np.float32)
        np.savez_compressed(str(f), **d)
    sessions = load_all(str(tmp_path))
    q = quantify(sessions, ["temporal"], "enc12", TASK, CONTRAST)
    assert abs(q["split_half_ceiling_r"]) < 0.5
    assert not q["ceiling_usable"]
    assert np.isnan(q["normalized"])


def test_retrieval_is_at_chance_when_subjects_share_nothing(tmp_path) -> None:
    rng = np.random.default_rng(4)
    for s in (1, 2, 3, 4):
        _write_classes(tmp_path, s, 0, TEMPORAL, 10,
                       {0: rng.normal(size=(T, C)), 1: rng.normal(size=(T, C))})
    sessions = load_all(str(tmp_path))
    r = retrieval(sessions, ["temporal"], "enc12", TASK)
    assert r["chance"] == 1.0 / T
    # median rank near the middle of the list is what "no temporal identity" looks like
    assert r["median_rank"] > T * 0.2, r["median_rank"]


def test_retrieval_is_perfect_when_the_contrast_is_shared(tmp_path) -> None:
    shared_diff = _pattern(2)
    profile = 20.0 * _pattern(0)
    for s in (1, 2, 3, 4):
        _write_classes(tmp_path, s, 0, TEMPORAL, 10,
                       {0: profile - 0.5 * shared_diff, 1: profile + 0.5 * shared_diff})
    sessions = load_all(str(tmp_path))
    r = retrieval(sessions, ["temporal"], "enc12", TASK)
    assert r["top1"] > 0.99, r["top1"]
    assert r["median_rank"] == 0.0


def test_unit_scale_removes_amplitude_but_not_shape() -> None:
    class _S:
        key = "S1T0"
    m = np.arange(12, dtype=float).reshape(1, 4, 3)
    (_, small), (_, big) = unit_scale([(_S(), m), (_S(), 100.0 * m)])
    assert abs(np.linalg.norm(small) - 1.0) < 1e-9
    assert np.allclose(small, big), "shape must survive; only the scale is removed"


def test_figure_tasks_keeps_a_dead_task_small_relative_to_a_live_one(tmp_path) -> None:
    """Per-session scaling, not per-task: a task with no signal must NOT be renormalized
    back up to the size of a real one -- that is the entire point of the panel."""
    live = _pattern(1)
    profile = 20.0 * _pattern(0)
    rng = np.random.default_rng(8)
    for s in (1, 2, 3, 4):
        out = {}
        for cls in (0, 1):
            sign = 1.0 if cls == 1 else -1.0
            out[cls] = profile + sign * 0.5 * live
        _write_classes(tmp_path, s, 0, TEMPORAL, 10, out)
        # add a second, dead task to the same file
        f = tmp_path / f"red_s{s}_t0_hga.npz"
        d = dict(np.load(f))
        for cls in (0, 1):
            for name in ("all", "h0", "h1"):
                d[f"enc12/dead/c{cls}/{name}"] = (
                    profile + 0.001 * rng.normal(size=(T, C)))[None].astype(np.float32)
                d[f"n/dead/c{cls}/{name}"] = np.int64(50)
            d[f"count/dead/c{cls}"] = np.int64(50)
        np.savez_compressed(str(f), **d)
    sessions = load_all(str(tmp_path))
    info = figure_tasks(sessions, ["temporal"], "enc12", [TASK, "dead"],
                        str(tmp_path / "t.png"))
    assert info["align_3pc"][TASK] > 0.9
    assert abs(info["align_3pc"]["dead"]) < 0.5


def test_figure_depth_reports_the_slope_of_the_rows_it_was_given(tmp_path) -> None:
    """The ladder must draw the numbers the quant/retrieval passes produced, not its own.
    If it ever recomputed, the figure and the printed table could quietly diverge."""
    from scripts.neuroprobe.viz_figures import figure_depth
    taps = ["enc0", "enc6", "enc12"]
    retr = [{"tap": t, "task": "onset", "top1": v, "chance": 1 / 32}
            for t, v in zip(taps, (0.10, 0.18, 0.26))]
    retr += [{"tap": t, "task": "frame_brightness", "top1": v, "chance": 1 / 32}
             for t, v in zip(taps, (0.03, 0.03, 0.04))]
    quant = [{"tap": t, "task": "onset", "class": CONTRAST, "normalized": v}
             for t, v in zip(taps, (0.2, 0.4, 0.53))]
    info = figure_depth(quant, retr, taps, ["onset", "frame_brightness"],
                        str(tmp_path / "d.png"))
    assert abs(info["top1_first_to_last"]["onset"] - 0.16) < 1e-9
    assert abs(info["top1_first_to_last"]["frame_brightness"] - 0.01) < 1e-9
    assert info["chance"] == 1 / 32 and info["n_tasks"] == 2


def test_figure_depth_tolerates_a_tap_with_no_row(tmp_path) -> None:
    from scripts.neuroprobe.viz_figures import figure_depth
    taps = ["enc0", "enc6", "enc12"]
    retr = [{"tap": "enc0", "task": "onset", "top1": 0.1, "chance": 1 / 32},
            {"tap": "enc12", "task": "onset", "top1": 0.3, "chance": 1 / 32}]
    info = figure_depth([], retr, taps, ["onset"], str(tmp_path / "d.png"))
    assert abs(info["top1_first_to_last"]["onset"] - 0.2) < 1e-9


def test_figure_identity_reports_the_cost_of_removing_identity(tmp_path) -> None:
    from scripts.neuroprobe.viz_figures import figure_identity
    rows = [{"tap": "enc0", "task": TASK, "identity_rank": 3, "n_sessions": 4,
             "identity_var_frac": 0.7, "cross_subject_r": 0.20,
             "cross_subject_r_identity_removed": 0.26},
            {"tap": "enc12", "task": TASK, "identity_rank": 3, "n_sessions": 4,
             "identity_var_frac": 0.6, "cross_subject_r": 0.42,
             "cross_subject_r_identity_removed": 0.40}]
    info = figure_identity(rows, str(tmp_path / "i.png"))
    assert info["taps"] == ["enc0", "enc12"]
    np.testing.assert_allclose(info["delta_r_after_removal"], [0.06, -0.02], atol=1e-9)


def test_figure_identity_skips_rows_with_no_variance_split(tmp_path) -> None:
    from scripts.neuroprobe.viz_figures import figure_identity
    assert figure_identity([{"tap": "enc0", "identity_var_frac": float("nan")}],
                           str(tmp_path / "i.png")) == {}
