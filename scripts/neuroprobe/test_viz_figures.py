"""End-to-end on synthetic reduced sessions, where the right answer is known by construction.

The figures are the deliverable, so the failure that matters is a pipeline that draws a
confident picture of nothing. Each test plants a known structure (identical subjects,
independent subjects, a known ceiling) and checks the number that comes back.
"""
from __future__ import annotations

import numpy as np

from scripts.neuroprobe.viz_common import load_all
from scripts.neuroprobe.viz_figures import (
    CONTRAST, _corr, figure_tasks, quantify, retrieval,
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



def test_corr_is_scale_and_offset_invariant() -> None:
    x = np.random.default_rng(1).normal(size=64)
    assert _corr(x, 3 * x + 7) > 0.999
    assert _corr(x, -x) < -0.999



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



def _shape_corpus(tmp_path, course):
    """Four subjects whose contrast is `course` (T,) times a fixed channel pattern."""
    patt = np.array([1.0, -0.6, 0.3, 0.8, -0.2, 0.5])
    d = course[:, None] * patt[None, :]
    profile = 20.0 * _pattern(0)
    rng = np.random.default_rng(11)
    for s in (1, 2, 3, 4):
        jit = d + 0.02 * rng.normal(size=d.shape)
        _write_classes(tmp_path, s, 0, TEMPORAL, 10,
                       {0: profile - 0.5 * jit, 1: profile + 0.5 * jit})
    return load_all(str(tmp_path))


def _course(plateau: float) -> np.ndarray:
    """Rise at frame 8, peak 2.0 at 12, decay by 20 to `plateau`, hold. Pre-stimulus is 0."""
    c = np.zeros(T)
    c[8:12] = np.linspace(0.0, 2.0, 4)
    c[12:20] = np.linspace(2.0, plateau, 8)
    c[20:] = plateau
    return c


def test_baseline_origin_separates_a_response_that_returns_from_one_that_settles(tmp_path):
    """The reason the origin moved, stated as a test.

    `onset` decays to nothing and `speech` decays to a plateau. Under the window's own
    time-average as origin, the plateau is folded INTO the origin, so the sustained response
    is drawn ending nearer its start than the transient one -- the ordering literally
    inverts, and both read as closed loops. Against a pre-stimulus baseline the ordering is
    the physical one. This is the bug the demo was showing.
    """
    from scripts.neuroprobe.viz_figures import peak_settle

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    trans = _shape_corpus(tmp_path / "a", _course(0.0))
    sust = _shape_corpus(tmp_path / "b", _course(1.0))
    lobes = ["temporal"]

    kw = dict(hz=32.0, offset=-0.25)
    base_t = peak_settle(trans, lobes, "enc12", [TASK], n_pre=8, **kw)[TASK]
    base_s = peak_settle(sust, lobes, "enc12", [TASK], n_pre=8, **kw)[TASK]
    time_t = peak_settle(trans, lobes, "enc12", [TASK], n_pre=None, **kw)[TASK]
    time_s = peak_settle(sust, lobes, "enc12", [TASK], n_pre=None, **kw)[TASK]

    # the built-in check: against a pre-stimulus baseline the pre-stimulus radius IS zero
    assert base_t["baseline_frac"] < 0.05 and base_s["baseline_frac"] < 0.05
    # physical ordering: the transient returns, the sustained one does not
    assert base_t["settle_frac"] < 0.15 < 0.35 < base_s["settle_frac"]
    # and the time-mean origin gets it backwards, which is why it had to go
    assert time_s["settle_frac"] < time_t["settle_frac"]


def test_loso_basis_keeps_shared_structure_and_kills_independent_structure(tmp_path):
    """The double-dip control. The pooled basis is fit on the very tokens it then scores, so
    the number has to survive refitting with both scored subjects held out."""
    from scripts.neuroprobe.viz_figures import align_loso

    (tmp_path / "shared").mkdir()
    (tmp_path / "indep").mkdir()
    shared = _shape_corpus(tmp_path / "shared", _course(1.0))
    profile = 20.0 * _pattern(0)
    rng = np.random.default_rng(3)
    for s in (1, 2, 3, 4):
        d = rng.normal(size=(T, C))
        _write_classes(tmp_path / "indep", s, 0, TEMPORAL, 10,
                       {0: profile - 0.5 * d, 1: profile + 0.5 * d})
    indep = load_all(str(tmp_path / "indep"))
    lobes = ["temporal"]

    assert align_loso(shared, lobes, "enc12", [TASK], n_pre=8)[TASK] > 0.9
    assert abs(align_loso(indep, lobes, "enc12", [TASK], n_pre=8)[TASK]) < 0.5


def test_splithalf_basis_scores_the_half_it_was_not_fit_on(tmp_path):
    """h0 fits the basis, h1 is scored in it, so nothing reported was fit on itself."""
    from scripts.neuroprobe.viz_figures import align_splithalf

    (tmp_path / "s").mkdir()
    sess = _shape_corpus(tmp_path / "s", _course(1.0))
    assert align_splithalf(sess, ["temporal"], "enc12", [TASK], n_pre=8)[TASK] > 0.9
