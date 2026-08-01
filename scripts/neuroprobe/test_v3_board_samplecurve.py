"""v3_board_samplecurve — the label-efficiency curve's contract (TDD).

The failure mode here is that a subsample harness prints a perfectly plausible curve while
quietly changing the fit: standardizing on the wrong rows, letting the two taps see different
trials, or drifting off the published protocol at the anchor. Every one of those produces numbers
that look exactly like correct ones. So the tests pin the things that make the curve MEAN
something:

  * the N=full point IS the published board cell (computed by calling the real ``_ws_cell``);
  * both taps are fit on the SAME drawn rows, which is what licenses the paired comparison;
  * subsampling train changes only the ROWS, never the standardization contract;
  * test is never subsampled;
  * the `full` sentinel cannot crash the x-axis ordering.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.neuroprobe.test_v3_board_readout import _rec
from scripts.neuroprobe.v3_board_readout import BOARD_TASKS, _finite
from scripts.neuroprobe.v3_board_samplecurve import (
    COLUMNS,
    FULL,
    N_GRID,
    SEEDS_FOR_N,
    ANCHOR_TOL,
    _anchor_check,
    _anchor_verdict,
    _curve,
    _digest,
    _reach,
    _rng,
    _strat_draw,
    _ws_curve_cell,
    _x_order,
)

TAPS = ("enc12", "enc0")          # the synthetic fixture's taps stand in for the elec pair
SESS = (1, 1)


def _big_rec(n=256, seed=0):
    """Fixture wide enough that several N_GRID points fit inside the train half."""
    return _rec(n=n, n_parcels=4, feat=8, seed=seed)


# ── INVARIANT 1: the anchor ────────────────────────────────────────────────────────────────────

def test_N_full_reproduces_the_published_ws_cell_exactly() -> None:
    """The whole run is licensed by this. If the curve's N=full point is not the board's own
    number, the subsample harness perturbed the fit and every other point inherits the drift."""
    rec = _big_rec()
    task = BOARD_TASKS[0]
    pts, _ = _ws_curve_cell(rec, SESS, task, TAPS)
    rows = _anchor_check(rec, task, TAPS, pts)
    assert rows, "no anchor rows produced — the check would vacuously pass"
    for r in rows:
        assert r["absdiff"] == 0.0, f"anchor drifted: {r}"


def test_anchor_verdict_raises_on_an_empty_row_list() -> None:
    """A guard that reports 'no violations' because it compared nothing is worse than no guard.
    This is the only anchor failure mode that could otherwise pass silently."""
    with pytest.raises(AssertionError, match="VACUOUS"):
        _anchor_verdict([])


def test_anchor_verdict_passes_exact_rows_and_catches_drifted_ones() -> None:
    ok = [{"absdiff": 0.0}, {"absdiff": ANCHOR_TOL / 2}]
    assert _anchor_verdict(ok) == []
    drift = {"absdiff": 1e-4}
    assert _anchor_verdict(ok + [drift]) == [drift]


def test_anchor_covers_every_tap_and_every_column() -> None:
    rec = _big_rec()
    task = BOARD_TASKS[0]
    pts, _ = _ws_curve_cell(rec, SESS, task, TAPS)
    got = {(r["tap"], r["col"]) for r in _anchor_check(rec, task, TAPS, pts)}
    assert got == {(t, c) for t in TAPS for c in COLUMNS}


def test_both_columns_are_identical_at_N_full() -> None:
    """At N=full nothing is subsampled, so 'subsample train+val' and 'subsample train only' are
    the same experiment. If they ever differ there, the val subset is not the full val."""
    rec = _big_rec()
    pts, _ = _ws_curve_cell(rec, SESS, BOARD_TASKS[0], TAPS)
    full = [p for p in pts if p["n_is_full"]]
    assert full
    for tap in TAPS:
        for fold in {p["fold"] for p in full}:
            v = {p["col"]: p["test"] for p in full if p["tap"] == tap and p["fold"] == fold}
            assert v["both"] == v["trainonly"]


# ── INVARIANT 2: the pairing ───────────────────────────────────────────────────────────────────

def test_both_taps_are_fit_on_the_same_drawn_rows(monkeypatch) -> None:
    """The paired enc12-vs-enc0 comparison is void if the taps see different trials. The draw is
    made once outside the tap loop; this pins that it stays that way by recording the exact rows
    every gather was handed."""
    import scripts.neuroprobe.v3_board_samplecurve as M

    real = M._feat
    calls: list = []

    def spy(rec, enc, rows, col_idx=None):
        calls.append((enc, _digest(rows)))
        return real(rec, enc, rows, col_idx)

    monkeypatch.setattr(M, "_feat", spy)
    _ws_curve_cell(_big_rec(), SESS, BOARD_TASKS[0], TAPS)

    by_tap: dict = {}
    for enc, dig in calls:
        by_tap.setdefault(enc, []).append(dig)
    assert set(by_tap) == set(TAPS)
    assert by_tap[TAPS[0]] == by_tap[TAPS[1]], "the two taps were gathered on different rows"


def test_draw_is_deterministic_for_the_same_cell_key() -> None:
    rec = _big_rec()
    y = np.asarray(rec["labels"][BOARD_TASKS[0]], dtype=np.float64)
    tr = _finite(y, rec["ws_split"][BOARD_TASKS[0]][0]["train"])
    a = _strat_draw(y, tr, 16, _rng(SESS, "onset", 0, 16, 0))
    b = _strat_draw(y, tr, 16, _rng(SESS, "onset", 0, 16, 0))
    assert _digest(a) == _digest(b)


def test_different_seeds_give_different_draws() -> None:
    rec = _big_rec()
    y = np.asarray(rec["labels"][BOARD_TASKS[0]], dtype=np.float64)
    tr = _finite(y, rec["ws_split"][BOARD_TASKS[0]][0]["train"])
    a = _strat_draw(y, tr, 16, _rng(SESS, "onset", 0, 16, 0))
    b = _strat_draw(y, tr, 16, _rng(SESS, "onset", 0, 16, 1))
    assert _digest(a) != _digest(b)


def test_draw_seed_depends_on_the_session_so_shards_do_not_share_draws() -> None:
    rec = _big_rec()
    y = np.asarray(rec["labels"][BOARD_TASKS[0]], dtype=np.float64)
    tr = _finite(y, rec["ws_split"][BOARD_TASKS[0]][0]["train"])
    a = _strat_draw(y, tr, 16, _rng((1, 1), "onset", 0, 16, 0))
    b = _strat_draw(y, tr, 16, _rng((3, 0), "onset", 0, 16, 0))
    assert _digest(a) != _digest(b)


# ── INVARIANT 3: stratification ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_every_draw_keeps_both_classes(n) -> None:
    """An unstratified draw can come back single-class, and auroc() returns NaN on a single-class
    eval half — which would delete the hardest points of the curve instead of reporting them."""
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.12).astype(np.float64)     # deliberately skewed
    rows = np.arange(400)
    for seed in range(20):
        d = _strat_draw(y, rows, n, np.random.default_rng(seed))
        assert len(d) == n
        assert 0 < (y[d] > 0).sum() < n, f"draw is single-class at n={n}, seed={seed}"


def test_draw_tracks_the_parent_class_balance() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.3).astype(np.float64)
    rows = np.arange(2000)
    for n in (64, 128, 256):
        d = _strat_draw(y, rows, n, np.random.default_rng(1))
        assert abs((y[d] > 0).mean() - 0.3) < 0.02


def test_draw_is_sorted_and_a_subset_of_its_parent() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.5).astype(np.float64)
    rows = np.arange(50, 150)
    d = _strat_draw(y, rows, 32, np.random.default_rng(3))
    assert np.all(np.diff(d) > 0)
    assert set(d.tolist()) <= set(rows.tolist())


def test_full_draw_is_the_parent_element_for_element() -> None:
    """Asked for >= everything, the draw must be the parent ARRAY, not a reshuffle of it — that
    identity is what makes the N=full anchor bit-comparable to the published cell."""
    y = np.array([float(i % 2) for i in range(100)])
    rows = np.arange(100)
    assert np.array_equal(_strat_draw(y, rows, 100, np.random.default_rng(0)), rows)
    assert np.array_equal(_strat_draw(y, rows, 500, np.random.default_rng(0)), rows)


def test_contiguous_takes_a_prefix_not_a_random_subset() -> None:
    y = np.array([float(i % 2) for i in range(200)])
    rows = np.arange(200)
    d = _strat_draw(y, rows, 20, np.random.default_rng(0), contiguous=True)
    assert d.max() < 40, "a contiguous draw must come from the FRONT of the parent"


# ── the readout contract is not touched ────────────────────────────────────────────────────────

def test_test_half_is_never_subsampled() -> None:
    """Calibration budget is a TRAIN-side quantity. A shrinking test half would make the small-N
    points noisier for a reason that has nothing to do with the claim."""
    rec = _big_rec()
    task = BOARD_TASKS[0]
    _, census = _ws_curve_cell(rec, SESS, task, TAPS)
    y = np.asarray(rec["labels"][task], dtype=np.float64)
    for fold, sp in rec["ws_split"][task].items():
        want = len(_finite(y, sp["test"]))
        for c in census:
            if c["fold"] == fold:
                assert c["n_test"] == want


def test_val_shrinks_with_train_in_the_both_column_but_not_at_full() -> None:
    rec = _big_rec()
    task = BOARD_TASKS[0]
    _, census = _ws_curve_cell(rec, SESS, task, TAPS)
    small = [c for c in census if c["n"] == min(N_GRID) and not c["n_is_full"]]
    full = [c for c in census if c["n_is_full"]]
    assert small and full
    assert max(c["n_val"] for c in small) < min(c["n_val"] for c in full)


def test_smaller_N_never_silently_drops_a_task() -> None:
    """A NaN point must still BE a point. Dropping it would make the curve's left end look like
    it was measured on the same cells as its right end when it was not."""
    rec = _big_rec()
    task = BOARD_TASKS[0]
    pts, _ = _ws_curve_cell(rec, SESS, task, TAPS)
    per_n = {}
    for p in pts:
        per_n.setdefault(p["n"] if not p["n_is_full"] else FULL, set()).add((p["tap"], p["col"]))
    assert len({frozenset(v) for v in per_n.values()}) == 1, "tap x column coverage varies with N"


def test_seed_count_follows_the_declared_taper() -> None:
    rec = _big_rec(n=2048)
    task = BOARD_TASKS[0]
    pts, _ = _ws_curve_cell(rec, SESS, task, TAPS)
    for n in N_GRID:
        got = {p["seed"] for p in pts if p["n"] == n and not p["n_is_full"]}
        if got:
            assert len(got) == SEEDS_FOR_N[n], f"N={n} ran {len(got)} seeds, declared {SEEDS_FOR_N[n]}"


def test_signal_fixture_reaches_the_ceiling_by_the_full_point() -> None:
    """Sanity that the harness fits anything at all: the fixture's first column IS the label, so
    a working readout must be at AUROC 1.0 by N=full."""
    rec = _big_rec()
    pts, _ = _ws_curve_cell(rec, SESS, BOARD_TASKS[0], TAPS)
    full = [p["test"] for p in pts if p["n_is_full"] and p["tap"] == "enc12"]
    assert np.nanmean(full) == pytest.approx(1.0)


def test_pure_noise_stays_near_chance_at_every_N() -> None:
    """The opposite guard: with no signal the curve must not climb. A subsample harness that
    leaked test rows into train would show a rising noise curve."""
    rec = _rec(n=256, signal=False, seed=7)
    pts, _ = _ws_curve_cell(rec, SESS, BOARD_TASKS[0], TAPS)
    for n in {p["n"] for p in pts}:
        v = [p["test"] for p in pts if p["n"] == n and p["tap"] == "enc12"]
        assert np.nanmean(v) < 0.9, f"noise fixture scores {np.nanmean(v):.3f} at N={n}"


# ── reporting helpers ──────────────────────────────────────────────────────────────────────────

def test_x_order_puts_full_last_and_never_compares_str_to_int() -> None:
    keys = [FULL, 256, 16, 1024]
    assert sorted(keys, key=_x_order) == [16, 256, 1024, FULL]


def test_curve_would_crash_without_x_order() -> None:
    """Pins WHY _x_order exists: bare sorted() over the mixed grid is a TypeError, so a future
    edit that drops it fails here instead of at 3am in the merge step."""
    with pytest.raises(TypeError):
        sorted([FULL, 16, 256])


def test_reach_interpolates_in_log2_between_bracketing_points() -> None:
    curve = {16: 0.50, 32: 0.60, 64: 0.70}
    # target 0.65 sits halfway between the 32 and 64 points ⇒ 2^(5 + .5) = 45.25
    assert _reach(curve, 0.65) == pytest.approx(2.0 ** 5.5, rel=1e-6)


def test_reach_returns_None_when_the_curve_never_gets_there() -> None:
    """'Did not reach' and 'reached at the last point' are different results; clipping one to the
    other would manufacture a Tier-1 ratio out of a curve that has no crossing."""
    assert _reach({16: 0.5, 32: 0.55}, 0.9) is None


def test_reach_takes_the_FIRST_crossing_not_the_best_point() -> None:
    curve = {16: 0.80, 32: 0.60, 64: 0.85}
    assert _reach(curve, 0.75) == 16.0


def test_curve_aggregates_task_then_cell_not_a_flat_mean() -> None:
    """Board convention: mean over the 15 tasks of the cohort mean. A flat mean would let a task
    with more folds or more seeds weigh more than another."""
    pts = [
        {"tap": "t", "col": "both", "n_bucket": 16, "task": "a", "cell": "S1", "test": 1.0},
        {"tap": "t", "col": "both", "n_bucket": 16, "task": "a", "cell": "S1", "test": 1.0},
        {"tap": "t", "col": "both", "n_bucket": 16, "task": "a", "cell": "S1", "test": 1.0},
        {"tap": "t", "col": "both", "n_bucket": 16, "task": "b", "cell": "S1", "test": 0.0},
    ]
    assert _curve(pts, "t", "both")[16] == pytest.approx(0.5)   # not 0.75


def test_curve_ignores_other_taps_and_columns() -> None:
    pts = [
        {"tap": "t", "col": "both", "n_bucket": 16, "task": "a", "cell": "S1", "test": 1.0},
        {"tap": "u", "col": "both", "n_bucket": 16, "task": "a", "cell": "S1", "test": 0.0},
        {"tap": "t", "col": "trainonly", "n_bucket": 16, "task": "a", "cell": "S1", "test": 0.0},
    ]
    assert _curve(pts, "t", "both")[16] == pytest.approx(1.0)
