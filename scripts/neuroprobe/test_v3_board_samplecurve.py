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
from scripts.neuroprobe.v3_board_readout import BOARD_TASKS, CS_TRAIN_ANCHOR, _finite
from scripts.neuroprobe.v3_board_samplecurve import (
    COLUMNS,
    CURVE_TAPS,
    FULL,
    N_GRID,
    N_GRID_CS,
    SEEDS_FOR_N,
    SHARD_PREFIX,
    ANCHOR_TOL,
    _addmult,
    _anchor_check,
    _anchor_verdict,
    _cs_anchor_check,
    _cs_curve_cell,
    _csession_anchor_check,
    _csession_curve_cell,
    _curve,
    _digest,
    _merge,
    _reach,
    _rng,
    _strat_draw,
    _transfer_cols,
    _transfer_curve_cell,
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


def test_curve_runs_and_anchor_holds_when_the_val_split_is_UNSORTED() -> None:
    """REGRESSION, found on real data (20715642): `va` is _finite(y, split["val"]) and carries the
    cached split's own order, which is NOT guaranteed ascending. The board readout never noticed
    because it only uses va via _feat(rec, enc, va) and y[va] -- both order-agnostic. An earlier
    np.searchsorted here silently required sorted input and blew up the subset assert.

    The synthetic fixture builds val with np.arange, so ONLY an explicitly shuffled split covers it.
    """
    rec = _big_rec()
    rng = np.random.default_rng(0)
    for task in rec["ws_split"]:
        for fold in rec["ws_split"][task]:
            v = rec["ws_split"][task][fold]["val"].copy()
            rng.shuffle(v)
            assert not np.all(np.diff(v) > 0), "shuffle left val sorted — test would be vacuous"
            rec["ws_split"][task][fold]["val"] = v

    task = BOARD_TASKS[0]
    pts, _ = _ws_curve_cell(rec, SESS, task, TAPS)
    assert pts, "unsorted val produced no points"
    rows = _anchor_check(rec, task, TAPS, pts)
    assert _anchor_verdict(rows) == [], f"anchor drifted under an unsorted val split: {rows}"


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


# ── CROSS-SUBJECT MODE ─────────────────────────────────────────────────────────────────────────
#
# CS asks a DIFFERENT question with the same machinery, and the ways it can go quietly wrong are
# its own: subsampling the target's val (which would answer a calibration question the benchmark
# does not pose), keying the donor draw on the test cell (which would silently average over 10x
# more donor subsamples than the claim is about), or letting the anchor drift off `_cs_cell`.

def test_cs_N_full_reproduces_the_published_cs_cell_exactly() -> None:
    """The CS licence, same as the WS one: computed by calling the REAL `_cs_cell`."""
    a_rec, t_rec = _big_rec(seed=0), _big_rec(seed=1)
    task = BOARD_TASKS[0]
    pts, _ = _cs_curve_cell(a_rec, t_rec, task, TAPS)
    rows = _cs_anchor_check(a_rec, t_rec, task, TAPS, pts)
    assert rows, "no anchor rows produced — the check would vacuously pass"
    for r in rows:
        assert r["absdiff"] == 0.0, f"cs anchor drifted: {r}"


def test_cs_never_subsamples_the_targets_val_or_test() -> None:
    """val and test belong to the TARGET subject and to the protocol. If either moves with N, the
    curve is measuring a target-calibration budget instead of a donor-label budget."""
    a_rec, t_rec = _big_rec(seed=0), _big_rec(seed=1)
    _, census = _cs_curve_cell(a_rec, t_rec, BOARD_TASKS[0], TAPS)
    assert census
    assert len({c["n_val"] for c in census}) == 1, "val shrank with N"
    assert len({c["n_test"] for c in census}) == 1, "test shrank with N"


def test_cs_emits_only_the_trainonly_column() -> None:
    """`both` means 'train AND val subsampled'. In CS there is no such experiment, so emitting a
    `both` column would ship a duplicate of `trainonly` under a name that misstates what it is."""
    a_rec, t_rec = _big_rec(seed=0), _big_rec(seed=1)
    pts, _ = _cs_curve_cell(a_rec, t_rec, BOARD_TASKS[0], TAPS)
    assert pts and {p["col"] for p in pts} == {"trainonly"}


def test_cs_donor_draw_is_keyed_on_the_anchor_not_the_test_cell() -> None:
    """One donor subsample, evaluated on ten patients. Keying on the test cell would make each cell
    see different donor trials — a different, lower-variance estimand than the claim."""
    a_rec = _big_rec(seed=0)
    t1, t2 = _big_rec(seed=1), _big_rec(seed=2)
    task = BOARD_TASKS[0]
    y_a = np.asarray(a_rec["labels"][task], dtype=np.float64)
    tr = _finite(y_a, np.arange(len(y_a)))
    d1 = _digest(_strat_draw(y_a, tr, 16, _rng(CS_TRAIN_ANCHOR, task, 0, 16, 0)))
    seen = set()
    for t_rec in (t1, t2):
        pts, _ = _cs_curve_cell(a_rec, t_rec, task, TAPS)
        seen.add(len(pts))
    assert len(seen) == 1
    # the draw itself is a pure function of (anchor, task, n, seed) — no test cell in the key
    assert d1 == _digest(_strat_draw(y_a, tr, 16, _rng(CS_TRAIN_ANCHOR, task, 0, 16, 0)))
    assert d1 != _digest(_strat_draw(y_a, tr, 16, _rng((1, 1), task, 0, 16, 0)))


def test_cs_both_taps_are_fit_on_the_same_donor_rows(monkeypatch) -> None:
    import scripts.neuroprobe.v3_board_samplecurve as M

    real = M._feat
    calls: list = []

    def spy(rec, enc, rows, col_idx=None):
        calls.append((enc, _digest(rows)))
        return real(rec, enc, rows, col_idx)

    monkeypatch.setattr(M, "_feat", spy)
    _cs_curve_cell(_big_rec(seed=0), _big_rec(seed=1), BOARD_TASKS[0], TAPS)
    by_tap: dict = {}
    for enc, dig in calls:
        by_tap.setdefault(enc, []).append(dig)
    assert set(by_tap) == set(TAPS)
    assert by_tap[TAPS[0]] == by_tap[TAPS[1]], "the two taps were gathered on different rows"


def test_cs_grid_extends_one_doubling_past_the_ws_grid() -> None:
    """CS fits the WHOLE anchor session, ~2x a WS train half, so the extra point is reachable there
    and only there. Points above the parent are filtered at runtime, so an unreachable entry is a
    no-op rather than a crash."""
    assert N_GRID_CS[:len(N_GRID)] == N_GRID
    assert len(N_GRID_CS) == len(N_GRID) + 1
    assert N_GRID_CS[-1] == 2 * N_GRID[-1]
    assert all(n in SEEDS_FOR_N for n in N_GRID_CS)


def test_cs_taps_are_the_parcel_mean_pair_not_the_electrode_pair() -> None:
    """Electrode identity is not shared across subjects, so a CS curve on `*_elec` would be
    measuring nothing. The regime picks its own unit; the default must not be inherited."""
    assert CURVE_TAPS["cs"] == ("enc0", "enc12")
    assert CURVE_TAPS["ws"] == ("enc0_elec", "enc12_elec")
    assert SHARD_PREFIX["ws"] != SHARD_PREFIX["cs"]


# ── TIER 3: the additive-vs-multiplicative decision ────────────────────────────────────────────
#
# This is the estimator the whole verdict is read off, so it is tested against data with a KNOWN
# law. A test that only checks it runs would not catch a sign error or a swapped intercept/slope.

def _planted(a, k, taps=("enc0", "enc12"), n_subj=6, noise=0.0, seed=0):
    """Points whose macro curve obeys (enc12-.5) = a + k*(enc0-.5) by construction."""
    rng = np.random.default_rng(seed)
    pts = []
    for si in range(n_subj):
        for ni, n in enumerate((16, 32, 64, 128, 256, 512)):
            x = 0.02 + 0.03 * ni                      # enc0 headroom grows with N
            for task in ("onset", "speech"):
                e = rng.normal(scale=noise) if noise else 0.0
                for tap, v in ((taps[0], .5 + x), (taps[1], .5 + a + k * x + e)):
                    pts.append({"tap": tap, "col": "trainonly", "n_bucket": n, "n_is_full": False,
                                "task": task, "cell": f"S{si}T0", "test": v})
    return pts


def test_addmult_recovers_a_planted_ADDITIVE_law() -> None:
    r = _addmult(_planted(a=0.02, k=1.0), "enc0", "enc12", "trainonly", nboot=200)
    assert r["a"] == pytest.approx(0.02, abs=1e-6)
    assert r["k"] == pytest.approx(1.0, abs=1e-6)


def test_addmult_recovers_a_planted_MULTIPLICATIVE_law() -> None:
    r = _addmult(_planted(a=0.0, k=1.4), "enc0", "enc12", "trainonly", nboot=200)
    assert r["a"] == pytest.approx(0.0, abs=1e-6)
    assert r["k"] == pytest.approx(1.4, abs=1e-6)


def test_addmult_does_not_confuse_the_intercept_with_the_slope() -> None:
    """a and k are not interchangeable: a pure multiplier and a pure offset must not both come back
    looking like the same fit. Pinning the CROSS terms is what catches a swapped np.polyfit unpack."""
    add = _addmult(_planted(a=0.03, k=1.0), "enc0", "enc12", "trainonly", nboot=100)
    mul = _addmult(_planted(a=0.0, k=1.5), "enc0", "enc12", "trainonly", nboot=100)
    assert add["a"] > mul["a"] and mul["k"] > add["k"]


def test_addmult_bootstrap_CI_widens_with_noise_and_covers_the_truth() -> None:
    """The CI is the whole decision rule. If noise does not widen it, the verdict is decided by an
    error bar that is not measuring anything."""
    quiet = _addmult(_planted(a=0.02, k=1.0, noise=0.000), "enc0", "enc12", "trainonly", nboot=400)
    loud = _addmult(_planted(a=0.02, k=1.0, noise=0.02, seed=3), "enc0", "enc12", "trainonly",
                    nboot=400)
    w = lambda r: r["a_ci"][1] - r["a_ci"][0]
    assert w(loud) > w(quiet)
    assert loud["a_ci"][0] <= 0.02 <= loud["a_ci"][1]


def test_addmult_resamples_subjects_not_cells() -> None:
    """Two sessions of one patient are not two independent draws. The bootstrap unit has to be the
    subject, or the CI is too tight and the verdict is called on an error bar that overstates n."""
    pts = _planted(a=0.02, k=1.0, n_subj=4)
    pts += [dict(p, cell=p["cell"].replace("T0", "T1")) for p in pts]   # 4 subjects, 8 cells
    assert len({p["cell"] for p in pts}) == 8
    r = _addmult(pts, "enc0", "enc12", "trainonly", nboot=50)
    assert r["n_subjects"] == 4, "the bootstrap resampled cells, not subjects"


def test_merge_globs_only_its_own_regimes_shards(tmp_path) -> None:
    """One shard dir can hold both families. Merging without naming the regime would silently
    average two different units — different taps, different train sets — into a number of no
    regime at all, and it would not look wrong on inspection."""
    import json as _json
    base = {"name": "S1T1", "tag": "t", "contiguous": False,
            "points": [], "census": [], "anchor": []}
    (tmp_path / "cscurve_S1T1.json").write_text(
        _json.dumps(dict(base, kind="cscurve", regime="cs")))
    (tmp_path / "wscurve_S1T1.json").write_text(
        _json.dumps(dict(base, kind="wscurve", regime="ws")))
    assert _merge(str(tmp_path), "cs")["regime"] == "cs"
    assert _merge(str(tmp_path), "ws")["regime"] == "ws"


def test_merge_rejects_a_shard_whose_regime_contradicts_its_filename(tmp_path) -> None:
    """Belt to the glob's braces: a hand-copied or legacy shard under the wrong prefix must fail
    loudly rather than be averaged into the wrong regime's curve."""
    import json as _json
    (tmp_path / "cscurve_S1T1.json").write_text(_json.dumps(
        {"kind": "cscurve", "regime": "ws", "name": "S1T1", "tag": "t", "contiguous": False,
         "points": [], "census": [], "anchor": []}))
    with pytest.raises(AssertionError):
        _merge(str(tmp_path), "cs")


def test_merge_raises_rather_than_reporting_an_empty_curve(tmp_path) -> None:
    with pytest.raises(SystemExit):
        _merge(str(tmp_path), "cs")


# ── CROSS-SESSION MODE ─────────────────────────────────────────────────────────────────────────
#
# csession is the CONTROL that gives the ws-vs-cs contrast its meaning: same patient, different
# session. Without it, any ws/cs difference could be "cs is simply harder" rather than anything
# about the subject boundary. It shares _transfer_curve_cell with cs, so what needs pinning is the
# two things that DIFFER: the train record is the sibling (not the fixed anchor), and the draw key
# moves with it.
#
# ⚠️ COVERAGE NOTE: the synthetic fixture's taps are parcel-named, so these exercise the PARCEL
# branch of _transfer_cols. In production csession runs on the ELECTRODE taps. That branch is
# guarded by the anchor check against the real _csession_cell, which fails fast and loudly on the
# first task -- which is the design, not an omission.

def test_csession_N_full_reproduces_the_published_csession_cell_exactly() -> None:
    sib, tst = _big_rec(seed=0), _big_rec(seed=1)
    task = BOARD_TASKS[0]
    pts, _ = _csession_curve_cell(sib, tst, task, TAPS, (1, 2))
    rows = _csession_anchor_check(sib, tst, task, TAPS, pts)
    assert rows, "no anchor rows produced — the check would vacuously pass"
    for r in rows:
        assert r["absdiff"] == 0.0, f"csession anchor drifted: {r}"


def test_csession_draw_is_keyed_on_the_sibling_so_it_MOVES_with_the_cell() -> None:
    """The opposite of cs. There the donor is one fixed session for all ten cells, so the draw must
    NOT move. Here the train session IS per-cell, so two cells with different siblings must get
    different draws — otherwise every cell would be fit on rows drawn for someone else's sibling."""
    rec = _big_rec()
    y = np.asarray(rec["labels"][BOARD_TASKS[0]], dtype=np.float64)
    tr = _finite(y, np.arange(len(y)))
    a = _digest(_strat_draw(y, tr, 16, _rng((1, 2), "onset", 0, 16, 0)))
    b = _digest(_strat_draw(y, tr, 16, _rng((3, 1), "onset", 0, 16, 0)))
    assert a != b
    # and it is NOT the cs key, or csession would silently reuse the donor's draws
    assert a != _digest(_strat_draw(y, tr, 16, _rng(CS_TRAIN_ANCHOR, "onset", 0, 16, 0)))


def test_csession_never_subsamples_the_targets_val_or_test() -> None:
    sib, tst = _big_rec(seed=0), _big_rec(seed=1)
    _, census = _csession_curve_cell(sib, tst, BOARD_TASKS[0], TAPS, (1, 2))
    assert census
    assert len({c["n_val"] for c in census}) == 1
    assert len({c["n_test"] for c in census}) == 1


def test_csession_emits_only_the_trainonly_column() -> None:
    sib, tst = _big_rec(seed=0), _big_rec(seed=1)
    pts, _ = _csession_curve_cell(sib, tst, BOARD_TASKS[0], TAPS, (1, 2))
    assert pts and {p["col"] for p in pts} == {"trainonly"}


def test_csession_keeps_the_ELECTRODE_taps_because_electrodes_are_shared_within_subject() -> None:
    """cs must drop to parcel means (electrode identity is not shared across subjects); csession
    must NOT, or it would throw away the per-electrode resolution it is entitled to and the ws/cs
    ladder would confound 'crossed a subject boundary' with 'changed unit'."""
    assert CURVE_TAPS["csession"] == CURVE_TAPS["ws"] == ("enc0_elec", "enc12_elec")
    assert CURVE_TAPS["cs"] == ("enc0", "enc12")
    assert len({SHARD_PREFIX[r] for r in ("ws", "cs", "csession")}) == 3


def test_transfer_cols_routes_by_TAP_not_by_regime() -> None:
    """A parcel tap must never take the electrode branch and vice versa. _is_elec is name-based, so
    this is the seam where a wrong pairing would hide."""
    import scripts.neuroprobe.v3_board_samplecurve as M

    a, t = _big_rec(seed=0), _big_rec(seed=1)
    seen = []
    real_parcel = M._parcel_cols
    M_R = M.R
    real_elec = M_R._elec_cols

    def spy_parcel(x, y):
        seen.append("parcel"); return real_parcel(x, y)

    def spy_elec(x, y):
        seen.append("elec"); return real_elec(x, y)

    M._parcel_cols = spy_parcel
    M_R._elec_cols = spy_elec
    try:
        M._transfer_cols(a, t, "enc12")
        M._transfer_cols(a, t, "enc12_elec")
    finally:
        M._parcel_cols = real_parcel
        M_R._elec_cols = real_elec
    assert seen == ["parcel", "elec"], seen


def test_transfer_regimes_share_one_cell_function() -> None:
    """cs and csession differ ONLY in the train record and the draw key. If they ever stop sharing
    _transfer_curve_cell, the two curves can drift apart for reasons that are not the regime."""
    sib, tst = _big_rec(seed=0), _big_rec(seed=1)
    task = BOARD_TASKS[0]
    a, _ = _cs_curve_cell(sib, tst, task, TAPS)
    b, _ = _transfer_curve_cell(sib, tst, task, TAPS, CS_TRAIN_ANCHOR)
    assert [p["test"] for p in a] == [p["test"] for p in b]
