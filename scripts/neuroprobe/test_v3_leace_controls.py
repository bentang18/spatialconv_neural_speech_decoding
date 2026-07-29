"""The controls are only worth running if they can distinguish the two readings they exist for.

`test_geometry_separates_a_planted_pc1_concept_from_a_planted_low_variance_concept` is the
load-bearing one: it plants identity ONCE on the top principal component (the "it is just common
mode" world) and ONCE on a deliberately low-variance direction (the "the model relocated it"
world), and requires the diagnostic to come back with opposite answers. A diagnostic that reports
the same numbers in both worlds cannot adjudicate anything, however plausible its definition.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import v3_board_readout as B
import v3_leace_controls as C

from speech_decoding.experiments.leace import fit_leace

TASK = "onset"
TAPS = ("enc0", "enc12")


def _rec(seed, n, parcels, feat_dim=8, shift=0.0, with_split=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(parcels), feat_dim)).astype(np.float32) + shift
    y = np.asarray(rng.integers(0, 2, size=n), dtype=np.float64)
    x[:, 0, 0] += 1.5 * y
    rec = {
        "labels": {TASK: y},
        "present_parcels": np.asarray(parcels, dtype=np.int64),
        "feats": {t: {"raw": torch.from_numpy(x).to(torch.float16)} for t in TAPS},
    }
    if with_split:
        half = n // 2
        rec["cs_split"] = {TASK: {"val": np.arange(half), "test": np.arange(half, n)}}
    return rec


@pytest.fixture
def pair():
    return _rec(0, 160, [3, 5, 9, 12]), _rec(1, 120, [5, 9, 12, 20], shift=2.0, with_split=True)


def _planted(concept_on_pc1: bool, n=400, d=40, seed=0):
    """Features with a steep variance spectrum, and a domain concept planted either ON the top
    principal component or on a direction deliberately starved of variance.

    The low-variance amplitude has to clear the CHANCE covariance between the domain label and the
    leading components (~s0/sqrt(n)), or LEACE latches onto sampling noise at the top of the
    spectrum instead of the planted axis. That is not an artifact of the fixture -- it is a real
    small-n hazard for the experiment itself, which is why the amplitude is set explicitly here.
    """
    rng = np.random.default_rng(seed)
    z = np.r_[np.zeros(n // 2), np.ones(n - n // 2)].astype(int)
    base = rng.normal(size=(n, d)) * np.linspace(6.0, 0.05, d)   # steep spectrum
    axis = np.zeros(d)
    axis[0 if concept_on_pc1 else d - 1] = 1.0
    amp = 25.0 if concept_on_pc1 else 3.0
    return base + amp * np.outer(z - z.mean(), axis), z


def _geom_of(x, z):
    return C._geometry(fit_leace(x, z), x, z)


def test_geometry_separates_a_planted_pc1_concept_from_a_planted_low_variance_concept():
    """The whole point. Same estimator, two worlds, opposite verdicts."""
    hi = _geom_of(*_planted(True))
    lo = _geom_of(*_planted(False))

    # World B: the concept IS the top principal component.
    assert hi["cos_pc1"] > 0.95, hi
    assert hi["pc_participation"] < 2.0, "a single principal axis, not a spread direction"
    assert hi["pc_com"] < 1.0, "sits at the very top of the spectrum"

    # World A: the concept is a real direction that is NOT where the variance is.
    # Some chance alignment with PC1 survives at finite n -- the point is that it is SMALL, and
    # an order of magnitude below the world where the concept genuinely is PC1.
    assert lo["cos_pc1"] < 0.25, lo
    assert hi["cos_pc1"] > 5 * lo["cos_pc1"], "the two worlds must not be a matter of degree"
    assert lo["pc_com"] > 10.0, "sits far down the spectrum"
    assert lo["var_along_dir"] < hi["var_along_dir"] / 10


def test_the_erased_direction_is_exactly_the_between_domain_mean_shift():
    """An algebraic identity for a binary concept, so it is a self-check and never evidence.
    Asserted here so that if it ever stops holding we learn it from a test, not from a claim."""
    for seed in range(4):
        x, z = _planted(bool(seed % 2), seed=seed)
        assert _geom_of(x, z)["cos_domain_mean_shift"] == pytest.approx(1.0, abs=1e-9)


def test_between_within_split_tells_an_offset_apart_from_trial_variation():
    """The load-bearing decomposition: a direction that is purely a domain offset must report
    between_frac ~ 1, and one that varies trial-to-trial within each domain must report ~0."""
    rng = np.random.default_rng(7)
    n, d = 400, 12
    z = np.r_[np.zeros(200), np.ones(200)].astype(int)
    axis = np.zeros(d); axis[0] = 1.0

    offset = rng.normal(size=(n, d)) * 0.01 + 5.0 * np.outer(z - z.mean(), axis)
    assert C._along(offset, offset.mean(0), axis, z)["between_frac"] > 0.99

    within = rng.normal(size=(n, d))            # no domain structure at all
    assert C._along(within, within.mean(0), axis, z)["between_frac"] < 0.05


def test_common_mode_is_detected_when_it_is_actually_there():
    """cos_common_mode must respond to a genuine uniform offset, or it is decoration."""
    rng = np.random.default_rng(1)
    n, d = 160, 30
    z = np.r_[np.zeros(n // 2), np.ones(n - n // 2)].astype(int)
    x = rng.normal(size=(n, d)) + 8.0 * np.outer(z - z.mean(), np.ones(d) / np.sqrt(d))
    assert _geom_of(x, z)["cos_common_mode"] > 0.95

    x2 = rng.normal(size=(n, d))
    x2[:, 3] += 8.0 * (z - z.mean())                     # a single feature, not common mode
    assert _geom_of(x2, z)["cos_common_mode"] < 0.4


def test_top_pc_eraser_removes_exactly_the_leading_component():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 20)) * np.linspace(5, 0.1, 20)
    mean = x.mean(0)
    u, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    er = C._top_pc_eraser(mean, vt.T, s)
    assert er.var_removed == pytest.approx(s[0] ** 2 / (s**2).sum(), rel=1e-12)
    # No variance may survive along PC1, and the rest must be untouched.
    e = er(x)
    assert abs(float(((e - e.mean(0)) @ vt[0]).std())) < 1e-9
    assert np.allclose((e - e.mean(0)) @ vt[1], (x - mean) @ vt[1], atol=1e-9)


def test_shared_svd_arm_reproduces_the_standalone_eraser():
    """leace here must be the SAME eraser v3_cs_leace fits, or the controls are not comparable
    to the published arm."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(90, 30))
    z = np.r_[np.zeros(45), np.ones(45)].astype(int)
    x[:, 5] += 2.0 * z
    want = fit_leace(x, z)
    got = fit_leace(x, z, svd=np.linalg.svd(x - x.mean(0), full_matrices=False))
    assert got.var_removed == pytest.approx(want.var_removed, rel=1e-12)


def test_std_arm_still_matches_the_board(pair):
    """Same parity guarantee as v3_cs_leace: without it the control deltas float free."""
    anchor, test = pair
    board = B._cs_cell(anchor, test, TASK, TAPS)["cells"]
    mine = C._cell_arms(anchor, test, TASK, TAPS, None)["cells"]
    for k in [k for k in board if k.endswith("|std")]:
        assert mine[k]["test"] == board[k]["test"], f"{k} diverged from the leaderboard baseline"


def test_all_four_arms_are_reported_with_geometry(pair):
    anchor, test = pair
    res = C._cell_arms(anchor, test, TASK, TAPS, None)
    for tap in TAPS:
        for arm in C.ARMS:
            assert f"{tap}|{arm}" in res["cells"], f"{tap}|{arm} missing"
        for key in ("cos_pc1", "pc_participation", "cos_common_mode", "cos_domain_mean_shift",
                    "var_removed_leace_shuf", "var_removed_leace_toppc"):
            assert key in res["checks"][tap], f"{tap} missing {key}"


def test_var_removed_can_approach_100_percent_at_exactly_zero_cost():
    """`var_removed` is NOT a difficulty measure, and this is the proof.

    When the erased direction is a pure between-domain offset, AUROC cannot see it: the score
    changes by a constant on every test row and ranks are unchanged. The algebra only covers a
    FIXED w, so this drives the real standardize+ridge path end to end, INCLUDING the refit on
    erased anchor features -- the part the algebra does not cover. At a large enough offset the
    eraser destroys >99% of total variance and the score is bit-identical, in a model with no
    learning at all. Any claim of the form "erasing X% of the variance cost nothing" has to be
    read against this.
    """
    d, n_a, n_t = 40, 300, 200
    r = np.random.default_rng(0)
    u = r.normal(size=d); u /= np.linalg.norm(u)                  # offset axis
    v = r.normal(size=d); v -= (v @ u) * u; v /= np.linalg.norm(v)  # content axis, _|_ to it

    def make(n, off, seed):
        rr = np.random.default_rng(seed)
        y = np.r_[np.zeros(n // 2), np.ones(n - n // 2)]
        rr.shuffle(y)
        x = rr.normal(size=(n, d))
        x -= np.outer(x @ u, u)                                   # zero WITHIN-domain var along u
        x += 2.0 * np.outer(y - 0.5, v)
        return (x + off * u).astype(np.float32), y

    va, te = np.arange(n_t // 2), np.arange(n_t // 2, n_t)
    x_a, y_a = make(n_a, 0.0, 1)
    x_t, y_t = make(n_t, 200.0, 2)
    er = fit_leace(np.vstack([x_a, x_t]), np.r_[np.zeros(n_a), np.ones(n_t)].astype(int))

    def sel(fa, ft):
        a, (b, c) = B._standardize_inplace(fa.copy(), [ft[va].copy(), ft[te].copy()])
        return B._select_lam(B._lam_grid(a, y_a, {"val": (b, y_t[va]),
                                                  "test": (c, y_t[te])}))["test"]

    assert er.var_removed > 0.99, f"fixture must delete nearly everything, got {er.var_removed}"
    before = sel(x_a, x_t)
    after = sel(er(x_a).astype(np.float32), er(x_t).astype(np.float32))
    assert after == before, f"a pure offset must be free, moved {after - before:+.3e}"


def test_the_shuffled_control_erases_a_different_direction_than_identity(pair):
    """If shuffling produced the same eraser the control would be vacuous."""
    anchor, test = pair
    res = C._cell_arms(anchor, test, TASK, TAPS, None)
    ck = res["checks"]["enc12"]
    assert ck["var_removed_leace_shuf"] < ck["var_removed"], (
        "a shuffled concept must ride LESS variance than the real one")
