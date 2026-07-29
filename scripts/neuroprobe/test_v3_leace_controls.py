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


def _spectrum_world(alpha: float, ratio: float, shift: float, n=600, seed=0):
    """No group structure unless ``shift`` says so; eigenvalues decay as i^-alpha and the feature
    count is ``ratio * n``. Total variance is 1, so ``shift`` is in units of the TOTAL sd.

    Those units matter. Scaled to the leading eigenvalue instead, any offset large enough to notice
    also BECOMES pc1 and drags ``pc1_var_frac`` up with it, so the excess pins at 1 and the fixture
    can never exhibit the regime the real features are in -- pc1 carrying only ~19% of variance
    while the offset lies exactly along it.
    """
    rng = np.random.default_rng(seed)
    d = int(round(ratio * n))
    lam = np.arange(1, d + 1, dtype=float) ** (-alpha)
    lam /= lam.sum()
    x = rng.normal(size=(n, d)) * np.sqrt(lam)
    z = (np.arange(n) >= n // 2).astype(int)
    if shift:
        v = rng.normal(size=d) * np.sqrt(lam)   # a generic direction with the data's anisotropy
        x[z == 1] += shift * v / np.linalg.norm(v)
    return x, z


def _null_excess(alpha: float, ratio: float, reps=24, **kw) -> list:
    """`cos_pc1_excess` over independent draws.

    Averaged deliberately. Under the null ``cos_pc1^2`` is driven by ONE chi-square-1 variate, so a
    single draw at a fixed spectrum swings over most of [0, 1] -- 0.32 and 0.86 both occur at
    alpha=3. Only the EXPECTATION obeys the law, which is also why a per-cell `cos_pc1` near 1 is
    weak on its own and 10/10 cells pinned at 0.9999 is not.
    """
    return [_geom_of(*_spectrum_world(alpha=alpha, ratio=ratio, seed=s, **kw))["cos_pc1_excess"]
            for s in range(reps)]


def test_a_steep_spectrum_alone_inflates_cos_pc1_but_never_the_excess_over_its_own_null():
    """The reason `cos_pc1_null` exists, and the reason the excess is the number to read.

    Anisotropy alone lifts raw `cos_pc1` most of the way to 1 with NO group structure whatever, so
    the raw number is not evidence. It lifts `pc1_var_frac` at least as fast, so the RATIO stays at
    or below 1 across the whole decay range. That ceiling is what makes an excess above it mean
    something.
    """
    for alpha in (0.0, 0.5, 1.5, 3.0):
        g = [_geom_of(*_spectrum_world(alpha=alpha, ratio=13.3, shift=0.0, seed=s))
             for s in range(12)]
        ex = float(np.mean([q["cos_pc1_excess"] for q in g]))
        assert ex < 1.6, f"null excess must stay near/below 1, alpha={alpha} gave {ex:.2f}"
    assert np.mean([_geom_of(*_spectrum_world(alpha=3.0, ratio=13.3, shift=0.0, seed=s))["cos_pc1"]
                    for s in range(12)]) > 0.4, "the trap is real: no group structure, yet cos ~ .67"


def test_a_one_sd_rigid_offset_reproduces_the_observed_enc12_signature():
    """The regime the real enc12 features are in, built from a known cause.

    A near-flat within-session spectrum plus a rigid between-session offset of ~1 total sd gives
    pc1_var .20 / cos_pc1 1.000 / excess 4.96 / participation 1.00, against observed .1914 / .9999
    / 5.22 / 1.0003. So the enc12 geometry has a quantitative account -- and the offset's SIZE is
    what the excess reads out: the same fixture at half the offset doubles the excess, at double
    the offset halves it, because pc1_var tracks the offset while cos_pc1 is already pinned.
    """
    got = {sh: [_geom_of(*_spectrum_world(alpha=0.5, ratio=13.3, shift=sh, seed=s))
                for s in range(8)] for sh in (0.5, 1.0, 2.0)}
    m = {sh: {k: float(np.mean([q[k] for q in g])) for k in g[0]} for sh, g in got.items()}

    one = m[1.0]
    assert one["pc1_var_frac"] == pytest.approx(0.1914, abs=0.05), one
    assert one["cos_pc1"] > 0.999 and one["pc_participation"] < 1.05, one
    assert one["cos_pc1_excess"] == pytest.approx(5.2, rel=0.25), one
    # Monotone in the offset, which is what licenses reading the excess as a magnitude.
    assert m[0.5]["cos_pc1_excess"] > one["cos_pc1_excess"] > m[2.0]["cos_pc1_excess"]


def test_high_dn_biases_the_null_downward_so_an_excess_stays_conservative():
    """`E[cos_pc1^2] = pc1_var_frac` is the population law, but the SAMPLE pc1 eigenvalue is
    inflated at high d/n, so the measured null excess falls BELOW 1 there (~.6 at d/n=13.3 vs ~1.0
    at .29). It moves in the safe direction: enc12 is the high-d/n tap, so its excess of 5.2 is if
    anything understated, and enc0-vs-enc12 cannot be an artifact of their d/n gap."""
    lo = float(np.mean(_null_excess(alpha=1.5, ratio=0.29, shift=0.0)))
    hi = float(np.mean(_null_excess(alpha=1.5, ratio=13.3, shift=0.0)))
    assert lo < 1.6 and hi < 1.6, f"neither may exceed the ceiling: {lo:.3f}, {hi:.3f}"
    assert hi < lo, f"high d/n must not INFLATE the null: {lo:.3f} -> {hi:.3f}"


def test_white_features_at_high_dn_do_not_fake_a_principal_axis():
    """The originally-feared artifact, measured: high d/n on its own sends cos_pc1 to ~0 and
    participation to the hundreds. Dimensionality was never the mechanism -- anisotropy was."""
    g = _geom_of(*_spectrum_world(alpha=0.0, ratio=13.3, shift=0.0))
    assert g["cos_pc1"] < 0.2, g
    assert g["pc_participation"] > 50.0, g
    assert g["pc_participation_null"] > 50.0, g


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


def test_the_shuffled_arm_carries_a_full_geometry_null(pair):
    """dir_between_frac is biased upward by d/n, so the shuffled arm has to report the SAME
    statistics as the real one -- otherwise there is nothing to compare enc0 and enc12 against."""
    anchor, test = pair
    ck = C._cell_arms(anchor, test, TASK, TAPS, None)["checks"]["enc12"]
    for stat in ("dir_between_frac", "cos_pc1", "pc_participation", "var_removed"):
        assert f"{stat}_leace_shuf" in ck, f"no null for {stat}"
        assert f"{stat}_leace" in ck, f"no treatment value for {stat}"


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


def _two_session_world(kind: str, n=600, d=60, k=8, offset=40.0, seed=0):
    """Two sessions whose within-session structure is related in a controlled way.

    `scale`   -- identical axes, different amount of travel along each (Ben's read of the
                 trajectory figures: same path, different size).
    `rotate`  -- identical SUBSPACE, but the axes inside it are rotated. This is the case a
                 subspace-overlap number cannot tell apart from `scale`, and the whole reason
                 `diag_k` exists.
    `unrelated` -- disjoint subspaces; the model shares nothing.

    A large between-session offset is added in every world, because the real data has one and it
    must not be able to reach these statistics through the per-domain centering.
    """
    r = np.random.default_rng(seed)
    q = np.linalg.qr(r.normal(size=(d, 2 * k)))[0]
    qa, qb_alt = q[:, :k], q[:, k:]
    lam = np.geomspace(5.0, 0.5, k)

    if kind == "scale":
        qb, lb = qa, lam * np.geomspace(3.0, 0.4, k)
    elif kind == "rotate":
        rot = np.linalg.qr(r.normal(size=(k, k)))[0]
        qb, lb = qa @ rot, lam
    elif kind == "unrelated":
        qb, lb = qb_alt, lam
    else:
        raise ValueError(kind)

    a = (r.normal(size=(n, k)) * lam) @ qa.T
    b = (r.normal(size=(n, k)) * lb) @ qb.T + offset * q[:, 0]
    return np.vstack([a, b]), np.r_[np.zeros(n), np.ones(n)].astype(int)


def test_alignment_tells_a_shared_coordinate_system_from_a_rotated_one():
    """Overlap alone cannot adjudicate "same trajectory, different scale" -- a rotation inside the
    same subspace scores identically. `diag_k` is what separates them, and this is its proof."""
    k = 8
    got = {kind: C._subspace_alignment(*_two_session_world(kind), ks=(k,))
           for kind in ("scale", "rotate", "unrelated")}

    # Both related worlds recover essentially the whole shared subspace...
    for kind in ("scale", "rotate"):
        assert got[kind][f"align_k{k}_frac"] > 0.9, (kind, got[kind])
    # ...and an unrelated pair sits near the analytic floor, nowhere near the ceiling.
    assert got["unrelated"][f"align_k{k}_frac"] < 0.25, got["unrelated"]

    # The discrimination overlap cannot make. Read against the random-rotation reference, never
    # an absolute threshold -- a fully rotated steep spectrum still keeps most of its mass on the
    # diagonal, so "diag is high" alone means nothing.
    assert got["scale"][f"diag_k{k}"] > 0.9, got["scale"]
    assert got["scale"][f"diag_k{k}"] > 1.4 * got["scale"][f"diag_k{k}_rot"], got["scale"]
    assert got["rotate"][f"diag_k{k}"] == pytest.approx(
        got["rotate"][f"diag_k{k}_rot"], abs=0.12), got["rotate"]


def _task_world(shared: bool, orthogonal: bool, n=800, d=300, k=40, offset=40.0, seed=0):
    """Two sessions with a big rigid offset, each carrying a binary task.

    ``shared`` -- both sessions encode the task along the SAME axis, or along unrelated ones.
    ``orthogonal`` -- that axis is orthogonal to the session offset, or lies along it.

    The isotropic floor is not decoration. Without it the rows span only ~k dims, and two random
    directions in a dozen dimensions have cos ~ 0.4, so the label-shuffle null sits high and the
    fixture cannot exercise the regime the real features are in (rank 7000, null ~ 1/sqrt(rank)).
    """
    r = np.random.default_rng(seed)
    q = np.linalg.qr(r.normal(size=(d, k + 3)))[0]
    off, t_a, t_b = q[:, 0], q[:, 1], q[:, 2]
    if not orthogonal:
        t_a = t_b = off
    elif shared:
        t_b = t_a
    x = (r.normal(size=(2 * n, k)) * np.geomspace(5.0, 0.5, k)) @ q[:, 3:].T
    x += 0.5 * r.normal(size=(2 * n, d))                     # full-rank floor
    dom = (np.arange(2 * n) >= n).astype(int)
    lab = (np.arange(2 * n) % 2).astype(float)
    x += offset * np.outer(dom, off)
    x += 4.0 * np.outer((lab - 0.5) * (dom == 0), t_a)
    x += 4.0 * np.outer((lab - 0.5) * (dom == 1), t_b)
    return x, dom, lab


def _task_of(x, dom, lab):
    z = x - x.mean(0)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    return C._task_alignment(u * s, dom, lab)


def test_task_alignment_tells_a_shared_task_axis_from_a_session_specific_one():
    """The falsifier for 'the task-locked component is shared even though the covariance is not'."""
    shared = _task_of(*_task_world(shared=True, orthogonal=True))
    private = _task_of(*_task_world(shared=False, orthogonal=True))

    assert shared["task_cos"] > 0.85 and shared["task_cos_p"] == 0.0, shared
    assert shared["task_cos_frac"] > 4.0, shared
    assert private["task_cos"] < 0.1 and private["task_cos_p"] > 0.5, private
    assert private["task_cos"] < private["task_cos_null"], "a private axis sits BELOW its own null"


def test_task_alignment_reports_whether_the_task_rides_the_session_offset():
    """`task_vs_sess` is the honest form of the separability claim -- a measured overlap, not an
    inference from an erasure that costs nothing downstream."""
    apart = _task_of(*_task_world(shared=True, orthogonal=True))
    along = _task_of(*_task_world(shared=True, orthogonal=False))

    for key in ("task_vs_sess_a", "task_vs_sess_t"):
        assert apart[key] < apart["task_vs_sess_chance"], (key, apart)
        assert along[key] > 0.9, (key, along)


def test_the_task_null_shuffles_within_session_so_it_keeps_the_offset():
    """A null that shuffled labels ACROSS sessions would leak the session offset into the task
    direction and inflate the null toward 1, hiding a real shared axis."""
    small = _task_of(*_task_world(shared=False, orthogonal=True, offset=40.0))
    huge = _task_of(*_task_world(shared=False, orthogonal=True, offset=5000.0))
    assert huge["task_cos_null"] < 0.25, huge
    assert huge["task_cos_null"] == pytest.approx(small["task_cos_null"], abs=0.02), (small, huge)


def test_alignment_is_blind_to_the_between_session_offset():
    """The offset is the one thing we already know is there, so it must not be able to
    manufacture alignment -- otherwise this metric repeats the LEACE mistake."""
    k = 8
    small = C._subspace_alignment(*_two_session_world("scale", offset=0.0), ks=(k,))
    huge = C._subspace_alignment(*_two_session_world("scale", offset=5000.0), ks=(k,))
    for stat in (f"align_k{k}", f"diag_k{k}"):
        assert small[stat] == pytest.approx(huge[stat], abs=1e-9), stat


def test_the_shuffled_control_erases_a_different_direction_than_identity(pair):
    """If shuffling produced the same eraser the control would be vacuous."""
    anchor, test = pair
    res = C._cell_arms(anchor, test, TASK, TAPS, None)
    ck = res["checks"]["enc12"]
    assert ck["var_removed_leace_shuf"] < ck["var_removed"], (
        "a shuffled concept must ride LESS variance than the real one")
