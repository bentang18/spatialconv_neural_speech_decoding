import numpy as np
import pytest

from speech_decoding.experiments.leace import LeaceEraser, fit_leace


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y == 1], scores[y == 0]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def _probe_auc(x_tr, y_tr, x_te, y_te) -> float:
    """rcond is load-bearing: erased features are rank-deficient, and pinv's default tolerance
    inverts the erased direction (singular value ~1e-15) into a 1e14-norm weight vector, which
    reads as 'erasure failed' when it is really numerical blow-up in the probe."""
    w = np.linalg.pinv(x_tr - x_tr.mean(0), rcond=1e-8) @ (y_tr - y_tr.mean())
    return _auc((x_te - x_tr.mean(0)) @ w, y_te)


def _concept_data(seed=0, n=600, d=40, sep=3.0, n_classes=2):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, n_classes, size=n)
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    x = rng.normal(size=(n, d)) + sep * (z[:, None] - z.mean()) * direction
    return x, z, direction


def test_covariance_with_concept_is_exactly_zeroed():
    x, z, _ = _concept_data()
    eraser = fit_leace(x, z)
    xe = eraser(x)
    zc = (z[:, None] == np.unique(z)[None, :]).astype(float)
    cov = (xe - xe.mean(0)).T @ (zc - zc.mean(0)) / (len(x) - 1)
    assert np.abs(cov).max() < 1e-8


def test_rank_is_at_most_n_classes_minus_one():
    for n_classes in (2, 3, 5):
        x, z, _ = _concept_data(seed=n_classes, n_classes=n_classes)
        assert fit_leace(x, z).rank <= n_classes - 1


def test_linear_probe_falls_to_chance_on_held_out_data():
    x, z, _ = _concept_data(seed=1, n=800)
    tr, te = slice(0, 500), slice(500, 800)

    before = _probe_auc(x[tr], z[tr], x[te], z[te])
    assert before > 0.95, f"concept must be decodable pre-erasure, got {before}"

    eraser = fit_leace(x[tr], z[tr])
    after = _probe_auc(eraser(x[tr]), z[tr], eraser(x[te]), z[te])
    assert abs(after - 0.5) < 0.05, f"erasure fitted on train did not transfer, AUC {after}"


def test_uncorrelated_concept_still_costs_the_rank_budget():
    """LEACE always spends C-1 directions, even on a pure-noise concept, so the damage floor is
    (C-1)/rank rather than zero. Anything much above this floor means the concept sits on
    high-variance directions -- that is how var_removed should be read downstream."""
    rng = np.random.default_rng(7)
    n, d = 400, 30
    x = rng.normal(size=(n, d))
    z = rng.integers(0, 2, size=n)
    eraser = fit_leace(x, z)
    floor = 1 / min(n - 1, d)
    assert 0.3 * floor < eraser.var_removed < 3 * floor


def test_content_orthogonal_to_the_concept_survives():
    x, z, direction = _concept_data(seed=3, d=40)
    content = np.random.default_rng(11).normal(size=40)
    content -= content @ direction * direction
    content /= np.linalg.norm(content)

    xe = fit_leace(x, z)(x)
    before, after = x @ content, xe @ content
    assert np.corrcoef(before, after)[0, 1] > 0.99


def test_var_removed_is_reported_and_bounded():
    x, z, _ = _concept_data(seed=5, d=40, sep=6.0)
    eraser = fit_leace(x, z)
    assert 0.0 < eraser.var_removed < 1.0


def test_full_rank_fit_reports_exact_erasure():
    x, z, _ = _concept_data(seed=9, n=300, d=80)
    assert fit_leace(x, z).residual_cov < 1e-12


def test_truncation_trades_exactness_for_conditioning():
    """n_components is the shrinkage knob for ill-conditioned covariances, but the concept can
    have a component outside the retained basis, so erasure stops being exact. residual_cov is
    what makes that visible instead of silent."""
    x, z, _ = _concept_data(seed=9, n=300, d=80)
    truncated = fit_leace(x, z, n_components=25)
    assert truncated.basis.shape[1] == 25
    assert truncated.residual_cov > 1e-6
    assert truncated.residual_cov < fit_leace(x, z, n_components=5).residual_cov


def test_eraser_is_reusable_across_arrays():
    x, z, _ = _concept_data(seed=13)
    eraser = fit_leace(x, z)
    assert isinstance(eraser, LeaceEraser)
    assert np.allclose(eraser(x[:10]), eraser(x)[:10])


@pytest.mark.parametrize(
    "x, z, msg",
    [
        (np.zeros((10, 3)), np.zeros(10, dtype=int), "at least 2 classes"),
        (np.zeros((10, 3)), np.arange(9), "9 labels"),
        (np.zeros((2, 3)), np.arange(2), "at least 3 samples"),
        (np.zeros(10), np.arange(10), "2-D"),
    ],
)
def test_rejects_malformed_input(x, z, msg):
    with pytest.raises(ValueError, match=msg):
        fit_leace(x, z)


def test_rejects_wrong_feature_width_at_apply_time():
    x, z, _ = _concept_data(seed=17)
    eraser = fit_leace(x, z)
    with pytest.raises(ValueError, match="expected"):
        eraser(np.zeros((5, 3)))


def test_removed_dir_spans_exactly_what_erasure_subtracts():
    """The diagnostic that asks WHERE the erased direction sits is only meaningful if the vector
    it inspects is the one actually subtracted. Check it against the residual itself rather than
    against the internals that produced it."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 25))
    z = np.asarray(rng.integers(0, 2, size=60))
    x[:, 4] += 3.0 * z                                  # plant the concept on a real axis
    er = fit_leace(x, z)

    d = er.removed_dir
    assert d.shape == (25, 1), "binary concept => rank 1"
    assert np.allclose(d.T @ d, np.eye(1), atol=1e-10), "must be orthonormal"

    r = x - er(x)                                       # what erasure actually removed
    assert np.linalg.norm(r) > 1e-6, "nothing was removed; test is vacuous"
    resid = r - (r @ d) @ d.T                           # component outside the claimed span
    assert np.linalg.norm(resid) / np.linalg.norm(r) < 1e-10


def test_singular_values_line_up_with_the_basis():
    """`sv` exists so callers can weight `basis` columns by variance. Misalignment would silently
    mis-locate the erased direction in the spectrum."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 15)) @ np.diag(np.linspace(5, 0.1, 15))
    er = fit_leace(x, np.asarray(rng.integers(0, 2, size=40)))
    assert er.sv.shape[0] == er.basis.shape[1]
    assert np.all(np.diff(er.sv) <= 1e-9), "singular values must stay descending"
    xc = x - x.mean(0)
    assert np.allclose(np.linalg.norm(xc @ er.basis, axis=0), er.sv, rtol=1e-8)


def test_a_precomputed_svd_reproduces_the_eraser_exactly():
    """Sharing one factorisation across several concepts is only sound if it changes nothing."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(50, 20))
    z = np.asarray(rng.integers(0, 2, size=50))
    x[:, 2] += 2.0 * z
    want = fit_leace(x, z)
    got = fit_leace(x, z, svd=np.linalg.svd(x - x.mean(0), full_matrices=False))
    assert np.allclose(got.proj, want.proj, atol=1e-12)
    assert got.var_removed == pytest.approx(want.var_removed, rel=1e-12)
    assert np.allclose(np.abs(got.removed_dir), np.abs(want.removed_dir), atol=1e-12)


def test_a_mismatched_precomputed_svd_is_rejected_not_used():
    """Silently accepting a stale factorisation would produce a plausible, wrong eraser."""
    rng = np.random.default_rng(6)
    x = rng.normal(size=(30, 12))
    z = np.asarray(rng.integers(0, 2, size=30))
    with pytest.raises(ValueError, match="do not match a thin SVD"):
        fit_leace(x, z, svd=np.linalg.svd(rng.normal(size=(30, 9)), full_matrices=False))
