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
