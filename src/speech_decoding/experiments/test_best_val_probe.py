"""Tests for the S06 best-val probe callback.

Covers:

* :func:`fit_linear_probe_score` ridge → r² (linear signal → r² ≈ 1.0;
  noise → r² ≈ 0; constant target → r² = 0.0 not nan).
* :func:`fit_linear_probe_score` logistic → AUROC (separable signal →
  AUROC ≈ 1.0; random labels → ≈ 0.5).
* Validation rejects: non-2D features, label/feature mismatch, mode
  outside the allowed set, non-{0,1} labels for classification.
* :class:`BestValProbeR2Callback` collects features + labels from a
  toy probe DataLoader using the default mean-pool M4 extractor.
* The callback's ``on_save_checkpoint`` hook injects the EMA teacher's
  state_dict under ``"ema_teacher"`` (S06 §3 co-save contract).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import Tensor

from speech_decoding.experiments.best_val_probe import (
    BestValProbeR2Callback,
    DEFAULT_N_FOLDS,
    PROBE_RIDGE_LAMBDA,
    ProbeScore,
    fit_linear_probe_score,
)


# ---------------------------------------------------------------------------
# fit_linear_probe_score — regression
# ---------------------------------------------------------------------------


def test_fit_linear_probe_score_regression_recovers_linear_target() -> None:
    """y = Xw + ε with small ε → ridge probe recovers near-perfect r²."""
    rng = np.random.default_rng(0)
    n, d = 200, 5
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d)
    y = X @ w_true + 0.01 * rng.standard_normal(n)
    score = fit_linear_probe_score(X, y, mode="regression", seed=0)
    assert score.mode == "regression"
    assert score.n_folds == DEFAULT_N_FOLDS
    assert score.score > 0.98


def test_fit_linear_probe_score_regression_noise_yields_near_zero_r2() -> None:
    """Random features uncorrelated with target → r² near 0."""
    rng = np.random.default_rng(1)
    n, d = 150, 8
    X = rng.standard_normal((n, d))
    y = rng.standard_normal(n)
    score = fit_linear_probe_score(X, y, mode="regression", seed=0)
    assert score.score < 0.3


def test_fit_linear_probe_score_regression_constant_target_returns_zero() -> None:
    """Constant target → r² is 0.0 not nan (callback wouldn't crash)."""
    X = np.random.default_rng(2).standard_normal((40, 4))
    y = np.full(40, 7.0)
    score = fit_linear_probe_score(X, y, mode="regression", seed=0)
    assert score.score == 0.0


# ---------------------------------------------------------------------------
# fit_linear_probe_score — binary classification
# ---------------------------------------------------------------------------


def test_fit_linear_probe_score_binary_separable_yields_auroc_near_one() -> None:
    """Linearly separable binary labels → AUROC ≈ 1.0."""
    rng = np.random.default_rng(3)
    n, d = 100, 4
    X = rng.standard_normal((n, d))
    y = (X[:, 0] > 0).astype(np.float64)
    score = fit_linear_probe_score(
        X, y, mode="binary_classification", seed=0,
        logistic_iter=300,
    )
    assert score.mode == "binary_classification"
    assert score.score > 0.95


def test_fit_linear_probe_score_binary_random_yields_auroc_near_half() -> None:
    """Random labels → AUROC near 0.5."""
    rng = np.random.default_rng(4)
    n, d = 200, 6
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n).astype(np.float64)
    score = fit_linear_probe_score(
        X, y, mode="binary_classification", seed=0,
    )
    # AUROC ~ 0.5 ± a few percent under finite-sample noise.
    assert 0.35 < score.score < 0.65


# ---------------------------------------------------------------------------
# Validation rejects
# ---------------------------------------------------------------------------


def test_fit_linear_probe_score_rejects_non_2d_features() -> None:
    with pytest.raises(ValueError, match=r"\(N, d\)"):
        fit_linear_probe_score(np.zeros((4, 5, 3)), np.zeros(4))


def test_fit_linear_probe_score_rejects_label_mismatch() -> None:
    with pytest.raises(ValueError, match="align with features"):
        fit_linear_probe_score(np.zeros((10, 3)), np.zeros(7))


def test_fit_linear_probe_score_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="regression"):
        fit_linear_probe_score(
            np.zeros((10, 3)), np.zeros(10), mode="multi_class",  # type: ignore[arg-type]
        )


def test_fit_linear_probe_score_rejects_non_binary_labels_for_classification() -> None:
    X = np.zeros((6, 3))
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.float64)
    with pytest.raises(ValueError, match=r"labels in \{0, 1\}"):
        fit_linear_probe_score(X, y, mode="binary_classification")


def test_fit_linear_probe_score_rejects_insufficient_samples_for_folds() -> None:
    with pytest.raises(ValueError, match="smaller than n_folds"):
        fit_linear_probe_score(
            np.zeros((3, 2)), np.zeros(3), mode="regression", n_folds=5,
        )


def test_fit_linear_probe_score_accepts_torch_tensors() -> None:
    """Torch features + labels should be handled identically to numpy."""
    rng = np.random.default_rng(5)
    n, d = 80, 4
    X = torch.from_numpy(rng.standard_normal((n, d)))
    w = rng.standard_normal(d)
    y = torch.from_numpy(X.numpy() @ w + 0.01 * rng.standard_normal(n))
    score = fit_linear_probe_score(X, y, mode="regression", seed=0)
    assert score.score > 0.95


# ---------------------------------------------------------------------------
# BestValProbeR2Callback
# ---------------------------------------------------------------------------


class _ToyEncoder(torch.nn.Module):
    """Encoder that returns an M4 tap shaped (B, P, T, d) so the default
    mean-pool extractor exercises the (parcel, time) reduction."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(d, d)

    def forward(self, **kwargs: Tensor) -> dict[str, Tensor]:
        x = kwargs["electrode_tokens"]                    # (B, C, T, F)
        B, _, T, _ = x.shape
        P, d = 2, self.linear.out_features
        # Use the first F dim flattened through Linear → (B, P, T, d).
        proj = self.linear(x.mean(dim=1).mean(dim=-1, keepdim=True).expand(B, T, d))
        m4 = proj.unsqueeze(1).expand(B, P, T, d).contiguous()
        return {"M2": m4, "M3": m4, "M4": m4}


class _ToyTeacher(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flag = torch.nn.Parameter(torch.tensor(1.0))

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return {"flag": self.flag.detach().clone()}


class _ToyLitModule:
    def __init__(self, d: int, n_clips: int) -> None:
        self.student = _ToyEncoder(d)
        self.teacher = SimpleNamespace(model=_ToyTeacher())
        self.logged: dict[str, float] = {}
        self._n_clips = n_clips

    def eval(self) -> None:
        self.student.eval()

    def log(self, key: str, value: float, **kwargs) -> None:  # noqa: ARG002
        self.logged[key] = float(value)


def _make_probe_loader(
    *,
    n_clips: int = 24, C: int = 3, T: int = 4, F: int = 2,
) -> list[SimpleNamespace]:
    """Toy probe loader yielding 2 batches of {electrode_tokens,
    support, probe_label}."""
    torch.manual_seed(0)
    half = n_clips // 2
    batches = []
    for b in range(2):
        electrode_tokens = torch.randn(half, C, T, F)
        support = torch.eye(C)[:, :2].unsqueeze(0).expand(half, C, 2).clone()
        valid_mask = torch.ones(half, C, dtype=torch.bool)
        probe_label = torch.randint(0, 2, (half,)).float()
        batches.append(SimpleNamespace(data={
            "electrode_tokens": electrode_tokens,
            "support": support,
            "valid_mask": valid_mask,
            "probe_label": probe_label,
        }))
    return batches


def test_best_val_probe_callback_collects_features_and_labels() -> None:
    loader = _make_probe_loader(n_clips=20)
    module = _ToyLitModule(d=8, n_clips=20)
    callback = BestValProbeR2Callback(loader, mode="binary_classification")
    feats, lbls = callback.collect_features_and_labels(module)
    assert feats.shape == (20, 8)
    assert lbls.shape == (20,)
    assert callback.metric_name == "val_probe_auroc"


def test_best_val_probe_callback_default_metric_name_for_regression() -> None:
    callback = BestValProbeR2Callback([], mode="regression")
    assert callback.metric_name == "val_probe_r2"


def test_best_val_probe_callback_validation_epoch_end_logs_probe_score() -> None:
    loader = _make_probe_loader(n_clips=20)
    module = _ToyLitModule(d=8, n_clips=20)
    callback = BestValProbeR2Callback(loader, mode="binary_classification")
    callback.on_validation_epoch_end(trainer=None, pl_module=module)
    assert "val_probe_auroc" in module.logged
    # AUROC must land in [0, 1].
    assert 0.0 <= module.logged["val_probe_auroc"] <= 1.0


def test_best_val_probe_callback_co_saves_ema_teacher() -> None:
    """S06 §3 lock: every checkpoint must carry the EMA teacher state."""
    module = _ToyLitModule(d=4, n_clips=8)
    callback = BestValProbeR2Callback([], mode="regression")
    ckpt: dict = {}
    callback.on_save_checkpoint(trainer=None, pl_module=module, checkpoint=ckpt)
    assert "ema_teacher" in ckpt
    assert "flag" in ckpt["ema_teacher"]


def test_best_val_probe_callback_co_save_off_skips_ema_state() -> None:
    """When ``co_save_ema_teacher=False`` the hook is a no-op."""
    module = _ToyLitModule(d=4, n_clips=8)
    callback = BestValProbeR2Callback(
        [], mode="regression", co_save_ema_teacher=False,
    )
    ckpt: dict = {}
    callback.on_save_checkpoint(trainer=None, pl_module=module, checkpoint=ckpt)
    assert "ema_teacher" not in ckpt


def test_best_val_probe_callback_rejects_loader_missing_probe_label() -> None:
    bad_loader = [SimpleNamespace(data={
        "electrode_tokens": torch.zeros(2, 3, 4, 2),
        "support": torch.zeros(2, 3, 2),
    })]
    module = _ToyLitModule(d=4, n_clips=2)
    callback = BestValProbeR2Callback(bad_loader, mode="regression")
    with pytest.raises(KeyError, match="probe_label"):
        callback.collect_features_and_labels(module)


def test_best_val_probe_callback_rejects_empty_loader() -> None:
    module = _ToyLitModule(d=4, n_clips=0)
    callback = BestValProbeR2Callback([], mode="regression")
    with pytest.raises(RuntimeError, match="zero batches"):
        callback.collect_features_and_labels(module)


def test_best_val_probe_callback_default_ridge_lambda_locked() -> None:
    assert PROBE_RIDGE_LAMBDA == 1.0


def test_probe_score_returns_per_fold_breakdown() -> None:
    rng = np.random.default_rng(6)
    X = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)
    score = fit_linear_probe_score(X, y, mode="regression", n_folds=5)
    assert isinstance(score, ProbeScore)
    assert len(score.per_fold) == 5
