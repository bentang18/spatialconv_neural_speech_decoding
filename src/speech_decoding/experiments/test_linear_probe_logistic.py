"""Tests for the shared logistic linear-probe core (linear_probe_logistic).

Covers the upstream-parity recipe (StandardScaler + L2 LogisticRegression, AUROC
from predict_proba) on the two label regimes the offline probes use — ±1 binary
(the three probe tasks) and integer multiclass (upstream tasks) — plus the
single-class NaN guard, determinism, and the WS contiguous-2-fold / CS anchor→test
shapes that mirror online_probe.
"""

from __future__ import annotations

import numpy as np

from speech_decoding.experiments.linear_probe_logistic import (
    cs_auroc_logistic,
    logistic_auroc,
    ws_auroc_2fold_logistic,
)


def _separable_binary(n=200, d=8, *, seed=0):
    """±1 labels with the first feature carrying the class so AUROC ≈ 1."""
    rng = np.random.RandomState(seed)
    y = np.array([1, -1] * (n // 2), dtype=np.float64)
    z = rng.randn(n, d)
    z[:, 0] += 3.0 * y
    return z, y


def test_logistic_auroc_separable_binary_is_high():
    z, y = _separable_binary()
    # train on the first half, test on the second (both class-balanced)
    a = logistic_auroc(z[:100], y[:100], z[100:], y[100:])
    assert a > 0.95


def test_logistic_auroc_handles_zero_one_labels_like_pm1():
    # {0,1} integer binary must score identically to the ±1 encoding of the same split.
    z, y = _separable_binary(seed=1)
    a_pm1 = logistic_auroc(z[:100], y[:100], z[100:], y[100:])
    y01 = (y > 0).astype(int)
    a_01 = logistic_auroc(z[:100], y01[:100], z[100:], y01[100:])
    assert abs(a_pm1 - a_01) < 1e-9


def test_logistic_auroc_single_class_train_or_test_is_nan():
    z, y = _separable_binary()
    yconst = np.ones_like(y)
    assert np.isnan(logistic_auroc(z[:100], yconst[:100], z[100:], y[100:]))   # train 1-class
    assert np.isnan(logistic_auroc(z[:100], y[:100], z[100:], yconst[100:]))   # test 1-class


def test_logistic_auroc_multiclass_ovr_macro():
    # 3 well-separated gaussian blobs -> OVR-macro AUROC near 1.
    rng = np.random.RandomState(0)
    centers = np.array([[0, 0], [6, 0], [0, 6]], dtype=float)
    y = np.repeat([0, 1, 2], 60)
    z = np.vstack([rng.randn(60, 2) + c for c in centers])
    perm = rng.permutation(len(y))
    z, y = z[perm], y[perm]
    a = logistic_auroc(z[:120], y[:120], z[120:], y[120:])
    assert 0.9 < a <= 1.0


def test_logistic_auroc_deterministic():
    z, y = _separable_binary()
    assert logistic_auroc(z[:100], y[:100], z[100:], y[100:]) == logistic_auroc(
        z[:100], y[:100], z[100:], y[100:]
    )


def test_ws_auroc_2fold_logistic_structured():
    z, y = _separable_binary(n=240)
    a = ws_auroc_2fold_logistic(z, y)
    assert 0.9 < a <= 1.0


def test_cs_auroc_logistic_matches_direct_fit():
    z, y = _separable_binary(seed=3)
    direct = logistic_auroc(z[:100], y[:100], z[100:], y[100:])
    cs = cs_auroc_logistic(z[:100], y[:100], z[100:], y[100:])
    assert cs == direct
