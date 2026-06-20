"""Logistic linear-probe scoring (upstream-parity), shared by the offline probes.

The online probe (:mod:`online_probe`) uses dual/kernel ridge — cheap, closed-form,
feature-dim-free — because it fits every ``cadence`` steps inside a live run. The
OFFLINE probes (raw-feature baseline, encoder↔raw head-to-head, Neuroprobe-Lite
baseline) instead fit an **L2 LogisticRegression** so their AUROCs sit on the SAME
axis as the upstream "logistic-on-Multi-STFT 0.663" line
(``scripts/neuroprobe/probe_v14_frozen_logistic.py::fit_sklearn``). Same recipe:
``StandardScaler`` fit on train only, ``LogisticRegression(C=1.0, max_iter=10000,
tol=1e-3)`` (default L2 / lbfgs), AUROC from ``predict_proba`` — binary uses the
positive-class column, multiclass is OVR macro-averaged.

Pure numpy + sklearn, no torch / BT / DCC — laptop-TDD'd in
``test_linear_probe_logistic``. The WS contiguous-2-fold / CS anchor→test SHAPES
mirror :func:`online_probe.ws_auroc_2fold` / :func:`online_probe.cs_auroc` exactly,
so swapping ridge→logistic changes only the estimator, not the protocol.
"""

from __future__ import annotations

import numpy as np

from speech_decoding.experiments.online_probe import contiguous_folds

__all__ = [
    "DEFAULT_C",
    "cs_auroc_logistic",
    "logistic_auroc",
    "ws_auroc_2fold_logistic",
]

DEFAULT_C: float = 1.0


def logistic_auroc(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    *,
    C: float = DEFAULT_C,
    seed: int = 0,
    max_iter: int = 10000,
) -> float:
    """Fit ``StandardScaler`` + L2 ``LogisticRegression`` on ``(z_train, y_train)``,
    return test-set ROC-AUC (upstream-parity ``fit_sklearn`` recipe).

    Labels may be ±1 (the probe tasks) or integer classes (multiclass upstream
    tasks). Binary → ``roc_auc_score`` against the positive class (``classes_[1]``)
    using ``proba[:, 1]``; multiclass → OVR macro. Returns NaN when the train OR
    test fold is single-class (AUROC undefined) so callers nan-mean over
    folds/subjects, matching :func:`online_probe.auroc`.

    ``max_iter`` bounds the lbfgs iterations. The default (10000) is the upstream
    recipe and converges for the raw d=1 and parcel-pooled matrices; the offline
    encoder head-to-head (:mod:`offline_probe_bench`) passes a smaller cap for the
    p≫n per-electrode d=256 fits, applied identically across taps so the
    raw↔encoder comparison stays apples-to-apples."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return float("nan")

    scaler = StandardScaler().fit(z_train)
    clf = LogisticRegression(C=C, random_state=seed, max_iter=max_iter, tol=1e-3)
    clf.fit(scaler.transform(z_train), y_train)
    proba = clf.predict_proba(scaler.transform(z_test))

    if len(clf.classes_) == 2:
        pos = clf.classes_[1]
        return float(roc_auc_score((y_test == pos).astype(int), proba[:, 1]))
    return float(
        roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=clf.classes_)
    )


def ws_auroc_2fold_logistic(
    z: np.ndarray, y: np.ndarray, *, C: float = DEFAULT_C, seed: int = 0,
    k_folds: int = 2, max_iter: int = 10000,
) -> float:
    """Within-session AUROC, logistic: contiguous (time-ordered) ``k_folds`` split,
    fit on each train half / score the held-out half, nan-mean the folds. Same
    split as :func:`online_probe.ws_auroc_2fold` (``contiguous_folds``), only the
    estimator differs (logistic, not ridge)."""
    scores: list[float] = []
    for train, test in contiguous_folds(len(y), k_folds):
        if len(train) == 0 or len(test) == 0:
            continue
        scores.append(
            logistic_auroc(z[train], y[train], z[test], y[test], C=C, seed=seed,
                           max_iter=max_iter)
        )
    return float(np.nanmean(scores)) if scores else float("nan")


def cs_auroc_logistic(
    z_anchor: np.ndarray,
    y_anchor: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    *,
    C: float = DEFAULT_C,
    seed: int = 0,
    max_iter: int = 10000,
) -> float:
    """Cross-subject/-session AUROC, logistic: fit on the anchor, score the test
    subject (caller restricts both to the shared parcel set). Logistic counterpart
    of :func:`online_probe.cs_auroc`."""
    return logistic_auroc(z_anchor, y_anchor, z_test, y_test, C=C, seed=seed,
                          max_iter=max_iter)
