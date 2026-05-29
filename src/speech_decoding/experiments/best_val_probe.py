"""Best-val probe callback (S06): downstream linear probe r² / AUROC.

Spec: ``docs/neuroprobe/v14_blockers.md §S06`` (Checkpoint cadence +
best-val criterion, 2026-05-26).

S06 verbatim:
    "Every 5k steps; best-3 by downstream probe r²; co-save EMA teacher
    state. Primary criterion = downstream probe r² on tiny held-out
    Neuroprobe task at every 5k save. Default: linear logistic
    regression on `speech-onset binary` (sub-1, M07 session-held-out,
    5-fold CV) from frozen `M4` mean-pooled-T."

This module owns two things:

* :func:`fit_linear_probe_score` — pure function that takes
  ``(features, labels)`` and returns the held-out probe score. Two modes:
  ``regression`` (ridge → r²) and ``binary_classification`` (logistic →
  AUROC). Pure-numpy fold split + cross-validated score, no sklearn
  dependency.
* :class:`BestValProbeR2Callback` — Lightning callback that, at every
  validation epoch end, (i) iterates a probe DataLoader, (ii) extracts
  PMA-pooled or mean-pooled M4 features from the student encoder, (iii)
  calls :func:`fit_linear_probe_score`, (iv) logs the score as
  ``val_probe_r2`` (regression) or ``val_probe_auroc`` (classification),
  and (v) co-saves the EMA teacher in every checkpoint (data2vec 2.0 /
  V-JEPA precedent).

The probe DataLoader and the M07 session-held-out split itself are
out-of-scope for this module — they live in the multi-corpus loader
build. This module gives the callback its scoring + checkpoint surface
so the rest of the pipeline can call into it as soon as the loader
exists.
"""

from __future__ import annotations

import dataclasses
import typing as tp

import numpy as np
import torch
from torch import Tensor


PROBE_RIDGE_LAMBDA: float = 1.0
"""Default ridge regularizer for the regression probe (~standard for
small-batch linear probes; the held-out r² is insensitive to λ in the
[0.1, 10] band for our scales)."""

DEFAULT_N_FOLDS: int = 5
"""S06 spec: 5-fold CV."""


@dataclasses.dataclass(frozen=True)
class ProbeScore:
    """Held-out probe score returned by :func:`fit_linear_probe_score`."""

    score: float
    mode: str
    n_folds: int
    per_fold: tuple[float, ...]


def _kfold_indices(
    n: int, *, n_folds: int, generator: tp.Optional[np.random.Generator] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, val_idx)`` for each of ``n_folds`` folds.

    Deterministic given ``generator`` (or a fixed default). Splits are
    contiguous shuffled blocks — adequate for clip-level probes where
    nothing inside a clip leaks across the split.
    """
    if n < n_folds:
        raise ValueError(
            f"n={n} samples is smaller than n_folds={n_folds}; "
            "use fewer folds or supply more probe clips"
        )
    rng = generator if generator is not None else np.random.default_rng(0)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        val = folds[k]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        out.append((train, val))
    return out


def _ridge_fit(X: np.ndarray, y: np.ndarray, *, lam: float) -> np.ndarray:
    """Closed-form ridge: ``w = (XᵀX + λI)⁻¹ Xᵀy``. ``X`` is
    ``(n, d+1)`` with the bias column already appended."""
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d, dtype=X.dtype)
    b = X.T @ y
    return np.linalg.solve(A, b)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination ``1 - SSR / SST``. Returns ``0.0``
    when the target is constant (SST = 0) — keeps the score finite and
    non-informative without crashing the callback."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _logistic_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lam: float,
    n_iter: int = 100,
    lr: float = 0.1,
) -> np.ndarray:
    """L2-regularized logistic regression via gradient descent.

    Pure numpy so we don't pull sklearn into the inner loop. Bias column
    expected in ``X`` already. ``y ∈ {0, 1}``.
    """
    d = X.shape[1]
    w = np.zeros(d, dtype=X.dtype)
    n = X.shape[0]
    for _ in range(n_iter):
        z = X @ w
        # Numerically stable sigmoid via piecewise.
        p = np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z)),
        )
        grad = X.T @ (p - y) / n + lam * w
        w = w - lr * grad
    return w


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUROC. Constant-label edge case → returns 0.5."""
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def fit_linear_probe_score(
    features: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
    *,
    mode: tp.Literal["regression", "binary_classification"] = "regression",
    n_folds: int = DEFAULT_N_FOLDS,
    ridge_lambda: float = PROBE_RIDGE_LAMBDA,
    logistic_lambda: float = 1e-3,
    logistic_iter: int = 200,
    logistic_lr: float = 0.1,
    seed: int = 0,
) -> ProbeScore:
    """Fit a linear probe and return the cross-validated held-out score.

    Parameters
    ----------
    features : ``(N, d)`` clip-level features (numpy or torch). For the
        S06 default this is the student's PMA-pooled or mean-pooled-T M4
        tap on the probe DataLoader's held-out clips.
    labels : ``(N,)`` per-clip target. For ``regression`` any float
        target; for ``binary_classification`` ``{0, 1}`` (or convertible
        to such).
    mode : ``"regression"`` → ridge probe, score = r². ``"binary_classification"``
        → logistic probe, score = AUROC.
    n_folds : Cross-validation folds. Default ``5`` per S06.
    ridge_lambda / logistic_lambda : Per-mode L2 regularizer.
    logistic_iter / logistic_lr : Logistic GD knobs (constant-LR; no
        sklearn dep).
    seed : Seed for the fold permutation. Same seed across val epochs →
        reproducible probe score.

    Returns
    -------
    :class:`ProbeScore` — the mean held-out score across folds, the
    per-fold scores, and the chosen ``mode``.
    """
    if mode not in ("regression", "binary_classification"):
        raise ValueError(
            f"mode must be 'regression' or 'binary_classification'; "
            f"got {mode!r}"
        )
    x_np = features.detach().cpu().numpy() if isinstance(features, Tensor) else np.asarray(features)
    y_np = labels.detach().cpu().numpy() if isinstance(labels, Tensor) else np.asarray(labels)
    x_np = x_np.astype(np.float64, copy=False)
    y_np = y_np.astype(np.float64, copy=False)
    if x_np.ndim != 2:
        raise ValueError(
            f"features must be (N, d); got shape {x_np.shape}"
        )
    if y_np.ndim != 1 or y_np.shape[0] != x_np.shape[0]:
        raise ValueError(
            f"labels must be (N,) and align with features; got shape "
            f"{y_np.shape} for N={x_np.shape[0]}"
        )
    if mode == "binary_classification":
        unique = np.unique(y_np)
        if not np.all(np.isin(unique, [0.0, 1.0])):
            raise ValueError(
                "binary_classification requires labels in {0, 1}; "
                f"got unique values {unique.tolist()}"
            )

    n = x_np.shape[0]
    X = np.concatenate([x_np, np.ones((n, 1), dtype=x_np.dtype)], axis=1)
    folds = _kfold_indices(n, n_folds=n_folds, generator=np.random.default_rng(seed))
    per_fold: list[float] = []
    for train_idx, val_idx in folds:
        if mode == "regression":
            w = _ridge_fit(X[train_idx], y_np[train_idx], lam=ridge_lambda)
            y_pred = X[val_idx] @ w
            per_fold.append(_r2(y_np[val_idx], y_pred))
        else:
            w = _logistic_fit(
                X[train_idx], y_np[train_idx],
                lam=logistic_lambda,
                n_iter=logistic_iter,
                lr=logistic_lr,
            )
            scores = X[val_idx] @ w
            per_fold.append(_auroc(y_np[val_idx], scores))

    return ProbeScore(
        score=float(np.mean(per_fold)),
        mode=mode,
        n_folds=n_folds,
        per_fold=tuple(per_fold),
    )


class BestValProbeR2Callback:
    """Lightning callback wiring :func:`fit_linear_probe_score` into the
    joint-phase val loop.

    Surface matches Lightning's callback hooks but is intentionally not
    inheriting from :class:`pytorch_lightning.Callback` here — the
    callback gets registered at dispatch time, and inheriting at module
    import time would pull Lightning into the test surface unnecessarily.
    Lightning's callback registration accepts any object with the
    matching method names (duck-typed `on_validation_epoch_end` etc.).

    Parameters
    ----------
    probe_dataloader: iterable of ``(features, labels)`` tuples or of
        ``batch`` objects whose ``.data`` dict contains a
        ``"electrode_tokens"`` tensor + a ``"probe_label"`` tensor. The
        callback iterates this loader with the student encoder set to
        eval mode and accumulates per-clip features.
    feature_extractor: optional callable ``(pl_module, batch_data) ->
        Tensor[N, d]`` that selects the probe features from the module's
        encoder output. Default mean-pools the M4 tap over the time and
        parcel axes.
    mode: ``"regression"`` (r²) or ``"binary_classification"`` (AUROC).
    metric_name: key used to log the probe score. Default matches the
        mode: ``"val_probe_r2"`` / ``"val_probe_auroc"``.
    co_save_ema_teacher: when ``True``, hooks
        ``on_save_checkpoint`` to inject the EMA teacher's
        ``state_dict`` under ``"ema_teacher"`` (S06 §3).
    seed: deterministic fold permutation seed. Same value across epochs
        so the probe score is reproducible.
    """

    def __init__(
        self,
        probe_dataloader: tp.Iterable[tp.Any],
        *,
        feature_extractor: tp.Optional[
            tp.Callable[[tp.Any, dict[str, Tensor]], Tensor]
        ] = None,
        mode: tp.Literal["regression", "binary_classification"] = "regression",
        metric_name: tp.Optional[str] = None,
        n_folds: int = DEFAULT_N_FOLDS,
        ridge_lambda: float = PROBE_RIDGE_LAMBDA,
        co_save_ema_teacher: bool = True,
        seed: int = 0,
    ) -> None:
        self._probe_dataloader = probe_dataloader
        self._feature_extractor = (
            feature_extractor if feature_extractor is not None
            else _default_mean_pool_m4_feature_extractor
        )
        self._mode = mode
        self._metric_name = metric_name or (
            "val_probe_r2" if mode == "regression" else "val_probe_auroc"
        )
        self._n_folds = n_folds
        self._ridge_lambda = ridge_lambda
        self._co_save_ema_teacher = co_save_ema_teacher
        self._seed = seed

    @property
    def metric_name(self) -> str:
        return self._metric_name

    def collect_features_and_labels(
        self,
        pl_module: tp.Any,
    ) -> tuple[Tensor, Tensor]:
        """Iterate the probe loader and stack per-clip features + labels."""
        feats: list[Tensor] = []
        lbls: list[Tensor] = []
        pl_module.eval()
        with torch.no_grad():
            for batch in self._probe_dataloader:
                batch_data = (
                    batch.data if hasattr(batch, "data") else batch
                )
                if not isinstance(batch_data, dict):
                    raise TypeError(
                        "probe_dataloader items must be dicts or have a "
                        f".data dict attribute; got {type(batch).__name__}"
                    )
                if "probe_label" not in batch_data:
                    raise KeyError(
                        "probe_dataloader batch missing 'probe_label' key; "
                        "S06's probe target column must be supplied by the "
                        "loader builder"
                    )
                features = self._feature_extractor(pl_module, batch_data)
                if features.dim() != 2:
                    raise ValueError(
                        f"feature_extractor must return (N, d); got shape "
                        f"{tuple(features.shape)}"
                    )
                feats.append(features.detach().cpu())
                lbls.append(batch_data["probe_label"].detach().cpu())
        if not feats:
            raise RuntimeError(
                "probe_dataloader yielded zero batches; ensure the "
                "held-out probe set is non-empty before val epoch end"
            )
        return torch.cat(feats, dim=0), torch.cat(lbls, dim=0)

    def on_validation_epoch_end(self, trainer: tp.Any, pl_module: tp.Any) -> None:  # noqa: ARG002
        """Lightning hook: fit the probe + log the score."""
        features, labels = self.collect_features_and_labels(pl_module)
        result = fit_linear_probe_score(
            features, labels,
            mode=self._mode,
            n_folds=self._n_folds,
            ridge_lambda=self._ridge_lambda,
            seed=self._seed,
        )
        pl_module.log(
            self._metric_name, float(result.score),
            on_epoch=True, prog_bar=True,
        )

    def on_save_checkpoint(  # noqa: D401
        self, trainer: tp.Any, pl_module: tp.Any, checkpoint: dict[str, tp.Any],
    ) -> None:
        """S06 §3: co-save the EMA teacher state in every checkpoint."""
        if not self._co_save_ema_teacher:
            return
        teacher = getattr(pl_module, "teacher", None)
        if teacher is None:
            return
        inner = getattr(teacher, "model", teacher)
        try:
            checkpoint["ema_teacher"] = inner.state_dict()
        except Exception:
            # Don't crash a 5k-step save because the EMA mirror's
            # state_dict surface drifted; log on the module and move on.
            pl_module.log(
                f"{self._metric_name}_ema_save_skipped",
                1.0,
                on_epoch=True, prog_bar=False,
            )


def _default_mean_pool_m4_feature_extractor(
    pl_module: tp.Any, batch_data: dict[str, Tensor],
) -> Tensor:
    """Default S06 feature extractor: ``mean(M4, dim=(parcel, time))``.

    Calls ``pl_module.student(...)`` with the same kwarg surface used by
    the joint module's ``_step``. Returns ``(B, d)``.
    """
    kwargs = {
        "electrode_tokens": batch_data["electrode_tokens"],
        "support": batch_data["support"],
        "valid_mask": batch_data.get("valid_mask"),
    }
    taps = pl_module.student(**{k: v for k, v in kwargs.items() if v is not None})
    if not isinstance(taps, dict) or "M4" not in taps:
        raise KeyError(
            "default_mean_pool_m4_feature_extractor expected the student "
            "encoder to return a dict with an 'M4' tap"
        )
    m4 = taps["M4"]  # (B, P, T, d) per v14 layout
    if m4.dim() != 4:
        raise ValueError(
            f"M4 tap must be (B, P, T, d); got shape {tuple(m4.shape)}"
        )
    return m4.mean(dim=(1, 2))


__all__ = [
    "BestValProbeR2Callback",
    "DEFAULT_N_FOLDS",
    "PROBE_RIDGE_LAMBDA",
    "ProbeScore",
    "fit_linear_probe_score",
]
