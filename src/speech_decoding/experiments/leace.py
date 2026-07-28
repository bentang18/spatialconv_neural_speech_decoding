"""LEACE — least-squares concept erasure (Belrose et al., arXiv 2306.03819).

Closed-form affine map ``r(x) = x - P(x - mu)`` that drives ``Cov(r(X), Z)`` to exactly zero, so
no linear classifier on ``r(X)`` predicts the concept ``Z`` better than a constant. ``P`` has rank
at most ``rank(Sigma_XZ) <= C - 1`` for ``C`` concept classes: the erasure spends at most ``C - 1``
directions no matter how the concept sits in the variance spectrum.

Fitted in the row space of the centered data via SVD. For ``n << d`` -- our regime, ``n ~ 1e3``
trials against ``d = 13312`` features -- forming the ``d x d`` covariance is both wasteful and
rank-deficient. Directions orthogonal to the row space carry zero variance and zero covariance
with ``Z``, so erasure there is the identity and dropping them is exact, not an approximation.

The guarantee is LINEAR ONLY: a nonlinear probe can still recover ``Z``. That is the correct scope
here because the cross-subject readout is ridge, so the erasure and the readout share a function
class. It would prove nothing about the CNN/MLP decoders.

Fit on train subjects, apply the same map to train and test, so the readout sees one consistent
space. An unchanged downstream score is only interpretable once you have confirmed the erasure
actually transferred -- re-probe the concept on the ERASED TEST features and require chance.
``var_removed`` reports the fraction of total feature variance the map destroyed, which turns the
minimum-damage property from a citation into a measured number. Note the floor: the map always
spends its full rank budget, so an uncorrelated concept still costs ``(C - 1) / rank`` of the
variance. A ``var_removed`` far above that floor means the concept rides high-variance directions.

``residual_cov`` is the self-check -- concept covariance remaining on the FIT data, as a fraction
of what was there before. It is ~0 at full rank, but ``n_components`` truncation trades exactness
for conditioning and leaves a real residual, because the concept can have a component outside the
retained basis. Read it before trusting any downstream null result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LeaceEraser", "fit_leace"]


def _one_hot(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    if z.ndim == 2:
        return z.astype(np.float64)
    if z.ndim != 1:
        raise ValueError(f"concept labels must be 1-D or 2-D, got shape {z.shape}")
    classes = np.unique(z)
    if classes.size < 2:
        raise ValueError("concept needs at least 2 classes to erase")
    return (z[:, None] == classes[None, :]).astype(np.float64)


@dataclass(frozen=True)
class LeaceEraser:
    """Affine eraser. Call it on any array with the fitted feature dimension."""

    mean: np.ndarray
    basis: np.ndarray
    proj: np.ndarray
    var_removed: float
    residual_cov: float

    @property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.proj))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.mean.size:
            raise ValueError(f"expected (n, {self.mean.size}), got {x.shape}")
        return x - ((x - self.mean) @ self.basis) @ self.proj.T @ self.basis.T


def fit_leace(
    x: np.ndarray,
    z: np.ndarray,
    *,
    n_components: int | None = None,
    rtol: float = 1e-10,
) -> LeaceEraser:
    """Fit the eraser that removes concept ``z`` from features ``x``.

    ``n_components`` truncates the row-space basis before fitting, which is the shrinkage knob
    for ill-conditioned covariances; ``None`` keeps every direction above ``rtol``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {x.shape}")
    n = x.shape[0]
    zc = _one_hot(z)
    if zc.shape[0] != n:
        raise ValueError(f"got {n} feature rows but {zc.shape[0]} labels")
    if n < 3:
        raise ValueError("need at least 3 samples to estimate a covariance")

    mean = x.mean(0)
    xc = x - mean
    zc = zc - zc.mean(0)

    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    keep = s > (s[0] * rtol if s[0] > 0 else 0.0)
    if n_components is not None:
        keep &= np.arange(s.size) < n_components
    if not keep.any():
        raise ValueError("features have no variance to erase from")

    basis = vt[keep].T
    s = s[keep]
    y = u[:, keep] * s

    scale = np.sqrt(n - 1)
    sigma_yz = y.T @ zc / (n - 1)
    a = (scale / s)[:, None] * sigma_yz

    ua, sa, _ = np.linalg.svd(a, full_matrices=False)
    live = sa > (sa[0] * rtol if sa.size and sa[0] > 0 else 0.0)
    if not live.any():
        proj = np.zeros((s.size, s.size))
    else:
        qa = ua[:, live]
        proj = (s / scale)[:, None] * (qa @ qa.T) * (scale / s)[None, :]

    removed = (y @ proj.T)
    total_var = float((s**2).sum() / (n - 1))
    var_removed = float((removed**2).sum() / (n - 1) / total_var) if total_var > 0 else 0.0

    sigma_xz = xc.T @ zc / (n - 1)
    after = sigma_xz - basis @ (proj @ (basis.T @ sigma_xz))
    before_max = float(np.abs(sigma_xz).max())
    residual_cov = float(np.abs(after).max() / before_max) if before_max > 0 else 0.0

    return LeaceEraser(
        mean=mean,
        basis=basis,
        proj=proj,
        var_removed=var_removed,
        residual_cov=residual_cov,
    )
