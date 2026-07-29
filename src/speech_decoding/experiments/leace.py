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
    sv: np.ndarray = None  # type: ignore[assignment]
    """Singular values of the centered fit data, aligned with ``basis`` columns.

    Kept so a caller can locate the erased direction in the VARIANCE SPECTRUM without paying for a
    second SVD of an (n, 93184) matrix -- the spectrum and the eraser come from one factorisation.
    """

    removed_dir: np.ndarray = None  # type: ignore[assignment]
    """(d, k) orthonormal basis of the subspace actually SUBTRACTED from x, in feature space.

    ``var_removed`` says how much variance the erasure destroyed but not WHERE that direction
    sits, and those are different questions. A rank-1 direction carrying 20% of the variance of an
    iEEG representation is equally consistent with "the model concentrated identity into one axis"
    and with "this is the global common-mode/amplitude axis, which differs across subjects for
    trivial recording reasons and which the decoder never used". Distinguishing them needs the
    direction itself -- its overlap with the leading principal components, and with the
    between-domain mean shift that per-feature standardisation already removes.

    This is the RANGE of the (oblique) projector, i.e. the thing removed, not the linear functional
    that detects the concept. For ``proj = D qa qaᵀ D⁻¹`` in basis coordinates the range is
    ``span(D qa)``, so in feature space it is ``basis @ (D qa)``, orthonormalised.
    """

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
    svd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> LeaceEraser:
    """Fit the eraser that removes concept ``z`` from features ``x``.

    ``n_components`` truncates the row-space basis before fitting, which is the shrinkage knob
    for ill-conditioned covariances; ``None`` keeps every direction above ``rtol``.

    ``svd`` accepts a precomputed thin SVD of the CENTERED ``x`` as ``(u, s, vt)`` and skips the
    factorisation. The SVD depends only on ``x``, never on the concept, so several erasers over the
    same features -- the real concept, a shuffled control, a variance-matched control -- cost ONE
    factorisation between them instead of one each. That is what makes a control arm affordable at
    d=93184, where the SVD is the entire bill and the ridge that follows is ~10% of it.
    CALLER'S RESPONSIBILITY: it must be the SVD of this exact centered matrix. Only the shapes are
    checked, and passing a stale factorisation yields a wrong eraser silently.
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

    if svd is None:
        u, s, vt = np.linalg.svd(xc, full_matrices=False)
    else:
        u, s, vt = svd
        k = min(xc.shape)
        if u.shape != (n, k) or s.shape != (k,) or vt.shape != (k, xc.shape[1]):
            raise ValueError(
                f"precomputed svd shapes {u.shape}/{s.shape}/{vt.shape} do not match a thin SVD "
                f"of {xc.shape}")
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
        removed_dir = np.zeros((x.shape[1], 0))
    else:
        qa = ua[:, live]
        proj = (s / scale)[:, None] * (qa @ qa.T) * (scale / s)[None, :]
        # Range of the projector, carried back to feature space and orthonormalised. QR rather
        # than a normalise: rank > 1 columns are not orthogonal to each other.
        removed_dir, _ = np.linalg.qr(basis @ ((s / scale)[:, None] * qa))

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
        sv=s,
        removed_dir=removed_dir,
    )
