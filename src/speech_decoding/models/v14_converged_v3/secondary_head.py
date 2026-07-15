"""v14_converged_v3 r4 — secondary-head STATE distribution + Gaussian NLL.

The secondary (perceiver) objective predicts, per present (parcel, 4 Hz slot), the
model-free parcel-state 6-vector

    x = [z_slow_mu, z_mid_mu, z_hga_mu, z_slow_sd, z_mid_sd, z_hga_sd]

(the 3 per-band parcel MEANS + the 3 per-band within-parcel STDs; z-scored per
(subject, parcel, dim), common-mode-removed — contract project-r4-contract-2026-07-15 §7).

HEAD FORM — single full-covariance Gaussian NLL, noise-floored (Ben-locked 2026-07-15,
from first principles): the target is a smooth, low-dim, near-Gaussian vector with a
KNOWN per-dim measurement-noise floor and dominantly-LINEAR cross-dim coupling (M17:
hist-TC 0.166 <= gauss-TC 0.200 => no nonlinear structure a Gaussian would miss). So a
single 6-D Gaussian (an MDN / "GMM" at ONE component) captures the joint EXACTLY through
its covariance. Go to a mixture only on measured multimodality (none). NOT soft-bins:
discretizing a smooth Gaussian is machinery to approximate what a Gaussian models natively.

    Sigma = L Lᵀ + N        L = softplus-diagonal lower-tri Cholesky the head emits
    N     = diag(sigma²_noise)  FIXED = 1 − reliability (means M11, stds M17)

N is the S-JEPA "honest ambiguity" property expressed as an additive observation-noise
covariance: the target is signal + noise, noise ~ N(0, N), so x ~ N(mu, L Lᵀ + N). The
loss can never reward resolving below the measurement floor (Sigma's marginals >= N),
and a perfect predictor's NLL floors at the target's own differential entropy (the
ceiling). Because N is strictly positive Sigma is always PD => the NLL is numerically
safe with no jitter.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# Per-dim measurement-noise variance = 1 − split-half reliability, in the z-scored units
# the target lives in (unit marginal variance per dim). Order:
#   [slow_mu, mid_mu, hga_mu, slow_sd, mid_sd, hga_sd].
# Refreshed to the ACTUAL target rate + processing (#28 cm-removed 4 Hz sweep, 2026-07-15):
# means  reliability slow .881 / mid .796 / hga .828  (HGA-mean RISES vs .561 native — the 4 Hz
#   temporal window-average denoises HGA rather than losing it; the 8 Hz arm was worse on every
#   dim, headroom 3.53 < 3.82 nats, so 4 Hz is the evidence-based grid).
# stds   reliability slow .652 / mid .496 / hga .546.
NOISE_VAR: tuple[float, ...] = (0.119, 0.204, 0.172, 0.348, 0.504, 0.454)
STATE_DIM: int = len(NOISE_VAR)

# --- count-dependent noise floor (Ben 2026-07-15: "weight dependent") --------------------
# A parcel's within-parcel STD is a sample statistic over its n electrodes: measurement is
# WORSE for small parcels. The distribution probe found 28% of std-loss terms come from
# 2-4 electrode parcels, so a FIXED floor over-trusts a quarter of the std targets. Make the
# 3 STD floors depend on n via the SAMPLING VARIANCE of the sample std (Var(s) ∝ 1/(n−1) —
# the "1/√(n−1)" law), anchored on the reliability MEASURED at the reference electrode count.
# The 3 MEAN dims keep a FIXED floor for v1: the parcel mean is robust, and Spearman-Brown's
# r(n→∞)→1 idealization (ignoring shared biological noise) is shakier than the std sampling
# law — mean-count-dependence is a clean HELD option, not built for v1.
#
# REFRESHED anchors (#28 cm-removed 4 Hz sweep, 2026-07-15): std reliability at the sweep's
# effective reference electron count. STD_REF_RELIABILITY = 1 − NOISE_VAR[3:] (consistent by
# construction); N_REF is the mean n_elec the reliability was averaged over in that sweep.
STD_REF_RELIABILITY: tuple[float, ...] = (0.652, 0.496, 0.546)  # slow/mid/hga std reliab @ n_ref
N_REF: float = 13.35


def count_dependent_noise_var(
    n_elec: Tensor,
    *,
    mean_var: tuple[float, ...] = NOISE_VAR[:3],
    std_r_ref: tuple[float, ...] = STD_REF_RELIABILITY,
    n_ref: float = N_REF,
) -> Tensor:
    """Per-(parcel) 6-vector noise floor N, keyed on the parcel's electrode count.

    ``n_elec`` (...,) long ≥ 1. Returns (..., 6) = [3 FIXED mean floors, 3 count-dependent
    std floors]. The std floor uses the sample-std sampling-variance law in RELIABILITY
    space (unit-invariant, so the z-scoring doesn't touch it): with the noise fraction of
    the unit-variance std target scaling as 1/(n−1),

        r_σ(n) = 1 / (1 + (1−r_ref)/r_ref · (n_ref−1)/(n−1)),   N_σ(n) = 1 − r_σ(n)

    so N_σ(n_ref) = 1 − r_ref (recovers the anchor) and N_σ decreases with n. At n=1 the std
    is UNDEFINED (present-masked out upstream); the denominator is clamped so the value stays
    finite (→ the max floor, but never scored). Mean floors are broadcast unchanged."""
    if len(mean_var) != 3 or len(std_r_ref) != 3:
        raise ValueError("mean_var and std_r_ref must each have 3 entries (slow/mid/hga)")
    n = n_elec.to(torch.float32)
    denom = (n - 1.0).clamp_min(1.0)  # n=1 → 1 (std masked out anyway; keeps it finite)
    r_ref = n_elec.new_tensor(std_r_ref, dtype=torch.float32)  # (3,)
    ratio = (1.0 - r_ref) / r_ref * (n_ref - 1.0)  # (3,) the noise/signal at n_ref, ×(n_ref−1)
    # broadcast over leading dims: (..., 1) with (3,) → (..., 3)
    r_sigma = 1.0 / (1.0 + ratio / denom.unsqueeze(-1))  # (..., 3)
    n_sigma = 1.0 - r_sigma  # (..., 3) std floor
    n_mean = n_elec.new_tensor(mean_var, dtype=torch.float32).expand(*n.shape, 3)  # (..., 3)
    return torch.cat([n_mean, n_sigma], dim=-1)  # (..., 6)


class GaussianStateHead(nn.Module):
    """Maps a per-(parcel, slot) feature to a full-covariance Gaussian over the state.

    Emits ``mu`` (dim) and a lower-triangular Cholesky factor ``L`` (dim·(dim+1)/2 free
    params; softplus on the diagonal keeps it a valid factor). The predicted covariance
    is ``Sigma = L Lᵀ + N`` with ``N`` the FIXED measured noise floor (a buffer — no
    gradient, not weight-decayed). The head owns no distributional hyper-params beyond
    what the locked K=1 Gaussian needs.
    """

    def __init__(
        self,
        d_in: int,
        *,
        dim: int = STATE_DIM,
        noise_var: tuple[float, ...] = NOISE_VAR,
    ) -> None:
        super().__init__()
        if len(noise_var) != dim:
            raise ValueError(f"noise_var has {len(noise_var)} entries, expected dim={dim}")
        if any(v <= 0.0 for v in noise_var):
            raise ValueError("noise_var must be strictly positive (it is the PD floor)")
        self.dim = dim
        self.n_tril = dim * (dim + 1) // 2
        self.mu_head = nn.Linear(d_in, dim)
        self.chol_head = nn.Linear(d_in, self.n_tril)
        # fixed measurement-noise floor; buffer => saved, moved with .to(), no grad/decay.
        self.register_buffer("noise_var", torch.tensor(noise_var, dtype=torch.float32))
        # lower-tri scatter indices, computed once.
        ii, jj = torch.tril_indices(dim, dim)
        self.register_buffer("_tril_i", ii, persistent=False)
        self.register_buffer("_tril_j", jj, persistent=False)
        # which of the n_tril entries are the diagonal (softplus'd for positivity).
        self.register_buffer("_diag_sel", (ii == jj), persistent=False)

    def forward(
        self, feat: Tensor, noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """``feat`` (..., d_in) → (``mu`` (..., dim), ``cov`` (..., dim, dim)).

        ``noise`` (..., dim) optional PER-POSITION PD floor — the count-dependent floor
        keyed on each query's parcel electrode count (:func:`count_dependent_noise_var`,
        threaded by the objective, which is the only place n_elec is known). When None,
        the FIXED measured buffer floors every position (backward-compatible default).
        Either way ``cov = L Lᵀ + diag(floor)`` stays strictly PD, so the head remains the
        SOLE owner of positive-definiteness (the objective never sees a singular cov)."""
        mu = self.mu_head(feat)
        # Assemble the covariance in fp32. Under bf16 autocast the chol_head Linear emits
        # bf16, and a bf16 ``L @ Lᵀ`` (+ floor) is NOT reliably PD for a 6×6 covariance once
        # training drifts L ill-conditioned — cholesky then fails mid-run (a STOCHASTIC crash
        # the .float() in _nll_terms cannot undo, because the precision is already lost in the
        # bf16 matmul). Do the factor→cov build in fp32; mu keeps the autocast dtype (the NLL
        # upcasts it). [bf16 autocast landmine #3, caught by the λ-drift probe 2026-07-15]
        raw = self.chol_head(feat).float()  # (..., n_tril) fp32
        # softplus the diagonal entries (strictly positive) — a valid Cholesky factor.
        diag_soft = torch.nn.functional.softplus(raw)
        vals = torch.where(self._diag_sel, diag_soft, raw)
        L = raw.new_zeros(*raw.shape[:-1], self.dim, self.dim)  # fp32
        L[..., self._tril_i, self._tril_j] = vals
        cov = L @ L.transpose(-1, -2)  # fp32 — PD-stable
        if noise is None:
            cov = cov + torch.diag(self.noise_var.float())
        else:
            if noise.shape[-1] != self.dim:
                raise ValueError(
                    f"noise last dim {noise.shape[-1]} != head dim {self.dim}"
                )
            cov = cov + torch.diag_embed(noise.float())  # (..., dim, dim)
        return mu, cov


def _nll_terms(mu: Tensor, cov: Tensor, x: Tensor) -> Tensor:
    """Per-position full-covariance Gaussian NLL (NO reduction). ``mu``/``x`` (..., D),
    ``cov`` (..., D, D) PD → (...,). Cholesky solve (cov is PD by the noise floor).

    The covariance solve is forced to fp32: cholesky / cholesky_solve are numerically unsafe
    in bf16 (a near-singular cov loses all precision), and under bf16 autocast the head emits
    bf16 mu/cov while the model-free target x stays fp32 (elementwise reductions are not
    autocast-downcast) — cholesky_solve then hard-errors on the A/b dtype mismatch. Casting all
    three to fp32 fixes both the correctness and the crash; the scalar loss rejoins the bf16
    graph on return."""
    mu = mu.float()
    cov = cov.float()
    x = x.float()
    d = x.shape[-1]
    r = (x - mu).unsqueeze(-1)  # (..., D, 1)
    chol = torch.linalg.cholesky(cov)  # (..., D, D) lower
    sol = torch.cholesky_solve(r, chol)  # cov^{-1} r
    maha = (r * sol).sum((-2, -1))  # (...,)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)  # (...,)
    return 0.5 * (maha + logdet + d * math.log(2.0 * math.pi))  # (...,)


def gaussian_nll(mu: Tensor, cov: Tensor, x: Tensor) -> Tensor:
    """Mean full-covariance Gaussian negative log-likelihood over present cells.

    ``mu`` (..., D), ``cov`` (..., D, D) PD, ``x`` (..., D) the model-free target. Uses a
    Cholesky solve (``cov`` is PD by the noise floor, so no jitter). Returns the scalar
    mean over all leading (parcel, slot) positions — the full-field domain.
    """
    return _nll_terms(mu, cov, x).mean()


def present_masked_nll(
    mu: Tensor, cov: Tensor, x: Tensor, present: Tensor
) -> Tensor:
    """Mean MARGINAL Gaussian NLL over present (parcel, slot) positions.

    A joint-Gaussian MARGINAL over a dim subset IS the ``(μ_S, Σ_SS)`` sub-block (drop the
    absent rows/cols) — it depends on ``Σ_SS`` ONLY, never the cross-cov to the absent dims
    (that is what makes it a marginal and not a conditional). The r4 state
    (state_target.dim_presence) has exactly two presence patterns: the 3 mean dims are
    ALWAYS present, the 3 std dims are present all-or-none iff ``n_elec ≥ 2``. So every
    position scores either its full 6-D NLL (n≥2) or its 3-D mean-marginal NLL (n=1). Both
    marginals are computed DENSELY over every position and selected by presence — no
    data-dependent grouping, so the shape is static (compile-once-per-session) and there is
    no host sync. The scalar is the mean over ALL positions (each on its own present dims).

    ``mu``/``x`` (..., D), ``cov`` (..., D, D) PD, ``present`` (..., D) bool. D must be 6
    with the mean/std split at 3 (the locked r4 state layout)."""
    D = x.shape[-1]
    n_mean = D // 2
    # fail loud on a non-conforming present mask (a wiring bug, not a data condition):
    # mean dims [0:n_mean] always on; std dims [n_mean:] all-or-none.
    mean_on = present[..., :n_mean].all(-1)
    std_block = present[..., n_mean:]
    std_all = std_block.all(-1)
    std_none = (~std_block).all(-1)
    if not bool(mean_on.all()):
        raise ValueError("present: the mean dims must always be present")
    if not bool((std_all | std_none).all()):
        raise ValueError("present: the std dims must be present all-or-none per position")
    full = _nll_terms(mu, cov, x)  # (...,) 6-D
    mean_marg = _nll_terms(mu[..., :n_mean], cov[..., :n_mean, :n_mean], x[..., :n_mean])
    per_pos = torch.where(std_all, full, mean_marg)  # (...,)
    return per_pos.mean()


def gaussian_entropy(cov: Tensor) -> Tensor:
    """Differential entropy ½·log((2πe)^D · det cov) of a Gaussian — the analytic floor
    machinery for the M12-style collapse tripwire (ceiling / common-mode / marginal).
    ``cov`` (..., D, D) → (...,)."""
    cov = cov.float()  # fp32 cholesky (bf16-autocast safe; see _nll_terms)
    d = cov.shape[-1]
    chol = torch.linalg.cholesky(cov)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    return 0.5 * (d * math.log(2.0 * math.pi * math.e) + logdet)
