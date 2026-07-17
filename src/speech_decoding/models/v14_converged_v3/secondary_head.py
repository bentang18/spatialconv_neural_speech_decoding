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

# r5 Arm 2 (floor-off) ONLY — the floor the head gets when --no-nll-floor replaces the
# measured NOISE_VAR. Not a floor: L already carries a softplus diagonal, so ``L Lᵀ`` is
# strictly PD on its own and this is bf16 conditioning insurance for the case where that
# softplus underflows toward 0. Ben-set 1e-6 (2026-07-16), ~5 orders below the smallest
# measured entry (0.119), so it cannot act as a floor in disguise and contaminate the
# Arm1-vs-Arm2 contrast — that separation is the arm's whole point and
# test_arm2_jitter_cannot_masquerade_as_a_floor pins it.
NLL_FLOOR_JITTER: float = 1e-6

# --- count-dependent noise floor (Ben 2026-07-15: "weight dependent"; means added same day) --
# A parcel summary is a sample statistic over its n electrodes: measurement is WORSE for small
# parcels, and the distribution probe found 28% of the loss terms come from 2-4 electrode
# parcels — a FIXED floor over-trusts a quarter of the targets. BOTH summaries get a
# count-dependent floor, each on its OWN sampling law in RELIABILITY space (unit-invariant, so
# the z-scoring doesn't touch it):
#   MEAN — a mean of n electrodes has sampling variance ∝ 1/n (the SEM), so the noise fraction
#     of the unit-variance mean target ∝ 1/n:   r_μ(n) = 1/(1 + (1−r_ref)/r_ref · n_ref/n).
#     Defined at n=1 (a mean IS defined for a singleton). DIRECTLY MEASURED at small n by
#     probe_v3_mean_floor_vs_count (disjoint size-n subsets, cm-removed, 4 Hz): the anchored law
#     predicts r_μ(1)=0.357 (meas 0.370), r_μ(2)=0.526 (meas 0.510) — tight where unconfounded.
#     [The measured curve plateaus BELOW the #28 SB full-parcel number (~0.72 vs 0.881 slow);
#     that gap is Spearman-Brown-extrapolation vs direct + a big-parcel confound at large n, so
#     we anchor at #28 (frozen ref, recovered exactly at n_ref) and add count-dependence BELOW
#     it rather than lower the large-n floor. See project-r4-mean-floor-count-dependent-*.]
#   STD — a sample std has sampling variance ∝ 1/(n−1), so its noise fraction ∝ 1/(n−1):
#     r_σ(n) = 1/(1 + (1−r_ref)/r_ref · (n_ref−1)/(n−1)). At n=1 the std is UNDEFINED
#     (present-masked upstream); the denominator is clamped so it stays finite (never scored).
# Both recover the measured anchor exactly at n_ref: N(n_ref) = 1 − r_ref.
#
# Anchors (#28 cm-removed 4 Hz sweep, 2026-07-15). *_REF_RELIABILITY = 1 − NOISE_VAR[...]
# (consistent by construction); N_REF = mean n_elec the reliabilities were averaged over.
MEAN_REF_RELIABILITY: tuple[float, ...] = (0.881, 0.796, 0.828)  # slow/mid/hga mean reliab @ n_ref
STD_REF_RELIABILITY: tuple[float, ...] = (0.652, 0.496, 0.546)   # slow/mid/hga std  reliab @ n_ref
N_REF: float = 13.35


def count_dependent_noise_var(
    n_elec: Tensor,
    *,
    mean_r_ref: tuple[float, ...] = MEAN_REF_RELIABILITY,
    std_r_ref: tuple[float, ...] = STD_REF_RELIABILITY,
    n_ref: float = N_REF,
) -> Tensor:
    """Per-(parcel) 6-vector noise floor N, keyed on the parcel's electrode count.

    ``n_elec`` (...,) long ≥ 1. Returns (..., 6) = [3 count-dependent mean floors, 3
    count-dependent std floors]. Each summary uses its own sampling law in RELIABILITY space:

        r_μ(n) = 1 / (1 + (1−r_ref)/r_ref · n_ref/n),         N_μ(n) = 1 − r_μ(n)   (mean, ∝1/n)
        r_σ(n) = 1 / (1 + (1−r_ref)/r_ref · (n_ref−1)/(n−1)), N_σ(n) = 1 − r_σ(n)   (std, ∝1/(n−1))

    Both recover the anchor exactly at n_ref (N(n_ref) = 1 − r_ref) and decrease with n. The
    mean is defined at n=1; the std is present-masked upstream at n=1 but the denominator is
    clamped so it stays finite (→ the max floor, never scored)."""
    if len(mean_r_ref) != 3 or len(std_r_ref) != 3:
        raise ValueError("mean_r_ref and std_r_ref must each have 3 entries (slow/mid/hga)")
    n = n_elec.to(torch.float32)
    # MEAN floor — SEM sampling law ∝ 1/n; a mean is defined at n=1 so the denominator is n.
    denom_mean = n.clamp_min(1.0)
    r_ref_m = n_elec.new_tensor(mean_r_ref, dtype=torch.float32)  # (3,)
    ratio_m = (1.0 - r_ref_m) / r_ref_m * n_ref  # (3,) noise/signal at n_ref, ×n_ref
    r_mu = 1.0 / (1.0 + ratio_m / denom_mean.unsqueeze(-1))  # (..., 3)
    n_mean = 1.0 - r_mu  # (..., 3) mean floor
    # STD floor — sample-std sampling law ∝ 1/(n−1).
    denom = (n - 1.0).clamp_min(1.0)  # n=1 → 1 (std masked out anyway; keeps it finite)
    r_ref = n_elec.new_tensor(std_r_ref, dtype=torch.float32)  # (3,)
    ratio = (1.0 - r_ref) / r_ref * (n_ref - 1.0)  # (3,) the noise/signal at n_ref, ×(n_ref−1)
    r_sigma = 1.0 / (1.0 + ratio / denom.unsqueeze(-1))  # (..., 3)
    n_sigma = 1.0 - r_sigma  # (..., 3) std floor
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
        point_only: bool = False,
    ) -> None:
        super().__init__()
        if len(noise_var) != dim:
            raise ValueError(f"noise_var has {len(noise_var)} entries, expected dim={dim}")
        if any(v <= 0.0 for v in noise_var):
            raise ValueError("noise_var must be strictly positive (it is the PD floor)")
        self.dim = dim
        self.n_tril = dim * (dim + 1) // 2
        # r5 Arm 3 (point loss): the head emits mu ONLY and forward returns cov=None.
        # chol_head is NOT CONSTRUCTED rather than merely unused — a point loss touches no
        # covariance parameter, so leaving the Linear in place would hand DDP a parameter
        # that never receives a gradient, which is the X1 failure class that killed r3
        # (find_unused_parameters=False asserts every param contributes to the loss). Not
        # constructing it also makes the arm's claim structural: Arm 3 IS "the second
        # moment does not exist", not "the second moment exists and is ignored".
        self.point_only = bool(point_only)
        self.mu_head = nn.Linear(d_in, dim)
        self.chol_head = None if self.point_only else nn.Linear(d_in, self.n_tril)
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
    ) -> tuple[Tensor, Tensor | None]:
        """``feat`` (..., d_in) → (``mu`` (..., dim), ``cov`` (..., dim, dim)).

        Under ``point_only`` (r5 Arm 3) ``cov`` is None — there is no covariance head.

        ``noise`` (..., dim) optional PER-POSITION PD floor — the count-dependent floor
        keyed on each query's parcel electrode count (:func:`count_dependent_noise_var`,
        threaded by the objective, which is the only place n_elec is known). When None,
        the FIXED measured buffer floors every position (backward-compatible default).
        Either way ``cov = L Lᵀ + diag(floor)`` stays strictly PD, so the head remains the
        SOLE owner of positive-definiteness (the objective never sees a singular cov)."""
        mu = self.mu_head(feat)
        if self.chol_head is None:  # r5 Arm 3 — point head, there is no second moment
            return mu, None
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
    # ``present`` comes from state_target.dim_presence, which CONSTRUCTS exactly the two legal
    # patterns — mean dims always on, std dims all-or-none per position — so the layout is
    # well-formed BY CONSTRUCTION (pinned by test_dim_presence_layout_wellformed). We trust it
    # here and do NOT re-validate per step: the old ``bool(mean_on.all())`` /
    # ``bool((std_all|std_none).all())`` guards were two host syncs (graph breaks) inside the
    # compiled forward over a session-invariant structure.
    std_all = present[..., n_mean:].all(-1)  # (...,) select 6-D vs 3-D-mean marginal per pos
    full = _nll_terms(mu, cov, x)  # (...,) 6-D
    mean_marg = _nll_terms(mu[..., :n_mean], cov[..., :n_mean, :n_mean], x[..., :n_mean])
    per_pos = torch.where(std_all, full, mean_marg)  # (...,)
    return per_pos.mean()


def present_masked_l1(mu: Tensor, x: Tensor, present: Tensor) -> Tensor:
    """Mean per-position SUM of |x − mu| over PRESENT dims — r5 Arm 3's POINT loss.

    The reduction is the exact twin of :func:`present_masked_nll`: that scores each
    position on its present dims (summing over them inside ``_nll_terms``) and then means
    over positions, so this sums |r| over the present dims and means over positions. Same
    denominator, same masking ⇒ the ONLY difference between Arm 3 and Arm 1 is the loss
    form, which is what makes it an ablation rather than two unrelated runs.

    WHY L1 AND NOT L2 (M18, measured 2026-07-16 — probe_v3_residual_tailweight).
    L2 is not a neutral default: at Σ=I the Gaussian NLL IS ``0.5·‖r‖² + const`` (see
    ``_nll_terms``), so L2 asserts GAUSSIAN residuals and L1 asserts LAPLACE ones. M18
    measured the assertion instead of arguing it — split each parcel's electrodes in half,
    take the difference of the two half-states (the parcel's true state cancels, leaving
    pure sampling noise with the model removed) and read its excess kurtosis against
    synthetic gauss/laplace controls pushed through the identical pipeline:

        dim      gauss ctl      REAL      laplace ctl
        slow_mu     -0.005    + 3.646        +0.440
        mid_mu      -0.008    + 2.476        +0.404
        hga_mu      +0.000    +10.364        +0.411

    5/6 dims sit ABOVE the laplace pole by ≥3 SEM. So the residual is heavier-tailed than
    Laplace, L1 is strictly the closer of the two, and L2 is the worse choice. (Neither is
    the TRUE noise model — a Huber/Student-t would fit better, but that adds a second axis
    to an ablation whose whole point is isolating one. Logged as a follow-up, not smuggled
    in here.) The controls are load-bearing: the state is a MEAN over electrodes, so the
    CLT drags a Laplace electrode's kurtosis 3.0 down to +0.4 by the time it reaches the
    parcel state — reading the real data against a hard-coded 3.0 would have been wrong.

    ``mu``/``x`` (..., D), ``present`` (..., D) bool. Absent dims contribute 0 — the target
    is SET to 0 there by ``state_target.normalize_target`` while ``mu`` is free, so the
    mask is what keeps the head from being scored on a value that carries no information."""
    r = (x - mu).abs() * present.to(x.dtype)
    return r.sum(-1).mean()


def gaussian_entropy(cov: Tensor) -> Tensor:
    """Differential entropy ½·log((2πe)^D · det cov) of a Gaussian — the analytic floor
    machinery for the M12-style collapse tripwire (ceiling / common-mode / marginal).
    ``cov`` (..., D, D) → (...,)."""
    cov = cov.float()  # fp32 cholesky (bf16-autocast safe; see _nll_terms)
    d = cov.shape[-1]
    chol = torch.linalg.cholesky(cov)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    return 0.5 * (d * math.log(2.0 * math.pi * math.e) + logdet)
