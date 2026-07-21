"""v14_converged_v3 r5-mod — secondary-head STATE distribution + Gaussian NLL.

The secondary (perceiver) objective predicts, per present (parcel, 4 Hz slot), the
model-free parcel-state 5-vector

    x = [z_slow_mu, z_mid_mu, z_hga_mu, z_relmod48, z_relmod816]

(the 3 per-band parcel MEANS + the 2 HGA MODULATION dims — phase-free relative modulation
power in 4-8 Hz and 8-16 Hz; z-scored per (subject, parcel, dim), common-mode-removed —
see state_target.py). The 3 within-parcel STDs of the r4 target are DROPPED: r4 flatlined
on mean+std (the means are richly reachable from JEPA, the stds are noisy). Modulation is a
normalized SECOND moment, quadratic-in-envelope and NOT linearly reachable from the latent,
so the encoder must BUILD it (above-MAE work); validity probe
(project-modulation-target-validity-2026-07-17) confirms it is a weak-but-real per-clip
regional signal.

HEAD FORM for THIS OFAT — frozen-DIAGONAL Gaussian NLL (point head + fixed per-dim σ²).
The r4 full-covariance head let the model LEARN σ; the r4 secondary then flatlined by
inflating σ (raising its own NLL floor) instead of predicting better. Freezing σ² closes
that hatch — the ONLY way the loss drops is a better μ — which is what makes the OFAT a fair
test of whether the modulation target moves downstream CS transfer. σ² is the measured
per-dim floor, so the weak modulation dims (σ²≈0.82/0.76) contribute ~¼ the gradient of a
reliable mean dim (σ²≈0.12) — auto-downweighted, no tuned HP. Diagonal (not full-cov)
because with σ frozen there is no learned cross-dim coupling to carry, and the point head
emits no covariance parameter (DDP-safe: no unused params, the r3 X1 failure class).

    x ~ N(μ, diag(σ²_noise))    μ = the point head's output; σ² FIXED (count-dependent floor)

σ² is the S-JEPA "honest ambiguity" as additive observation noise: target = signal + noise,
noise ~ N(0, σ²). The loss can never reward resolving below the measurement floor, and a
perfect μ floors the NLL at the target's differential entropy. σ² strictly positive ⇒ always
numerically safe. (The full-covariance :class:`GaussianStateHead` + :func:`present_masked_nll`
below remain for the retired r5 full-cov arms; the diagonal path is what this build wires.)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# Per-dim measurement-noise variance = 1 − split-half reliability, in the z-scored units
# the target lives in (unit marginal variance per dim ⇒ σ² = 1 − r directly). Order:
#   [slow_mu, mid_mu, hga_mu, relmod48, relmod816].
# means   reliability slow .881 / mid .796 / hga .828  (#28 cm-removed 4 Hz sweep, 2026-07-15;
#   HGA-mean RISES vs .561 native — the 4 Hz window-average denoises HGA rather than losing it).
# modulation reliability relmod48 .179 / relmod816 .239  (parcel-spec per-clip, SB-corrected,
#   mean over the 3 board subjects — project-modulation-target-validity-2026-07-17). WEAK but
#   positive in all 3 subjects: σ²≈0.82/0.76 ⇒ NLL weight (1/σ²)≈1.2/1.3 vs ~5-8 for a mean dim
#   ⇒ modulation contributes ~¼ the gradient, auto-ignored if it were noise, kept as an honest
#   weak pin. No tuned HP.
NOISE_VAR: tuple[float, ...] = (0.119, 0.204, 0.172, 0.821, 0.761)
STATE_DIM: int = len(NOISE_VAR)

# r5 Arm 2 (floor-off) ONLY — the floor the head gets when --no-nll-floor replaces the
# measured NOISE_VAR. Not a floor: L already carries a softplus diagonal, so ``L Lᵀ`` is
# strictly PD on its own and this is bf16 conditioning insurance for the case where that
# softplus underflows toward 0. Ben-set 1e-6 (2026-07-16), ~5 orders below the smallest
# measured entry (0.119), so it cannot act as a floor in disguise and contaminate the
# Arm1-vs-Arm2 contrast — that separation is the arm's whole point and
# test_arm2_jitter_cannot_masquerade_as_a_floor pins it.
NLL_FLOOR_JITTER: float = 1e-6

# --- count-dependent noise floor (Ben 2026-07-15: "weight dependent") -------------------------
# Every dim of the 5-vector is now a MEAN over the parcel's n electrodes — the 3 band means, and
# the 2 modulation dims (per-electrode relmod, averaged over the parcel). A mean of n electrodes
# has sampling variance ∝ 1/n (the SEM), so ALL five dims share ONE law in RELIABILITY space
# (unit-invariant, so the z-scoring doesn't touch it):
#     r(n) = 1 / (1 + (1−r_ref)/r_ref · n_ref/n),   N(n) = 1 − r(n).
# Defined at n=1 (a mean IS defined for a singleton — modulation is present-masked upstream at
# n=1 anyway). Recovers the anchor exactly at n_ref: N(n_ref) = 1 − r_ref. Measurement is WORSE
# for small parcels (the distribution probe found 28% of the loss terms come from 2-4 electrode
# parcels), so a FIXED floor over-trusts a quarter of the targets — hence the count law.
# [The r4 STD dims used a separate ∝1/(n−1) law; with the stds dropped there is no second law.]
#
# Anchors: means #28 cm-removed 4 Hz sweep (2026-07-15); modulation the validity probe
# (2026-07-17). REF_RELIABILITY = 1 − NOISE_VAR (consistent by construction); N_REF = mean
# n_elec the reliabilities were averaged over (shared: both probes pooled full-parcel electrodes).
REF_RELIABILITY: tuple[float, ...] = (0.881, 0.796, 0.828, 0.179, 0.239)  # = 1 − NOISE_VAR
N_REF: float = 13.35


def count_dependent_noise_var(
    n_elec: Tensor,
    *,
    r_ref: tuple[float, ...] = REF_RELIABILITY,
    n_ref: float = N_REF,
) -> Tensor:
    """Per-(parcel) 5-vector noise floor N, keyed on the parcel's electrode count.

    ``n_elec`` (...,) long ≥ 1. Returns (..., 5). Every dim is a mean over the parcel's
    electrodes, so all share the SEM sampling law ∝ 1/n in reliability space:

        r(n) = 1 / (1 + (1−r_ref)/r_ref · n_ref/n),   N(n) = 1 − r(n).

    Recovers the anchor exactly at n_ref (N(n_ref) = 1 − r_ref) and decreases with n. Defined
    at n=1 (a mean is; modulation is present-masked upstream there)."""
    if len(r_ref) != STATE_DIM:
        raise ValueError(f"r_ref must have {STATE_DIM} entries, got {len(r_ref)}")
    n = n_elec.to(torch.float32)
    denom = n.clamp_min(1.0)  # a mean is defined at n=1
    r_ref_t = n_elec.new_tensor(r_ref, dtype=torch.float32)  # (5,)
    ratio = (1.0 - r_ref_t) / r_ref_t * n_ref  # (5,) noise/signal at n_ref, ×n_ref
    r = 1.0 / (1.0 + ratio / denom.unsqueeze(-1))  # (..., 5)
    return 1.0 - r  # (..., 5) floor


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


def present_masked_diag_nll(
    mu: Tensor, x: Tensor, present: Tensor, noise: Tensor
) -> Tensor:
    """Mean frozen-DIAGONAL Gaussian NLL over present (parcel, slot) dims — this build's loss.

        L = mean_pos Σ_{d ∈ present}  ½[ (x_d − μ_d)² / σ²_d  +  log(2π σ²_d) ]

    ``mu``/``x`` (..., D) (D=5), ``present`` (..., D) bool, ``noise`` (..., D) the FIXED per-dim
    σ² (the count-dependent floor from :func:`count_dependent_noise_var`, keyed on each query's
    parcel electrode count). σ² is FROZEN — the head emits only μ — so the ONLY way to lower L
    is a better μ; the r4 hatch of inflating σ to raise the NLL floor is closed.

    Reduction matches :func:`present_masked_nll`/:func:`present_masked_l1`: sum over the present
    dims of a position, mean over positions. Diagonal ⇒ no cross-dim term and no Cholesky, so
    the marginal over present dims is just the sum of the present per-dim terms — no sub-block
    logic, no data-dependent grouping, static shape (compile-once, no host sync). Absent dims are
    masked to 0 (``normalize_target`` also sets the target 0 there); μ is free at absent dims and
    never scored. fp32 throughout (x is fp32; μ may be bf16 under autocast) — the log/reciprocal
    of a strictly-positive σ² is numerically safe with no jitter."""
    mu = mu.float()
    x = x.float()
    noise = noise.float()
    per_dim = 0.5 * ((x - mu) ** 2 / noise + torch.log(2.0 * math.pi * noise))  # (..., D)
    per_dim = per_dim * present.to(per_dim.dtype)
    return per_dim.sum(-1).mean()


def gaussian_entropy(cov: Tensor) -> Tensor:
    """Differential entropy ½·log((2πe)^D · det cov) of a Gaussian — the analytic floor
    machinery for the M12-style collapse tripwire (ceiling / common-mode / marginal).
    ``cov`` (..., D, D) → (...,)."""
    cov = cov.float()  # fp32 cholesky (bf16-autocast safe; see _nll_terms)
    d = cov.shape[-1]
    chol = torch.linalg.cholesky(cov)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    return 0.5 * (d * math.log(2.0 * math.pi * math.e) + logdet)
