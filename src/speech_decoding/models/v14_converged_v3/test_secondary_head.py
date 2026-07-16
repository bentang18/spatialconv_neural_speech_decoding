"""v14_converged_v3 r4 — adversarial tests for the Gaussian-NLL secondary head (#33).

Ben was wary of this head ("scared, new territory"). The launch gate is that a WRONG or
COLLAPSED head is DETECTABLE and the locked properties hold BY CONSTRUCTION. Each test
names the invariant, asserts it, and prints a ``[check]`` line (feedback-build-the-
invariant-into-the-probe):

  1. Sigma is PD and every marginal variance >= its measured noise floor (the S-JEPA floor).
  2. A correctly-fit unconditional head reaches the target's differential ENTROPY (the
     ceiling) and cannot beat it — a proper scoring rule.
  3. The full covariance ACTUALLY captures cross-dim coupling: on correlated data it beats
     the diagonal (marginal) floor by the total-correlation nats. This is the 0.200-nats
     property the whole joint-target decision (M17) rests on — collapse to marginals is
     visible as NLL sitting at the diagonal floor.
  4. The noise floor is honored under overconfident pressure: data with variance BELOW the
     floor cannot be fit below the floor's entropy — the head cannot manufacture confidence
     the measurement doesn't support.
  5. The floor carries no gradient (a buffer, not a parameter).
"""

from __future__ import annotations

import math

import torch

from speech_decoding.models.v14_converged_v3.secondary_head import (
    MEAN_REF_RELIABILITY,
    N_REF,
    NLL_FLOOR_JITTER,
    NOISE_VAR,
    STD_REF_RELIABILITY,
    GaussianStateHead,
    count_dependent_noise_var,
    gaussian_entropy,
    gaussian_nll,
    present_masked_nll,
)

D = len(NOISE_VAR)


def _fit_unconditional(head: GaussianStateHead, x: torch.Tensor, steps: int = 900) -> float:
    """Train the head on a CONSTANT feature (learns one unconditional Gaussian) to minimize
    mean NLL over ``x``; return the final train NLL."""
    feat = torch.ones(1, head.mu_head.in_features)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        opt.zero_grad()
        mu, cov = head(feat)  # (1, D), (1, D, D)
        loss = gaussian_nll(mu.expand(x.shape[0], D), cov.expand(x.shape[0], D, D), x)
        loss.backward()
        opt.step()
    return float(loss)


def _psd_data(cov: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    L = torch.linalg.cholesky(cov)
    z = torch.randn(n, cov.shape[0], generator=g)
    return z @ L.T


def test_sigma_is_pd_and_honors_noise_floor() -> None:
    torch.manual_seed(0)
    head = GaussianStateHead(d_in=8)
    feat = torch.randn(64, 8)
    mu, cov = head(feat)
    assert mu.shape == (64, D) and cov.shape == (64, D, D)
    # symmetric
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6)
    # PD: cholesky succeeds on every cell
    torch.linalg.cholesky(cov)
    # every marginal variance >= its measured noise floor, by construction (Sigma = LLᵀ + N)
    floor = torch.tensor(NOISE_VAR)
    marg = torch.diagonal(cov, dim1=-2, dim2=-1)  # (64, D)
    min_slack = (marg - floor).min().item()
    ok = min_slack >= -1e-6
    print(f"[check] Sigma PD + marginal var >= noise floor (min slack {min_slack:+.4f}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_fit_reaches_entropy_ceiling_and_cannot_beat_it() -> None:
    torch.manual_seed(1)
    # target cov = floor + a reachable correlated part (so the head CAN express it: LLᵀ = base).
    base = torch.randn(D, D) * 0.35
    cov_true = torch.diag(torch.tensor(NOISE_VAR)) + base @ base.T
    x = _psd_data(cov_true, n=20000, seed=7)
    h_true = float(gaussian_entropy(cov_true))
    achieved = _fit_unconditional(GaussianStateHead(d_in=8), x)
    # proper scoring rule: NLL >= H(sample) always; a well-fit head reaches ~H(cov_true).
    gap = achieved - h_true
    ok = abs(gap) < 0.05
    print(f"[check] fit NLL {achieved:.4f} ~ target entropy {h_true:.4f} "
          f"(gap {gap:+.4f}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_full_cov_beats_marginal_floor_by_total_correlation() -> None:
    torch.manual_seed(2)
    # strongly correlated target => big total correlation (the coupling the joint must keep).
    base = torch.randn(D, D) * 0.5
    cov_true = torch.diag(torch.tensor(NOISE_VAR)) + base @ base.T
    x = _psd_data(cov_true, n=20000, seed=11)
    # analytic floors: ceiling = H(full); marginal floor = H(product of marginals).
    h_full = float(gaussian_entropy(cov_true))
    h_marg = float(gaussian_entropy(torch.diag(torch.diagonal(cov_true))))
    tc = h_marg - h_full  # total correlation in nats (>0 by Hadamard)
    achieved = _fit_unconditional(GaussianStateHead(d_in=8), x)
    # the fit full-cov head must sit at the ceiling, i.e. BELOW the marginal floor by ~TC.
    beats = h_marg - achieved
    ok = tc > 0.1 and beats > 0.7 * tc
    print(f"[check] TC={tc:.3f} nats; full-cov NLL {achieved:.3f} beats marginal floor "
          f"{h_marg:.3f} by {beats:.3f} (>{0.7*tc:.3f}?) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_noise_floor_caps_confidence_on_sub_floor_data() -> None:
    torch.manual_seed(3)
    # data far BELOW the floor: the head must NOT be able to fit it confidently.
    tiny = torch.eye(D) * 0.01
    x = _psd_data(tiny, n=20000, seed=13)
    h_tiny = float(gaussian_entropy(tiny))  # what a floor-free model could reach
    h_floor = float(gaussian_entropy(torch.diag(torch.tensor(NOISE_VAR))))  # the cap
    achieved = _fit_unconditional(GaussianStateHead(d_in=8), x)
    # achieved is pinned near the floor entropy, FAR above the (much lower) tiny-data entropy.
    ok = achieved > h_tiny + 1.0 and achieved <= h_floor + 0.05
    print(f"[check] sub-floor data: NLL {achieved:.3f} pinned near floor {h_floor:.3f}, "
          f"far above sub-floor entropy {h_tiny:.3f} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_noise_floor_is_a_buffer_no_grad() -> None:
    head = GaussianStateHead(d_in=8)
    names = dict(head.named_buffers())
    assert "noise_var" in names
    assert not names["noise_var"].requires_grad
    # not exposed as a trainable parameter (won't be weight-decayed by the optimizer).
    assert "noise_var" not in dict(head.named_parameters())
    print("[check] noise floor is a non-trainable buffer (no grad, no decay) OK")


def test_gaussian_entropy_matches_closed_form_identity() -> None:
    ent = float(gaussian_entropy(torch.eye(D)))
    expect = 0.5 * D * math.log(2.0 * math.pi * math.e)
    assert abs(ent - expect) < 1e-5
    print(f"[check] H(I_{D}) = {ent:.4f} == {expect:.4f} OK")


# --- count-dependent noise floor (Ben "weight dependent" 2026-07-15; means added same day) --
def test_count_floor_recovers_anchor_at_reference_n() -> None:
    # By construction N(n_ref) = 1 − r_ref for BOTH summaries: the count-dep floor must EQUAL
    # the measured anchor exactly at the reference electrode count (mean AND std).
    n = torch.tensor([round(N_REF)])
    N = count_dependent_noise_var(n, n_ref=float(round(N_REF)))
    mean_floor, std_floor = N[0, :3], N[0, 3:]
    exp_mean = 1.0 - torch.tensor(MEAN_REF_RELIABILITY)
    exp_std = 1.0 - torch.tensor(STD_REF_RELIABILITY)
    ok = torch.allclose(mean_floor, exp_mean, atol=1e-6) and torch.allclose(std_floor, exp_std, atol=1e-6)
    # the mean anchor is exactly the fixed NOISE_VAR[:3] the head buffer uses (consistency).
    consistent = torch.allclose(mean_floor, torch.tensor(NOISE_VAR[:3]), atol=1e-6)
    print(f"[check] floor at n_ref={round(N_REF)}: mean {mean_floor.tolist()}==1−r_ref, "
          f"std {std_floor.tolist()}==1−r_ref, mean==NOISE_VAR[:3]={consistent} "
          f"{'OK' if ok and consistent else 'VIOLATED'}")
    assert ok and consistent


def test_count_floor_both_decrease_with_n() -> None:
    # More electrodes ⇒ less-noisy mean AND std estimates ⇒ STRICTLY lower floor on every dim;
    # every floor stays in (0,1). (Mean is now count-dependent too — was fixed pre-2026-07-15.)
    n = torch.tensor([1, 2, 3, 4, 6, 10, 20, 30])
    N = count_dependent_noise_var(n)  # (8, 6)
    mean, std = N[:, :3], N[:, 3:]
    mean_mono = bool((mean[1:] < mean[:-1] - 1e-7).all())  # mean defined at n=1, strictly ↓
    # std is undefined at n=1 (clamped/masked); check strict decrease from n=2 onward.
    std_mono = bool((std[2:] < std[1:-1] - 1e-7).all())
    in_unit = bool((N > 0).all() and (N < 1.0 + 1e-6).all())
    ok = mean_mono and std_mono and in_unit
    print(f"[check] mean floor ↓ with n (n=1 {mean[0].tolist()} → n=30 {mean[-1].tolist()}); "
          f"std floor ↓ with n; in (0,1)={in_unit} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_count_floor_obeys_sampling_variance_law() -> None:
    # Each floor's noise/signal ratio follows its OWN sampling law: mean ∝ 1/n, std ∝ 1/(n−1).
    # So mean ratio(a)/ratio(b) == b/a, std ratio(a)/ratio(b) == (b−1)/(a−1) — anchor-free.
    a, b = 3, 9
    N = count_dependent_noise_var(torch.tensor([a, b]))
    mean_ratio = N[:, :3] / (1.0 - N[:, :3])
    std_ratio = N[:, 3:] / (1.0 - N[:, 3:])
    got_mean = mean_ratio[0] / mean_ratio[1]
    got_std = std_ratio[0] / std_ratio[1]
    exp_mean = b / a               # 9/3 = 3   (mean ∝ 1/n)
    exp_std = (b - 1) / (a - 1)    # 8/2 = 4   (std  ∝ 1/(n−1))
    ok = (torch.allclose(got_mean, torch.full((3,), float(exp_mean)), rtol=1e-5)
          and torch.allclose(got_std, torch.full((3,), float(exp_std)), rtol=1e-5))
    print(f"[check] mean noise/signal ∝ 1/n: ratio(3)/ratio(9) {got_mean.tolist()}=={exp_mean}; "
          f"std ∝ 1/(n−1): {got_std.tolist()}=={exp_std} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_count_floor_finite_at_singleton_and_shape_preserved() -> None:
    # n=1 std is undefined (present-masked upstream) but the floor must stay FINITE (no div0),
    # and leading dims are preserved: (B, Q) → (B, Q, 6).
    N1 = count_dependent_noise_var(torch.tensor([1]))
    finite = bool(torch.isfinite(N1).all())
    N2 = count_dependent_noise_var(torch.tensor([[2, 6, 10], [3, 4, 30]]))
    ok = finite and N2.shape == (2, 3, 6)
    print(f"[check] n=1 finite={finite} (floor {N1[0,3:].tolist()}), "
          f"shape (2,3)→{tuple(N2.shape)} {'OK' if ok else 'VIOLATED'}")
    assert ok


# --- Q1-A: objective-computed per-position floor threaded into the head -------------------
def test_noise_override_none_matches_fixed_buffer() -> None:
    # noise=None must reproduce the fixed-buffer path EXACTLY (backward compatible default).
    torch.manual_seed(0)
    head = GaussianStateHead(16)
    feat = torch.randn(4, 5, 16)
    mu0, cov0 = head(feat)
    mu1, cov1 = head(feat, noise=None)
    same = torch.equal(mu0, mu1) and torch.equal(cov0, cov1)
    # and the fixed floor IS the buffer on the diagonal (LLᵀ has 0 added elsewhere from N).
    print(f"[check] noise=None == fixed-buffer path={same} {'OK' if same else 'VIOLATED'}")
    assert same


def test_noise_override_sets_per_position_floor_and_stays_pd() -> None:
    # A per-position floor must (a) replace the fixed buffer on the diagonal EXACTLY —
    # cov(noise) − cov(None) = diag(noise − buffer) at every position — and (b) keep cov PD.
    torch.manual_seed(1)
    head = GaussianStateHead(16)
    feat = torch.randn(3, 7, 16)
    _, cov_fixed = head(feat)
    noise = 0.1 + torch.rand(3, 7, D)  # strictly positive per-position floor
    _, cov_ovr = head(feat, noise=noise)
    delta = cov_ovr - cov_fixed  # should be diag(noise − buffer)
    expect = torch.diag_embed(noise - NOISE_VAR_T())
    diag_ok = torch.allclose(delta, expect, atol=1e-5)
    pd = bool((torch.linalg.eigvalsh(cov_ovr).min() > 0))
    ok = diag_ok and pd
    print(f"[check] per-position floor = diag(noise−buffer)={diag_ok}; cov PD={pd} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def NOISE_VAR_T() -> torch.Tensor:
    return torch.tensor(NOISE_VAR, dtype=torch.float32)


# --- #33 launch-gate: present-masked MARGINAL NLL (3-D for 1-elec parcels) ----------------
def _rand_gaussians(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(n, D, generator=g)
    a = torch.randn(n, D, D, generator=g)
    cov = a @ a.transpose(-1, -2) + torch.eye(D) * 0.5  # PD
    x = torch.randn(n, D, generator=g)
    return mu, cov, x


def test_present_masked_all_present_equals_full_nll() -> None:
    # With every dim present, the marginal NLL is exactly the full 6-D gaussian_nll.
    mu, cov, x = _rand_gaussians(20, 0)
    present = torch.ones(20, D, dtype=torch.bool)
    a = present_masked_nll(mu, cov, x, present)
    b = gaussian_nll(mu, cov, x)
    ok = torch.allclose(a, b, atol=1e-6)
    print(f"[check] all-present marginal == full NLL ({float(a):.4f} vs {float(b):.4f}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_nll_handles_bf16_cov_vs_fp32_target() -> None:
    # REGRESSION (bf16 launch): under autocast the head emits bf16 mu/cov while the model-free
    # target x stays fp32 (reductions are not autocast-downcast). cholesky_solve then errors
    # "Expected b and A to have the same dtype". The NLL must force fp32 internally (also the
    # numerically-correct choice for a covariance solve) and return a finite fp32 scalar.
    mu, cov, x = _rand_gaussians(20, 7)
    mu_bf, cov_bf, x_fp = mu.bfloat16(), cov.bfloat16(), x.float()
    present = torch.ones(20, D, dtype=torch.bool)
    full = gaussian_nll(mu_bf, cov_bf, x_fp)         # must NOT raise on dtype mismatch
    marg = present_masked_nll(mu_bf, cov_bf, x_fp, present)
    ok = (full.dtype == torch.float32 and torch.isfinite(full)
          and marg.dtype == torch.float32 and torch.isfinite(marg))
    print(f"[check] bf16 cov / fp32 target NLL finite fp32: full {float(full):.3f} "
          f"marg {float(marg):.3f} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_head_assembles_cov_in_fp32_under_bf16_autocast() -> None:
    # REGRESSION (bf16 launch landmine #3, caught by the λ-drift probe): if the head assembles
    # the covariance UNDER autocast, ``L @ Lᵀ`` runs in bf16 and a 6×6 covariance is not
    # reliably PD once training drifts L — cholesky crashes mid-run (a stochastic ~hour-N kill
    # the .float() in _nll_terms can't undo, since precision is lost in the bf16 matmul). The
    # head must build the factor→cov in fp32 REGARDLESS of the autocast dtype. Lock the
    # invariant that was violated: cov.dtype is fp32 and cov is PD under bf16 autocast, both
    # for the fixed buffer floor and the per-position count-dependent floor.
    torch.manual_seed(0)
    head = GaussianStateHead(d_in=16)
    feat = torch.randn(128, 16)
    noise = count_dependent_noise_var(torch.randint(1, 40, (128,)))  # per-position floor
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, cov_fix = head(feat)                 # fixed buffer floor
        _, cov_cnt = head(feat, noise=noise)    # count-dependent floor
    for tag, cov in [("fixed", cov_fix), ("count-dep", cov_cnt)]:
        is_fp32 = cov.dtype == torch.float32
        torch.linalg.cholesky(cov)  # raises if the bf16-assembly regression returns
        min_eig = float(torch.linalg.eigvalsh(cov).min())
        print(f"[check] autocast cov {tag}: dtype={cov.dtype} min_eig={min_eig:.4f} "
              f"{'OK' if is_fp32 and min_eig > 0 else 'VIOLATED'}")
        assert is_fp32 and min_eig > 0


def test_present_masked_marginal_matches_analytic_subblock() -> None:
    # A 1-electrode parcel scores the 3-D MEAN marginal = the exact (μ[:3], Σ[:3,:3]) NLL.
    mu, cov, x = _rand_gaussians(15, 2)
    present = torch.zeros(15, D, dtype=torch.bool)
    present[:, :3] = True  # mean-only (n_elec=1)
    got = present_masked_nll(mu, cov, x, present)
    want = gaussian_nll(mu[:, :3], cov[:, :3, :3], x[:, :3])  # analytic sub-block
    ok = torch.allclose(got, want, atol=1e-6)
    print(f"[check] mean-only marginal == analytic 3-D sub-block "
          f"({float(got):.4f} vs {float(want):.4f}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_present_masked_marginal_ignores_absent_and_crosscov() -> None:
    # THE sharp marginal-vs-conditional check. A marginal over the mean dims depends ONLY on
    # Σ_PP = Σ[:3,:3]. Perturbing (a) the std TARGET x[3:], (b) the std MEAN μ[3:], (c) the
    # std cov BLOCK Σ[3:,3:], AND (d) the CROSS-cov Σ[:3,3:]/Σ[3:,:3] must leave a mean-only
    # position's NLL BIT-IDENTICAL. A conditional (the wrong impl) WOULD move under (d).
    mu, cov, x = _rand_gaussians(12, 3)
    present = torch.zeros(12, D, dtype=torch.bool)
    present[:, :3] = True
    base = present_masked_nll(mu, cov, x, present)

    g = torch.Generator().manual_seed(99)
    mu2, x2 = mu.clone(), x.clone()
    x2[:, 3:] += 7.0 * torch.randn(12, 3, generator=g)  # (a) std target
    mu2[:, 3:] += 7.0 * torch.randn(12, 3, generator=g)  # (b) std mean
    # Rebuild cov2 with Σ_PP = Σ[:3,:3] held EXACTLY, but a DIFFERENT cross block (d) and std
    # block (c). Schur construction guarantees the full 6×6 is PD while touching neither the
    # present block nor its inverse: [[A, B],[Bᵀ, BᵀA⁻¹B + S]] ≻ 0 for any PD S.
    A = cov[:, :3, :3]  # the marginal's ONLY dependency — kept bit-identical
    B = 0.2 * torch.randn(12, 3, 3, generator=g)  # (d) new cross-cov (present × absent)
    s = torch.randn(12, 3, 3, generator=g)
    S = s @ s.transpose(-1, -2) + torch.eye(3) * 0.5  # (c) PD Schur complement → new std block
    Dblk = B.transpose(-1, -2) @ torch.linalg.inv(A) @ B + S
    cov2 = torch.zeros_like(cov)
    cov2[:, :3, :3] = A
    cov2[:, :3, 3:] = B
    cov2[:, 3:, :3] = B.transpose(-1, -2)
    cov2[:, 3:, 3:] = Dblk
    moved = present_masked_nll(mu2, cov2, x2, present)

    invariant = torch.allclose(base, moved, atol=1e-6)
    # sanity: those same std/cross perturbations DO move the full 6-D NLL (perturbation real).
    full_moved = not torch.allclose(gaussian_nll(mu, cov, x), gaussian_nll(mu2, cov2, x2))
    ok = invariant and full_moved
    print(f"[check] mean marginal invariant to std+cross-cov={invariant} "
          f"(Δ={(base-moved).abs().item():.2e}); full NLL moved={full_moved} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


# NOTE: the former test_present_masked_rejects_noncprefix_present is gone — present_masked_nll
# no longer validates the present layout per step (those two bool(...all()) checks were host
# syncs inside the compiled forward). The layout is guaranteed by dim_presence's construction
# and pinned in test_state_target.test_dim_presence_layout_wellformed instead.


# --- r5 Arm 2: floor-off -----------------------------------------------------
# Arm 2 removes the measured floor so the head must learn Sigma itself. It exists because
# r4 (wandb y365e10t) showed the entropy gap at only ~0.36 nats over a 5.46 floor -- N is
# most of Sigma -- and the raw NLL flatlined from 16k (slope -0.0016/1000 steps over
# 20k-26k with lambda pinned at 0.02) while JEPA kept falling. The head may be echoing the
# floor we handed it. These two tests pin the properties that make the arm interpretable.


def test_arm2_jitter_cannot_masquerade_as_a_floor() -> None:
    """THE invariant protecting the Arm1-vs-Arm2 contrast. The jitter exists only so a
    bf16-underflowed softplus diagonal cannot make Sigma singular; if it were anywhere near
    the measured floor it would BE a floor, and Arm 2 would silently test 'smaller floor'
    instead of 'no floor' -- the contrast would be uninterpretable and we would not see it
    in the CS number. Ben set 1e-6 (2026-07-16) against a measured floor of 0.119..0.504."""
    smallest_measured = min(NOISE_VAR)
    orders = math.log10(smallest_measured / NLL_FLOOR_JITTER)
    ok = orders >= 4.0
    print(f"[check] jitter {NLL_FLOOR_JITTER:g} vs smallest measured floor "
          f"{smallest_measured:g} -> {orders:.1f} orders below (need >=4) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_arm2_floor_off_keeps_sigma_pd_when_L_underflows() -> None:
    """The one real risk of floor-off: L's softplus diagonal underflows toward 0 in bf16,
    and with no measured floor Sigma leans entirely on the jitter. Drive the head's chol
    params hard negative (softplus -> ~0) and assert Sigma is STILL PD and the NLL finite.
    Without the jitter this is a singular Sigma and a NaN loss ~40 h into a queued run."""
    torch.manual_seed(0)
    head = GaussianStateHead(d_in=16)
    with torch.no_grad():  # force softplus(diag) -> ~0: the underflow regime
        head.chol_head.weight.zero_()
        head.chol_head.bias.fill_(-30.0)
    feat = torch.randn(8, 16)
    jitter = torch.full((8, len(NOISE_VAR)), NLL_FLOOR_JITTER)
    mu, cov = head(feat, noise=jitter)
    eig = torch.linalg.eigvalsh(cov.float())
    pd = bool((eig > 0).all())
    chol_ok = True
    try:
        torch.linalg.cholesky(cov.float())
    except RuntimeError:
        chol_ok = False
    nll = gaussian_nll(mu, cov, torch.randn_like(mu))
    finite = bool(torch.isfinite(nll).all())
    ok = pd and chol_ok and finite
    print(f"[check] floor-off under softplus underflow: min eig={eig.min():.3e} PD={pd} "
          f"cholesky={chol_ok} finite NLL={finite} {'OK' if ok else 'VIOLATED'}")
    assert ok
