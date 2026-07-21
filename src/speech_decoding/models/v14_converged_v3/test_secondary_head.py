"""v14_converged_v3 r5-mod — tests for the secondary head + frozen-diagonal NLL.

This build's LIVE loss is :func:`present_masked_diag_nll` over the 5-dim state
[slow_mu, mid_mu, hga_mu, relmod48, relmod816] with FROZEN per-dim σ² (the count-dependent
floor). The launch gate is that a WRONG or COLLAPSED head is DETECTABLE and the locked
properties hold BY CONSTRUCTION. Each test names the invariant, asserts it, prints a
``[check]`` line (feedback-build-the-invariant-into-the-probe):

  Frozen-diagonal loss (the live path):
  - the NLL equals the hand-written diagonal Gaussian formula over present dims;
  - with σ² FROZEN, the loss is minimised ONLY at μ=x and floors there at the target's own
    diagonal entropy — the r4 "inflate σ to raise the floor" hatch is closed;
  - absent dims (the modulation dims of a <2-electrode parcel) carry no gradient.
  - count floor: every dim is a mean-over-electrodes ⇒ ONE SEM (∝1/n) law, recovering the
    measured anchor at n_ref and decreasing with n.

  Full-covariance head (retained for the retired r5 full-cov arms; generic D-dim checks):
  - Sigma PD with every marginal variance >= its measured floor; a fit head reaches the
    entropy ceiling and cannot beat it; full-cov beats the marginal floor by the total
    correlation; sub-floor data cannot be fit below the floor; the floor is a no-grad buffer;
    bf16-autocast cov assembly stays fp32 + PD.
"""

from __future__ import annotations

import math

import torch

from speech_decoding.models.v14_converged_v3.secondary_head import (
    N_REF,
    NLL_FLOOR_JITTER,
    NOISE_VAR,
    REF_RELIABILITY,
    GaussianStateHead,
    count_dependent_noise_var,
    gaussian_entropy,
    gaussian_nll,
    present_masked_diag_nll,
    present_masked_nll,
)

D = len(NOISE_VAR)  # 5


# =========================================================================================
# Frozen-diagonal NLL — THIS BUILD'S LIVE LOSS
# =========================================================================================
def test_diag_nll_matches_hand_formula() -> None:
    # The loss must be the literal diagonal Gaussian NLL summed over present dims, meaned over
    # positions: mean_pos Σ_{d∈present} ½[(x−μ)²/σ² + log(2π σ²)].
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(4, 7, D, generator=g)
    x = torch.randn(4, 7, D, generator=g)
    present = torch.ones(4, 7, D, dtype=torch.bool)
    noise = 0.1 + torch.rand(4, 7, D, generator=g)
    got = present_masked_diag_nll(mu, x, present, noise)
    want = (0.5 * ((x - mu) ** 2 / noise + torch.log(2.0 * math.pi * noise))).sum(-1).mean()
    ok = torch.allclose(got, want, atol=1e-6)
    print(f"[check] diag NLL == hand formula ({float(got):.5f} vs {float(want):.5f}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_diag_nll_minimised_at_mu_equals_x_and_floors_at_frozen_entropy() -> None:
    # THE property that closes the r4 flatline hatch: σ² is FROZEN, so the loss can drop ONLY
    # by moving μ toward x. Its global minimum is at μ=x and equals the target's own frozen
    # diagonal entropy ½Σ log(2π σ²) — the head cannot manufacture a lower loss by inflating σ
    # (it owns no σ). Any μ≠x scores strictly higher.
    g = torch.Generator().manual_seed(1)
    x = torch.randn(64, D, generator=g)
    noise = count_dependent_noise_var(torch.randint(1, 40, (64,), generator=g))  # frozen σ²
    present = torch.ones(64, D, dtype=torch.bool)
    at_min = present_masked_diag_nll(x, x, present, noise)         # μ=x
    frozen_entropy = (0.5 * torch.log(2.0 * math.pi * noise)).sum(-1).mean()
    worse = present_masked_diag_nll(x + 0.5, x, present, noise)    # μ off by 0.5
    ok = torch.allclose(at_min, frozen_entropy, atol=1e-6) and float(worse) > float(at_min) + 1e-3
    print(f"[check] diag NLL min at μ=x = frozen entropy {float(at_min):.4f}; "
          f"μ≠x worse {float(worse):.4f} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_diag_nll_masks_absent_dims() -> None:
    # A <2-electrode parcel has its modulation dims [3:5] present=False. Perturbing BOTH the
    # target and the prediction at those absent dims must leave the loss BIT-IDENTICAL — the
    # head is never scored on a dim the measurement doesn't define.
    g = torch.Generator().manual_seed(2)
    mu = torch.randn(20, D, generator=g)
    x = torch.randn(20, D, generator=g)
    noise = 0.2 + torch.rand(20, D, generator=g)
    present = torch.zeros(20, D, dtype=torch.bool)
    present[:, :3] = True  # mean-only (1-electrode parcel pattern)
    base = present_masked_diag_nll(mu, x, present, noise)
    mu2, x2 = mu.clone(), x.clone()
    mu2[:, 3:] += 9.0 * torch.randn(20, 2, generator=g)  # absent-dim prediction
    x2[:, 3:] += 9.0 * torch.randn(20, 2, generator=g)   # absent-dim target
    moved = present_masked_diag_nll(mu2, x2, present, noise)
    ok = torch.allclose(base, moved, atol=1e-6)
    print(f"[check] diag NLL invariant to absent (modulation) dims "
          f"(Δ={(base - moved).abs().item():.2e}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_diag_nll_bf16_mu_fp32_target_is_finite_fp32() -> None:
    # REGRESSION (bf16 launch): under autocast the point head emits bf16 μ while the model-free
    # target x + the σ² floor stay fp32. The loss must upcast internally and return a finite
    # fp32 scalar (no dtype-mismatch error, no bf16 precision loss in the reciprocal/log).
    g = torch.Generator().manual_seed(7)
    mu = torch.randn(30, D, generator=g)
    x = torch.randn(30, D, generator=g)
    noise = 0.1 + torch.rand(30, D, generator=g)
    present = torch.ones(30, D, dtype=torch.bool)
    out = present_masked_diag_nll(mu.bfloat16(), x.float(), present, noise.float())
    ok = out.dtype == torch.float32 and bool(torch.isfinite(out))
    print(f"[check] diag NLL bf16 μ / fp32 target → finite fp32 ({float(out):.3f}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_point_only_head_emits_mu_only_no_cov_param() -> None:
    # The diagonal path uses point_only=True: the head emits μ and forward returns cov=None.
    # chol_head is NOT constructed (not merely unused) — a diagonal loss touches no covariance
    # parameter, so leaving a Linear would hand DDP an unused param (the r3 X1 crash class).
    head = GaussianStateHead(d_in=16, point_only=True)
    feat = torch.randn(4, 5, 16)
    mu, cov = head(feat)
    no_chol = head.chol_head is None
    no_tril_param = not any("chol" in n for n, _ in head.named_parameters())
    ok = mu.shape == (4, 5, D) and cov is None and no_chol and no_tril_param
    print(f"[check] point_only head: μ only, cov=None ({cov is None}), no chol param "
          f"({no_tril_param}) {'OK' if ok else 'VIOLATED'}")
    assert ok


# =========================================================================================
# Count-dependent noise floor — unified SEM (∝1/n) law over all 5 dims
# =========================================================================================
def test_count_floor_recovers_anchor_at_reference_n() -> None:
    # By construction N(n_ref) = 1 − r_ref for every dim, and equals the fixed NOISE_VAR buffer
    # the head uses (consistency). Every dim is a mean-over-electrodes ⇒ one law, no split.
    n = torch.tensor([round(N_REF)])
    N = count_dependent_noise_var(n, n_ref=float(round(N_REF)))
    exp = 1.0 - torch.tensor(REF_RELIABILITY)
    anchor = torch.allclose(N[0], exp, atol=1e-6)
    consistent = torch.allclose(N[0], torch.tensor(NOISE_VAR), atol=1e-6)
    ok = anchor and consistent
    print(f"[check] floor at n_ref={round(N_REF)}: {N[0].tolist()} == 1−r_ref == NOISE_VAR "
          f"({consistent}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_count_floor_decreases_with_n_all_dims() -> None:
    # More electrodes ⇒ less-noisy parcel mean on EVERY dim ⇒ strictly lower floor; all in (0,1).
    # Every dim is defined at n=1 (a mean is), so the decrease is strict from n=1.
    n = torch.tensor([1, 2, 3, 4, 6, 10, 20, 30])
    N = count_dependent_noise_var(n)  # (8, 5)
    mono = bool((N[1:] < N[:-1] - 1e-7).all())
    in_unit = bool((N > 0).all() and (N < 1.0 + 1e-6).all())
    ok = mono and in_unit
    print(f"[check] floor ↓ with n on all 5 dims (n=1 {N[0].tolist()} → n=30 {N[-1].tolist()}); "
          f"in (0,1)={in_unit} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_count_floor_obeys_sem_law() -> None:
    # Noise/signal ratio ∝ 1/n on every dim ⇒ ratio(a)/ratio(b) == b/a, anchor-free.
    a, b = 3, 9
    N = count_dependent_noise_var(torch.tensor([a, b]))
    ratio = N / (1.0 - N)
    got = ratio[0] / ratio[1]  # (5,)
    ok = torch.allclose(got, torch.full((D,), float(b) / a), rtol=1e-5)
    print(f"[check] noise/signal ∝ 1/n: ratio(3)/ratio(9) {got.tolist()} == {b/a} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_count_floor_finite_at_singleton_and_shape_preserved() -> None:
    # n=1 stays finite (a mean is defined; modulation is present-masked upstream), and leading
    # dims are preserved: (B, Q) → (B, Q, 5).
    N1 = count_dependent_noise_var(torch.tensor([1]))
    finite = bool(torch.isfinite(N1).all())
    N2 = count_dependent_noise_var(torch.tensor([[2, 6, 10], [3, 4, 30]]))
    ok = finite and N2.shape == (2, 3, D)
    print(f"[check] n=1 finite={finite} (floor {N1[0].tolist()}), shape (2,3)→{tuple(N2.shape)} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


# =========================================================================================
# Full-covariance head — retained for the retired r5 full-cov arms (generic D-dim checks)
# =========================================================================================
def _fit_unconditional(head: GaussianStateHead, x: torch.Tensor, steps: int = 900) -> float:
    feat = torch.ones(1, head.mu_head.in_features)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        opt.zero_grad()
        mu, cov = head(feat)
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
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6)
    torch.linalg.cholesky(cov)
    floor = torch.tensor(NOISE_VAR)
    marg = torch.diagonal(cov, dim1=-2, dim2=-1)
    min_slack = (marg - floor).min().item()
    ok = min_slack >= -1e-6
    print(f"[check] Sigma PD + marginal var >= noise floor (min slack {min_slack:+.4f}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_fit_reaches_entropy_ceiling_and_cannot_beat_it() -> None:
    torch.manual_seed(1)
    base = torch.randn(D, D) * 0.35
    cov_true = torch.diag(torch.tensor(NOISE_VAR)) + base @ base.T
    x = _psd_data(cov_true, n=20000, seed=7)
    h_true = float(gaussian_entropy(cov_true))
    achieved = _fit_unconditional(GaussianStateHead(d_in=8), x)
    gap = achieved - h_true
    ok = abs(gap) < 0.05
    print(f"[check] fit NLL {achieved:.4f} ~ target entropy {h_true:.4f} (gap {gap:+.4f}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_full_cov_beats_marginal_floor_by_total_correlation() -> None:
    torch.manual_seed(2)
    base = torch.randn(D, D) * 0.5
    cov_true = torch.diag(torch.tensor(NOISE_VAR)) + base @ base.T
    x = _psd_data(cov_true, n=20000, seed=11)
    h_full = float(gaussian_entropy(cov_true))
    h_marg = float(gaussian_entropy(torch.diag(torch.diagonal(cov_true))))
    tc = h_marg - h_full
    achieved = _fit_unconditional(GaussianStateHead(d_in=8), x)
    beats = h_marg - achieved
    ok = tc > 0.1 and beats > 0.7 * tc
    print(f"[check] TC={tc:.3f} nats; full-cov NLL {achieved:.3f} beats marginal floor "
          f"{h_marg:.3f} by {beats:.3f} (>{0.7*tc:.3f}?) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_noise_floor_caps_confidence_on_sub_floor_data() -> None:
    torch.manual_seed(3)
    tiny = torch.eye(D) * 0.01
    x = _psd_data(tiny, n=20000, seed=13)
    h_tiny = float(gaussian_entropy(tiny))
    h_floor = float(gaussian_entropy(torch.diag(torch.tensor(NOISE_VAR))))
    achieved = _fit_unconditional(GaussianStateHead(d_in=8), x)
    ok = achieved > h_tiny + 1.0 and achieved <= h_floor + 0.05
    print(f"[check] sub-floor data: NLL {achieved:.3f} pinned near floor {h_floor:.3f}, "
          f"far above sub-floor entropy {h_tiny:.3f} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_noise_floor_is_a_buffer_no_grad() -> None:
    head = GaussianStateHead(d_in=8)
    names = dict(head.named_buffers())
    assert "noise_var" in names
    assert not names["noise_var"].requires_grad
    assert "noise_var" not in dict(head.named_parameters())
    print("[check] noise floor is a non-trainable buffer (no grad, no decay) OK")


def test_gaussian_entropy_matches_closed_form_identity() -> None:
    ent = float(gaussian_entropy(torch.eye(D)))
    expect = 0.5 * D * math.log(2.0 * math.pi * math.e)
    assert abs(ent - expect) < 1e-5
    print(f"[check] H(I_{D}) = {ent:.4f} == {expect:.4f} OK")


def test_noise_override_none_matches_fixed_buffer() -> None:
    torch.manual_seed(0)
    head = GaussianStateHead(16)
    feat = torch.randn(4, 5, 16)
    mu0, cov0 = head(feat)
    mu1, cov1 = head(feat, noise=None)
    same = torch.equal(mu0, mu1) and torch.equal(cov0, cov1)
    print(f"[check] noise=None == fixed-buffer path={same} {'OK' if same else 'VIOLATED'}")
    assert same


def _noise_var_t() -> torch.Tensor:
    return torch.tensor(NOISE_VAR, dtype=torch.float32)


def test_noise_override_sets_per_position_floor_and_stays_pd() -> None:
    torch.manual_seed(1)
    head = GaussianStateHead(16)
    feat = torch.randn(3, 7, 16)
    _, cov_fixed = head(feat)
    noise = 0.1 + torch.rand(3, 7, D)
    _, cov_ovr = head(feat, noise=noise)
    delta = cov_ovr - cov_fixed
    expect = torch.diag_embed(noise - _noise_var_t())
    diag_ok = torch.allclose(delta, expect, atol=1e-5)
    pd = bool((torch.linalg.eigvalsh(cov_ovr).min() > 0))
    ok = diag_ok and pd
    print(f"[check] per-position floor = diag(noise−buffer)={diag_ok}; cov PD={pd} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_head_assembles_cov_in_fp32_under_bf16_autocast() -> None:
    torch.manual_seed(0)
    head = GaussianStateHead(d_in=16)
    feat = torch.randn(128, 16)
    noise = count_dependent_noise_var(torch.randint(1, 40, (128,)))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, cov_fix = head(feat)
        _, cov_cnt = head(feat, noise=noise)
    for tag, cov in [("fixed", cov_fix), ("count-dep", cov_cnt)]:
        is_fp32 = cov.dtype == torch.float32
        torch.linalg.cholesky(cov)
        min_eig = float(torch.linalg.eigvalsh(cov).min())
        print(f"[check] autocast cov {tag}: dtype={cov.dtype} min_eig={min_eig:.4f} "
              f"{'OK' if is_fp32 and min_eig > 0 else 'VIOLATED'}")
        assert is_fp32 and min_eig > 0


def test_present_masked_all_present_equals_full_nll() -> None:
    # present_masked_nll (full-cov, retired path) with every dim present == the full D-D NLL.
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(20, D, generator=g)
    a = torch.randn(20, D, D, generator=g)
    cov = a @ a.transpose(-1, -2) + torch.eye(D) * 0.5
    x = torch.randn(20, D, generator=g)
    present = torch.ones(20, D, dtype=torch.bool)
    va = present_masked_nll(mu, cov, x, present)
    vb = gaussian_nll(mu, cov, x)
    ok = torch.allclose(va, vb, atol=1e-6)
    print(f"[check] all-present marginal == full NLL ({float(va):.4f} vs {float(vb):.4f}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


# --- r5 Arm 2: floor-off (full-cov head) --------------------------------------------------
def test_arm2_jitter_cannot_masquerade_as_a_floor() -> None:
    smallest_measured = min(NOISE_VAR)
    orders = math.log10(smallest_measured / NLL_FLOOR_JITTER)
    ok = orders >= 4.0
    print(f"[check] jitter {NLL_FLOOR_JITTER:g} vs smallest measured floor "
          f"{smallest_measured:g} -> {orders:.1f} orders below (need >=4) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_arm2_floor_off_keeps_sigma_pd_when_L_underflows() -> None:
    torch.manual_seed(0)
    head = GaussianStateHead(d_in=16)
    with torch.no_grad():
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
