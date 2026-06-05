"""Tests for the VICReg variance-floor anti-collapse term (``ssl/variance.py``).

The term exists to make the P1 paradigm-A constant-collapse a HIGH-loss point
(see the module docstring). These tests pin its three load-bearing properties:

* a collapsed (rank-1, all-rows-identical) representation → loss ≈ ``gamma``
  (maximal hinge) — the collapse the capstone run fell into is now penalised;
* a diverse representation with per-dim std ≥ ``gamma`` → loss == 0 AND zero
  gradient (one-sided: a generous coefficient never distorts a healthy rep);
* scaling cannot cheat it — multiplying constant rows by any LN affine scale
  keeps their per-dim std at 0, so the floor is unsatisfiable without genuine
  cross-token diversity.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.ssl.variance import covariance_loss, variance_floor_loss


def _rankme_norm(z: torch.Tensor) -> float:
    """Normalised effective rank — the collapse metric the floor protects."""
    z = z.float() - z.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(z)
    p = s / (s.sum() + 1e-12)
    ent = -(p * (p + 1e-12).log()).sum()
    return float(torch.exp(ent) / z.shape[1])


def test_collapsed_tokens_give_max_hinge() -> None:
    """All rows identical (rank 1) ⇒ per-dim std → 0 ⇒ loss ≈ gamma."""
    tokens = torch.ones(64, 16)  # every row the same vector
    loss, mean_std = variance_floor_loss(tokens, gamma=1.0)
    assert loss.item() == pytest.approx(1.0, abs=1e-2)
    # eps under the sqrt floors mean_std at sqrt(eps) ≈ 0.01 — collapsed, far
    # below a healthy ≈ gamma.
    assert mean_std.item() < 0.05


def test_diverse_tokens_above_floor_give_zero_loss_and_grad() -> None:
    """Per-dim std ≥ gamma ⇒ hinge is 0 AND the gradient is exactly 0 (the
    one-sided property: the term only activates below the floor, so it cannot
    distort an already-healthy representation)."""
    torch.manual_seed(0)
    tokens = (torch.randn(2048, 16) * 3.0).requires_grad_(True)  # std ≈ 3 ≫ 1
    loss, mean_std = variance_floor_loss(tokens, gamma=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert mean_std.item() > 1.0
    loss.backward()
    assert tokens.grad is not None
    assert float(tokens.grad.abs().sum()) == pytest.approx(0.0, abs=1e-6)


def test_partial_floor_is_one_sided_per_dim() -> None:
    """A representation diverse on some dims, collapsed on others: only the
    below-floor dims contribute. With half the dims at std≈0 and half at
    std≫gamma, loss ≈ gamma * (fraction below floor) = 0.5."""
    torch.manual_seed(0)
    n, d = 1024, 16
    z = torch.randn(n, d) * 3.0
    z[:, : d // 2] = 7.0  # first half constant ⇒ std 0 ⇒ full hinge
    loss, _ = variance_floor_loss(z, gamma=1.0)
    assert loss.item() == pytest.approx(0.5, abs=2e-2)


def test_scaling_a_collapsed_rep_cannot_cheat_the_floor() -> None:
    """A LayerNorm affine scale multiplies a constant rep but cannot create
    cross-token variance — std stays 0, the hinge stays maximal. This is the
    reason the floor (not LN) is the right collapse guard."""
    constant_rows = torch.full((128, 8), 0.3)
    for scale in (1.0, 10.0, 100.0):
        loss, mean_std = variance_floor_loss(constant_rows * scale, gamma=1.0)
        assert loss.item() == pytest.approx(1.0, abs=1e-2)
        assert mean_std.item() < 0.05  # sqrt(eps) floor, scale-invariant


def test_gradient_pushes_collapsed_rep_apart() -> None:
    """One descent step on the variance loss strictly increases the per-dim
    std of a (near-)collapsed representation — the term actively un-collapses,
    not merely flags."""
    torch.manual_seed(0)
    z = (torch.ones(256, 16) + 1e-3 * torch.randn(256, 16)).requires_grad_(True)
    loss0, std0 = variance_floor_loss(z, gamma=1.0)
    loss0.backward()
    with torch.no_grad():
        z_next = z - 10.0 * z.grad
    _, std1 = variance_floor_loss(z_next, gamma=1.0)
    assert std1.item() > std0.item()


def test_gamma_sets_the_floor_height() -> None:
    """The hinge is ``gamma - std`` clamped at 0, so a fully collapsed rep
    scores ≈ gamma for any gamma."""
    tokens = torch.zeros(32, 8)
    for gamma in (0.5, 1.0, 2.0):
        loss, _ = variance_floor_loss(tokens, gamma=gamma)
        assert loss.item() == pytest.approx(gamma, abs=1e-2)


def test_fewer_than_two_rows_is_exact_zero_no_nan() -> None:
    """std is undefined for < 2 rows ⇒ exact 0 (graph-connected), never NaN."""
    for n in (0, 1):
        tokens = torch.randn(n, 16, requires_grad=True)
        loss, mean_std = variance_floor_loss(tokens)
        assert float(loss.detach()) == 0.0
        assert torch.isfinite(loss)
        assert float(mean_std) == 0.0


def test_single_row_loss_stays_connected_to_graph() -> None:
    """The N<2 zero is ``tokens.sum() * 0`` so it carries a grad_fn — a
    backward never hits a detached-leaf surprise."""
    tokens = torch.randn(1, 16, requires_grad=True)
    loss, _ = variance_floor_loss(tokens)
    loss.backward()
    assert tokens.grad is not None
    assert float(tokens.grad.abs().sum()) == 0.0


def test_rejects_non_2d_tokens() -> None:
    with pytest.raises(ValueError, match=r"\(N, d\)"):
        variance_floor_loss(torch.randn(2, 3, 4))


def test_loss_is_finite_at_exact_collapse() -> None:
    """eps under the sqrt keeps the gradient finite at std → 0 exactly."""
    tokens = torch.zeros(64, 16, requires_grad=True)
    loss, _ = variance_floor_loss(tokens, gamma=1.0, eps=1e-4)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(tokens.grad).all()


def test_bf16_input_is_computed_in_fp32() -> None:
    """A bf16 representation (autocast) carries ~3 sig digits — too few for a
    variance. The loss casts to fp32 internally, so a diverse bf16 rep still
    reads as above-floor (std accurate) rather than spuriously triggering."""
    torch.manual_seed(0)
    tokens = (torch.randn(1024, 16) * 3.0).to(torch.bfloat16)
    loss, mean_std = variance_floor_loss(tokens, gamma=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)
    assert mean_std.item() > 1.0


# ---------------------------------------------------------------------------
# Covariance decorrelation — the dimensional-collapse guard the variance
# floor is blind to (4-agent audit, 2026-06-04)
# ---------------------------------------------------------------------------


def test_decorrelated_rep_has_near_zero_covariance() -> None:
    """Independent feature dims ⇒ off-diagonal covariance ≈ 0."""
    torch.manual_seed(0)
    z = torch.randn(8192, 16)  # iid dims, no cross-correlation
    assert covariance_loss(z).item() < 0.05


def test_dimensional_collapse_evades_variance_but_covariance_fires() -> None:
    """THE headline: a rank-k≪d subspace, rescaled so every per-dim std ≥ gamma,
    sits deep in the RankMe alarm band yet the variance floor reads it as
    healthy (loss ≈ 0). The covariance term is what catches it — it must be
    large. This is the exact hole the audit reproduced; the test pins that
    covariance closes it."""
    torch.manual_seed(0)
    n, d, k = 4096, 64, 4
    basis = torch.randn(d, k)
    z = (torch.randn(n, k) @ basis.T)
    z = z / (z.std(dim=0, keepdim=True) + 1e-6)  # force per-dim std ≈ 1
    var_loss, mean_std = variance_floor_loss(z, gamma=1.0)
    cov = covariance_loss(z)
    assert _rankme_norm(z) < 0.25          # a hard RankMe alarm (dim-collapsed)
    assert var_loss.item() < 0.05          # variance floor is BLIND to it
    assert mean_std.item() > 0.9
    assert cov.item() > 1.0                 # covariance fires hard


def test_covariance_gradient_decorrelates() -> None:
    """One descent step on the covariance loss strictly reduces the off-diagonal
    correlation energy of a correlated representation — it actively decorrelates,
    raising effective rank."""
    torch.manual_seed(0)
    n, d, k = 2048, 32, 4
    basis = torch.randn(d, k)
    z = (torch.randn(n, k) @ basis.T)
    z = (z / (z.std(dim=0, keepdim=True) + 1e-6)).requires_grad_(True)
    rank0 = _rankme_norm(z.detach())
    loss0 = covariance_loss(z)
    loss0.backward()
    with torch.no_grad():
        z_next = z - 0.1 * z.grad
    assert covariance_loss(z_next).item() < loss0.item()
    assert _rankme_norm(z_next) > rank0


def test_covariance_penalizes_correlation_not_diagonal() -> None:
    """Only the OFF-diagonal is penalised: a correlated rep scores far higher
    than an UNcorrelated rep with the SAME per-dim (diagonal) variances. The
    diagonal — per-dim variance — is the variance floor's job, so the
    covariance loss is driven by cross-dim correlation, not diagonal
    magnitude. (The term does scale with overall magnitude, as C ∝ zᵀz; at the
    operating point the variance floor + LayerNorm hold per-dim std ≈ 1, the
    regime cov_coeff is calibrated against.)"""
    torch.manual_seed(0)
    n, d = 16384, 16
    scales = torch.linspace(0.5, 2.0, d)
    indep = torch.randn(n, d) * scales  # heteroscedastic, uncorrelated
    factors = torch.randn(n, 2) @ torch.randn(2, d)  # 2 shared factors
    corr = factors / (factors.std(dim=0, keepdim=True) + 1e-6) * scales
    assert covariance_loss(corr).item() > 10.0 * covariance_loss(indep).item()


def test_covariance_fewer_than_two_rows_is_exact_zero() -> None:
    for n in (0, 1):
        z = torch.randn(n, 16, requires_grad=True)
        loss = covariance_loss(z)
        assert float(loss.detach()) == 0.0
        assert torch.isfinite(loss)


def test_covariance_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match=r"\(N, d\)"):
        covariance_loss(torch.randn(2, 3, 4))


def test_covariance_is_fp32_under_bf16() -> None:
    torch.manual_seed(0)
    z = torch.randn(2048, 16).to(torch.bfloat16)
    loss = covariance_loss(z)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_variance_and_covariance_divide_labor() -> None:
    """The two terms cover disjoint collapse modes:
      * constant (rank-1): variance fires, covariance is ~0 (no spread to
        correlate);
      * dimensional (rank-k, per-dim std normalised): variance ~0, covariance
        fires.
    Neither term alone covers both — the pair does."""
    torch.manual_seed(0)
    n, d = 4096, 64
    const = torch.ones(n, d) * torch.randn(d)
    v_const, _ = variance_floor_loss(const, gamma=1.0)
    assert v_const.item() > 0.9 and covariance_loss(const).item() < 0.05

    basis = torch.randn(d, 4)
    dim = torch.randn(n, 4) @ basis.T
    dim = dim / (dim.std(dim=0, keepdim=True) + 1e-6)
    v_dim, _ = variance_floor_loss(dim, gamma=1.0)
    assert v_dim.item() < 0.05 and covariance_loss(dim).item() > 1.0
