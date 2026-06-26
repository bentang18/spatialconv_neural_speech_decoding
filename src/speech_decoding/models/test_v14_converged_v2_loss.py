"""P2.7 tests for the converged-v2 shared-denominator dual-depth JEPA loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from speech_decoding.models.v14_converged_v2 import converged_v2_loss


def _batch(Q2=20, Q4=12, d=16, seed=0):
    torch.manual_seed(seed)
    m2_pred = torch.randn(Q2, d)
    m2_target = torch.randn(Q2, d) * 7.0          # deliberately off-scale (shallow)
    m4_pred = torch.randn(Q4, d)
    m4_target = torch.randn(Q4, d) * 0.1          # off-scale the other way (deep)
    m4_weight = torch.randint(1, 5, (Q4,)).float()  # w_p·n_p per query
    m4_tubed = torch.zeros(Q4, dtype=torch.bool)
    m4_tubed[: Q4 // 2] = True
    return m2_pred, m2_target, m4_pred, m4_target, m4_weight, m4_tubed


def test_loss_scalar_and_finite():
    out = converged_v2_loss(*_batch())
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    for key in ("loss_m2", "loss_m4", "loss_m4_tubed", "loss_m4_untubed"):
        assert torch.isfinite(out[key])


def test_perfect_prediction_zero_loss():
    """pred == LN(target) ⇒ every error 0 ⇒ loss 0."""
    Q2, Q4, d = 10, 8, 16
    m2_target = torch.randn(Q2, d) * 5
    m4_target = torch.randn(Q4, d) * 0.3
    m2_pred = F.layer_norm(m2_target, (d,))
    m4_pred = F.layer_norm(m4_target, (d,))
    w = torch.rand(Q4) + 0.5
    tubed = torch.zeros(Q4, dtype=torch.bool)
    out = converged_v2_loss(m2_pred, m2_target, m4_pred, m4_target, w, tubed)
    assert out["loss"].abs() < 1e-6


def test_legibility_constant_error_equals_e_independent_of_weights():
    """The crux: if EVERY per-scalar error == e, loss == e EXACTLY, for ANY weights.
    This is what the shared (weighted num / weighted denom) buys — the old weighted-
    num / unweighted-denom scheme would report e·(mean weight) ≠ e."""
    Q2, Q4, d, e = 14, 9, 16, 0.37
    m2_target = torch.randn(Q2, d) * 9
    m4_target = torch.randn(Q4, d) * 0.05
    m2_pred = F.layer_norm(m2_target, (d,)) + e         # constant +e per scalar
    m4_pred = F.layer_norm(m4_target, (d,)) + e
    tubed = torch.zeros(Q4, dtype=torch.bool)
    loss_a = converged_v2_loss(
        m2_pred, m2_target, m4_pred, m4_target, torch.ones(Q4), tubed
    )["loss"]
    loss_b = converged_v2_loss(
        m2_pred, m2_target, m4_pred, m4_target, torch.full((Q4,), 3.0), tubed
    )["loss"]
    loss_c = converged_v2_loss(
        m2_pred, m2_target, m4_pred, m4_target, torch.randint(1, 9, (Q4,)).float(), tubed
    )["loss"]
    assert torch.allclose(loss_a, torch.tensor(e), atol=1e-5)
    assert torch.allclose(loss_b, torch.tensor(e), atol=1e-5)   # ×3 does NOT shrink loss
    assert torch.allclose(loss_c, torch.tensor(e), atol=1e-5)


def test_ln_puts_targets_on_same_order():
    """Refinement 5(a): despite raw targets differing 70× in scale, after LN the
    per-scalar M2 and M4 errors are same-order ⇒ loss_m2 ≈ loss_m4 within 10×."""
    out = converged_v2_loss(*_batch())
    r = out["ratio_m2_m4"].item()
    assert 0.1 < r < 10.0


def test_diagnostics_match_manual():
    m2_pred, m2_target, m4_pred, m4_target, w, tubed = _batch()
    out = converged_v2_loss(m2_pred, m2_target, m4_pred, m4_target, w, tubed)
    d = m2_pred.shape[-1]
    err_m4 = (m4_pred - F.layer_norm(m4_target, (d,))).abs().sum(-1)
    man_tubed = (w[tubed] * err_m4[tubed]).sum() / (w[tubed].sum() * d)
    man_unt = (w[~tubed] * err_m4[~tubed]).sum() / (w[~tubed].sum() * d)
    assert torch.allclose(out["loss_m4_tubed"], man_tubed, atol=1e-6)
    assert torch.allclose(out["loss_m4_untubed"], man_unt, atol=1e-6)


def test_gradient_flows_to_preds_not_targets():
    m2_pred, m2_target, m4_pred, m4_target, w, tubed = _batch()
    m2_pred.requires_grad_(True)
    m4_pred.requires_grad_(True)
    m2_target.requires_grad_(True)
    m4_target.requires_grad_(True)
    converged_v2_loss(m2_pred, m2_target, m4_pred, m4_target, w, tubed)["loss"].backward()
    assert m2_pred.grad is not None and torch.isfinite(m2_pred.grad).all()
    assert m4_pred.grad is not None
    assert m2_target.grad is None        # targets detached (stop-grad)
    assert m4_target.grad is None


def test_empty_tubed_group_safe():
    """A 1-parcel-ish batch with no tubed queries ⇒ tubed diagnostic 0, no div0."""
    m2_pred, m2_target, m4_pred, m4_target, w, _ = _batch()
    tubed = torch.zeros(m4_pred.shape[0], dtype=torch.bool)   # nothing tubed
    out = converged_v2_loss(m2_pred, m2_target, m4_pred, m4_target, w, tubed)
    assert out["loss_m4_tubed"] == 0.0
    assert torch.isfinite(out["loss"])


def test_feature_dim_mismatch_rejected():
    with pytest.raises(ValueError, match="feature dim mismatch"):
        converged_v2_loss(
            torch.randn(4, 16), torch.randn(4, 16),
            torch.randn(4, 8), torch.randn(4, 8),
            torch.ones(4), torch.zeros(4, dtype=torch.bool),
        )


def test_m4_weight_length_mismatch_rejected():
    with pytest.raises(ValueError, match="m4_weight length"):
        converged_v2_loss(
            torch.randn(4, 16), torch.randn(4, 16),
            torch.randn(5, 16), torch.randn(5, 16),
            torch.ones(4), torch.zeros(5, dtype=torch.bool),
        )
