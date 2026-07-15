"""MON-LOSS-GRAD-RATIO helper tests (colocated). Invariants named + asserted + printed
(feedback-build-the-invariant-into-the-probe):

  1. The reported ‖g_jepa‖ / ‖g_nll‖ EQUAL an independent manual autograd reference.
  2. retain_graph holds: after loss_grad_ratio, a normal ``total.backward()`` still runs
     and populates .grad (the live wire-point — the monitor must not consume the graph).
  3. grad_ratio_weighted == lambda_nll · grad_ratio (the effective pull identity).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from speech_decoding.experiments.monitors.grad_ratio import loss_grad_ratio


def _toy():
    torch.manual_seed(0)
    tower = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 4))  # the "shared" trunk
    x = torch.randn(16, 8)
    h = tower(x)
    # two objectives that BOTH pull on the tower via h, plus a JEPA-only head.
    jepa_head = nn.Linear(4, 3)
    jepa_loss = jepa_head(h).abs().mean()
    nll_loss = (h.pow(2).sum(-1) - 1.0).mean()  # sign-indefinite, like the real NLL
    return tower, jepa_head, jepa_loss, nll_loss


def _ref_norm(loss, params) -> float:
    grads = torch.autograd.grad(loss, list(params), retain_graph=True, allow_unused=True)
    sq = sum((g.pow(2).sum() for g in grads if g is not None), start=torch.zeros(()))
    return float(sq.sqrt())


def test_grad_ratio_matches_manual_reference_and_retains_graph() -> None:
    tower, jepa_head, jepa_loss, nll_loss = _toy()
    params = list(tower.parameters())
    lam = 0.2

    out = loss_grad_ratio(jepa_loss, nll_loss, params, lambda_nll=lam)

    ref_gj = _ref_norm(jepa_loss, params)
    ref_gn = _ref_norm(nll_loss, params)
    gj_ok = math.isclose(out["loss_g_jepa"], ref_gj, rel_tol=1e-5, abs_tol=1e-7)
    gn_ok = math.isclose(out["loss_g_nll"], ref_gn, rel_tol=1e-5, abs_tol=1e-7)
    ratio_ok = math.isclose(out["grad_ratio"], ref_gn / ref_gj, rel_tol=1e-5)
    wt_ok = math.isclose(out["grad_ratio_weighted"], lam * out["grad_ratio"], rel_tol=1e-9)

    # retain_graph invariant: the total backward must still run after the probe.
    total = jepa_loss + lam * nll_loss
    total.backward()
    grad_flows = all(p.grad is not None for p in params)

    ok = gj_ok and gn_ok and ratio_ok and wt_ok and grad_flows
    print(
        f"[check] grad-ratio vs manual: g_jepa {out['loss_g_jepa']:.5f}(={ref_gj:.5f}) "
        f"g_nll {out['loss_g_nll']:.5f}(={ref_gn:.5f}) ratio {out['grad_ratio']:.4f} "
        f"λ·ratio {out['grad_ratio_weighted']:.4f} | backward-after-probe grads="
        f"{grad_flows} → {'OK' if ok else 'VIOLATED'}"
    )
    assert ok


def test_grad_ratio_flags_degenerate_zero_jepa_grad() -> None:
    # a jepa_loss disconnected from the tower ⇒ ‖g_jepa‖ = 0 ⇒ ratio reported as inf,
    # not a silent divide-by-zero.
    tower, _jh, _jl, nll_loss = _toy()
    params = list(tower.parameters())
    detached = torch.zeros((), requires_grad=True)
    jepa_loss = detached * 2.0  # touches `detached`, NOT the tower
    out = loss_grad_ratio(jepa_loss, nll_loss, params, lambda_nll=0.2)
    ok = out["loss_g_jepa"] == 0.0 and math.isinf(out["grad_ratio"])
    print(f"[check] zero g_jepa → ratio=inf (not a crash): {out['grad_ratio']} → {'OK' if ok else 'VIOLATED'}")
    assert ok
