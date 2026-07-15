"""monitor_taps reducers (#40/#41/#42) — every invariant named, asserted, PRINTED
(feedback-build-the-invariant-into-the-probe). The weighted (compile-safe) reductions
must equal the plain boolean-indexed reference they replace, so the launch monitors report
the same numbers a naive offline computation would.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.monitor_taps import (
    cov_entropy_vs_floor,
    per_band_jepa_stats,
    per_band_nll,
)
from speech_decoding.models.v14_converged_v3.secondary_head import _nll_terms


def _ref_band_jepa(pred, tgt, w, band_ids, b, eps=1e-8):
    """Plain boolean-indexed reference: gather the band's scored tokens, then
    mean_f Var_tok exactly (no weighting trick)."""
    sel = (w > 0) & (band_ids.unsqueeze(0) == b)  # (B, total) bool
    p = pred[sel]  # (Nsel, F)
    t = tgt[sel]
    r = t - p
    var_t = t.var(0, unbiased=False).mean()  # mean over F of per-feat var
    var_p = p.var(0, unbiased=False).mean()
    var_r = r.var(0, unbiased=False).mean()
    return {
        "explained_var": float(1.0 - var_r / (var_t + eps)),
        "pred_target_var_ratio": float(var_p / (var_t + eps)),
        "l1": float(r.abs().mean()),
    }


def test_per_band_jepa_matches_boolean_reference() -> None:
    torch.manual_seed(0)
    B, total, F = 2, 20, 16
    pred = torch.randn(B, total, F)
    tgt = pred + 0.3 * torch.randn(B, total, F)  # partly predictable ⇒ EV in (0,1)
    band_ids = torch.randint(0, 3, (total,))
    # scored mask: a random ~60% of tokens are in-loss (weight 1), rest 0.
    w = (torch.rand(B, total) < 0.6).float()
    got = per_band_jepa_stats(pred, tgt, w, band_ids)

    worst = 0.0
    for b, name in enumerate(("slow", "mid", "hga")):
        ref = _ref_band_jepa(pred, tgt, w, band_ids, b)
        for key, rv in ref.items():
            gv = float(got[f"jepa_{name}_{key}"])
            worst = max(worst, abs(gv - rv))
    ok = worst < 1e-4
    print(f"[check] per-band JEPA weighted==boolean: max|Δ|={worst:.2e} → {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_per_band_jepa_perfect_prediction_gives_unit_ev_zero_l1() -> None:
    # invariant: pred==tgt ⇒ EV=1, L1=0, var-ratio=1 in every band (a sanity floor).
    torch.manual_seed(1)
    B, total, F = 2, 18, 8
    tgt = torch.randn(B, total, F)
    band_ids = torch.randint(0, 3, (total,))
    w = torch.ones(B, total)
    got = per_band_jepa_stats(tgt.clone(), tgt, w, band_ids)
    ok = all(
        abs(float(got[f"jepa_{n}_explained_var"]) - 1.0) < 1e-5
        and abs(float(got[f"jepa_{n}_l1"])) < 1e-6
        and abs(float(got[f"jepa_{n}_pred_target_var_ratio"]) - 1.0) < 1e-5
        for n in ("slow", "mid", "hga")
    )
    print(f"[check] perfect-pred ⇒ EV=1,L1=0,ratio=1 all bands → {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_per_band_nll_matches_marginal_subblock_reference() -> None:
    # invariant: the per-band scalar equals _nll_terms on the boolean-selected present
    # positions of that band's 2-D marginal (dims [b, b+3]).
    torch.manual_seed(2)
    B, Q, D = 2, 12, 6
    mu = torch.randn(B, Q, D)
    L = torch.randn(B, Q, D, D) * 0.2
    cov = L @ L.transpose(-1, -2) + torch.eye(D) * 0.5  # PD
    target_q = torch.randn(B, Q, D)
    present_q = (torch.rand(B, Q, D) < 0.7).float()
    got = per_band_nll(mu, cov, target_q, present_q)

    worst = 0.0
    for b, name in enumerate(("slow", "mid", "hga")):
        idx = [b, b + 3]
        sub_cov = cov[..., idx, :][..., :, idx]
        nll_pos = _nll_terms(mu[..., idx], sub_cov, target_q[..., idx])  # (B,Q)
        wpos = present_q[..., b + 3]
        ref = float((nll_pos * wpos).sum() / wpos.sum().clamp_min(1.0))
        worst = max(worst, abs(float(got[f"nll_{name}"]) - ref))
    ok = worst < 1e-5
    print(f"[check] per-band NLL == marginal-subblock ref: max|Δ|={worst:.2e} → {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_cov_entropy_gap_nonnegative_and_floor_exact() -> None:
    # invariant 1: gap = H(cov) - H(diag(noise)) >= 0 (cov ⪰ diag(noise) by construction).
    # invariant 2: with L=0 the head IS the floor ⇒ gap == 0 exactly.
    torch.manual_seed(3)
    B, Q, D = 2, 10, 6
    noise = torch.rand(B, Q, D) * 0.3 + 0.1  # PD floor
    L = torch.randn(B, Q, D, D) * 0.4
    cov = L @ L.transpose(-1, -2) + torch.diag_embed(noise)
    out = cov_entropy_vs_floor(cov, noise)
    gap = float(out["cov_entropy_gap"])
    floor_only = cov_entropy_vs_floor(torch.diag_embed(noise), noise)
    gap0 = float(floor_only["cov_entropy_gap"])
    # cross-check the floor entropy against a closed-form diagonal Gaussian entropy.
    cf_floor = float((0.5 * (D * (1.0 + torch.log(torch.tensor(2 * torch.pi))) + torch.log(noise).sum(-1))).mean())
    floor_match = abs(float(out["cov_entropy_floor"]) - cf_floor) < 1e-4
    ok = gap >= -1e-5 and abs(gap0) < 1e-5 and floor_match
    print(
        f"[check] cov-entropy gap≥0 ({gap:.4f}), L=0⇒gap==0 ({gap0:.2e}), "
        f"floor==closed-form ({floor_match}) → {'OK' if ok else 'VIOLATED'}"
    )
    assert ok
