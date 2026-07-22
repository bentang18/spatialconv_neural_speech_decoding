"""monitor_taps reducers (#40/#41/#42) — every invariant named, asserted, PRINTED
(feedback-build-the-invariant-into-the-probe). The weighted (compile-safe) reductions
must equal the plain boolean-indexed reference they replace, so the launch monitors report
the same numbers a naive offline computation would.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.monitor_taps import per_band_jepa_stats


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
