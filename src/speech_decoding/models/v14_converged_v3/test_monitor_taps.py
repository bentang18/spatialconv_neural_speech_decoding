"""monitor_taps reducers (#40/#41/#42) — every invariant named, asserted, PRINTED
(feedback-build-the-invariant-into-the-probe). The weighted (compile-safe) reductions
must equal the plain boolean-indexed reference they replace, so the launch monitors report
the same numbers a naive offline computation would.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.monitor_taps import (
    nofusion_recon_stats,
    per_band_jepa_stats,
)


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


def test_nofusion_recon_perfect_prediction_gives_unit_ev() -> None:
    # invariant: pred==tgt (on each stream's valid feats) ⇒ EV=1, var-ratio=1 for both streams.
    torch.manual_seed(2)
    B, total, F_MAX, n_hga, n_lfs = 2, 16, 8, 8, 2
    band = torch.randint(0, 2, (total,))  # 0=HGA 1=LFS
    w = torch.ones(B, total)
    w_hga = w * (band == 0).float()[None]
    w_lfs = w * (band == 1).float()[None]
    tgt = torch.randn(B, total, F_MAX)
    tgt[:, band == 1, n_lfs:] = 0.0  # LFS pad slots zero in the target (as the gather produces)
    got = nofusion_recon_stats(tgt.clone(), tgt, w_hga, w_lfs, n_hga=n_hga, n_lfs=n_lfs)
    ok = all(
        abs(float(got[f"jepa_{n}_explained_var"]) - 1.0) < 1e-5
        and abs(float(got[f"jepa_{n}_pred_target_var_ratio"]) - 1.0) < 1e-5
        for n in ("hga", "lfs")
    )
    print(f"[check] nf perfect-pred ⇒ EV=1,ratio=1 both streams → {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_nofusion_recon_ignores_lfs_pad_slots() -> None:
    # LFS uses only the first n_lfs feats; garbage in the pad slots must NOT change LFS stats
    # (they'd deflate variance if included). HGA (all 8 valid) is untouched here.
    torch.manual_seed(3)
    B, total, F_MAX, n_hga, n_lfs = 2, 20, 8, 8, 2
    band = torch.zeros(total, dtype=torch.long)
    band[total // 2:] = 1  # half HGA, half LFS
    w = torch.ones(B, total)
    w_hga = w * (band == 0).float()[None]
    w_lfs = w * (band == 1).float()[None]
    tgt = torch.randn(B, total, F_MAX)
    pred = tgt + 0.3 * torch.randn(B, total, F_MAX)  # imperfect ⇒ EV<1
    base = nofusion_recon_stats(pred.clone(), tgt.clone(), w_hga, w_lfs, n_hga=n_hga, n_lfs=n_lfs)
    pred2, tgt2 = pred.clone(), tgt.clone()
    pred2[:, band == 1, n_lfs:] = 99.0  # poison LFS pad in pred
    tgt2[:, band == 1, n_lfs:] = -99.0  # and in tgt
    poisoned = nofusion_recon_stats(pred2, tgt2, w_hga, w_lfs, n_hga=n_hga, n_lfs=n_lfs)
    same = all(
        abs(float(base[k]) - float(poisoned[k])) < 1e-6
        for k in ("jepa_lfs_explained_var", "jepa_lfs_pred_target_var_ratio",
                  "jepa_hga_explained_var", "jepa_hga_pred_target_var_ratio")
    )
    print(f"[check] LFS pad slots ignored → {'OK' if same else 'VIOLATED'}")
    assert same


def test_visible_recon_gap_excludes_filler_and_equals_ev_difference() -> None:
    # invariant 1: filler tokens (all-zero target, always visible) must NOT enter the visible
    #   EV — else their zero variance deflates it. invariant 2: recon_gap == visible_EV −
    #   masked_EV per band. invariant 3: visible reconstructs BETTER than masked here (we build
    #   pred to match tgt more closely on visible tokens), so the gap is positive.
    from speech_decoding.models.v14_converged_v3.monitor_taps import (
        per_band_jepa_stats,
        visible_recon_gap_stats,
    )
    torch.manual_seed(7)
    B, total, F = 2, 30, 7
    band_ids = torch.randint(0, 3, (total,))
    w = (torch.rand(B, total) < 0.5).float()  # masked (scored) tokens
    tgt = torch.randn(B, total, F)
    # visible tokens predicted well (low resid), masked tokens predicted poorly (high resid).
    vis = (w == 0).float()[..., None]
    pred = tgt + (0.1 * vis + 0.6 * (1 - vis)) * torch.randn(B, total, F)
    # inject FILLER: 4 always-visible tokens with EXACT-zero target (w already 0 possible; force it)
    fill = torch.zeros(total, dtype=torch.bool); fill[:4] = True
    tgt[:, fill, :] = 0.0
    pred[:, fill, :] = 0.0
    w[:, fill] = 0.0  # filler is always visible

    stats = per_band_jepa_stats(pred, tgt, w, band_ids)
    gap = visible_recon_gap_stats(pred, tgt, w, stats, band_ids)

    # reference visible EV: energy>0 AND w==0, computed by the boolean path.
    energy = tgt.pow(2).sum(-1)
    vis_w = ((energy > 0) & (w == 0)).float()
    ref_vis = per_band_jepa_stats(pred, tgt, vis_w, band_ids)

    worst_ev = 0.0; worst_gap = 0.0; all_pos = True
    for b, name in enumerate(("slow", "mid", "hga")):
        gv = float(gap[f"jepa_{name}_visible_explained_var"])
        rv = float(ref_vis[f"jepa_{name}_explained_var"])
        worst_ev = max(worst_ev, abs(gv - rv))
        expect_gap = gv - float(stats[f"jepa_{name}_explained_var"])
        worst_gap = max(worst_gap, abs(float(gap[f"jepa_{name}_recon_gap"]) - expect_gap))
        all_pos = all_pos and float(gap[f"jepa_{name}_recon_gap"]) > 0

    # filler-inclusion control: if filler were folded into visible, EV would move.
    vis_w_naive = (w == 0).float()  # includes the 4 zero-energy filler tokens
    naive = per_band_jepa_stats(pred, tgt, vis_w_naive, band_ids)
    moved = any(
        abs(float(naive[f"jepa_{n}_explained_var"]) - float(gap[f"jepa_{n}_visible_explained_var"])) > 1e-6
        for n in ("slow", "mid", "hga")
    )
    ok = worst_ev < 1e-5 and worst_gap < 1e-5 and all_pos and moved
    print(f"[check] visible EV==bool ref max|Δ|={worst_ev:.2e}; gap==Δ max|Δ|={worst_gap:.2e}; "
          f"gap>0={all_pos}; filler-exclusion changes EV={moved} → {'OK' if ok else 'VIOLATED'}")
    assert ok
