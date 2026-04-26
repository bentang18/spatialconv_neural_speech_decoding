"""Z-score recipe comparison on PS patient.

PS's `production_highgamma.fif` is all-NaN (stub). Baseline and
productionMeanSub have real data, so we reconstruct:
    production = productionMeanSub + baseline_mean_per_channel
where baseline_mean_per_channel = baseline.mean(axis=(0, 2)).

Then compare recipes A (pooled-baseline mean/std), B (production-pool
median/MAD), B' (production-pool mean/std), C (baseline-pool median/MAD).

Handles BOTH trial-level and phoneme-level .fif paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("ERROR")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="S14")
    parser.add_argument(
        "--bids-root",
        default="/Users/bentang/Documents/Code/speech/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS",
    )
    parser.add_argument(
        "--out",
        default="/Users/bentang/Documents/Code/speech/reports/zscore_comparison_2026_04_18",
    )
    parser.add_argument("--level", choices=["phoneme", "trial"], default="phoneme")
    args = parser.parse_args()

    bids = Path(args.bids_root)
    pt = args.patient
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.level == "phoneme":
        deriv = bids / f"derivatives/epoch(phonemeLevel)(CAR)/sub-{pt}/epoch(band)(power)"
    else:
        deriv = bids / f"derivatives/epoch(CAR)/sub-{pt}/epoch(band)(power)"

    base = mne.read_epochs(
        str(deriv / f"sub-{pt}_task-PhonemeSequence_desc-baseline_highgamma.fif"),
        preload=True, verbose="ERROR",
    )
    meansub = mne.read_epochs(
        str(deriv / f"sub-{pt}_task-PhonemeSequence_desc-productionMeanSub_highgamma.fif"),
        preload=True, verbose="ERROR",
    )
    zsco = mne.read_epochs(
        str(deriv / f"sub-{pt}_task-PhonemeSequence_desc-productionZscore_highgamma.fif"),
        preload=True, verbose="ERROR",
    )

    B = base.get_data().astype(np.float64)
    MS = meansub.get_data().astype(np.float64)
    Z = zsco.get_data().astype(np.float64)

    n_trial, n_ch, L_b = B.shape
    n_phon, _, L_p = MS.shape
    print(f"=== PS {pt} ({args.level}-level): "
          f"n_baseline={n_trial}, n_{args.level}={n_phon}, n_ch={n_ch}, "
          f"L_baseline={L_b}, L_prod={L_p} ===")

    # Reconstruct P = MS + X_channel where X = baseline.mean(axis=(0,2))
    X_A = B.mean(axis=(0, 2))
    Y_A = B.std(axis=(0, 2), ddof=1)
    P = MS + X_A[None, :, None]

    # Verify recipe-A reconstruction
    z_A = (P - X_A[None, :, None]) / (Y_A[None, :, None] + 1e-12)
    # Equivalently z_A = MS / Y_A
    max_abs_diff_A = float(np.abs(z_A - Z).max())
    rmse_A = float(np.sqrt(((z_A - Z) ** 2).mean()))
    corr_AA = float(np.corrcoef(z_A.reshape(-1), Z.reshape(-1))[0, 1])
    print(
        f"  recipe-A reconstruction vs productionZscore: "
        f"max|Δ|={max_abs_diff_A:.3e}, RMSE={rmse_A:.3e}, corr={corr_AA:.6f}"
    )

    # Recipe B: production-pool median/MAD
    P_flat = P.transpose(1, 0, 2).reshape(n_ch, -1)
    X_B = np.median(P_flat, axis=1)
    MAD_B = np.median(np.abs(P_flat - X_B[:, None]), axis=1)
    Y_B = MAD_B * 1.4826
    z_B = (P - X_B[None, :, None]) / (Y_B[None, :, None] + 1e-12)

    # Recipe B': production-pool mean/std
    X_Bp = P_flat.mean(axis=1)
    Y_Bp = P_flat.std(axis=1, ddof=1)
    z_Bp = (P - X_Bp[None, :, None]) / (Y_Bp[None, :, None] + 1e-12)

    # Recipe C: baseline-pool median/MAD (robust analogue of A)
    B_flat = B.transpose(1, 0, 2).reshape(n_ch, -1)
    X_C = np.median(B_flat, axis=1)
    MAD_C = np.median(np.abs(B_flat - X_C[:, None]), axis=1)
    Y_C = MAD_C * 1.4826
    z_C = (P - X_C[None, :, None]) / (Y_C[None, :, None] + 1e-12)

    def summarize(name, X, Y, z_alt):
        loc_diff = (X - X_A) / (Y_A + 1e-12)
        scale_ratio = Y / (Y_A + 1e-12)
        per_ch = np.empty(n_ch)
        for c in range(n_ch):
            a = z_A[:, c, :].ravel()
            b = z_alt[:, c, :].ravel()
            ad = a - a.mean()
            bd = b - b.mean()
            denom = np.sqrt((ad * ad).sum() * (bd * bd).sum())
            per_ch[c] = (ad * bd).sum() / denom if denom > 0 else np.nan
        return {
            "name": name,
            "loc_diff_abs_median": float(np.median(np.abs(loc_diff))),
            "loc_diff_abs_p95": float(np.percentile(np.abs(loc_diff), 95)),
            "scale_ratio_median": float(np.median(scale_ratio)),
            "scale_ratio_p5": float(np.percentile(scale_ratio, 5)),
            "scale_ratio_p95": float(np.percentile(scale_ratio, 95)),
            "pearson_median": float(np.nanmedian(per_ch)),
            "pearson_p5": float(np.nanpercentile(per_ch, 5)),
            "rms": float(np.sqrt((z_alt ** 2).mean())),
        }

    sum_B = summarize("production_median_MAD", X_B, Y_B, z_B)
    sum_Bp = summarize("production_mean_std", X_Bp, Y_Bp, z_Bp)
    sum_C = summarize("baseline_median_MAD", X_C, Y_C, z_C)

    # Class separability (phoneme-level only, for PS 9-phoneme codes)
    codes = zsco.events[:, 2]
    tmin = float(zsco.tmin)
    sfreq = float(zsco.info["sfreq"])
    i0 = int(round((0.05 - tmin) * sfreq))
    i1 = int(round((0.30 - tmin) * sfreq))

    def class_sep(z):
        w = z[:, :, i0:i1].mean(axis=-1)
        classes = np.unique(codes)
        means = np.stack([w[codes == k].mean(axis=0) for k in classes])
        grand = w.mean(axis=0)
        between = ((means - grand) ** 2).mean(axis=0)
        within = np.stack([np.var(w[codes == k], axis=0, ddof=1) for k in classes]).mean(axis=0)
        return between / (within + 1e-12)

    eta_A = class_sep(z_A)
    eta_B = class_sep(z_B)

    report = {
        "patient": pt,
        "cohort": "PS",
        "level": args.level,
        "shapes": {
            "n_baseline": int(n_trial),
            "n_phon": int(n_phon),
            "n_channels": int(n_ch),
            "sfreq": float(sfreq),
            "production_samples": int(L_p),
            "baseline_samples": int(L_b),
        },
        "recipe_A_reconstruction_vs_productionZscore": {
            "max_abs_diff": max_abs_diff_A,
            "rmse": rmse_A,
            "corr": corr_AA,
        },
        "summaries": {
            "B_production_median_MAD": sum_B,
            "Bp_production_mean_std": sum_Bp,
            "C_baseline_median_MAD": sum_C,
        },
        "class_separability_eta_sq_med": {
            "A": float(np.median(eta_A)),
            "B": float(np.median(eta_B)),
        },
    }
    (out_dir / f"ps_{pt}_{args.level}.json").write_text(json.dumps(report, indent=2))

    print()
    print(f"=== SUMMARY: PS {pt} ({args.level}-level, {n_ch} ch) ===")
    print(f"  Recipe A ≡ productionZscore: max|Δ|={max_abs_diff_A:.2e}, corr={corr_AA:.4f}")
    for name, s in [
        ("B  prod median/MAD   ", sum_B),
        ("B' prod mean/std     ", sum_Bp),
        ("C  base median/MAD   ", sum_C),
    ]:
        print(
            f"  {name}: |locΔ|med={s['loc_diff_abs_median']:.3f}  "
            f"scale med={s['scale_ratio_median']:.3f} "
            f"[{s['scale_ratio_p5']:.2f}, {s['scale_ratio_p95']:.2f}]  "
            f"ρ_med={s['pearson_median']:.4f}, ρ_p5={s['pearson_p5']:.4f}  "
            f"rms={s['rms']:.3f}"
        )
    print(f"  class-sep η² med: A={np.median(eta_A):.4f}, B={np.median(eta_B):.4f} (should match)")
    print(f"  wrote {out_dir / f'ps_{pt}_{args.level}.json'}")


if __name__ == "__main__":
    main()
