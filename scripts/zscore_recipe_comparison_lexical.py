"""Z-score recipe comparison on lexical patient using the un-z-scored
baseline/production HG files.

The actual z-score recipe (inferred from productionZscore - production reverse
engineering, 2026-04-18): for each channel c,
    X_c = mean of baseline values across all trials and all baseline samples
    Y_c = std  of baseline values across all trials and all baseline samples
    z(x) = (x - X_c) / Y_c

So it is NOT a per-trial z-score; it is a per-channel z-score with statistics
pooled across the pre-auditory-window samples of every trial.

This test compares that recipe (A) to recording-level median/MAD (B) and
recording-level mean/std (B') computed over pooled production-window samples,
which is a plausible drop-in replacement for the Phase-1.5 SSL scale contract.

Reads:
  baseline_highgamma.fif   → per-trial un-z-scored HG over pre-auditory 500 ms
  production_highgamma.fif → per-phoneme un-z-scored HG over [-1.0, 1.5) s
  productionZscore_highgamma.fif → same shape, z-scored via recipe A
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
    parser.add_argument("--patient", default="S41")
    parser.add_argument(
        "--bids-root",
        default="/Users/bentang/Documents/Code/speech/BIDS_1.0_Lexical_µECoG/BIDS_1.0_Lexical_µECoG/BIDS",
    )
    parser.add_argument(
        "--out",
        default="/Users/bentang/Documents/Code/speech/reports/zscore_comparison_2026_04_18",
    )
    args = parser.parse_args()

    bids = Path(args.bids_root)
    pt = args.patient
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    deriv = bids / f"derivatives/epoch(phonemeLevel)(CAR)/sub-{pt}/epoch(band)(power)"
    base = mne.read_epochs(
        str(deriv / f"sub-{pt}_task-lexical_desc-baseline_highgamma.fif"),
        preload=True,
        verbose="ERROR",
    )
    prod = mne.read_epochs(
        str(deriv / f"sub-{pt}_task-lexical_desc-production_highgamma.fif"),
        preload=True,
        verbose="ERROR",
    )
    zsco = mne.read_epochs(
        str(deriv / f"sub-{pt}_task-lexical_desc-productionZscore_highgamma.fif"),
        preload=True,
        verbose="ERROR",
    )

    B = base.get_data().astype(np.float64)  # (T_tr, C, L_b)
    P = prod.get_data().astype(np.float64)  # (T_ph, C, L_p)
    Z = zsco.get_data().astype(np.float64)

    n_trial, n_ch, L_b = B.shape
    n_phon, _, L_p = P.shape
    print(f"=== {pt} (lexical): n_trial={n_trial}, n_phon={n_phon}, n_ch={n_ch} ===")

    # --- Recipe A: per-channel pooled-baseline mean/std (the actual recipe) ---
    X_A = B.mean(axis=(0, 2))  # (C,)
    Y_A = B.std(axis=(0, 2), ddof=1)  # (C,)
    z_A = (P - X_A[None, :, None]) / (Y_A[None, :, None] + 1e-12)

    # Reconstruction sanity: does recipe A == productionZscore?
    max_abs_diff_A = float(np.abs(z_A - Z).max())
    rmse_A = float(np.sqrt(((z_A - Z) ** 2).mean()))
    # Correlate with productionZscore
    z_A_flat = z_A.reshape(-1)
    Z_flat = Z.reshape(-1)
    corr_AA = float(np.corrcoef(z_A_flat, Z_flat)[0, 1])
    print(
        f"  recipe-A reconstruction vs productionZscore:  max|Δ|={max_abs_diff_A:.3e}, "
        f"RMSE={rmse_A:.3e}, corr={corr_AA:.6f}"
    )

    # --- Recipe B: per-channel recording-level median/MAD over production-window samples ---
    P_flat = P.transpose(1, 0, 2).reshape(n_ch, -1)
    X_B_med = np.median(P_flat, axis=1)  # (C,)
    MAD_B = np.median(np.abs(P_flat - X_B_med[:, None]), axis=1)
    Y_B_est = MAD_B * 1.4826
    z_B = (P - X_B_med[None, :, None]) / (Y_B_est[None, :, None] + 1e-12)

    # --- Recipe B': per-channel recording-level mean/std over production-window samples ---
    X_Bp = P_flat.mean(axis=1)
    Y_Bp = P_flat.std(axis=1, ddof=1)
    z_Bp = (P - X_Bp[None, :, None]) / (Y_Bp[None, :, None] + 1e-12)

    # --- Recipe C: baseline-pool median/MAD (robust variant of A) ---
    B_flat = B.transpose(1, 0, 2).reshape(n_ch, -1)
    X_C = np.median(B_flat, axis=1)
    MAD_C = np.median(np.abs(B_flat - X_C[:, None]), axis=1)
    Y_C = MAD_C * 1.4826
    z_C = (P - X_C[None, :, None]) / (Y_C[None, :, None] + 1e-12)

    # Per-channel divergence summaries (A vs each alternative)
    def summarize(name: str, X: np.ndarray, Y: np.ndarray, z_alt: np.ndarray):
        loc_diff = (X - X_A) / (Y_A + 1e-12)  # in recipe-A z-units
        scale_ratio = Y / (Y_A + 1e-12)
        # Pearson per-channel between z_A and z_alt
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
            "pearson_min": float(np.nanmin(per_ch)),
            "rms": float(np.sqrt((z_alt ** 2).mean())),
        }

    sum_B = summarize("recording_median_MAD", X_B_med, Y_B_est, z_B)
    sum_Bp = summarize("recording_mean_std", X_Bp, Y_Bp, z_Bp)
    sum_C = summarize("baseline_median_MAD", X_C, Y_C, z_C)

    # Class-separability (between-class / within-class variance ratio) in
    # [0.05, 0.30) s post-production-onset window
    tmin = float(prod.tmin)
    sfreq = float(prod.info["sfreq"])
    i0 = int(round((0.05 - tmin) * sfreq))
    i1 = int(round((0.30 - tmin) * sfreq))
    codes = prod.events[:, 2]

    def class_sep(z):
        w = z[:, :, i0:i1].mean(axis=-1)  # (T, C)
        classes = np.unique(codes)
        means = np.stack([w[codes == k].mean(axis=0) for k in classes])
        grand = w.mean(axis=0)
        between = ((means - grand) ** 2).mean(axis=0)
        within = np.stack(
            [np.var(w[codes == k], axis=0, ddof=1) for k in classes]
        ).mean(axis=0)
        return between / (within + 1e-12)

    eta_A = class_sep(z_A)
    eta_B = class_sep(z_B)
    eta_Bp = class_sep(z_Bp)
    eta_C = class_sep(z_C)

    report = {
        "patient": pt,
        "cohort": "lexical",
        "shapes": {
            "n_trial": int(n_trial),
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
            "B_recording_median_MAD": sum_B,
            "Bp_recording_mean_std": sum_Bp,
            "C_baseline_median_MAD": sum_C,
        },
        "class_separability": {
            "A_pooled_baseline_mean_std": {
                "median": float(np.median(eta_A)),
                "p95": float(np.percentile(eta_A, 95)),
            },
            "B_recording_median_MAD": {
                "median": float(np.median(eta_B)),
                "p95": float(np.percentile(eta_B, 95)),
            },
            "Bp_recording_mean_std": {
                "median": float(np.median(eta_Bp)),
                "p95": float(np.percentile(eta_Bp, 95)),
            },
            "C_baseline_median_MAD": {
                "median": float(np.median(eta_C)),
                "p95": float(np.percentile(eta_C, 95)),
            },
        },
    }

    (out_dir / f"lex_{pt}.json").write_text(json.dumps(report, indent=2))

    print()
    print(f"=== SUMMARY: {pt} (lexical, {n_ch} ch, {n_phon} phonemes) ===")
    print(
        f"  Recipe A (pooled-baseline mean/std) ≡ productionZscore:   "
        f"max|Δ|={max_abs_diff_A:.2e}, corr={corr_AA:.4f}"
    )
    for name, s in [
        ("B  rec median/MAD    ", sum_B),
        ("B' rec mean/std      ", sum_Bp),
        ("C  base median/MAD   ", sum_C),
    ]:
        print(
            f"  {name}: |locΔ|med={s['loc_diff_abs_median']:.3f}  "
            f"scale med={s['scale_ratio_median']:.3f} "
            f"[{s['scale_ratio_p5']:.2f}, {s['scale_ratio_p95']:.2f}]  "
            f"ρ_med={s['pearson_median']:.4f}, ρ_p5={s['pearson_p5']:.4f}  "
            f"rms={s['rms']:.3f}"
        )
    print()
    print(
        f"Class separability η² (per-channel between/within variance, [0.05, 0.30) s):"
    )
    for name, eta in [
        ("A  pooled-baseline mean/std", eta_A),
        ("B  recording median/MAD   ", eta_B),
        ("B' recording mean/std     ", eta_Bp),
        ("C  baseline median/MAD    ", eta_C),
    ]:
        print(
            f"  {name}: median={np.median(eta):.4f}, p95={np.percentile(eta, 95):.4f}"
        )
    print()
    print(f"wrote {out_dir / f'lex_{pt}.json'}")


if __name__ == "__main__":
    main()
