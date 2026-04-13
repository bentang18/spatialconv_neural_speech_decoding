#!/usr/bin/env python3
"""Analyze real uECOG data statistics from multiple source patients.

Computes temporal, spatial, amplitude, trial envelope, and cross-trial
statistics to inform a matched-statistics synthetic generator.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import signal, stats

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speech_decoding.data.bids_dataset import load_patient_data

# ── Config ──────────────────────────────────────────────────────────────────
PATIENTS = ["S16", "S22", "S23", "S32", "S33", "S39", "S57", "S58", "S62"]
PATIENTS_8x16 = ["S16", "S22", "S23"]
FS = 200  # Hz
TMIN = -0.5
TMAX = 1.0

with open(Path(__file__).resolve().parent.parent / "configs" / "paths.yaml") as f:
    paths = yaml.safe_load(f)
BIDS_ROOT = paths["ps_bids_root"]
RESULTS_DIR = Path(paths["results_dir"])
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_patients():
    """Load data for all patients, return dict of pid -> (grid_data, grid_shape)."""
    patient_data = {}
    for pid in PATIENTS:
        print(f"Loading {pid}...", end=" ", flush=True)
        try:
            ds = load_patient_data(pid, BIDS_ROOT, task="PhonemeSequence",
                                   n_phons=3, tmin=TMIN, tmax=TMAX)
            gd = ds.grid_data  # (n_trials, H, W, T)
            print(f"OK  shape={gd.shape}  grid={ds.grid_shape}")
            patient_data[pid] = (gd, ds.grid_shape)
        except Exception as e:
            print(f"FAILED: {e}")
    return patient_data


def temporal_statistics(patient_data):
    """Compute temporal autocorrelation and PSD statistics."""
    print("\n" + "=" * 70)
    print("TEMPORAL STATISTICS (per electrode, averaged)")
    print("=" * 70)

    lags = [1, 2, 5, 10, 20]
    psd_freqs_target = [1, 5, 10, 20, 50, 100]

    all_autocorrs = {lag: [] for lag in lags}
    all_psd_at_freqs = {f: [] for f in psd_freqs_target}
    all_trial_stds = []

    for pid, (gd, gs) in patient_data.items():
        n_trials, H, W, T = gd.shape
        # Flatten spatial dims: (n_trials, H*W, T)
        flat = gd.reshape(n_trials, H * W, T)

        # Temporal autocorrelation per electrode, averaged across trials
        for lag in lags:
            # For each trial and electrode, compute autocorrelation at this lag
            x = flat[:, :, :T - lag]  # (n_trials, n_elec, T-lag)
            y = flat[:, :, lag:]       # (n_trials, n_elec, T-lag)
            # Normalize per-electrode per-trial
            x_mean = x.mean(axis=-1, keepdims=True)
            y_mean = y.mean(axis=-1, keepdims=True)
            x_std = x.std(axis=-1, keepdims=True) + 1e-10
            y_std = y.std(axis=-1, keepdims=True) + 1e-10
            corr = ((x - x_mean) * (y - y_mean)).mean(axis=-1) / (x_std.squeeze(-1) * y_std.squeeze(-1))
            # Mean across all electrodes and trials
            all_autocorrs[lag].append(float(np.nanmean(corr)))

        # PSD per electrode (average across trials first for cleaner estimate)
        trial_mean = flat.mean(axis=0)  # (n_elec, T)
        for elec_idx in range(trial_mean.shape[0]):
            freqs, psd = signal.welch(trial_mean[elec_idx], fs=FS, nperseg=min(128, T))
            for ft in psd_freqs_target:
                idx = np.argmin(np.abs(freqs - ft))
                all_psd_at_freqs[ft].append(float(psd[idx]))

        # Also compute PSD on individual trials for better statistics
        # (but just report pooled-mean PSD for each freq)

        # Temporal std per trial
        trial_stds = flat.std(axis=-1).mean(axis=-1)  # std over time, mean over electrodes -> per trial
        all_trial_stds.extend(trial_stds.tolist())

    # Print autocorrelation results
    print("\nAutocorrelation at lags (samples @ 200Hz):")
    autocorr_results = {}
    for lag in lags:
        vals = all_autocorrs[lag]
        mean_val = np.mean(vals)
        print(f"  Lag {lag:2d} ({lag/FS*1000:5.1f} ms): mean r = {mean_val:.4f}")
        autocorr_results[f"lag_{lag}"] = {
            "mean": float(mean_val),
            "per_patient": {pid: float(v) for pid, v in zip(patient_data.keys(), vals)},
        }

    # Print PSD results
    print("\nMean PSD (log10) at target frequencies:")
    psd_results = {}
    for ft in psd_freqs_target:
        vals = all_psd_at_freqs[ft]
        mean_val = np.mean(vals)
        log_mean = np.log10(mean_val + 1e-20)
        print(f"  {ft:3d} Hz: mean PSD = {mean_val:.6f} (log10 = {log_mean:.3f})")
        psd_results[f"{ft}Hz"] = {"mean": float(mean_val), "log10_mean": float(log_mean)}

    # Print trial std
    trial_std_mean = np.mean(all_trial_stds)
    trial_std_std = np.std(all_trial_stds)
    print(f"\nTemporal std per trial (mean over electrodes):")
    print(f"  Mean = {trial_std_mean:.4f}, Std = {trial_std_std:.4f}")

    return {
        "autocorrelation": autocorr_results,
        "psd": psd_results,
        "trial_temporal_std": {"mean": float(trial_std_mean), "std": float(trial_std_std)},
    }


def spatial_statistics(patient_data):
    """Compute spatial covariance/correlation vs distance for 8x16 patients."""
    print("\n" + "=" * 70)
    print("SPATIAL STATISTICS")
    print("=" * 70)

    # Correlation vs Manhattan distance for 8x16 patients
    print("\nSpatial correlation vs Manhattan distance (8x16 patients only):")
    max_dist = 10
    dist_corrs = {d: [] for d in range(1, max_dist + 1)}

    for pid in PATIENTS_8x16:
        if pid not in patient_data:
            print(f"  {pid}: not loaded, skipping")
            continue
        gd, gs = patient_data[pid]
        if gs != (8, 16):
            print(f"  {pid}: grid {gs} is not 8x16, skipping")
            continue

        n_trials, H, W, T = gd.shape
        # Flatten to (n_trials, n_elec, T), then compute time-averaged spatial cov
        flat = gd.reshape(n_trials, H * W, T)

        # Identify live electrodes (non-zero variance in at least some trials)
        elec_var = flat.var(axis=-1).mean(axis=0)  # (n_elec,)
        live_mask = elec_var > 1e-10

        # Mean across trials -> (n_elec, T), only live electrodes
        mean_data = flat.mean(axis=0)  # (n_elec, T)
        # Compute correlation only for live electrodes
        live_indices = np.where(live_mask)[0]
        live_data = mean_data[live_indices]  # (n_live, T)
        corr_mat_live = np.corrcoef(live_data)  # (n_live, n_live)

        # Compute Manhattan distance for each pair of live electrodes
        for ii, i in enumerate(live_indices):
            ri, ci = divmod(int(i), W)
            for jj in range(ii + 1, len(live_indices)):
                j = live_indices[jj]
                rj, cj = divmod(int(j), W)
                d = abs(ri - rj) + abs(ci - cj)
                if 1 <= d <= max_dist:
                    val = corr_mat_live[ii, jj]
                    if not np.isnan(val):
                        dist_corrs[d].append(float(val))

    corr_vs_dist = {}
    for d in range(1, max_dist + 1):
        vals = dist_corrs[d]
        if vals:
            mean_c = np.mean(vals)
            std_c = np.std(vals)
            print(f"  Distance {d:2d}: mean r = {mean_c:.4f} +/- {std_c:.4f}  (n={len(vals)})")
            corr_vs_dist[str(d)] = {"mean": float(mean_c), "std": float(std_c), "n": len(vals)}
        else:
            print(f"  Distance {d:2d}: no pairs")
            corr_vs_dist[str(d)] = {"mean": None, "std": None, "n": 0}

    # Per-electrode variance distribution (all patients)
    print("\nPer-electrode variance distribution (all patients pooled):")
    all_elec_vars = []
    per_patient_var_stats = {}
    for pid, (gd, gs) in patient_data.items():
        n_trials, H, W, T = gd.shape
        flat = gd.reshape(n_trials, H * W, T)
        # Variance over time, then mean over trials -> per electrode
        elec_var = flat.var(axis=-1).mean(axis=0)  # (n_elec,)
        all_elec_vars.extend(elec_var.tolist())
        per_patient_var_stats[pid] = {
            "mean": float(np.mean(elec_var)),
            "std": float(np.std(elec_var)),
            "min": float(np.min(elec_var)),
            "max": float(np.max(elec_var)),
        }

    all_elec_vars = np.array(all_elec_vars)
    kurt = float(stats.kurtosis(all_elec_vars, fisher=True))
    print(f"  Mean  = {np.mean(all_elec_vars):.4f}")
    print(f"  Std   = {np.std(all_elec_vars):.4f}")
    print(f"  Min   = {np.min(all_elec_vars):.4f}")
    print(f"  Max   = {np.max(all_elec_vars):.4f}")
    print(f"  Kurt  = {kurt:.4f} (excess)")

    return {
        "correlation_vs_manhattan_distance_8x16": corr_vs_dist,
        "electrode_variance": {
            "pooled": {
                "mean": float(np.mean(all_elec_vars)),
                "std": float(np.std(all_elec_vars)),
                "min": float(np.min(all_elec_vars)),
                "max": float(np.max(all_elec_vars)),
                "kurtosis": kurt,
            },
            "per_patient": per_patient_var_stats,
        },
    }


def amplitude_statistics(patient_data):
    """Compute amplitude distribution statistics."""
    print("\n" + "=" * 70)
    print("AMPLITUDE STATISTICS")
    print("=" * 70)

    all_values = []
    per_patient_stats = {}

    for pid, (gd, gs) in patient_data.items():
        vals = gd.flatten()
        per_patient_stats[pid] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "skewness": float(stats.skew(vals)),
            "kurtosis": float(stats.kurtosis(vals, fisher=True)),
        }
        all_values.append(vals)

    all_vals = np.concatenate(all_values)
    overall_mean = float(np.mean(all_vals))
    overall_std = float(np.std(all_vals))
    overall_skew = float(stats.skew(all_vals))
    overall_kurt = float(stats.kurtosis(all_vals, fisher=True))

    print(f"\nPooled amplitude distribution (z-scored values):")
    print(f"  Mean     = {overall_mean:.4f}")
    print(f"  Std      = {overall_std:.4f}")
    print(f"  Skewness = {overall_skew:.4f}")
    print(f"  Kurtosis = {overall_kurt:.4f} (excess, Gaussian=0)")

    # Fraction exceeding thresholds
    sigma_thresholds = [2, 3, 4]
    exceed = {}
    print(f"\nFraction of values exceeding thresholds:")
    for sigma in sigma_thresholds:
        frac = float(np.mean(np.abs(all_vals) > sigma * overall_std))
        # Also compute what Gaussian would predict
        gauss_expected = float(2 * (1 - stats.norm.cdf(sigma)))
        print(f"  |z| > {sigma}sigma: {frac:.6f}  (Gaussian expected: {gauss_expected:.6f}, ratio: {frac/gauss_expected:.2f}x)")
        exceed[f"{sigma}sigma"] = {
            "fraction": frac,
            "gaussian_expected": gauss_expected,
            "ratio": float(frac / gauss_expected),
        }

    print(f"\nPer-patient amplitude stats:")
    for pid, st in per_patient_stats.items():
        print(f"  {pid}: mean={st['mean']:.3f} std={st['std']:.3f} skew={st['skewness']:.3f} kurt={st['kurtosis']:.3f}")

    return {
        "pooled": {
            "mean": overall_mean,
            "std": overall_std,
            "skewness": overall_skew,
            "kurtosis": overall_kurt,
        },
        "exceedance": exceed,
        "per_patient": per_patient_stats,
    }


def trial_envelope(patient_data):
    """Compute RMS across electrodes at each time point, averaged across trials."""
    print("\n" + "=" * 70)
    print("TRIAL ENVELOPE (RMS across electrodes)")
    print("=" * 70)

    # Time axis
    n_times = None
    for gd, gs in patient_data.values():
        n_times = gd.shape[-1]
        break
    times = np.linspace(TMIN, TMAX, n_times)
    target_times = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

    per_patient_envelopes = {}
    all_envelopes = []

    for pid, (gd, gs) in patient_data.items():
        n_trials, H, W, T = gd.shape
        flat = gd.reshape(n_trials, H * W, T)
        # RMS across electrodes at each time point, per trial
        rms = np.sqrt((flat ** 2).mean(axis=1))  # (n_trials, T)
        mean_rms = rms.mean(axis=0)  # (T,)
        all_envelopes.append(mean_rms)

        # Sample at target times
        envelope_at_targets = {}
        for tt in target_times:
            idx = np.argmin(np.abs(times - tt))
            envelope_at_targets[f"{tt:.2f}s"] = float(mean_rms[idx])
        per_patient_envelopes[pid] = envelope_at_targets

    # Pooled average
    pooled_rms = np.mean(all_envelopes, axis=0)
    pooled_at_targets = {}
    print(f"\nPooled RMS envelope (mean across {len(patient_data)} patients):")
    print(f"  {'Time':>8s}  {'RMS':>8s}")
    print(f"  {'----':>8s}  {'---':>8s}")
    for tt in target_times:
        idx = np.argmin(np.abs(times - tt))
        val = float(pooled_rms[idx])
        pooled_at_targets[f"{tt:.2f}s"] = val
        marker = " <-- response onset" if tt == 0.0 else ""
        print(f"  {tt:>8.2f}s  {val:>8.4f}{marker}")

    # Also compute min/max/ratio
    print(f"\n  Min RMS = {pooled_rms.min():.4f} at t={times[pooled_rms.argmin()]:.3f}s")
    print(f"  Max RMS = {pooled_rms.max():.4f} at t={times[pooled_rms.argmax()]:.3f}s")
    print(f"  Max/Min ratio = {pooled_rms.max()/pooled_rms.min():.2f}")

    print(f"\nPer-patient RMS at t=0.0s (response onset):")
    for pid, env in per_patient_envelopes.items():
        print(f"  {pid}: RMS = {env['0.00s']:.4f}")

    return {
        "pooled": pooled_at_targets,
        "pooled_min": float(pooled_rms.min()),
        "pooled_max": float(pooled_rms.max()),
        "pooled_min_time": float(times[pooled_rms.argmin()]),
        "pooled_max_time": float(times[pooled_rms.argmax()]),
        "per_patient": per_patient_envelopes,
    }


def cross_trial_variability(patient_data):
    """Compute cross-trial variability measures."""
    print("\n" + "=" * 70)
    print("CROSS-TRIAL VARIABILITY")
    print("=" * 70)

    per_patient_results = {}

    for pid, (gd, gs) in patient_data.items():
        n_trials, H, W, T = gd.shape
        flat = gd.reshape(n_trials, H * W, T)

        # 1. Trial-mean spatial pattern: mean over time -> (n_trials, H*W)
        trial_spatial_means = flat.mean(axis=-1)  # (n_trials, n_elec)

        # Std of trial-mean spatial patterns across trials -> (n_elec,)
        spatial_pattern_std = trial_spatial_means.std(axis=0)  # per electrode
        mean_spatial_std = float(np.mean(spatial_pattern_std))

        # 2. Correlation between consecutive trials
        consec_corrs = []
        for i in range(n_trials - 1):
            r = np.corrcoef(trial_spatial_means[i], trial_spatial_means[i + 1])[0, 1]
            if not np.isnan(r):
                consec_corrs.append(float(r))

        mean_consec_corr = float(np.mean(consec_corrs)) if consec_corrs else float('nan')
        std_consec_corr = float(np.std(consec_corrs)) if consec_corrs else float('nan')

        per_patient_results[pid] = {
            "n_trials": n_trials,
            "spatial_pattern_std_mean": mean_spatial_std,
            "consecutive_trial_corr_mean": mean_consec_corr,
            "consecutive_trial_corr_std": std_consec_corr,
        }

    print(f"\n{'Patient':>8s}  {'Trials':>6s}  {'Spatial Std':>11s}  {'Consec Corr':>11s}")
    print(f"{'-------':>8s}  {'------':>6s}  {'-----------':>11s}  {'-----------':>11s}")
    for pid, res in per_patient_results.items():
        print(f"  {pid:>6s}  {res['n_trials']:>6d}  {res['spatial_pattern_std_mean']:>11.4f}  "
              f"{res['consecutive_trial_corr_mean']:>8.4f} +/- {res['consecutive_trial_corr_std']:.4f}")

    # Pooled summaries
    all_spatial_stds = [r["spatial_pattern_std_mean"] for r in per_patient_results.values()]
    all_consec_corrs = [r["consecutive_trial_corr_mean"] for r in per_patient_results.values()
                        if not np.isnan(r["consecutive_trial_corr_mean"])]

    print(f"\nPooled spatial pattern std: mean = {np.mean(all_spatial_stds):.4f}, "
          f"range = [{np.min(all_spatial_stds):.4f}, {np.max(all_spatial_stds):.4f}]")
    print(f"Pooled consecutive trial corr: mean = {np.mean(all_consec_corrs):.4f}, "
          f"range = [{np.min(all_consec_corrs):.4f}, {np.max(all_consec_corrs):.4f}]")

    return {
        "per_patient": per_patient_results,
        "pooled_spatial_std": {
            "mean": float(np.mean(all_spatial_stds)),
            "min": float(np.min(all_spatial_stds)),
            "max": float(np.max(all_spatial_stds)),
        },
        "pooled_consecutive_corr": {
            "mean": float(np.mean(all_consec_corrs)),
            "min": float(np.min(all_consec_corrs)),
            "max": float(np.max(all_consec_corrs)),
        },
    }


def patient_summary(patient_data):
    """Print summary of loaded patients."""
    print("=" * 70)
    print("PATIENT SUMMARY")
    print("=" * 70)
    summary = {}
    total_trials = 0
    for pid, (gd, gs) in patient_data.items():
        n_trials, H, W, T = gd.shape
        total_trials += n_trials
        print(f"  {pid}: grid={H}x{W}, T={T} ({T/FS:.2f}s), trials={n_trials}")
        summary[pid] = {
            "grid_shape": [H, W],
            "n_times": T,
            "n_trials": n_trials,
            "duration_s": float(T / FS),
        }
    print(f"  Total: {len(patient_data)} patients, {total_trials} trials")
    summary["_total"] = {"n_patients": len(patient_data), "n_trials": total_trials}
    return summary


def main():
    print("Real uECOG Data Statistics Analysis")
    print("=" * 70)

    patient_data = load_all_patients()
    if not patient_data:
        print("ERROR: No patients loaded!")
        sys.exit(1)

    results = {}
    results["patient_summary"] = patient_summary(patient_data)
    results["temporal"] = temporal_statistics(patient_data)
    results["spatial"] = spatial_statistics(patient_data)
    results["amplitude"] = amplitude_statistics(patient_data)
    results["trial_envelope"] = trial_envelope(patient_data)
    results["cross_trial"] = cross_trial_variability(patient_data)

    # Save to JSON
    out_path = RESULTS_DIR / "real_data_stats.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {out_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
