"""Z-score recipe comparison — pre-auditory per-epoch vs recording-level.

Runs end-to-end on a single PS patient:
  EDF → CAR → HG envelope (ieeg.timefreq.gamma) → downsample to 200 Hz →
  fetch trial onsets from events.tsv → compute:
    A. per-epoch pre-auditory baseline mean/std (500 ms before auditory onset)
    B. recording-level median/MAD (converted to mean/std scale)
  → apply both to the production-locked epochs → compare distributions.

The objective is to answer: is recipe B a drop-in semantic replacement for
recipe A, or a materially different transformation? We compare:
  (1) Per-channel offset & scale divergence (a_c, b_c pairs)
  (2) Per-trial variance in recipe A (if low, A ≈ B up to per-channel affine)
  (3) Pearson correlation of z_A(x) vs z_B(x) on production epochs

Outputs JSON to reports/zscore_comparison_2026_04_18/<patient>.json plus a
one-screen stdout summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import mne
from ieeg.timefreq.gamma import extract as hg_extract

mne.set_log_level("ERROR")


def load_bids_edf(bids_root: Path, patient: str) -> mne.io.BaseRaw:
    edf = (
        bids_root
        / f"sub-{patient}"
        / "ieeg"
        / f"sub-{patient}_task-phoneme_acq-01_run-01_ieeg.edf"
    )
    return mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")


def _events_tsv(bids_root: Path, patient: str) -> Path:
    """Audio-clock production events TSV (`sample` is ECoG sample index)."""

    return (
        bids_root
        / "derivatives"
        / "audio"
        / f"sub-{patient}"
        / "events"
        / f"sub-{patient}_task-phoneme_acq-01_run-01_desc-production_events.tsv"
    )


def load_response_sample_indices(bids_root: Path, patient: str) -> np.ndarray:
    """Response onsets (ECoG sample) from the production events TSV.

    Each PS token has three phoneme rows plus a `response` anchor. We use the
    `response/...` rows (one per trial) as the production-time origin.
    """

    with open(_events_tsv(bids_root, patient)) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    samples: list[int] = []
    for r in rows:
        if r.get("trial_type", "").startswith("response"):
            samples.append(int(r["sample"]))
    return np.asarray(sorted(set(samples)), dtype=np.int64)


def run_hg_pipeline(raw: mne.io.BaseRaw) -> tuple[np.ndarray, float]:
    """CAR → HG envelope → downsample to 200 Hz. Returns (hg, sfreq_final)."""

    data = raw._data.astype(np.float64)  # (n_ch, n_samples) @ 2 kHz
    fs = float(raw.info["sfreq"])
    # CAR
    data -= data.mean(axis=0, keepdims=True)
    # HG envelope (8-band filterbank sum, matches ieeg pipeline default)
    t0 = time.time()
    hg = hg_extract(data, fs=fs, passband=(70, 150), n_jobs=-1, verbose=False)
    print(f"  HG extract: {time.time()-t0:.1f}s, shape={hg.shape}, dtype={hg.dtype}")
    # Decimate 2000 → 200 Hz (10×). Use mne.filter.resample with anti-alias.
    t0 = time.time()
    hg_ds = mne.filter.resample(hg.astype(np.float64), down=10.0, npad="auto", n_jobs=-1)
    print(f"  resample 2000→200: {time.time()-t0:.1f}s, shape={hg_ds.shape}")
    return hg_ds.astype(np.float32), 200.0


def epoch_around_samples(
    hg: np.ndarray,
    sfreq: float,
    sample_indices_2khz: np.ndarray,
    tmin: float,
    tmax: float,
    src_sfreq: float = 2000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut fixed windows around each sample index.

    `sample_indices_2khz` are in the 2 kHz source clock (events.tsv). We rescale
    to the post-resample 200 Hz grid with integer division.

    Returns (epochs, kept_mask) where epochs is (n_trials, n_ch, T) and
    kept_mask is the boolean mask over `sample_indices_2khz` of trials whose
    window fit inside the recording.
    """

    n_ch, n_t = hg.shape
    pre = int(round(abs(tmin) * sfreq))
    post = int(round(tmax * sfreq))
    T = pre + post
    idx_ds = np.round(sample_indices_2khz * (sfreq / src_sfreq)).astype(np.int64)
    starts = idx_ds - pre
    ends = idx_ds + post
    mask = (starts >= 0) & (ends <= n_t)
    keep_starts = starts[mask]
    out = np.empty((keep_starts.size, n_ch, T), dtype=np.float32)
    for i, s in enumerate(keep_starts):
        out[i] = hg[:, s : s + T]
    return out, mask


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
    args = parser.parse_args()

    bids = Path(args.bids_root)
    pt = args.patient
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== z-score recipe comparison: {pt} ===")
    t0 = time.time()
    raw = load_bids_edf(bids, pt)
    print(f"  EDF loaded: {time.time()-t0:.1f}s, {raw._data.shape}, sfreq={raw.info['sfreq']}")

    hg, sfreq = run_hg_pipeline(raw)

    # Use auditory-onset = response_onset - 1.1 s (PS canonical stim→response
    # delay, CLAUDE.md). Pre-auditory baseline window = [aud_onset - 0.5,
    # aud_onset - 0.005) s.
    resp_samples_2khz = load_response_sample_indices(bids, pt)
    print(f"  {resp_samples_2khz.size} response onsets from events.tsv")
    # Shift to auditory-onset ≈ 1.1 s before response
    aud_samples_2khz = resp_samples_2khz - int(round(1.1 * 2000))
    # Baseline window around auditory onset: [-0.5, -0.005)
    base_epochs, base_mask = epoch_around_samples(
        hg, sfreq, aud_samples_2khz, tmin=-0.5, tmax=-0.005
    )
    # Production-locked epochs: [-0.5, 1.0)  (matches trial-level .fif grid)
    prod_epochs, prod_mask = epoch_around_samples(
        hg, sfreq, resp_samples_2khz, tmin=-0.5, tmax=1.0
    )
    # Align: keep trials where BOTH windows fit
    both = base_mask & prod_mask
    print(
        f"  base_mask {base_mask.sum()}, prod_mask {prod_mask.sum()}, both {both.sum()}"
    )
    # Re-epoch using the intersected mask so ordering is consistent
    base_epochs, _ = epoch_around_samples(
        hg, sfreq, aud_samples_2khz[both], tmin=-0.5, tmax=-0.005
    )
    prod_epochs, _ = epoch_around_samples(
        hg, sfreq, resp_samples_2khz[both], tmin=-0.5, tmax=1.0
    )
    n_trials, n_ch, _ = prod_epochs.shape
    print(f"  aligned epochs: n_trials={n_trials}, n_ch={n_ch}")

    # Recipe A: per-epoch per-channel pre-auditory mean/std (ddof=1)
    base_mean = base_epochs.mean(axis=-1)  # (T, C)
    base_std = base_epochs.std(axis=-1, ddof=1) + 1e-10  # (T, C)
    z_A = (prod_epochs - base_mean[:, :, None]) / base_std[:, :, None]  # (T, C, L)

    # Recipe B: recording-level median/MAD → mean/std via consistent-est factor
    rec_median = np.median(hg, axis=1, keepdims=False)  # (C,)
    rec_mad = np.median(np.abs(hg - rec_median[:, None]), axis=1)  # (C,)
    rec_std_est = rec_mad * 1.4826  # robust std estimate
    z_B = (prod_epochs - rec_median[None, :, None]) / (
        rec_std_est[None, :, None] + 1e-10
    )

    # Per-channel affine relation: z_A = (z_B + beta_c_t) * alpha_c_t is NOT
    # quite right. More useful: compare the implied (loc, scale) of the two
    # recipes per channel.
    # Recipe A loc_c,t = base_mean[t,c]; scale_c,t = base_std[t,c]
    # Recipe B loc_c = rec_median[c];   scale_c = rec_std_est[c]
    # "How much does A vary around B within a channel?"
    base_mean_per_ch_mean = base_mean.mean(axis=0)  # (C,)
    base_std_per_ch_mean = base_std.mean(axis=0)  # (C,)

    loc_rel_diff = (base_mean_per_ch_mean - rec_median) / (rec_std_est + 1e-10)
    scale_ratio = base_std_per_ch_mean / (rec_std_est + 1e-10)

    # Per-channel trial-level variability of the A recipe
    base_mean_across_trials_sd = base_mean.std(axis=0, ddof=1) / (
        rec_std_est + 1e-10
    )  # in z-B units
    base_std_across_trials_cv = base_std.std(axis=0, ddof=1) / (
        base_std_per_ch_mean + 1e-10
    )  # coefficient of variation

    # Per-sample correlation on production epochs
    # Flatten to (C, T*L) per channel
    pearson_per_ch = np.empty(n_ch, dtype=np.float64)
    for c in range(n_ch):
        a = z_A[:, c, :].ravel()
        b = z_B[:, c, :].ravel()
        a_dm = a - a.mean()
        b_dm = b - b.mean()
        denom = np.sqrt((a_dm * a_dm).sum() * (b_dm * b_dm).sum())
        pearson_per_ch[c] = (a_dm * b_dm).sum() / denom if denom > 0 else np.nan

    report = {
        "patient": pt,
        "n_trials": int(n_trials),
        "n_channels": int(n_ch),
        "sfreq": float(sfreq),
        "recording_duration_sec": float(raw.times[-1]),
        "stats": {
            "loc_rel_diff": {
                "abs_mean": float(np.abs(loc_rel_diff).mean()),
                "abs_median": float(np.median(np.abs(loc_rel_diff))),
                "abs_p95": float(np.percentile(np.abs(loc_rel_diff), 95)),
                "abs_max": float(np.abs(loc_rel_diff).max()),
            },
            "scale_ratio": {
                "mean": float(scale_ratio.mean()),
                "median": float(np.median(scale_ratio)),
                "p5": float(np.percentile(scale_ratio, 5)),
                "p95": float(np.percentile(scale_ratio, 95)),
                "min": float(scale_ratio.min()),
                "max": float(scale_ratio.max()),
            },
            "base_mean_trial_sd_per_ch": {
                "mean": float(base_mean_across_trials_sd.mean()),
                "median": float(np.median(base_mean_across_trials_sd)),
                "p95": float(np.percentile(base_mean_across_trials_sd, 95)),
            },
            "base_std_trial_cv_per_ch": {
                "mean": float(base_std_across_trials_cv.mean()),
                "median": float(np.median(base_std_across_trials_cv)),
                "p95": float(np.percentile(base_std_across_trials_cv, 95)),
            },
            "pearson_zA_zB": {
                "mean": float(np.nanmean(pearson_per_ch)),
                "median": float(np.nanmedian(pearson_per_ch)),
                "p5": float(np.nanpercentile(pearson_per_ch, 5)),
                "min": float(np.nanmin(pearson_per_ch)),
            },
        },
    }

    # Also compare activation-contrast: in z_A the pre-auditory window mean
    # should be ≈ 0 by construction; in z_B it carries the activation relative
    # to recording median.
    aud_window_mean_A = base_epochs.mean(axis=-1)  # (T, C)
    base_centered_B = (base_epochs - rec_median[None, :, None]) / (
        rec_std_est[None, :, None] + 1e-10
    )
    aud_window_mean_B = base_centered_B.mean(axis=-1)  # (T, C)
    report["stats"]["base_window_mean"] = {
        "A_mean_abs": float(np.abs(aud_window_mean_A).mean()),
        "B_mean_abs": float(np.abs(aud_window_mean_B).mean()),
        "B_across_ch_range_mean_abs": float(np.abs(aud_window_mean_B.mean(0)).mean()),
    }

    (out_dir / f"{pt}.json").write_text(json.dumps(report, indent=2))
    print()
    print("=== SUMMARY ===")
    print(f"patient {pt}: {n_trials} trials × {n_ch} ch × {prod_epochs.shape[-1]} samples at {sfreq:.0f} Hz")
    print()
    print("Per-channel *recipe-A vs recipe-B* divergence:")
    print(
        f"  |loc_A - loc_B|/scale_B: median={report['stats']['loc_rel_diff']['abs_median']:.3f}, "
        f"p95={report['stats']['loc_rel_diff']['abs_p95']:.3f}"
    )
    print(
        f"  scale_A / scale_B:       median={report['stats']['scale_ratio']['median']:.3f}, "
        f"range=[{report['stats']['scale_ratio']['p5']:.3f}, {report['stats']['scale_ratio']['p95']:.3f}]"
    )
    print(
        f"  pearson(z_A, z_B):       median={report['stats']['pearson_zA_zB']['median']:.4f}, "
        f"p5={report['stats']['pearson_zA_zB']['p5']:.4f}"
    )
    print()
    print("Within-channel trial-to-trial variability of recipe A:")
    print(
        f"  SD(base_mean) / scale_B: median={report['stats']['base_mean_trial_sd_per_ch']['median']:.3f}"
    )
    print(
        f"  CV(base_std):            median={report['stats']['base_std_trial_cv_per_ch']['median']:.3f}"
    )
    print()
    print(f"wrote {out_dir / f'{pt}.json'}")


if __name__ == "__main__":
    main()
