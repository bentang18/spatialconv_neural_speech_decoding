"""First-principles physics of the per-band robust-z tail: what VOLTAGE produces a
high-|z| cell, and is the tail a clean signal continuum or a separable artifact knee?

Motivation (#264 WINSOR): the WINSOR cap W clamps per-cell |robust-z|. We labelled
P99.99 "artifact onset" heuristically. This measures the truth on real BT voltage:

1. CLEAN-SIGNAL BASELINE. A stationary Gaussian band-limited signal has Rayleigh
   |STFT|, so robust-z percentiles are FIXED: P99.9 -> z=3.82, P99.99 -> z=4.68.
   Empirical z far above that == non-stationary bursts + epileptiform + artifact.

2. PHYSICAL µV. Band-pass the CAR'd voltage to each band (LFS 2-56, HGA 64-160 Hz),
   take the Hilbert envelope = instantaneous band amplitude in µV. Report the median
   (typical band amplitude) and the envelope at the frames carrying the high-|z|
   |STFT| cells -> "a z=30 cell is ~X µV", plus the BROADBAND µV at those frames
   (a band-local burst vs a broadband step distinguishes physiology from a pop).

3. KNEE. Fine |z| tail percentiles (P99 ... P99.999, max) + a log-survival curve.
   A straight log-survival = one heavy-tailed population (no clean cut); a break =
   a separable artifact population, and THAT break is the principled W.

Reuses precompute_bad_windows for the exact production load+filter+CAR+STFT+robust-z
(verified byte-faithful to the cache). DCC-only (needs BT voltage). One session.

    ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank \
      .venv/bin/python scripts/neuroprobe/measure_band_amplitude_physics.py \
        --session 1 1 --frontend 2band
"""

from __future__ import annotations

import argparse
import gc
import json

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, stft

# precompute_bad_windows holds the exact production load+filter+CAR + band specs +
# robust_z wrapper. scripts/ is not a package -> import by path.
import importlib.util
import pathlib

_PBW_PATH = pathlib.Path(__file__).resolve().parent / "precompute_bad_windows.py"
_spec = importlib.util.spec_from_file_location("_pbw", _PBW_PATH)
assert _spec and _spec.loader
pbw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pbw)

# Rayleigh (stationary-Gaussian |STFT|) robust-z constants, derived analytically:
#   median = 1.1774 sigma, 1.4826*MAD = 0.665 sigma, R_p/sigma = sqrt(-2 ln(1-p))
#   z_p = (R_p/sigma - 1.1774) / (0.665/1.1774 * 1.1774)  == (R_p/sigma - 1.1774)/0.665
_RAY_MED_OVER_SIGMA = 1.1774
_RAY_SCALE_OVER_SIGMA = 0.665


def _rayleigh_z(p: float) -> float:
    r_over_sigma = float(np.sqrt(-2.0 * np.log1p(-p)))
    return (r_over_sigma - _RAY_MED_OVER_SIGMA) / _RAY_SCALE_OVER_SIGMA


def _bandpass_env(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """Instantaneous band amplitude (µV) = |Hilbert(bandpass(x))|, per electrode.
    ``x`` is (C, N) CAR'd voltage; returns (C, N) envelope."""
    nyq = 0.5 * fs
    hi = min(hi, nyq * 0.99)
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    xb = sosfiltfilt(sos, x, axis=-1)
    return np.abs(hilbert(xb, axis=-1)).astype(np.float32)


def measure(subject_id: int, trial_id: int, frontend: str, max_minutes: float = 0.0) -> dict:
    band_specs = pbw._make_band_specs(pbw._FRONTEND_BANDS[frontend])

    from neuroprobe.braintreebank_subject import BrainTreebankSubject
    import mne
    mne.set_log_level("ERROR")

    bt = BrainTreebankSubject(subject_id=subject_id, cache=False, coordinates_type="cortical")
    data, ch_names, sfreq = pbw.bt_load_raw(bt, trial_id=trial_id, subject_id=subject_id)
    ch_names = list(ch_names)
    sfreq = float(sfreq)
    data = np.asarray(data, dtype=np.float32)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg", verbose=False)
    raw = mne.io.RawArray(data.astype(np.float64), info, verbose=False)
    raw.load_data()
    raw.notch_filter(pbw.NOTCH_HZ, phase="zero", filter_length="auto", verbose=False)
    raw.filter(pbw.HPF_HZ, None, verbose=False)
    filt = np.asarray(raw.get_data(), dtype=np.float32)  # (C, N) pre-CAR
    del raw, data
    gc.collect()

    shafts = [pbw.parse_shaft(c)[0] for c in ch_names]
    for sh in set(shafts):
        rows = np.array([i for i, s in enumerate(shafts) if s == sh])
        if rows.size:
            filt[rows] -= filt[rows].mean(axis=0, keepdims=True)

    if max_minutes and filt.shape[1] > int(max_minutes * 60 * sfreq):
        win = int(max_minutes * 60 * sfreq)
        start = (filt.shape[1] - win) // 2  # central window — skip lead-in/run-out
        filt = np.ascontiguousarray(filt[:, start : start + win])
        gc.collect()

    n_elec, n_samples = filt.shape
    broadband_mad = float(np.median(np.abs(filt - np.median(filt))) * 1.4826)
    broadband_env_med = float(np.median(np.abs(filt)))

    bands_out: dict[str, dict] = {}
    band_dicts = {n: pbw._BAND_DICTS[n] for n in pbw._FRONTEND_BANDS[frontend]}
    for name, nperseg, hop, k0, k1 in band_specs:
        bd = band_dicts[name]
        lo, hi = float(bd["band_f_lo_hz"]), float(bd["band_f_hi_hz"])
        # --- physical band amplitude (µV) via Hilbert envelope ---
        env = _bandpass_env(filt, sfreq, lo, hi)  # (C, N)
        env_flat = env.reshape(-1)
        env_pct = {p: float(np.percentile(env_flat, p)) for p in (50, 99, 99.9, 99.99, 99.999)}
        env_max = float(env_flat.max())
        del env, env_flat
        gc.collect()

        # --- production |STFT| robust-z per (elec, freq-bin) ---
        z_abs_all = []
        for i in range(n_elec):
            _, _, z_stft = stft(
                filt[i], fs=sfreq, nperseg=nperseg, noverlap=nperseg - hop,
                boundary=None, padded=False,  # type: ignore[arg-type]
            )
            mag = np.abs(z_stft[k0 : k1 + 1]).astype(np.float32)
            z = pbw._robust_z_perbin(mag)  # (F_band, T)
            z_abs_all.append(np.abs(z).reshape(-1))
            del z_stft, mag, z
        za = np.concatenate(z_abs_all)
        del z_abs_all
        gc.collect()

        z_pct = {p: float(np.percentile(za, p)) for p in (99, 99.9, 99.99, 99.999, 99.9999)}
        z_max = float(za.max())
        # log-survival curve for knee detection: P(|z|>t) at a grid of t
        grid = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
        n = za.size
        survival = {t: float((za > t).sum()) / n for t in grid}
        bands_out[name] = {
            "f_lo_hz": lo, "f_hi_hz": hi, "nperseg": nperseg, "n_freq_bins": k1 - k0 + 1,
            "env_uV": env_pct, "env_max_uV": env_max,
            "z_pct": z_pct, "z_max": z_max,
            "z_survival": survival,
            "n_cells": int(n),
        }
        del za
        gc.collect()

    return {
        "session": f"btbank{subject_id}_t{trial_id}",
        "subject_id": subject_id, "trial_id": trial_id, "frontend": frontend,
        "n_elec": n_elec, "duration_s": n_samples / sfreq, "fs": sfreq,
        "broadband_robust_sigma_uV": broadband_mad,
        "broadband_median_abs_uV": broadband_env_med,
        "rayleigh_baseline_z": {
            "P99.9": _rayleigh_z(0.999), "P99.99": _rayleigh_z(0.9999),
            "P99.999": _rayleigh_z(0.99999),
        },
        "bands": bands_out,
    }


def _print_report(r: dict) -> None:
    print(f"\n===== {r['session']}  ({r['n_elec']} elec, {r['duration_s']/60:.1f} min, fs={r['fs']:.0f}) =====")
    print(f"broadband robust-sigma = {r['broadband_robust_sigma_uV']:.2f} µV   "
          f"median|V| = {r['broadband_median_abs_uV']:.2f} µV")
    rb = r["rayleigh_baseline_z"]
    print(f"CLEAN-SIGNAL (Rayleigh) baseline z:  P99.9={rb['P99.9']:.2f}  "
          f"P99.99={rb['P99.99']:.2f}  P99.999={rb['P99.999']:.2f}")
    for name, b in r["bands"].items():
        e, z = b["env_uV"], b["z_pct"]
        print(f"\n-- {name}  ({b['f_lo_hz']:.0f}-{b['f_hi_hz']:.0f} Hz, {b['n_freq_bins']} bins) --")
        print(f"   band envelope µV:  median={e['50']:.2f}  P99={e['99']:.2f}  "
              f"P99.9={e['99.9']:.2f}  P99.99={e['99.99']:.2f}  P99.999={e['99.999']:.2f}  "
              f"max={b['env_max_uV']:.1f}")
        print(f"   envelope ratio (×median): P99.9={e['99.9']/e['50']:.1f}  "
              f"P99.99={e['99.99']/e['50']:.1f}  P99.999={e['99.999']/e['50']:.1f}  "
              f"max={b['env_max_uV']/e['50']:.0f}")
        print(f"   |STFT| robust-|z|: P99={z['99']:.2f}  P99.9={z['99.9']:.2f}  "
              f"P99.99={z['99.99']:.2f}  P99.999={z['99.999']:.2f}  "
              f"P99.9999={z['99.9999']:.2f}  max={b['z_max']:.0f}")
        print("   log-survival  P(|z|>t):")
        s = b["z_survival"]
        line = "     " + "  ".join(f"{t}:{s[t]:.1e}" for t in [4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 200])
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", nargs=2, type=int, metavar=("SUBJECT", "TRIAL"), required=True)
    ap.add_argument("--frontend", choices=("3stft", "2band"), default="2band")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="If >0, measure only the central N-min window (fast, well-conditioned FFT).")
    ap.add_argument("--out", default=None, help="optional JSON dump path")
    args = ap.parse_args()
    r = measure(args.session[0], args.session[1], args.frontend, max_minutes=args.max_minutes)
    _print_report(r)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(r, f, indent=2)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
