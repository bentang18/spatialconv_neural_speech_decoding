"""T7 — predictive-autocorrelation (τ) probe for the r5 Chang 2-stream 64 Hz cache.

Masked prediction is non-trivial iff the mask span exceeds the signal's predictive-
autocorr length τ. Under early fusion ONE mask covers both streams, so the block must be
sized to the SLOWER/SMOOTHER stream (LFS) in BOTH dims. This probe MEASURES τ from the
actual baked features — it does not set the mask; we pick the τ→width rule after seeing
the curves (Ben: "measure first, then decide T8").

For each band ∈ {v3hga (F=4, 64-160 Hz |STFT| mag), v3lfs (F=1, 1-30 Hz signed voltage)}:
  • TEMPORAL τ: normalized autocorrelation along time (per (elec,bin), FFT-based, averaged),
    reported in ms and in 32 Hz TOKENS (2 frames/token, stem stride-2). Token = 31.25 ms.
  • SPATIAL τ: within-shaft correlation vs survivor-index lag Δ (the axis the contact mask
    blocks), reported in contacts. LFS is volume-conducted ⇒ expect longer spatial τ.

τ reported at three thresholds so we choose the rule on real numbers: autocorr crossing
1/e (≈0.368), half-max (0.5), and first zero-crossing.

Invariants named + asserted + printed (feedback-build-the-invariant-into-the-probe):
  • features are post-robust-z (median/σ from the sidecar), finite;
  • temporal autocorr[0] == 1 exactly (normalization check);
  • lag axis monotone; τ(1/e) ≤ τ(half-max is larger threshold? no) — we just print, no false asserts.

Read-only. Runs where the cache lives (DCC). Saves curves to --out npz for plotting.
"""

from __future__ import annotations

import argparse

import numpy as np

from speech_decoding.extractors.reference import parse_shaft
from speech_decoding.models.v14_converged_v3.cache_index import (
    index_band_cache,
    parse_key_session,
)

FS_HZ = 64.0  # baked frame rate (hop=32)
FRAMES_PER_TOKEN = 2  # stem stride-2 (64→32 Hz tokens)
MS_PER_TOKEN = 1000.0 * FRAMES_PER_TOKEN / FS_HZ  # 31.25 ms

# 13 pretrain sessions (subjects {1,2,3,4,6,8,9}); hard-coded to avoid importing the
# study/manifest chain (keeps the probe dependency-light). Matches V14_PRETRAIN_SESSIONS.
PRETRAIN_SESSIONS = [
    (1, 0), (2, 1), (2, 2), (2, 3), (2, 5), (2, 6), (3, 2),
    (4, 2), (6, 0), (6, 1), (6, 4), (8, 0), (9, 0),
]


def _entry(index, sub, trial):
    hits = [e for k, e in index.items() if parse_key_session(k) == (sub, trial)]
    if len(hits) != 1:
        raise ValueError(f"session {sub}/{trial}: found {len(hits)} entries (want 1)")
    return hits[0]


def _load_zscored(entry, max_frames: int):
    """Post-robust-z (C, F, T) as the model sees it, a centered time window of ≤max_frames."""
    z = np.load(entry.stats_path)
    med = z["median"].astype(np.float32)   # (C, F, 1)
    sig = z["sigma"].astype(np.float32)    # (C, F, 1)
    mm = np.load(entry.npy_path, mmap_mode="r")  # (C, F, T)
    C, F, T = mm.shape
    n = min(max_frames, T)
    lo = (T - n) // 2  # centered window (skip edges)
    clip = np.asarray(mm[:, :, lo:lo + n], dtype=np.float32)  # (C, F, n)
    feat = (clip - med) / np.maximum(sig, 1e-6)
    assert np.isfinite(feat).all(), f"{entry.npy_path}: non-finite after robust-z"
    return feat  # (C, F, n)


def _temporal_autocorr(feat: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean normalized autocorr vs lag over all (elec,bin) series. FFT-based, unbiased-ish
    (divide by overlap count). Returns (max_lag+1,), ac[0]==1."""
    C, F, n = feat.shape
    x = feat.reshape(C * F, n)
    x = x - x.mean(1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    fx = np.fft.rfft(x, nfft, axis=1)
    ac = np.fft.irfft(fx * np.conj(fx), nfft, axis=1)[:, : max_lag + 1]  # (M, L+1)
    counts = (n - np.arange(max_lag + 1)).astype(np.float32)  # overlap per lag
    ac = ac / counts[None, :]
    var = ac[:, :1]
    good = (var[:, 0] > 0)
    acn = ac[good] / var[good]  # normalize each series so ac[0]=1
    return acn.mean(0)  # (L+1,)


def _spatial_autocorr(feat: np.ndarray, ch_names, max_lag: int) -> np.ndarray:
    """Within-shaft mean Pearson corr vs survivor-index lag Δ (the contact-mask axis).
    For HGA (F>1) correlate the per-bin mean series. Returns (max_lag+1,), s[0]==1."""
    C, F, n = feat.shape
    series = feat.mean(1)  # (C, n) — per-contact time course (bin-averaged for HGA)
    series = series - series.mean(1, keepdims=True)
    norm = np.sqrt((series ** 2).sum(1))  # (C,)
    # group survivor indices by shaft, in cache (survivor) order
    shafts: dict[str, list[int]] = {}
    for i, name in enumerate(ch_names):
        pref, _ = parse_shaft(name)
        shafts.setdefault(pref, []).append(i)
    sums = np.zeros(max_lag + 1, dtype=np.float64)
    cnts = np.zeros(max_lag + 1, dtype=np.float64)
    for idxs in shafts.values():
        m = len(idxs)
        for a in range(m):
            ia = idxs[a]
            if norm[ia] == 0:
                continue
            for b in range(a, m):
                d = b - a
                if d > max_lag:
                    break
                ib = idxs[b]
                if norm[ib] == 0:
                    continue
                r = float((series[ia] * series[ib]).sum() / (norm[ia] * norm[ib]))
                sums[d] += r
                cnts[d] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return out  # (L+1,) ; out[0]==1 by construction


def _tau(curve: np.ndarray) -> dict:
    """τ at 1/e, half-max, first zero-crossing, by linear interp on the lag axis."""
    def cross(level):
        for i in range(1, len(curve)):
            a, b = curve[i - 1], curve[i]
            if np.isnan(a) or np.isnan(b):
                continue
            if (a - level) >= 0 and (b - level) < 0:
                return (i - 1) + (a - level) / (a - b)  # fractional lag
        return float("nan")
    return {
        "1/e": cross(1.0 / np.e),
        "half_max": cross(0.5),
        "first_zero": cross(0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hga-dir", required=True, help="band_v3hga leaf-containing dir")
    ap.add_argument("--lfs-dir", required=True, help="band_v3lfs leaf-containing dir")
    ap.add_argument("--max-frames", type=int, default=19200, help="centered window (300s @64Hz)")
    ap.add_argument("--max-lag-time", type=int, default=192, help="temporal lag (frames; 3s)")
    ap.add_argument("--max-lag-space", type=int, default=12, help="spatial lag (contacts)")
    ap.add_argument("--out", default="tau_autocorr_r5.npz")
    args = ap.parse_args()

    bands = {"v3hga": args.hga_dir, "v3lfs": args.lfs_dir}
    indexes = {b: index_band_cache(d) for b, d in bands.items()}
    for b, idx in indexes.items():
        print(f"[check] {b}: indexed {len(idx)} sessions in {bands[b]}")

    curves: dict[str, list] = {}
    for b in bands:
        temporal, spatial = [], []
        for sub, trial in PRETRAIN_SESSIONS:
            e = _entry(indexes[b], sub, trial)
            feat = _load_zscored(e, args.max_frames)
            tc = _temporal_autocorr(feat, args.max_lag_time)
            sc = _spatial_autocorr(feat, e.ch_names, args.max_lag_space)
            assert abs(tc[0] - 1.0) < 1e-4, f"{b} {sub}/{trial}: temporal ac[0]={tc[0]}"
            temporal.append(tc)
            spatial.append(sc)
            print(f"  {b} {sub}/{trial}: C={feat.shape[0]} F={feat.shape[1]} "
                  f"n={feat.shape[2]}")
        curves[f"{b}_temporal"] = np.stack(temporal)  # (S, Lt+1)
        curves[f"{b}_spatial"] = np.stack(spatial)     # (S, Ls+1)

    np.savez(args.out, **curves,
             ms_per_token=MS_PER_TOKEN, fs_hz=FS_HZ,
             frames_per_token=FRAMES_PER_TOKEN)

    print("\n=================== τ SUMMARY (session-mean autocorr) ===================")
    print(f"token = {FRAMES_PER_TOKEN} frames = {MS_PER_TOKEN:.2f} ms @ {FS_HZ:.0f} Hz\n")
    for b in bands:
        tc = np.nanmean(curves[f"{b}_temporal"], 0)
        sc = np.nanmean(curves[f"{b}_spatial"], 0)
        tt, st = _tau(tc), _tau(sc)
        print(f"── {b} ──")
        print("  TEMPORAL τ (frames → ms → tokens):")
        for k, v in tt.items():
            ms = v * 1000.0 / FS_HZ
            tok = v / FRAMES_PER_TOKEN
            print(f"    {k:11s}: {v:6.2f} fr  {ms:7.1f} ms  {tok:5.2f} tok")
        print("  SPATIAL τ (contacts, survivor-index lag):")
        for k, v in st.items():
            print(f"    {k:11s}: {v:6.2f} contacts")
        # print the first few lags for eyeballing the decay shape
        print("  temporal ac[0:8]:", np.array2string(tc[:8], precision=3))
        print("  spatial  ac[0:6]:", np.array2string(sc[:6], precision=3))
        print()
    print(f"[saved] curves → {args.out}")


if __name__ == "__main__":
    main()
