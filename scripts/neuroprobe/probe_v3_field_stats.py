"""M1 / M2 / M3 in one pass over the v3 3-band spec cache — the r4 gating measurements.

Three statistics, one data load, on the EXACT caches + normalization r1/r2/r3 train on
(robust-z with the frozen per-session median/sigma, winsor SLOW/MID 15 HGA 20, guard-2
spans excluded via the model's own clip sampler). Model-FREE: no checkpoint is loaded.

  M1 EIGENSPECTRUM   Per band, build the (N_contacts x T) band-power matrix and SVD it.
                     The number of components holding 80/90/95% of variance IS the
                     effective rank of the instantaneous spatial field. Run RAW and
                     POST-SHAFT-CAR: component 1 is the common mode, and the honest
                     number is the rank of the RESIDUAL after it is removed.
                     -> sets M (Perceiver-L2 latents) and k (secondary-head seeds).
                        If M/k < this rank, the latents collapse to the common mode.

  M2 AUTOCORRELATION (a) TIME-lag ACF, per contact, within-clip (never across the clip
                         boundary).                       -> sets block_w_time
                     (b) ALONG-SHAFT depth-lag correlation. -> sets block_w_space
                     Today masking.py asserts block_w_time=7 from the HGA |STFT|
                     support and block_w_space=4 from "HGA along-shaft autocorr".
                     The time floor has never actually been measured.

  M3 VARIOGRAM       Correlation vs 3D distance, all pairs, binned — SEPARATELY per band.
                     -> tells us what SLOW/MID actually donate spatially, i.e. whether
                        demoting them to context-only (never scored) in r4 throws away
                        real structure or just the common mode.

Coordinates are used RELATIONALLY ONLY (pairwise distances, native subject space). No
MNI, no shared frame.

Run on the Delta CPU partition (Delta mounts the DeltaAI /work/nvme caches):

  ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
  ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank \
  .venv/bin/python -m scripts.neuroprobe.probe_v3_field_stats \
      --band-root $ROOT \
      --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
      --bt-root /projects/bhqk/htang13/braintreebank \
      --out /projects/bhqk/htang13/probe_out_v3/field_stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from speech_decoding.models.v14_converged_v3.clip_sampler import sample_clip_start
from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
from speech_decoding.studies.braintreebank.anatomy import (
    aligned_voltage_coords,
    aligned_voltage_support,
)

# The 13 pretrain sessions r1/r2/r3 train on (v3_launch_r2.sbatch).
V3_SESSIONS: list[tuple[int, int]] = [
    (1, 0), (2, 1), (2, 2), (2, 3), (2, 5), (2, 6), (3, 2),
    (4, 2), (6, 0), (6, 1), (6, 4), (8, 0), (9, 0),
]
BAND_DIRS = ("band_v3slow", "band_v3mid", "band_hga")  # v3 concat order
BAND_NAMES = ("slow", "mid", "hga")
WINSOR = (15.0, 15.0, 20.0)  # --session-z-winsor-slow/mid/hga, r2 argv
FPS = 32  # hop 64 @ 2048 Hz -> 31.25 ms slots

# Variogram distance bins (mm), matching the 2026-07-08 probe's reporting granularity.
DIST_EDGES = np.array([0, 5, 10, 15, 20, 30, 40, 55, 70, 1e9], dtype=float)


def _parcel_fn(bt_root: str):
    def fn(subject_id: int, trial_id: int, labels):
        sup = aligned_voltage_support(bt_root, subject_id, trial_id)  # (n_v, P) one-hot
        return torch.as_tensor(np.asarray(sup).argmax(1))
    return fn


def _read_frames(spec, n_clips: int, clip_frames: int, seed: int) -> list[np.ndarray]:
    """n_clips clips, drawn with the MODEL'S OWN sampler (so guard-2 spans are excluded
    exactly as in training). Returns one (N, F, T) fp32 array per band, per clip, already
    robust-z + winsor normalized. Shape: list[band] of (n_clips, N, F, T)."""
    keep = spec.keep_idx.numpy()
    out: list[list[np.ndarray]] = [[] for _ in spec.band_paths]
    mms = [np.load(p, mmap_mode="r") for p in spec.band_paths]
    for i in range(n_clips):
        g = torch.Generator().manual_seed(seed + i)
        t0 = sample_clip_start(
            n_frames=spec.n_frames,
            clip_frames=clip_frames,
            bad_spans_s=spec.bad_spans_s,
            fps=FPS,
            generator=g,
        )
        for b, (mm, norm) in enumerate(zip(mms, spec.band_norms)):
            clip = np.asarray(mm[keep, :, t0 : t0 + clip_frames], dtype=np.float32)
            out[b].append(norm.transform(torch.from_numpy(clip)).numpy())
    del mms
    return [np.stack(o, 0) for o in out]  # per band: (n_clips, N, F, T)


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise z-score of (N, T). Rows with ~0 variance are zeroed, not NaN'd."""
    mu = x.mean(1, keepdims=True)
    sd = x.std(1, keepdims=True)
    return np.where(sd > 1e-8, (x - mu) / np.maximum(sd, 1e-8), 0.0)


def _shaft_car(x: np.ndarray, shaft_id: np.ndarray) -> np.ndarray:
    """Subtract the per-shaft mean ACROSS CONTACTS at each time slot. (N, T) -> (N, T).

    NOTE this is shaft-CAR on BAND POWER, which is NOT the same operation as the
    existing CARIeegExtractor._apply_car(car="shaft") — that one runs on raw voltage in
    the time domain, BEFORE the STFT. Written fresh here on purpose; do not conflate."""
    out = x.copy()
    for s in np.unique(shaft_id):
        idx = np.where(shaft_id == s)[0]
        out[idx] -= out[idx].mean(0, keepdims=True)
    return out


def _eigenspectrum(x: np.ndarray) -> dict:
    """SVD of the row-z-scored (N, T) matrix -> variance-explained curve."""
    z = _zscore_rows(x)
    sv = np.linalg.svd(z, compute_uv=False)
    e = sv**2
    frac = e / max(e.sum(), 1e-12)
    cum = np.cumsum(frac)
    pr = float(e.sum() ** 2 / max((e**2).sum(), 1e-12))  # participation ratio
    return {
        "n_contacts": int(x.shape[0]),
        "n80": int(np.searchsorted(cum, 0.80) + 1),
        "n90": int(np.searchsorted(cum, 0.90) + 1),
        "n95": int(np.searchsorted(cum, 0.95) + 1),
        "participation_ratio": round(pr, 2),
        "top5_var_frac": [round(float(f), 4) for f in frac[:5]],
    }


def _time_acf(env: np.ndarray, max_lag: int) -> np.ndarray:
    """Within-clip temporal ACF, per contact, averaged over clips+contacts.

    env: (n_clips, N, T). Never crosses a clip boundary. Returns (max_lag+1,)."""
    t = env.shape[2]
    z = env - env.mean(2, keepdims=True)
    sd = z.std(2, keepdims=True)
    z = np.where(sd > 1e-8, z / np.maximum(sd, 1e-8), 0.0)
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[0] = 1.0
            continue
        a = z[:, :, : t - lag]
        b = z[:, :, lag:]
        acf[lag] = float((a * b).mean())
    return acf


def _crossing(acf: np.ndarray, thresh: float) -> float:
    """First lag (linearly interpolated) at which the ACF drops below `thresh`."""
    below = np.where(acf < thresh)[0]
    if len(below) == 0:
        return float(len(acf) - 1)
    i = int(below[0])
    if i == 0:
        return 0.0
    y0, y1 = acf[i - 1], acf[i]
    return float(i - 1 + (y0 - thresh) / max(y0 - y1, 1e-9))


def _pair_corr(sig_z: np.ndarray) -> np.ndarray:
    """(N, T) row-z-scored -> (N, N) Pearson correlation."""
    t = sig_z.shape[1]
    return (sig_z @ sig_z.T) / t


def _depth_lag(corr, shaft_id, depth, coords, max_lag):
    """Within-shaft correlation binned by |delta depth| (contacts)."""
    rows = []
    for lag in range(1, max_lag + 1):
        vals, mms = [], []
        for s in np.unique(shaft_id):
            idx = np.where(shaft_id == s)[0]
            if len(idx) < 2:
                continue
            d = np.abs(depth[idx][:, None] - depth[idx][None, :])
            ii, jj = np.where(np.triu(d == lag, 1))
            for a, b in zip(ii, jj):
                vals.append(corr[idx[a], idx[b]])
                mms.append(float(np.linalg.norm(coords[idx[a]] - coords[idx[b]])))
        if vals:
            rows.append({
                "lag": lag, "n_pairs": len(vals),
                "corr_med": round(float(np.median(vals)), 4),
                "mm_med": round(float(np.median(mms)), 2),
            })
    return rows


def _variogram(corr, coords):
    """All-pairs correlation binned by 3D Euclidean distance (mm, native space)."""
    n = coords.shape[0]
    ii, jj = np.triu_indices(n, 1)
    d = np.linalg.norm(coords[ii] - coords[jj], axis=1)
    c = corr[ii, jj]
    rows = []
    for k in range(len(DIST_EDGES) - 1):
        lo, hi = DIST_EDGES[k], DIST_EDGES[k + 1]
        m = (d >= lo) & (d < hi)
        if m.sum() >= 5:
            rows.append({
                "lo_mm": float(lo), "hi_mm": (None if hi > 1e8 else float(hi)),
                "n_pairs": int(m.sum()),
                "corr_med": round(float(np.median(c[m])), 4),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--band-root", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", default=os.environ.get("ROOT_DIR_BRAINTREEBANK", ""))
    p.add_argument("--out", required=True)
    p.add_argument("--n-clips", type=int, default=256)
    p.add_argument("--clip-frames", type=int, default=96)  # --clip-len 3.0 @ 32 Hz
    p.add_argument("--max-time-lag", type=int, default=32)  # 1.0 s
    p.add_argument("--max-depth-lag", type=int, default=10)
    p.add_argument("--seed", type=int, default=33)
    args = p.parse_args()

    if not args.bt_root:
        sys.exit("need --bt-root or ROOT_DIR_BRAINTREEBANK")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = load_v3_sessions(
        sessions=V3_SESSIONS,
        band_cache_dirs=[os.path.join(args.band_root, b) for b in BAND_DIRS],
        span_dir=args.span_dir,
        parcel_fn=_parcel_fn(args.bt_root),
        lof_report_path=None,  # r2 passes no LOF report -> no guard-1 drops on BT
        winsor=WINSOR,
    )

    results = []
    for spec in specs:
        sid, tid = spec.session_key
        sc = spec.setup.sidecar
        keep = spec.keep_idx.numpy()
        shaft_id = sc.shaft_id.numpy()
        depth = sc.depth.numpy()

        coords_full = np.asarray(aligned_voltage_coords(args.bt_root, sid, tid), dtype=float)
        coords = coords_full[keep]
        if coords.shape[0] != len(sc.labels):
            sys.exit(f"{sid}/{tid}: coords {coords.shape} vs {len(sc.labels)} survivors")

        bands = _read_frames(spec, args.n_clips, args.clip_frames, args.seed)
        print(f"\n=== subject {sid} trial {tid} — {len(sc.labels)} contacts, "
              f"{sc.n_shafts} shafts, {args.n_clips}x{args.clip_frames} frames ===",
              flush=True)

        for b, name in enumerate(BAND_NAMES):
            env = bands[b].mean(2)                       # (n_clips, N, T) freq-average
            n_c = env.shape[1]
            flat = env.transpose(1, 0, 2).reshape(n_c, -1)  # (N, n_clips*T)

            eig_raw = _eigenspectrum(flat)
            eig_car = _eigenspectrum(_shaft_car(flat, shaft_id))

            acf = _time_acf(env, args.max_time_lag)
            sig_z = _zscore_rows(flat)
            corr = _pair_corr(sig_z)

            rec = {
                "subject_id": sid, "trial_id": tid, "band": name,
                "M1_eigenspectrum": {"raw": eig_raw, "shaft_car": eig_car},
                "M2a_time_acf": {
                    "acf": [round(float(v), 4) for v in acf],
                    "lag_half_slots": round(_crossing(acf, 0.5), 2),
                    "lag_1e_slots": round(_crossing(acf, 1 / np.e), 2),
                    "lag_02_slots": round(_crossing(acf, 0.2), 2),
                    "lag_half_ms": round(_crossing(acf, 0.5) * 1000 / FPS, 1),
                },
                "M2b_depth_lag": _depth_lag(corr, shaft_id, depth, coords, args.max_depth_lag),
                "M3_variogram": _variogram(corr, coords),
            }
            results.append(rec)

            print(f"  [{name}] rank(90%) raw={eig_raw['n90']}/{n_c} "
                  f"car={eig_car['n90']}/{n_c} | PR raw={eig_raw['participation_ratio']} "
                  f"car={eig_car['participation_ratio']} | top1 var "
                  f"raw={eig_raw['top5_var_frac'][0]:.3f} car={eig_car['top5_var_frac'][0]:.3f}",
                  flush=True)
            print(f"        ACF half-life {rec['M2a_time_acf']['lag_half_slots']:.1f} slots "
                  f"({rec['M2a_time_acf']['lag_half_ms']:.0f} ms) | "
                  f"depth-lag1 {rec['M2b_depth_lag'][0]['corr_med'] if rec['M2b_depth_lag'] else float('nan')}",
                  flush=True)

    with open(out_dir / "field_stats.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out_dir / 'field_stats.json'}  ({len(results)} band-session records)")

    # ---- pooled read: the numbers that actually set M/k, block_w_time, block_w_space ----
    print("\n" + "=" * 78)
    print("POOLED (median over the 13 sessions)")
    print("=" * 78)
    for name in BAND_NAMES:
        rs = [r for r in results if r["band"] == name]
        if not rs:
            continue
        n90r = np.median([r["M1_eigenspectrum"]["raw"]["n90"] for r in rs])
        n90c = np.median([r["M1_eigenspectrum"]["shaft_car"]["n90"] for r in rs])
        top1r = np.median([r["M1_eigenspectrum"]["raw"]["top5_var_frac"][0] for r in rs])
        top1c = np.median([r["M1_eigenspectrum"]["shaft_car"]["top5_var_frac"][0] for r in rs])
        half = np.median([r["M2a_time_acf"]["lag_half_slots"] for r in rs])
        print(f"\n[{name}]")
        print(f"  M1  rank@90%: raw {n90r:.0f}  -> shaft-CAR {n90c:.0f}   "
              f"(top-1 var {top1r:.3f} -> {top1c:.3f})")
        print(f"  M2a time ACF half-life: {half:.1f} slots = {half * 1000 / FPS:.0f} ms "
              f"(block_w_time is {7} today)")
        for lag in (1, 2, 3, 4, 5):
            vs = [d["corr_med"] for r in rs for d in r["M2b_depth_lag"] if d["lag"] == lag]
            if vs:
                print(f"  M2b depth lag {lag}: corr {np.median(vs):.3f}")
        print("  M3 variogram:")
        for k in range(len(DIST_EDGES) - 1):
            lo = float(DIST_EDGES[k])
            vs = [d["corr_med"] for r in rs for d in r["M3_variogram"] if d["lo_mm"] == lo]
            if vs:
                hi = DIST_EDGES[k + 1]
                lbl = f">{lo:.0f}mm" if hi > 1e8 else f"{lo:.0f}-{hi:.0f}mm"
                print(f"    {lbl:>10}: corr {np.median(vs):.3f}")


if __name__ == "__main__":
    main()
