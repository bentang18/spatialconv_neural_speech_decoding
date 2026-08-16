"""R42 — WITHIN-CONTACT per-band TIME autocorrelation r(lag), and what a masked block leaves.

The question (Ben 2026-08-15): *"There is no real significant correlation between adjacent HGA
tokens - do the test yourself to convince yourself."* The claim under test is mine, not his: I
asserted that adjacent HGA tokens are mechanically correlated because the STFT hop is half the
window, so neighbouring frames share raw samples, and that a width-4 time block exists to defeat
that redundancy. That is an assertion about the |STFT| magnitude the model actually tokenizes, and
it has never been measured. This measures it.

This is the TIME-axis twin of R20 (``v3_shaft_band_corr.py``), which measured the SPACE axis and
found r(d) cliffs at d=1 then goes flat. Same sessions, same clips, same ``_window_bands`` path, so
the two numbers are directly comparable: R20 reports HGA r(d=1) = .2746 across contacts. If HGA
r(lag=1) across time lands well below that, the time axis carries LESS local redundancy than the
space axis, and any argument that the time block width is set by a leak margin is arguing about a
quantity too small to matter.

TWO QUANTITIES, because r(lag) alone does not answer the design question:

  r(lag)   — Pearson r between token t and token t+lag, per frequency bin, per contact, pooled over
             (clip x t). Pairs never cross a clip boundary. Exact Pearson on the overlapping sample,
             not a biased autocorrelation estimate: a and b are centred and scaled on the pairs they
             actually contribute to.

  R2(w)    — the fraction of a MASKED token's variance that is linearly recoverable from the two
             nearest VISIBLE tokens under a contiguous masked run of width w. This is the number the
             mask design is actually about. For a run at positions p..p+w-1 the interior token p+1
             sees visible neighbours at p-1 and p+w, i.e. at distances 2 and w-2+1. w=1 gives the
             (t-1, t+1) pair. Computed as the exact 2-predictor OLS R^2 from a 3x3 correlation
             matrix built on the common overlap, so it needs no regression solve and cannot silently
             use mismatched sample windows.

R2 is reported for w=1 and w=4. The design claim "width 4 kills a copy path that width 1 leaves
open" predicts a large drop from R2(1) to R2(4). If both are near zero, there is no copy path on
the time axis at either width, and the width is doing something other than defeating interpolation.

Bands are read through ``_window_bands``, so they carry the SAME robust-z the model sees.
Correlating raw |STFT| would let per-contact amplitude drift masquerade as temporal structure.

CPU only, but dtai not Delta: ``speech_decoding.models.*`` imports the legacy zoo, which needs the
aarch64 project venv. → the ops note in v3_shaft_band_corr.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from scripts.neuroprobe.v3_mae_recon import clip_starts_seconds
from scripts.neuroprobe.v3_probe_encode_r4 import _lite_keep_labels_fn, _window_bands

BAND_NAMES = ("SLOW", "MID", "HGA")  # band_cache_dir order: slow, mid, hga
BAND_HZ = {"SLOW": 4.0, "MID": 16.0, "HGA": 32.0}


def lag_corr(x: np.ndarray, lag: int) -> np.ndarray:
    """Exact Pearson r between token t and t+lag, per (contact, bin). Returns (N, F).

    ``x`` is (W, N, F, T). Pairs are formed WITHIN a clip and pooled over (clip, t); a and b are
    each centred and scaled on the overlapping sample, so this is a true correlation and not the
    biased autocorrelation estimator that reuses the full-series mean.
    """
    if lag == 0:
        a = b = x
    else:
        a, b = x[..., :-lag], x[..., lag:]
    n, f = x.shape[1], x.shape[2]
    a = np.transpose(a, (1, 2, 0, 3)).reshape(n, f, -1).astype(np.float64)
    b = np.transpose(b, (1, 2, 0, 3)).reshape(n, f, -1).astype(np.float64)
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    out = np.zeros((n, f), dtype=np.float64)
    np.divide((a * b).sum(-1), den, out=out, where=den > 0)
    return out


def block_r2(x: np.ndarray, d_left: int, d_right: int) -> np.ndarray:
    """Exact 2-predictor OLS R^2 for x[t] from x[t-d_left] and x[t+d_right]. Returns (N, F).

    Built from the 3x3 correlation matrix of (y, x1, x2) on their COMMON overlap, which is what
    makes the three r's mutually consistent. R^2 = (r1^2 + r2^2 - 2 r1 r2 r12) / (1 - r12^2).
    """
    lo, hi = d_left, d_right
    t = x.shape[-1]
    y = x[..., lo:t - hi]
    x1 = x[..., 0:t - lo - hi]
    x2 = x[..., lo + hi:t]
    n, f = x.shape[1], x.shape[2]

    def flat(v: np.ndarray) -> np.ndarray:
        v = np.transpose(v, (1, 2, 0, 3)).reshape(n, f, -1).astype(np.float64)
        v = v - v.mean(-1, keepdims=True)
        sd = np.sqrt((v * v).sum(-1, keepdims=True))
        out = np.zeros_like(v)
        np.divide(v, sd, out=out, where=sd > 0)
        return out

    zy, z1, z2 = flat(y), flat(x1), flat(x2)
    r1 = (zy * z1).sum(-1)
    r2 = (zy * z2).sum(-1)
    r12 = (z1 * z2).sum(-1)
    num = r1 ** 2 + r2 ** 2 - 2.0 * r1 * r2 * r12
    den = 1.0 - r12 ** 2
    out = np.zeros((n, f), dtype=np.float64)
    np.divide(num, den, out=out, where=den > 1e-12)
    return np.clip(out, 0.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--band-cache-dir", dest="band_cache_dirs", action="append", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", required=True)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--electrode-set", choices=("pretrain", "lite"), default="pretrain")
    p.add_argument("--n-clips", type=int, default=64)
    p.add_argument("--clip-dur", type=float, default=2.0)
    p.add_argument("--max-lag", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    clip_frames = int(round(args.clip_dur * 32.0))
    print(f"[cfg] s{args.subject}t{args.trial} clips={args.n_clips} clip_frames={clip_frames} "
          f"max_lag={args.max_lag} electrode_set={args.electrode_set}", flush=True)

    from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
    from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions

    keep_labels_fn = _lite_keep_labels_fn(args.bt_root) if args.electrode_set == "lite" else None
    spec = load_v3_sessions(
        sessions=[(args.subject, args.trial)], band_cache_dirs=args.band_cache_dirs,
        span_dir=args.span_dir, parcel_fn=make_bt_parcel_fn(args.bt_root),
        lof_report_path=None, winsor=(15.0, 15.0, 20.0), keep_labels_fn=keep_labels_fn,
    )[0]

    starts = clip_starts_seconds(int(spec.n_frames), clip_frames, args.n_clips)
    bands = _window_bands(spec, starts, clip_frames, rate_mult=1)  # 3 x (W, N, F_b, T_b)
    print("[check] band shapes (W,N,F,T): "
          + " ".join(f"{n}={tuple(b.shape)}" for n, b in zip(BAND_NAMES, bands)), flush=True)

    rows, r2_rows = [], []
    for bi, name in enumerate(BAND_NAMES):
        x = bands[bi].numpy()  # (W, N, F, T)
        _, n, f, t = x.shape
        hz = BAND_HZ[name]

        # INVARIANT: r(0) must be exactly 1 for every non-degenerate (contact, bin).
        r0 = lag_corr(x, 0)
        live = r0 > 0
        assert np.allclose(r0[live], 1.0, atol=1e-9), \
            f"{name}: r(0) != 1 (max dev {float(np.abs(r0[live] - 1).max()):.2e})"
        print(f"[check] {name}: r(0)=1.000 for {int(live.sum())}/{r0.size} (contact,bin) cells, "
              f"{r0.size - int(live.sum())} constant", flush=True)

        max_lag = min(args.max_lag, t - 2)
        for lag in range(1, max_lag + 1):
            r = lag_corr(x, lag)          # (N, F)
            per_contact = r.mean(-1)      # mean over frequency bins, as in R20
            rows.append({
                "band": name, "lag": lag, "lag_ms": 1000.0 * lag / hz,
                "r_mean": float(per_contact.mean()), "r_sd": float(per_contact.std()),
                "r_p25": float(np.percentile(per_contact, 25)),
                "r_p75": float(np.percentile(per_contact, 75)),
                "r_max_contact": float(per_contact.max()),
                "n_contacts": int(n), "n_bins": int(f),
            })
            print(f"  {name:5s} lag={lag} ({1000.0 * lag / hz:6.1f} ms)  "
                  f"r={per_contact.mean():+.4f} +- {per_contact.std():.4f}  "
                  f"[p25 {np.percentile(per_contact, 25):+.3f} "
                  f"p75 {np.percentile(per_contact, 75):+.3f}]  "
                  f"max_contact {per_contact.max():+.3f}", flush=True)

        # what a contiguous masked run of width W leaves for its INTERIOR token
        for bw in (1, 4):
            # run at p..p+bw-1; the interior token p+1 sees visible at p-1 and p+bw.
            # bw=1 degenerates to the immediate neighbours (-1, +1).
            d_left = 1 if bw == 1 else 2
            d_right = 1 if bw == 1 else bw - 1
            if d_left + d_right >= t:
                continue
            r2 = block_r2(x, d_left, d_right).mean(-1)  # (N,)
            r2_rows.append({
                "band": name, "block_w": bw, "d_left": d_left, "d_right": d_right,
                "r2_mean": float(r2.mean()), "r2_sd": float(r2.std()),
                "r2_p90": float(np.percentile(r2, 90)), "r2_max": float(r2.max()),
            })
            print(f"  {name:5s} block_w={bw} (visible at -{d_left}, +{d_right})  "
                  f"R2={r2.mean():.4f} +- {r2.std():.4f}  "
                  f"[p90 {np.percentile(r2, 90):.3f} max {r2.max():.3f}]", flush=True)

    out = {
        "subject": args.subject, "trial": args.trial,
        "electrode_set": args.electrode_set,
        "n_clips": len(starts), "clip_frames": clip_frames,
        "rows": rows, "block_r2": r2_rows,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
