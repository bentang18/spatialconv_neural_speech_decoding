"""R20 — along-shaft per-band correlation r(d) on BrainTreebank. No model, no GPU, no training.

The question (Ben 2026-07-29): ``block_w_space = 4`` (masking.py:70) is HARD-CODED and, unlike
``block_w_band = 4``, is derived from nothing. The time width comes from the STFT hop-overlap leak
margin — adjacent frames share input samples, so a masked frame beside a visible one leaks. Contacts
do NOT share samples, so there is no mechanical leak on the space axis and no such derivation exists.
A reviewer asking "why 4?" currently has no answer. This measures the number that would give one.

🚫 The source CANNOT be Duraivel 2023 Fig 2d. That curve is µECoG measured TANGENTIALLY across a
surface grid — one cortical patch, same tissue, same depth from pia — and it is HGA-only. Our shafts
sample RADIALLY, crossing gray→white→gray and different gyral banks between consecutive contacts.
Transferring a grid correlation curve onto a shaft is the same error as reading a grid blob figure as
evidence about sEEG neighbours. The right source is our own data, which is what this measures.

THE DERIVATION IT FEEDS — equalize spatial DIFFICULTY across bands. Pick per-band width w_b such that
r_b(w_b) is roughly EQUAL across bands, anchored to HGA at the current width. One free parameter (the
common target r) instead of three arbitrary "uninformative" thresholds, and it is falsifiable: if
r_SLOW never falls to HGA's level within a shaft's length, the honest answer is "SLOW cannot be made
non-trivial by widening the block," which is itself a result.

Predicted shape, which INVERTS the dead R17 premise (R17 assumed HGA was too hard and wanted it
NARROWER): SLOW is volume-conducted and correlated over centimetres ⇒ at BrainTreebank's 3.61 mm pitch
the immediate neighbour is highly informative ⇒ SLOW needs a WIDE block. HGA decorrelates over
millimetres ⇒ width 1 is ALREADY non-trivial. R18 shows exactly this pattern already: SLOW-space
r .636 (still highly predictable AT width 4 ⇒ the block is too narrow to bite) vs HGA-space r .472.

TWO OFFSET AXES, both reported — they differ wherever a shaft has missing contacts, and deriving a
width from the wrong one is a silent error:

  slot  — position in the shaft's (S, C) grid. ``_cover_rank`` operates on THIS axis, so it is the
          unit ``block_w_space`` is actually expressed in. This is the one the derivation needs.
  depth — clinical contact number, gaps preserved (geometry.py:30). Physically meaningful; slot
          offset <= depth offset always, and the gap between them is the montage's missing contacts.

Bands are read through the normal ``_window_bands`` path, so they carry the SAME robust-z the model
sees. Correlating raw |STFT| instead would let per-contact amplitude scale masquerade as spatial
structure.

CPU only, on Delta. → feedback-cpu-probe-work-on-delta-not-dtai-gpu-2026-07-18
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from scripts.neuroprobe.v3_mae_recon import clip_starts_seconds
from scripts.neuroprobe.v3_probe_encode_r4 import _lite_keep_labels_fn, _window_bands

BAND_NAMES = ("SLOW", "MID", "HGA")  # band_cache_dir order: slow, mid, hga


def zscore_rows(x: np.ndarray) -> np.ndarray:
    """Z-score along the LAST axis. Rows with zero variance return all-zero (r contributes 0)."""
    mu = x.mean(-1, keepdims=True)
    sd = x.std(-1, keepdims=True)
    out = np.zeros_like(x)
    np.divide(x - mu, sd, out=out, where=sd > 0)
    return out


def pair_corr(z: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """Mean-over-bins Pearson r for each (ia[k], ib[k]) contact pair.

    ``z`` is (N, F, S) already z-scored along S, so r == mean_s(z_i * z_j) exactly. Returns (K,).
    """
    prod = z[ia] * z[ib]  # (K, F, S)
    r_per_bin = prod.mean(-1)  # (K, F)
    return r_per_bin.mean(-1)  # (K,)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--band-cache-dir", dest="band_cache_dirs", action="append", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", required=True)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--electrode-set", choices=("pretrain", "lite"), default="pretrain",
                   help="pretrain = the FULL montage the model trained on (default). The montage "
                        "sets shaft sizes, and shaft sizes ARE the spatial-mask geometry.")
    p.add_argument("--n-clips", type=int, default=64)
    p.add_argument("--clip-dur", type=float, default=2.0)
    p.add_argument("--max-offset", type=int, default=6)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    clip_frames = int(round(args.clip_dur * 32.0))
    print(f"[cfg] s{args.subject}t{args.trial} clips={args.n_clips} clip_frames={clip_frames} "
          f"max_offset={args.max_offset} electrode_set={args.electrode_set}", flush=True)

    from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
    from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions

    keep_labels_fn = _lite_keep_labels_fn(args.bt_root) if args.electrode_set == "lite" else None
    spec = load_v3_sessions(
        sessions=[(args.subject, args.trial)], band_cache_dirs=args.band_cache_dirs,
        span_dir=args.span_dir, parcel_fn=make_bt_parcel_fn(args.bt_root),
        lof_report_path=None, winsor=(15.0, 15.0, 20.0), keep_labels_fn=keep_labels_fn,
    )[0]

    geom = spec.setup.geom
    valid = geom.valid.numpy()          # (S, C)
    gather = geom.gather_idx.numpy()    # (S, C) -> contact index into N
    depth = geom.depth.numpy()          # (S, C) clinical contact number, gaps preserved
    cs = valid.sum(1)
    print(f"[check] montage: {int(valid.sum())} contacts / {geom.n_shafts} shafts, "
          f"contacts-per-shaft min={int(cs.min())} median={int(np.median(cs))} "
          f"max={int(cs.max())}", flush=True)

    starts = clip_starts_seconds(int(spec.n_frames), clip_frames, args.n_clips)
    bands = _window_bands(spec, starts, clip_frames, rate_mult=1)  # 3 x (W, N, F_b, T_b)
    print("[check] band shapes (W,N,F,T): "
          + " ".join(f"{n}={tuple(b.shape)}" for n, b in zip(BAND_NAMES, bands)), flush=True)

    # ── build the within-shaft pair index once; it is shared across bands ──
    # slot offset is the unit block_w_space lives in (_cover_rank runs on the (S,C) grid).
    pairs: dict[int, dict[str, np.ndarray]] = {}
    for d in range(1, args.max_offset + 1):
        ia, ib, dd, shaft = [], [], [], []
        for s in range(geom.n_shafts):
            slots = np.nonzero(valid[s])[0]
            for k in range(len(slots) - d):
                a, b = slots[k], slots[k + d]
                ia.append(int(gather[s, a]))
                ib.append(int(gather[s, b]))
                dd.append(int(depth[s, b]) - int(depth[s, a]))
                shaft.append(s)
        pairs[d] = {"ia": np.array(ia, dtype=np.int64), "ib": np.array(ib, dtype=np.int64),
                    "depth_off": np.array(dd, dtype=np.int64), "shaft": np.array(shaft)}

    # INVARIANT: slot offset d can never exceed the depth offset (gaps only ever widen it).
    for d in range(1, args.max_offset + 1):
        do = pairs[d]["depth_off"]
        if do.size:
            assert int(do.min()) >= d, f"slot offset {d} has depth offset {int(do.min())} < d"
    print("[check] slot_offset <= depth_offset holds for every pair at every d", flush=True)

    rows = []
    for bi, name in enumerate(BAND_NAMES):
        x = bands[bi].numpy()                    # (W, N, F, T)
        w, n, f, t = x.shape
        z = zscore_rows(np.transpose(x, (1, 2, 0, 3)).reshape(n, f, w * t).astype(np.float64))

        # INVARIANT: self-correlation must be exactly 1 for every non-degenerate contact.
        idx = np.arange(n)
        self_r = pair_corr(z, idx, idx)
        live = self_r > 0  # a contact that is constant across the whole sample yields 0, not 1
        assert np.allclose(self_r[live], 1.0, atol=1e-9), \
            f"{name}: r(0) != 1 (max dev {float(np.abs(self_r[live] - 1).max()):.2e})"
        print(f"[check] {name}: r(0)=1.000 for {int(live.sum())}/{n} contacts, "
              f"{n - int(live.sum())} constant", flush=True)

        prev = None
        for d in range(1, args.max_offset + 1):
            pr = pairs[d]
            if pr["ia"].size == 0:
                continue
            r = pair_corr(z, pr["ia"], pr["ib"])
            mono = "" if prev is None or float(r.mean()) <= prev + 1e-9 else "  ⚠️NON-MONOTONE"
            prev = float(r.mean())
            rows.append({
                "band": name, "slot_offset": d,
                "r_mean": float(r.mean()), "r_sd": float(r.std()),
                "r_p25": float(np.percentile(r, 25)), "r_p75": float(np.percentile(r, 75)),
                "n_pairs": int(r.size),
                "depth_offset_mean": float(pr["depth_off"].mean()),
                "depth_offset_median": float(np.median(pr["depth_off"])),
            })
            print(f"  {name:5s} d={d}  r={r.mean():.4f} +- {r.std():.4f}  "
                  f"[p25 {np.percentile(r, 25):.3f} p75 {np.percentile(r, 75):.3f}]  "
                  f"n={r.size:5d}  depth_off median={np.median(pr['depth_off']):.1f}{mono}",
                  flush=True)

    out = {
        "subject": args.subject, "trial": args.trial,
        "electrode_set": args.electrode_set,
        "n_clips": len(starts), "clip_frames": clip_frames,
        "n_contacts": int(valid.sum()), "n_shafts": int(geom.n_shafts),
        "contacts_per_shaft": [int(v) for v in cs],
        "rows": rows,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
