"""Outlier sweep + static-exclusion audit over the REBUILT 2STFT spec cache (#190,
Ben 2026-06-15 "final sweep over the new cache, scanning for bad electrodes / atrocious
outliers").

The 2026-06-15 stale-raw-cache bug (project_static_cache_stale_raw_exca_cache_bug) left
the static-excluded contacts IN the cache because the raw electrode_tokens exca cache was
reused. The cache was rebuilt with a fresh EXCA_EXTRACTOR_CACHE_FOLDER. This script reads
the rebuilt cache DIRECTLY (the bytes the run memmap-slices) and answers three questions:

  STATIC   Are all 11 statically-excluded contacts (anatomy._BT_V14_EXTRA_BAD_ELECTRODES)
           absent from every cached session's ch_names? (the bug was their PRESENCE.)
  OUTLIER  What is the residual per-electrode max |robust-z| over each whole movie? Is any
           electrode PERSISTENTLY giant (a missed bad contact), or are the giants confined
           to a few time windows (transients the CLIP/WINSOR layers handle)?
  CLIP-COV Does every catastrophic cell (|z| > clip_cell_thresh, default 5000) fall inside
           a Layer-2 bad-window sidecar span? (so the CLIP filter drops that pretrain clip;
           WINSOR independently clamps EVERY cell to the cap.)

It applies the cached per-session robust-z stats (``.stats.npz``, session-channel order,
written pre-global-scatter at view.py:1945 so they align row-for-row with the ``.npy``
memmap and ``.json`` ch_names) exactly as ``SessionRobustZNormalizer.transform`` does:
``z = (x - median) / max(sigma, floor)`` with constant-bin zeroing. No winsor here — the
point is to MEASURE the unclamped residual the two defenses must cover.

RUN ON DCC (reads /work caches; ~18 GB memmap I/O, so submit to a CPU node, not login):

    sbatch -p common -A coganlab -c 4 --mem 16G -t 00:30:00 \\
      -o sweep.out -e sweep.err --wrap "cd /work/ht203/repo/speech && \\
      ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank .venv/bin/python \\
      scripts/neuroprobe/sweep_static_cache_outliers.py \\
        --spec-cache-dir /work/ht203/cache_neuroai/v14_2stft_spec_cache_static_fixed \\
        --bad-window-dir /work/ht203/v14_bad_windows \\
        --report-json /work/ht203/sweep_static_cache_outliers.json"

Exit 0 iff STATIC passes (no excluded contact present) AND no electrode is flagged as a
persistent residual outlier. CLIP-COV is reported (informational: WINSOR covers the rest).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Cap the per-electrode max-over-freq |z| beyond which a TIME bin is "catastrophic".
# Matches the CLIP cell threshold (precompute_bad_windows cell_max > 5000).
_CLIP_CELL_THRESH = 5000.0
# An electrode whose |z| exceeds this in a FRACTION of windows above _PERSIST_FRAC is a
# persistent-bad candidate (a missed static exclusion), not a transient. The clean post-CAR
# |STFT| robust-z ceiling is ~2100 (anatomy.py docstring); use it as the "catastrophic
# amplitude" bar.
_CATASTROPHIC_Z = 2100.0
_PERSIST_FRAC = 0.02  # >2% of 5 s windows catastrophic ⇒ persistent, not a transient
_WINDOW_S = 5.0
_SIGMA_FLOOR = 1e-6  # view.MultiStftView.session_z_sigma_floor default
_TIME_CHUNK = 4096   # frames per memmap read chunk (bounds peak memory)


def _clean(label: str) -> str:
    from speech_decoding.studies.braintreebank.anatomy import clean_bt_electrode_label
    return clean_bt_electrode_label(label)


def _scan_one(npy_path: Path, meta: dict, stats_path: Path) -> dict:
    """Stream one session-band memmap, return per-electrode max|z|, catastrophic-window
    burden, and the movie-second timestamps of every catastrophic (>_CLIP_CELL_THRESH)
    frame. Chunked over time so peak memory is one (C, F, chunk) slice, not the whole
    movie."""
    with np.load(stats_path) as zf:
        median = zf["median"].astype(np.float32)          # (C, F, 1)
        sigma = zf["sigma"].astype(np.float32)            # (C, F, 1)
    safe_sigma = np.maximum(sigma, _SIGMA_FLOOR)
    const_bin = sigma < _SIGMA_FLOOR                       # (C, F, 1) zero these
    mm = np.load(npy_path, mmap_mode="r")                 # (C, F, T) raw |STFT| fp32
    n_chan, f_bins, total = mm.shape
    sr = float(meta["sample_rate"])
    hop = float(meta.get("band_hop") or meta["hop_length"])
    sec_per_frame = hop / sr

    per_elec_max = np.zeros(n_chan, dtype=np.float64)
    n_gt_1000 = np.zeros(n_chan, dtype=np.int64)          # time bins (max over freq)
    n_gt_catastrophic = np.zeros(n_chan, dtype=np.int64)
    catastrophic_frames: set[int] = set()                 # frames with ANY cell >clip thr

    for g0 in range(0, total, _TIME_CHUNK):
        x = np.asarray(mm[:, :, g0 : g0 + _TIME_CHUNK]).astype(np.float32)
        z = (x - median) / safe_sigma
        if const_bin.any():
            z = np.where(const_bin, 0.0, z)
        az = np.abs(z)                                    # (C, F, t)
        elec_time = az.max(axis=1)                        # (C, t) max over freq
        per_elec_max = np.maximum(per_elec_max, elec_time.max(axis=1))
        n_gt_1000 += (elec_time > 1000.0).sum(axis=1)
        n_gt_catastrophic += (elec_time > _CATASTROPHIC_Z).sum(axis=1)
        big = np.where(elec_time.max(axis=0) > _CLIP_CELL_THRESH)[0]
        for t in big:
            catastrophic_frames.add(int(g0 + t))
    del mm

    n_windows = max(1, int(round(total * sec_per_frame / _WINDOW_S)))
    return {
        "n_chan": int(n_chan),
        "f_bins": int(f_bins),
        "total_frames": int(total),
        "sec_per_frame": sec_per_frame,
        "n_windows": n_windows,
        "per_elec_max": per_elec_max,
        "n_gt_1000": n_gt_1000,
        "n_gt_catastrophic": n_gt_catastrophic,
        "catastrophic_secs": sorted(f * sec_per_frame for f in catastrophic_frames),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec-cache-dir", required=True)
    ap.add_argument("--bad-window-dir", default=None,
                    help="Layer-2 sidecar dir; enables the CLIP-COV cross-ref. Optional.")
    ap.add_argument("--report-json", default=None)
    ap.add_argument("--top", type=int, default=6, help="worst electrodes printed/session")
    args = ap.parse_args()

    from speech_decoding.studies.braintreebank.anatomy import (
        _BT_V14_EXTRA_BAD_ELECTRODES,
    )
    from speech_decoding.experiments.bad_windows import (
        bad_window_session_key,
        load_bad_windows,
    )
    bad_windows = load_bad_windows(args.bad_window_dir) if args.bad_window_dir else {}

    root = Path(args.spec_cache_dir)
    metas = sorted(root.rglob("*.json"))
    if not metas:
        print(f"FATAL: no *.json session metas under {root}", flush=True)
        return 2
    print(f"=== sweep over {len(metas)} session-band caches under {root} ===", flush=True)

    static_violations: list[str] = []
    persistent_outliers: list[str] = []
    report: list[dict] = []

    for json_path in metas:
        meta = json.loads(json_path.read_text())
        stem = json_path.with_suffix("")  # strip .json
        npy_path = Path(f"{stem}.npy")
        stats_path = Path(f"{stem}.stats.npz")
        band = "high" if "band_high" in str(json_path) else "low"
        key = json.loads(meta["key"]) if isinstance(meta.get("key"), str) and meta["key"].startswith("{") else {}
        tl = key.get("timeline", {}) if isinstance(key, dict) else {}
        subject_id = int(tl.get("subject_id", -1))
        trial_id = int(tl.get("trial_id", -1))
        ch_names = list(meta["ch_names"])
        if not (npy_path.exists() and stats_path.exists()):
            print(f"  [skip] {band} subj{subject_id} t{trial_id}: missing npy/stats", flush=True)
            continue

        # --- STATIC: every excluded contact for this subject must be ABSENT -----------
        cleaned_ch = {_clean(c) for c in ch_names}
        expected_drop = _BT_V14_EXTRA_BAD_ELECTRODES.get(subject_id, ())
        present_bad = [e for e in expected_drop if _clean(e) in cleaned_ch]
        if present_bad:
            static_violations.append(
                f"{band} subj{subject_id} t{trial_id}: excluded contacts STILL PRESENT "
                f"{present_bad}"
            )

        s = _scan_one(npy_path, meta, stats_path)
        order = np.argsort(s["per_elec_max"])[::-1]
        worst = [(ch_names[i], float(s["per_elec_max"][i]),
                  int(s["n_gt_catastrophic"][i]), int(s["n_gt_1000"][i])) for i in order[:args.top]]

        # --- OUTLIER: persistent vs transient -----------------------------------------
        frames_per_window = max(1, int(round(_WINDOW_S / s["sec_per_frame"])))
        persist_thresh_frames = _PERSIST_FRAC * s["n_windows"] * frames_per_window
        for i in order:
            if s["per_elec_max"][i] < _CATASTROPHIC_Z:
                break  # sorted desc — nothing else catastrophic
            if s["n_gt_catastrophic"][i] > persist_thresh_frames:
                persistent_outliers.append(
                    f"{band} subj{subject_id} t{trial_id}: electrode {ch_names[i]!r} "
                    f"max|z|={s['per_elec_max'][i]:.0f} catastrophic in "
                    f"{s['n_gt_catastrophic'][i]} frames "
                    f"(> {persist_thresh_frames:.0f} = {_PERSIST_FRAC:.0%} of session) "
                    f"⇒ PERSISTENT, not a transient"
                )

        # --- CLIP-COV: catastrophic cells inside a sidecar span -----------------------
        clip_cov = None
        if bad_windows and subject_id >= 0:
            spans = bad_windows.get(bad_window_session_key(subject_id, trial_id), [])
            cats = s["catastrophic_secs"]
            if cats:
                covered = sum(
                    1 for t in cats if any(b0 <= t < b1 for b0, b1 in spans)
                )
                clip_cov = (covered, len(cats))

        print(
            f"  [{band}] subj{subject_id} t{trial_id}: n_elec={s['n_chan']} "
            f"(dropped {len(expected_drop)} expected) | worst max|z|={worst[0][1]:.0f} "
            f"@ {worst[0][0]} | n_elec>2100={int((s['per_elec_max']>_CATASTROPHIC_Z).sum())}"
            + (f" | CLIP-cov {clip_cov[0]}/{clip_cov[1]} catastrophic frames in spans"
               if clip_cov else ""),
            flush=True,
        )
        for name, mx, ncat, n1k in worst:
            print(f"        {name:14s} max|z|={mx:8.0f}  frames>2100={ncat:5d}  >1000={n1k:6d}",
                  flush=True)

        report.append({
            "band": band, "subject_id": subject_id, "trial_id": trial_id,
            "n_elec": s["n_chan"], "expected_drop": list(expected_drop),
            "present_bad": present_bad,
            "worst": [{"ch": n, "max_abs_z": mx, "frames_gt_2100": c, "frames_gt_1000": k}
                      for n, mx, c, k in worst],
            "n_elec_gt_2100": int((s["per_elec_max"] > _CATASTROPHIC_Z).sum()),
            "clip_cov": list(clip_cov) if clip_cov else None,
        })

    print("\n=== VERDICT ===", flush=True)
    if static_violations:
        print("STATIC: FAIL — excluded contacts present in the rebuilt cache:", flush=True)
        for v in static_violations:
            print(f"  {v}", flush=True)
    else:
        print(f"STATIC: PASS — all {sum(len(v) for v in _BT_V14_EXTRA_BAD_ELECTRODES.values())} "
              "excluded contacts absent from every cached session.", flush=True)
    if persistent_outliers:
        print("OUTLIER: FAIL — persistent residual outlier electrode(s):", flush=True)
        for v in persistent_outliers:
            print(f"  {v}", flush=True)
    else:
        print("OUTLIER: PASS — no electrode is catastrophic across a persistent fraction of "
              "its session (residual giants are transients ⇒ CLIP drops the clip, WINSOR "
              "clamps the cell).", flush=True)

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2))
        print(f"\nreport written -> {args.report_json}", flush=True)

    ok = not static_violations and not persistent_outliers
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
