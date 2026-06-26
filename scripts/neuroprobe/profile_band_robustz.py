"""Per-band |robust-z| distribution profiler for WINSOR (#229) and CLIP (#231) tuning.

The bad-electrode WINSOR cap and the CLIP bad-window thresholds are set from the tail
of the POST-STATIC ``|robust-z|`` distribution, PER BAND (slow/beta/hg). With the
stronger per-session GUARD-1 STATIC drop, the residual distribution each downstream
layer sees has changed, and the three bands have different dynamic ranges — so one
scalar cap can't serve all three. This reads the rebuilt 3STFT spec cache directly
(the bytes the run memmap-slices) and reports, per band:

  CELL    percentiles of every (electrode, freq-bin, frame) |z|  -> sets WINSOR W
  ELECT   percentiles of per-(electrode, frame) max-over-freq |z| -> informs CLIP thr

The slow band is CARTESIAN (FE spec §4): the freq axis holds [Re bins ++ Im bins]
with the cache storing median=0 and sigma = [sigma_p ++ sigma_p] (the per-bin |STFT|
MAD shared across Re/Im). So the SAME generic reconstruction z = (x - median)/sigma
yields the scale-only (Re/sigma_p, Im/sigma_p) the model consumes — no read-path fork
(view.py _fit_session_stats). We additionally split Re vs Im as a diagnostic to confirm
they share scale (the shared-sigma design predicts they should).

WINSOR clamp is symmetric [-W, +W], so we profile |z| (abs) — correct for both the
signed slow band and the magnitude beta/HG bands.

RUN ON DCC (reads /work caches; submit to a CPU node, not login):

    sbatch -p common -A coganlab -c 4 --mem 16G -t 00:40:00 \\
      -o prof.out -e prof.err --wrap "cd /work/ht203/repo/speech && \\
      ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank .venv/bin/python \\
      scripts/neuroprobe/profile_band_robustz.py \\
        --spec-cache-dir /work/ht203/cache_neuroai/<rebuilt-3stft-cache> \\
        --report-json /work/ht203/profile_band_robustz.json"

The per-band CELL P99.9 over the clean sessions is the recommended WINSOR W floor; the
artifact tail (P99.99 .. max) is what CLIP must drop. The script PRINTS a recommendation
table but sets no guard — the values are Ben's call (discuss-before-code).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from speech_decoding.experiments.robustz_histogram import RobustZHistogram

_SIGMA_FLOOR = 1e-6   # view.MultiStftView.session_z_sigma_floor default
_TIME_CHUNK = 4096    # frames per memmap read chunk (bounds peak memory)
_REPORT_QS = (50.0, 90.0, 99.0, 99.9, 99.99)


def _discover_bands(root: Path) -> tuple[str, ...]:
    """The band tags actually present as ``band_<x>`` subdirs of the cache root.

    Band-agnostic so the SAME profiler serves the 3STFT cache (slow/beta/hg) and
    the 2-band converged-v2 cache (lfs/hga) — only the slow band carries the
    Cartesian Re/Im split, gated below on ``"slow" in bands``."""
    return tuple(
        sorted(p.name[len("band_") :] for p in root.glob("band_*") if p.is_dir())
    )


def _band_of(json_path: Path, bands: tuple[str, ...]) -> str | None:
    """Recover the band from the ``band_<x>`` cache subdir on the path."""
    for part in json_path.parts:
        if part.startswith("band_"):
            tag = part[len("band_") :]
            if tag in bands:
                return tag
    return None


def _session_of(meta: dict) -> tuple[int, int]:
    """(subject_id, trial_id) parsed from the exca event uid stored in meta['key']."""
    raw = meta.get("key", "")
    if isinstance(raw, str) and raw.startswith("{"):
        key, _ = json.JSONDecoder().raw_decode(raw)
    else:
        key = {}
    tl = key.get("timeline", {}) if isinstance(key, dict) else {}
    return int(tl.get("subject_id", -1)), int(tl.get("trial_id", -1))


def _accumulate(npy_path: Path, stats_path: Path, band: str, hists: dict) -> None:
    """Stream one session-band memmap, reconstruct robust-z exactly as the run does,
    and fold |z| into the per-band CELL, ELECT (and slow Re/Im) histograms."""
    with np.load(stats_path) as zf:
        median = zf["median"].astype(np.float32)   # (C, F, 1)
        sigma = zf["sigma"].astype(np.float32)      # (C, F, 1)
    safe_sigma = np.maximum(sigma, _SIGMA_FLOOR)
    const_bin = sigma < _SIGMA_FLOOR                # zero these (constant freq bins)
    mm = np.load(npy_path, mmap_mode="r")           # (C, F, T) raw |STFT| / [Re++Im]
    _, f_bins, total = mm.shape
    n_half = f_bins // 2                            # slow: Re = [:n_half], Im = [n_half:]

    for g0 in range(0, total, _TIME_CHUNK):
        x = np.asarray(mm[:, :, g0 : g0 + _TIME_CHUNK]).astype(np.float32)
        z = (x - median) / safe_sigma
        if const_bin.any():
            z = np.where(const_bin, 0.0, z)
        az = np.abs(z)                              # (C, F, t)
        hists["cell"][band].update(az)
        hists["elect"][band].update(az.max(axis=1))  # (C, t) max over freq
        if band == "slow":
            hists["slow_re"].update(az[:, :n_half, :])
            hists["slow_im"].update(az[:, n_half:, :])
    del mm


def _row(label: str, h: RobustZHistogram) -> dict:
    d: dict = {f"p{q}": round(h.percentile(q), 2) for q in _REPORT_QS}
    d.update(label=label, max=round(h.max_val, 1), n=h.total, overflow=h.overflow)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec-cache-dir", required=True)
    ap.add_argument(
        "--clean-sessions",
        default=None,
        help="optional 'subj:trial,subj:trial' allowlist; default = every session "
        "in the cache (all are clean of whole-bad electrodes post-STATIC).",
    )
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    allow: set[tuple[int, int]] | None = None
    if args.clean_sessions:
        allow = set()
        for tok in args.clean_sessions.split(","):
            s, t = tok.split(":")
            allow.add((int(s), int(t)))

    root = Path(args.spec_cache_dir)
    bands = _discover_bands(root)
    if not bands:
        print(f"FATAL: no band_<x> subdirs under {root}", flush=True)
        return 2
    print(f"bands discovered: {bands}", flush=True)
    metas = sorted(root.rglob("*.json"))
    if not metas:
        print(f"FATAL: no *.json session metas under {root}", flush=True)
        return 2

    hists = {
        "cell": {b: RobustZHistogram() for b in bands},
        "elect": {b: RobustZHistogram() for b in bands},
        "slow_re": RobustZHistogram(),
        "slow_im": RobustZHistogram(),
    }
    n_used = {b: 0 for b in bands}

    for json_path in metas:
        band = _band_of(json_path, bands)
        if band is None:
            continue
        meta = json.loads(json_path.read_text())
        subj, trial = _session_of(meta)
        if allow is not None and (subj, trial) not in allow:
            continue
        stem = json_path.with_suffix("")
        npy_path = Path(f"{stem}.npy")
        stats_path = Path(f"{stem}.stats.npz")
        if not (npy_path.exists() and stats_path.exists()):
            print(f"  [skip] {band} subj{subj} t{trial}: missing npy/stats", flush=True)
            continue
        _accumulate(npy_path, stats_path, band, hists)
        n_used[band] += 1
        print(f"  [{band}] folded subj{subj} t{trial}", flush=True)

    report: dict = {"bands": {}, "slow_components": {}}
    print("\n=== PER-BAND |robust-z| DISTRIBUTION ===", flush=True)
    for b in bands:
        cell = _row(f"{b}:cell", hists["cell"][b])
        elect = _row(f"{b}:elect", hists["elect"][b])
        report["bands"][b] = {"sessions": n_used[b], "cell": cell, "elect": elect}
        print(
            f"  {b:5s} n_sess={n_used[b]:2d} | CELL  P99={cell['p99.0']:>8} "
            f"P99.9={cell['p99.9']:>8} P99.99={cell['p99.99']:>9} max={cell['max']:>9}",
            flush=True,
        )
        print(
            f"  {b:5s}            | ELECT P99={elect['p99.0']:>8} "
            f"P99.9={elect['p99.9']:>8} P99.99={elect['p99.99']:>9} max={elect['max']:>9}",
            flush=True,
        )

    # Slow Re/Im split — confirm shared scale (shared-sigma design). Only the slow
    # band is Cartesian; the 2-band lfs/hga cache is magnitude-only, so skip it there.
    if "slow" in bands:
        re_row = _row("slow:Re", hists["slow_re"])
        im_row = _row("slow:Im", hists["slow_im"])
        report["slow_components"] = {"Re": re_row, "Im": im_row}
        print(
            f"\n  slow Re/Im split (shared-sigma check): "
            f"Re P99.9={re_row['p99.9']}  Im P99.9={im_row['p99.9']}",
            flush=True,
        )

    print("\n=== WINSOR W RECOMMENDATION (per-band CELL P99.9 = real-signal ceiling) ===",
          flush=True)
    for b in bands:
        p999 = report["bands"][b]["cell"]["p99.9"]
        p9999 = report["bands"][b]["cell"]["p99.99"]
        print(f"  {b:5s}: W >~ P99.9={p999}  (artifact tail starts ~P99.99={p9999}); "
              f"set W between them.", flush=True)
    print("  These are MEASUREMENTS, not a committed guard — Ben sets the final W.",
          flush=True)

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2))
        print(f"\nreport written -> {args.report_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
