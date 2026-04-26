#!/usr/bin/env python3
"""Compare snap-based and Euclidean-pool support caches per patient.

Reads both caches and reports:
- Per-patient argmax-change rate (electrodes whose dominant Tier-1 parcel
  differs between the two caches).
- Per-parcel stability: how many snap electrodes for each parcel kept that
  same parcel under the pool.
- Per-electrode Pearson correlation of support vectors (mean, p05, p95).
- Max-support magnitude comparison (median, p25, p75).

Writes a Markdown report to data/atlas/support_cache_euclidean/compare_report.md
and a CSV of per-electrode diffs per patient.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.support_cache import TIER1_COLUMNS, read_support_cache

DEFAULT_SNAP_DIR = PROJECT_ROOT / "data" / "atlas" / "support_cache"
DEFAULT_POOL_DIR = PROJECT_ROOT / "data" / "atlas" / "support_cache_euclidean"
DEFAULT_REPORT = DEFAULT_POOL_DIR / "compare_report.md"
DEFAULT_PATIENTS = ("S14", "S16", "S23", "S26", "S33", "S39", "S62")


def _per_electrode_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Pearson correlation per row. Rows with zero variance -> nan -> replaced with 0.
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    am = a - a.mean(axis=1, keepdims=True)
    bm = b - b.mean(axis=1, keepdims=True)
    num = (am * bm).sum(axis=1)
    den = np.sqrt((am ** 2).sum(axis=1) * (bm ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    return r


def compare_patient(snap_path: Path, pool_path: Path) -> dict:
    snap_names, snap = read_support_cache(snap_path)
    pool_names, pool = read_support_cache(pool_path)
    if snap_names != pool_names:
        raise ValueError(
            f"electrode order mismatch between {snap_path.name} and {pool_path.name}"
        )

    snap_argmax = snap.argmax(axis=1)
    pool_argmax = pool.argmax(axis=1)
    changed = snap_argmax != pool_argmax

    snap_max = snap.max(axis=1)
    pool_max = pool.max(axis=1)
    corr = _per_electrode_correlation(snap, pool)

    per_parcel: list[dict] = []
    n_parcels = len(TIER1_COLUMNS)
    for p_idx in range(n_parcels):
        snap_mask = snap_argmax == p_idx
        kept = int((snap_mask & (pool_argmax == p_idx)).sum())
        total = int(snap_mask.sum())
        per_parcel.append({
            "parcel": TIER1_COLUMNS[p_idx][len("support_"):],
            "snap_argmax_count": total,
            "kept_under_pool": kept,
            "retention": float(kept / total) if total else float("nan"),
        })

    return {
        "n_electrodes": int(snap.shape[0]),
        "n_changed": int(changed.sum()),
        "frac_changed": float(changed.mean()),
        "pearson_r_mean": float(corr.mean()),
        "pearson_r_p05": float(np.quantile(corr, 0.05)),
        "pearson_r_p95": float(np.quantile(corr, 0.95)),
        "snap_max_p25_p50_p75": tuple(float(x) for x in np.quantile(snap_max, [0.25, 0.5, 0.75])),
        "pool_max_p25_p50_p75": tuple(float(x) for x in np.quantile(pool_max, [0.25, 0.5, 0.75])),
        "per_parcel": per_parcel,
        "changed_mask": changed,
        "snap_argmax": snap_argmax,
        "pool_argmax": pool_argmax,
        "names": snap_names,
    }


def _write_diff_csv(out_path: Path, result: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("name,snap_argmax,pool_argmax,changed\n")
        for name, s, p, c in zip(
            result["names"],
            result["snap_argmax"],
            result["pool_argmax"],
            result["changed_mask"],
        ):
            snap_name = TIER1_COLUMNS[int(s)][len("support_"):]
            pool_name = TIER1_COLUMNS[int(p)][len("support_"):]
            f.write(f"{name},{snap_name},{pool_name},{int(c)}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", nargs="+", default=list(DEFAULT_PATIENTS))
    parser.add_argument("--snap-dir", type=Path, default=DEFAULT_SNAP_DIR)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    report_lines: list[str] = []
    report_lines.append("# Electrode support: snap vs Euclidean pool\n")
    report_lines.append(f"Snap cache: `{args.snap_dir}`\n")
    report_lines.append(f"Pool cache: `{args.pool_dir}`\n")
    report_lines.append("\n## Per-patient summary\n")
    report_lines.append(
        "| patient | n_elec | n_changed | frac | r mean | r p05 | r p95 | "
        "snap max p50 | pool max p50 |"
    )
    report_lines.append("|" + "---|" * 9)

    results: dict[str, dict] = {}
    for pt in args.patient:
        snap_path = args.snap_dir / f"{pt}_support_tier1.csv"
        pool_path = args.pool_dir / f"{pt}_support_tier1.csv"
        if not snap_path.exists():
            print(f"[skip] {pt}: missing {snap_path}")
            continue
        if not pool_path.exists():
            print(f"[skip] {pt}: missing {pool_path}")
            continue
        res = compare_patient(snap_path, pool_path)
        results[pt] = res
        _write_diff_csv(args.pool_dir / f"{pt}_argmax_diff.csv", res)
        report_lines.append(
            f"| {pt} | {res['n_electrodes']} | {res['n_changed']} | "
            f"{res['frac_changed']:.3f} | {res['pearson_r_mean']:.3f} | "
            f"{res['pearson_r_p05']:.3f} | {res['pearson_r_p95']:.3f} | "
            f"{res['snap_max_p25_p50_p75'][1]:.1f} | "
            f"{res['pool_max_p25_p50_p75'][1]:.1f} |"
        )

    # Per-parcel retention table pooled across patients.
    report_lines.append("\n## Per-parcel retention (pooled)\n")
    report_lines.append("| parcel | snap_argmax_count | kept_under_pool | retention |")
    report_lines.append("|" + "---|" * 4)
    totals = {col[len("support_"):]: [0, 0] for col in TIER1_COLUMNS}
    for res in results.values():
        for row in res["per_parcel"]:
            t = totals[row["parcel"]]
            t[0] += row["snap_argmax_count"]
            t[1] += row["kept_under_pool"]
    for parcel, (tot, kept) in totals.items():
        retention = (kept / tot) if tot else float("nan")
        report_lines.append(f"| {parcel} | {tot} | {kept} | {retention:.3f} |")

    report_lines.append("")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines))
    print(f"wrote {args.report}")
    for pt, res in results.items():
        print(
            f"  {pt}: n={res['n_electrodes']} changed={res['n_changed']} "
            f"({res['frac_changed']:.1%}) r_mean={res['pearson_r_mean']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
