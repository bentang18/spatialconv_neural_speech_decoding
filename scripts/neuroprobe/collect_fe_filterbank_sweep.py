"""Collect the FE filterbank sweep into a durable record.

Reads each job's `diagnostics.csv` (via the launch manifest), then writes:
  * `docs/experiments/fe_filterbank_sweep_<date>_per_task.csv` — one row per
    (cell × eval × task × subject × fold), the raw long table.
  * `docs/experiments/fe_filterbank_sweep_<date>_summary.csv` — one row per
    (cell × eval) with rank4_mean_auc, all15_mean_auc, and Δ-vs-control (cell 0).

The filterbank decision is read off `delta_rank4` (cross > within is the
broadband-power prediction). `delta_all15` is a confirm-only column.
Spec: `reports/fe_filterbank_sweep_build_2026_06_03.md` §8.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RANK4_TASKS = ("speech", "onset", "volume", "pitch")
CONTROL_CELL = "0"


def load_long_table(out_root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(out_root / "launch_manifest.csv")
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for _, m in manifest.iterrows():
        diag_path = Path(m["report_dir"]) / "diagnostics.csv"
        if not diag_path.exists():
            missing.append(str(diag_path))
            continue
        diag = pd.read_csv(diag_path)
        diag = diag.assign(
            cell=str(m["cell"]),
            eval=str(m["eval"]),
            ref_kind=m["ref_kind"],
            view_kind=m["view_kind"],
            f0_hz=m["f0_hz"],
            cap_hz=m["cap_hz"],
            octave_step=m["octave_step"],
            half_bw=m["half_bw"],
        )
        frames.append(diag)
    if missing:
        print(f"WARNING: {len(missing)}/{len(manifest)} jobs missing diagnostics.csv "
              f"(incomplete or failed). First few:\n  " + "\n  ".join(missing[:5]))
    if not frames:
        raise SystemExit("No diagnostics found — nothing to collect.")
    return pd.concat(frames, ignore_index=True)


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    """Mean and normal-approx 95% half-width across the given units."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan")
    half = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean, half


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    # Per (cell, eval, task, subject): mean AUROC over folds.
    by_subj = (
        long.groupby(["cell", "eval", "task", "subject_id"])["test_roc_auc"]
        .mean()
        .reset_index()
    )
    # Footprint = the exact (subject, task) pairs a (cell, eval) actually has.
    # A cell's mean/delta is only comparable to the control's when they cover the
    # SAME footprint; a failed or partial job otherwise yields a silently
    # non-comparable delta (BUG-COLLECT-4). We record means over what's present
    # but NaN out rank4/all15 (and their deltas) for any cell whose footprint
    # differs from the control's, and warn loudly. The full raw data survives in
    # the per_task.csv, so nothing is lost — only the headline is made safe.
    rows: list[dict[str, object]] = []
    r4_fp: dict[tuple[str, str], frozenset] = {}
    all_fp: dict[tuple[str, str], frozenset] = {}
    for (cell, ev), grp in by_subj.groupby(["cell", "eval"]):
        cell, ev = str(cell), str(ev)
        rank4 = grp[grp["task"].isin(RANK4_TASKS)]
        r4_fp[(cell, ev)] = frozenset(zip(rank4["subject_id"], rank4["task"]))
        all_fp[(cell, ev)] = frozenset(zip(grp["subject_id"], grp["task"]))
        # subject-level means (per task already folded), then average across the
        # rank-4 tasks per subject, then across subjects → CI across subjects.
        rank4_subj = rank4.groupby("subject_id")["test_roc_auc"].mean().to_numpy()
        all15_subj = grp.groupby("subject_id")["test_roc_auc"].mean().to_numpy()
        r4_mean, r4_ci = _mean_ci(rank4_subj)
        a15_mean, a15_ci = _mean_ci(all15_subj)
        rows.append({
            "cell": cell,
            "eval": ev,
            "n_subjects": int(grp["subject_id"].nunique()),
            "n_tasks": int(grp["task"].nunique()),
            "n_rank4_tasks": int(rank4["task"].nunique()),
            "rank4_mean_auc": r4_mean,
            "rank4_ci95": r4_ci,
            "all15_mean_auc": a15_mean,
            "all15_ci95": a15_ci,
        })
    summary = pd.DataFrame(rows)

    # Δ vs the shaft-CAR raw-bins control (cell 0), within each eval — but only
    # between footprint-identical cells.
    for ev in summary["eval"].unique():
        ctrl_key = (CONTROL_CELL, ev)
        if ctrl_key not in r4_fp:
            print(f"WARNING: [{ev}] control cell {CONTROL_CELL} absent — no Δ computed "
                  f"for this eval; every cell here is non-comparable.")
            continue
        ctrl_r4, ctrl_all = r4_fp[ctrl_key], all_fp[ctrl_key]
        ctrl_tasks = {t for _, t in ctrl_r4}
        if ctrl_tasks != set(RANK4_TASKS):
            print(f"WARNING: [{ev}] control cell {CONTROL_CELL} is itself missing rank-4 "
                  f"task(s) {sorted(set(RANK4_TASKS) - ctrl_tasks)} — every Δ for this "
                  f"eval is against a PARTIAL baseline.")
        base_r4 = float(summary.loc[
            (summary["eval"] == ev) & (summary["cell"] == CONTROL_CELL), "rank4_mean_auc"].iloc[0])
        base_a15 = float(summary.loc[
            (summary["eval"] == ev) & (summary["cell"] == CONTROL_CELL), "all15_mean_auc"].iloc[0])
        for i in summary.index[summary["eval"] == ev]:
            cell = str(summary.at[i, "cell"])
            key = (cell, ev)
            if r4_fp.get(key) != ctrl_r4:
                miss = len(ctrl_r4 - r4_fp.get(key, frozenset()))
                extra = len(r4_fp.get(key, frozenset()) - ctrl_r4)
                print(f"WARNING: [{ev}] cell {cell} rank-4 footprint differs from control "
                      f"(missing {miss} / extra {extra} subject×task pairs) — rank4_mean_auc "
                      f"& delta_rank4 set to NaN (non-comparable).")
                summary.at[i, "rank4_mean_auc"] = np.nan
                summary.at[i, "delta_rank4_vs_ctrl"] = np.nan
            else:
                summary.at[i, "delta_rank4_vs_ctrl"] = summary.at[i, "rank4_mean_auc"] - base_r4
            if all_fp.get(key) != ctrl_all:
                summary.at[i, "all15_mean_auc"] = np.nan
                summary.at[i, "delta_all15_vs_ctrl"] = np.nan
            else:
                summary.at[i, "delta_all15_vs_ctrl"] = summary.at[i, "all15_mean_auc"] - base_a15

    return summary.sort_values(["eval", "cell"]).reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    out_root = args.out_root
    long = load_long_table(out_root)
    summary = summarize(long)

    docs_dir = Path("docs/experiments")
    docs_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.date or out_root.name.replace("fe_filterbank_sweep_", "")
    per_task_path = docs_dir / f"fe_filterbank_sweep_{stamp}_per_task.csv"
    summary_path = docs_dir / f"fe_filterbank_sweep_{stamp}_summary.csv"
    long.to_csv(per_task_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {per_task_path} ({len(long)} rows)")
    print(f"Wrote {summary_path} ({len(summary)} cell×eval rows)\n")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        cols = ["cell", "eval", "n_subjects", "rank4_mean_auc", "delta_rank4_vs_ctrl",
                "all15_mean_auc", "delta_all15_vs_ctrl"]
        print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, required=True,
                   help="The sweep report dir (contains launch_manifest.csv).")
    p.add_argument("--date", default=None, help="Stamp for the output CSV name (default: from out-root).")
    return p.parse_args()


if __name__ == "__main__":
    main()
