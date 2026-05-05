"""Collect L.1 normalization sweep results into aggregate report.

Walks the per-cell-per-session directories produced by
`submit_l1_normalization_sweep.py`, reads each `diagnostics.csv` +
`metrics.json` + `experiment_record.json`, and writes:

- `aggregate_diagnostics.csv` — one row per cell × session × task × fold
- `aggregate_summary_by_cell.csv` — cell mean AUROC across all sessions × tasks
- `aggregate_summary_by_cell_task.csv` — cell × task mean AUROC across sessions
- `n0_vs_n1_delta.csv` — per-task and aggregate delta (per_window_z minus train_set_fixed)
- `cell_task_heatmap.png` — cell × task AUROC matrix with N0-vs-N1 column highlighted
- `cell_aggregate_bar.png` — bar chart of cell mean AUROC, with N1 reference line
- `summary.json` — pass/fail flags + headline numbers
- `README.md` — human-readable interpretation

Run from the project root:

    .venv/bin/python scripts/neuroprobe/collect_l1_normalization_sweep.py \\
        --sweep-root reports/neuroprobe_stage0_l1_normalization_2026_05_05/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CELL_ORDER: tuple[str, ...] = (
    "L.1.N0",
    "L.1.N1",
    "L.1.N2",
    "L.1.N3",
    "L.1.N4",
    "L.1.N5",
)

CELL_RECIPE: dict[str, str] = {
    "L.1.N0": "per_window_z",
    "L.1.N1": "train_set_fixed",
    "L.1.N2": "per_session_fixed",
    "L.1.N3": "train_set_scale_only",
    "L.1.N4": "none",
    "L.1.N5": "per_session_robust_mad",
}


def main() -> None:
    args = _parse_args()
    sweep_root = args.sweep_root.resolve()
    if not sweep_root.exists():
        raise FileNotFoundError(sweep_root)

    rows = collect_diagnostic_rows(sweep_root)
    if not rows:
        raise SystemExit(f"No diagnostics.csv rows found under {sweep_root}")
    diagnostics = pd.DataFrame(rows)

    diagnostics_path = sweep_root / "aggregate_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    print(f"wrote {diagnostics_path} ({len(diagnostics)} rows)")

    rename_map = {"mean": "mean_test_roc_auc", "std": "std_test_roc_auc", "count": "n"}
    sbc = cast(pd.DataFrame, diagnostics.groupby(["cell", "normalization"])["test_roc_auc"].agg(["mean", "std", "count"]).reset_index())
    summary_by_cell = sbc.rename(columns=rename_map).sort_values("cell")
    summary_by_cell.to_csv(sweep_root / "aggregate_summary_by_cell.csv", index=False)

    sbct = cast(pd.DataFrame, diagnostics.groupby(["cell", "normalization", "task"])["test_roc_auc"].agg(["mean", "std", "count"]).reset_index())
    summary_by_cell_task = sbct.rename(columns=rename_map)
    summary_by_cell_task.to_csv(
        sweep_root / "aggregate_summary_by_cell_task.csv", index=False
    )

    pivot = (
        summary_by_cell_task.pivot_table(
            index="cell", columns="task", values="mean_test_roc_auc"
        )
        .reindex([c for c in CELL_ORDER if c in summary_by_cell["cell"].values])
    )

    n0_vs_n1 = compute_n0_vs_n1(summary_by_cell_task)
    n0_vs_n1.to_csv(sweep_root / "n0_vs_n1_delta.csv", index=False)

    plot_cell_task_heatmap(pivot, sweep_root / "cell_task_heatmap.png")
    plot_cell_aggregate_bar(summary_by_cell, sweep_root / "cell_aggregate_bar.png")

    summary = build_summary(summary_by_cell, n0_vs_n1)
    (sweep_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    write_readme(sweep_root, summary, summary_by_cell)
    print(json.dumps(summary, indent=2, sort_keys=True))


def collect_diagnostic_rows(sweep_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for diag_path in sorted(sweep_root.rglob("diagnostics.csv")):
        record_path = diag_path.parent / "experiment_record.json"
        cell = ""
        if record_path.exists():
            record = json.loads(record_path.read_text())
            cell = str(record.get("cell", ""))
        if not cell:
            cell = _infer_cell_from_path(diag_path.parent, sweep_root)
        df = pd.read_csv(diag_path)
        df["cell"] = cell
        df["report_dir"] = str(diag_path.parent)
        rows.extend(df.to_dict("records"))
    return rows


def _infer_cell_from_path(report_dir: Path, sweep_root: Path) -> str:
    rel = report_dir.relative_to(sweep_root).parts
    return rel[0] if rel else ""


def compute_n0_vs_n1(summary_by_cell_task: pd.DataFrame) -> pd.DataFrame:
    n0_slice = cast(pd.DataFrame, summary_by_cell_task[summary_by_cell_task["cell"] == "L.1.N0"][["task", "mean_test_roc_auc"]])
    n0 = n0_slice.rename(columns={"mean_test_roc_auc": "n0_per_window_z"})
    n1_slice = cast(pd.DataFrame, summary_by_cell_task[summary_by_cell_task["cell"] == "L.1.N1"][["task", "mean_test_roc_auc"]])
    n1 = n1_slice.rename(columns={"mean_test_roc_auc": "n1_train_set_fixed"})
    merged = n0.merge(n1, on="task", how="outer")
    merged["delta_n0_minus_n1"] = merged["n0_per_window_z"] - merged["n1_train_set_fixed"]
    has_rows = len(merged.index) > 0
    aggregate = pd.DataFrame(
        [
            {
                "task": "ALL",
                "n0_per_window_z": float(merged["n0_per_window_z"].mean()) if has_rows else math.nan,
                "n1_train_set_fixed": float(merged["n1_train_set_fixed"].mean()) if has_rows else math.nan,
                "delta_n0_minus_n1": float(merged["delta_n0_minus_n1"].mean()) if has_rows else math.nan,
            }
        ]
    )
    return pd.concat([aggregate, merged.sort_values("task")], ignore_index=True)


def plot_cell_task_heatmap(pivot: pd.DataFrame, out_path: Path) -> None:
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(pivot.columns) + 4), 4 + 0.4 * len(pivot.index)))
    arr = pivot.values.astype(float)
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0.4, vmax=max(0.7, np.nanmax(arr)))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{c} ({CELL_RECIPE.get(c, '?')})" for c in pivot.index], fontsize=9)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.6 else "black")
    if "L.1.N0" in pivot.index and "L.1.N1" in pivot.index:
        n0_idx = list(pivot.index).index("L.1.N0")
        n1_idx = list(pivot.index).index("L.1.N1")
        for idx in (n0_idx, n1_idx):
            ax.axhline(idx - 0.5, color="red", linewidth=1.2, linestyle="--")
            ax.axhline(idx + 0.5, color="red", linewidth=1.2, linestyle="--")
    ax.set_title("L.1 cell × task mean test AUROC (red dashed = N0/N1 headline rows)")
    fig.colorbar(im, ax=ax, label="mean test AUROC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_cell_aggregate_bar(summary_by_cell: pd.DataFrame, out_path: Path) -> None:
    if summary_by_cell.empty:
        return
    cells_present = [c for c in CELL_ORDER if c in summary_by_cell["cell"].values]
    if not cells_present:
        return
    df = summary_by_cell.set_index("cell").loc[cells_present]
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(df))
    ax.bar(xs, df["mean_test_roc_auc"], yerr=df["std_test_roc_auc"], capsize=4, color="steelblue")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}\n{CELL_RECIPE.get(c, '?')}" for c in df.index], fontsize=9)
    n1_value = float(df.loc["L.1.N1", "mean_test_roc_auc"]) if "L.1.N1" in df.index else None
    if n1_value is not None:
        ax.axhline(n1_value, color="red", linestyle="--", linewidth=1, label=f"N1 (upstream)={n1_value:.3f}")
        ax.legend(loc="lower right")
    ax.set_ylabel("mean test AUROC (averaged over sessions × tasks × folds)")
    ax.set_title("L.1 normalization sweep — cell-level aggregate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def build_summary(
    summary_by_cell: pd.DataFrame, n0_vs_n1: pd.DataFrame
) -> dict[str, object]:
    by_cell = {
        row["cell"]: {
            "normalization": row["normalization"],
            "mean_test_roc_auc": float(row["mean_test_roc_auc"]),
            "std_test_roc_auc": float(row["std_test_roc_auc"]) if not bool(pd.isna(row["std_test_roc_auc"])) else None,
            "n_rows": int(row["n"]),
        }
        for _, row in summary_by_cell.iterrows()
    }
    aggregate_delta = cast(pd.DataFrame, n0_vs_n1[n0_vs_n1["task"] == "ALL"])
    headline_delta = (
        float(aggregate_delta["delta_n0_minus_n1"].iloc[0])
        if len(aggregate_delta.index) > 0
        else None
    )
    return {
        "headline_n0_minus_n1_delta": headline_delta,
        "load_bearing_threshold": 0.02,
        "load_bearing": (headline_delta is not None and headline_delta <= -0.02),
        "by_cell": by_cell,
        "n_cells": int(summary_by_cell["cell"].nunique()),
    }


def write_readme(
    sweep_root: Path,
    summary: dict[str, object],
    summary_by_cell: pd.DataFrame,
) -> None:
    lines = [
        "# Neuroprobe Stage 0 L.1 Normalization Sweep",
        "",
        "Linear baseline ablating per-channel normalization scope. Default split: CrossSession multiclass on all 12 BT Lite sessions × 15 tasks (D.0b protocol).",
        "",
        "## Headline",
        "",
        f"- N0 (per_window_z) − N1 (train_set_fixed) aggregate AUROC delta: **{summary['headline_n0_minus_n1_delta']}**",
        f"- Load-bearing threshold: |Δ| ≥ 0.02. Result: **{'load-bearing' if summary['load_bearing'] else 'not load-bearing'}**.",
        "",
        "## Cell aggregates",
        "",
        "| Cell | Recipe | Mean AUROC | SD | N |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in summary_by_cell.iterrows():
        sd = f"{row['std_test_roc_auc']:.4f}" if not bool(pd.isna(row['std_test_roc_auc'])) else ""
        lines.append(
            f"| {row['cell']} | `{row['normalization']}` | {row['mean_test_roc_auc']:.4f} | {sd} | {int(row['n'])} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "- `aggregate_diagnostics.csv` — every cell × session × task × fold row",
            "- `aggregate_summary_by_cell.csv` — cell mean across sessions × tasks",
            "- `aggregate_summary_by_cell_task.csv` — cell × task means",
            "- `n0_vs_n1_delta.csv` — per-task and aggregate N0-vs-N1 delta (the headline)",
            "- `cell_task_heatmap.png` — visual summary",
            "- `cell_aggregate_bar.png` — cell-level bar chart with N1 reference line",
            "",
            "## Stage-1 freeze",
            "Highest mean-AUROC cell is the Stage-1 default normalization. If two cells are within ~0.005 of each other, prefer the simpler/more standard recipe (N1 > N3 > N2 > N5 > N0 > N4) and record the tie in `docs/strategy/stage_1.md`.",
        ]
    )
    (sweep_root / "README.md").write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
