"""Collect L.2 reference × input-view sweep results into aggregate report.

Mirrors `collect_l1_normalization_sweep.py` but for the L.2 sweep grid.
Each cell = (reference, input_view). The L.2 baseline = `L.2.R4xI2`
(shaft_laplacian × stft_log_power), which is upstream's `Lap+spec` and the
linear D.0 default (multiclass mean ≈ 0.611 from Neuroprobe rebuttal).

Outputs at `--sweep-root`:
- `aggregate_diagnostics.csv`           — every cell × session × task × fold row
- `aggregate_summary_by_cell.csv`       — cell mean AUROC across sessions × tasks
- `aggregate_summary_by_cell_task.csv`  — cell × task means
- `cell_task_heatmap.png`               — cell × task AUROC matrix
- `cell_aggregate_bar.png`              — bar chart vs D.0 baseline reference line
- `paired_tests.csv`                    — paired Wilcoxon between every cell pair
- `summary.json`                        — headline numbers
- `README.md`                           — interpretation + freeze decision tree
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CELL_ORDER: tuple[str, ...] = (
    "L.2.R0xI0",
    "L.2.R0xI2",
    "L.2.R3xI2",
    "L.2.R3xI3",
    "L.2.R3xI4",
    "L.2.R4xI2",
    "L.2.R4xI3",
    "L.2.R4xI4",
    "L.2.R4xI5",
)

CELL_LABEL: dict[str, str] = {
    "L.2.R0xI0": "raw × voltage",
    "L.2.R0xI2": "raw × stft_log",
    "L.2.R3xI2": "bipolar × stft_log",
    "L.2.R3xI3": "bipolar × HG",
    "L.2.R3xI4": "bipolar × multi-band",
    "L.2.R4xI2": "shaftLap × stft_log [D.0]",
    "L.2.R4xI3": "shaftLap × HG",
    "L.2.R4xI4": "shaftLap × multi-band",
    "L.2.R4xI5": "shaftLap × wavelet",
}

BASELINE_CELL = "L.2.R4xI2"


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
    sbc = cast(
        pd.DataFrame,
        diagnostics.groupby("cell")["test_roc_auc"]
        .agg(["mean", "std", "count"])
        .reset_index(),
    )
    summary_by_cell = sbc.rename(columns=rename_map).sort_values(
        "mean_test_roc_auc", ascending=False
    )
    summary_by_cell.to_csv(sweep_root / "aggregate_summary_by_cell.csv", index=False)

    sbct = cast(
        pd.DataFrame,
        diagnostics.groupby(["cell", "task"])["test_roc_auc"]
        .agg(["mean", "std", "count"])
        .reset_index(),
    )
    summary_by_cell_task = sbct.rename(columns=rename_map)
    summary_by_cell_task.to_csv(
        sweep_root / "aggregate_summary_by_cell_task.csv", index=False
    )

    pivot = summary_by_cell_task.pivot_table(
        index="cell", columns="task", values="mean_test_roc_auc"
    ).reindex(index=[c for c in CELL_ORDER if c in summary_by_cell["cell"].values])

    paired = compute_paired_tests(diagnostics)
    paired.to_csv(sweep_root / "paired_tests.csv", index=False)

    plot_cell_task_heatmap(pivot, sweep_root / "cell_task_heatmap.png")
    plot_cell_aggregate_bar(summary_by_cell, sweep_root / "cell_aggregate_bar.png")

    summary = build_summary(summary_by_cell, paired)
    (sweep_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    write_readme(sweep_root, summary, summary_by_cell, paired)
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


def compute_paired_tests(diagnostics: pd.DataFrame) -> pd.DataFrame:
    cells = sorted(diagnostics["cell"].unique())
    keys = ["subject_id", "trial_id", "task", "fold"]
    available = [k for k in keys if k in diagnostics.columns]
    if not available:
        return pd.DataFrame(columns=["cell_a", "cell_b", "n_pairs", "delta", "p"])

    out: list[dict[str, object]] = []
    for a, b in combinations(cells, 2):
        df_a = diagnostics[diagnostics["cell"] == a].set_index(available)["test_roc_auc"]
        df_b = diagnostics[diagnostics["cell"] == b].set_index(available)["test_roc_auc"]
        common = df_a.index.intersection(df_b.index)
        if len(common) < 5:
            continue
        a_vals = df_a.loc[common].astype(float).values
        b_vals = df_b.loc[common].astype(float).values
        delta = float(np.mean(a_vals - b_vals))
        try:
            res = cast(Any, wilcoxon(a_vals, b_vals, zero_method="wilcox"))
            p = float(res.pvalue)
        except ValueError:
            p = float("nan")
        out.append(
            {"cell_a": a, "cell_b": b, "n_pairs": len(common), "delta": delta, "p": p}
        )
    return pd.DataFrame(out).sort_values("p")


def plot_cell_task_heatmap(pivot: pd.DataFrame, out_path: Path) -> None:
    if pivot.empty:
        return
    fig, ax = plt.subplots(
        figsize=(max(8, 0.6 * len(pivot.columns) + 4), 4 + 0.4 * len(pivot.index))
    )
    arr = pivot.values.astype(float)
    im = ax.imshow(
        arr,
        aspect="auto",
        cmap="viridis",
        vmin=0.4,
        vmax=max(0.7, np.nanmax(arr)),
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(
        [f"{c} ({CELL_LABEL.get(c, '?')})" for c in pivot.index], fontsize=9
    )
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v < 0.6 else "black",
                )
    if BASELINE_CELL in pivot.index:
        b_idx = list(pivot.index).index(BASELINE_CELL)
        ax.axhline(b_idx - 0.5, color="red", linewidth=1.2, linestyle="--")
        ax.axhline(b_idx + 0.5, color="red", linewidth=1.2, linestyle="--")
    ax.set_title(
        "L.2 cell × task mean test AUROC (red dashed = D.0 baseline shaftLap × stft_log)"
    )
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
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.arange(len(df))
    ax.bar(
        xs,
        df["mean_test_roc_auc"],
        yerr=df["std_test_roc_auc"],
        capsize=4,
        color="steelblue",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{c}\n{CELL_LABEL.get(c, '?')}" for c in df.index],
        fontsize=8,
        rotation=15,
        ha="right",
    )
    if BASELINE_CELL in df.index:
        baseline = float(df.loc[BASELINE_CELL, "mean_test_roc_auc"])
        ax.axhline(
            baseline,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"D.0 baseline ({BASELINE_CELL})={baseline:.3f}",
        )
        ax.legend(loc="lower right")
    ax.set_ylabel("mean test AUROC (sessions × tasks × folds)")
    ax.set_title("L.2 reference × input-view sweep — cell-level aggregate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def build_summary(
    summary_by_cell: pd.DataFrame, paired: pd.DataFrame
) -> dict[str, object]:
    by_cell = {
        row["cell"]: {
            "label": CELL_LABEL.get(row["cell"], "?"),
            "mean_test_roc_auc": float(row["mean_test_roc_auc"]),
            "std_test_roc_auc": (
                float(row["std_test_roc_auc"])
                if not bool(pd.isna(row["std_test_roc_auc"]))
                else None
            ),
            "n_rows": int(row["n"]),
        }
        for _, row in summary_by_cell.iterrows()
    }
    baseline = by_cell.get(BASELINE_CELL, {}).get("mean_test_roc_auc")
    winner = summary_by_cell.iloc[0]
    delta_winner_minus_baseline = (
        float(winner["mean_test_roc_auc"]) - float(baseline)
        if baseline is not None
        else None
    )
    return {
        "baseline_cell": BASELINE_CELL,
        "baseline_auc": baseline,
        "winner_cell": str(winner["cell"]),
        "winner_auc": float(winner["mean_test_roc_auc"]),
        "winner_minus_baseline": delta_winner_minus_baseline,
        "load_bearing_threshold": 0.02,
        "load_bearing": (
            delta_winner_minus_baseline is not None
            and delta_winner_minus_baseline >= 0.02
        ),
        "by_cell": by_cell,
        "n_cells": int(summary_by_cell["cell"].nunique()),
        "n_paired_tests": int(len(paired.index)),
    }


def write_readme(
    sweep_root: Path,
    summary: dict[str, object],
    summary_by_cell: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    lines = [
        "# Neuroprobe Stage 0 L.2 Reference × Input-View Sweep",
        "",
        "Linear baseline ablating reference scheme × spectral view. Default split: CrossSession multiclass on all 12 BT Lite sessions × 15 tasks (D.0b protocol).",
        "Normalization recipe is held at the L.1 winner (default `train_set_fixed`).",
        "",
        "## Headline",
        "",
        f"- Baseline cell: **{summary['baseline_cell']}** (`{CELL_LABEL.get(BASELINE_CELL, '?')}`) AUROC = {summary['baseline_auc']:.4f}",
        f"- Winner cell: **{summary['winner_cell']}** (`{CELL_LABEL.get(cast(str, summary['winner_cell']), '?')}`) AUROC = {summary['winner_auc']:.4f}",
        f"- Winner − baseline: **{summary['winner_minus_baseline']:+.4f}**",
        f"- Load-bearing threshold: |Δ| ≥ 0.02. Result: **{'load-bearing' if summary['load_bearing'] else 'not load-bearing'}**.",
        "",
        "## Cell aggregates (sorted by mean AUROC)",
        "",
        "| Cell | Recipe | Mean AUROC | SD | N |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in summary_by_cell.iterrows():
        sd = (
            f"{row['std_test_roc_auc']:.4f}"
            if not bool(pd.isna(row["std_test_roc_auc"]))
            else ""
        )
        cell_id = cast(str, row["cell"])
        lines.append(
            f"| {cell_id} | {CELL_LABEL.get(cell_id, '?')} | "
            f"{row['mean_test_roc_auc']:.4f} | {sd} | {int(row['n'])} |"
        )
    lines.append("")
    lines.append("## Top 10 paired Wilcoxon tests (most significant)")
    lines.append("")
    lines.append("| cell A | cell B | n pairs | Δ (A − B) | p |")
    lines.append("|---|---|---:|---:|---:|")
    for _, row in paired.head(10).iterrows():
        lines.append(
            f"| {row['cell_a']} | {row['cell_b']} | {int(row['n_pairs'])} | "
            f"{row['delta']:+.4f} | {row['p']:.2e} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "- `aggregate_diagnostics.csv` — every cell × session × task × fold row",
            "- `aggregate_summary_by_cell.csv` — cell mean across sessions × tasks",
            "- `aggregate_summary_by_cell_task.csv` — cell × task means",
            "- `paired_tests.csv` — paired Wilcoxon for every cell pair",
            "- `cell_task_heatmap.png` — visual summary",
            "- `cell_aggregate_bar.png` — cell-level bar chart with D.0 reference line",
            "",
            "## Stage-1 freeze",
            "Highest mean-AUROC cell is the Stage-1 default reference × view, **provided** "
            "its CI strictly dominates the baseline OR a paired Wilcoxon vs baseline is "
            "p < 0.001 with positive Δ ≥ 0.005. Otherwise default to D.0 baseline "
            "(`L.2.R4xI2 / shaftLap × stft_log`) for upstream parity. Record the "
            "decision in `docs/strategy/stage_1.md` along with the per-task "
            "winner table from `aggregate_summary_by_cell_task.csv`.",
        ]
    )
    (sweep_root / "README.md").write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
