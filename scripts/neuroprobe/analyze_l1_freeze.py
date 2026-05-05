"""Tier-A freeze analysis for the L.1 normalization sweep.

Reads `aggregate_diagnostics.csv` produced by
`collect_l1_normalization_sweep.py`, then emits:

- `per_task_heatmap.png` — cell × task AUROC matrix.
- `cell_ci_forest.png` — bootstrap 95% CIs of each cell's mean AUROC.
- `paired_tests.csv` — paired Wilcoxon signed-rank tests between every
  pair of cells, paired on (session, task).
- `freeze_analysis.md` — human-readable summary, including the
  per-session-stat transductive-leakage audit memo.
- `freeze_analysis.json` — machine-readable summary.

Usage:
    .venv/bin/python scripts/neuroprobe/analyze_l1_freeze.py \\
        --sweep-root reports/neuroprobe_stage0_l1_normalization_2026_05_05/
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402


N_BOOTSTRAP = 2000
RNG = np.random.default_rng(20260505)


CELL_RECIPE: dict[str, str] = {
    "L.1.N0": "per_window_z (BrainBERT/PopT recipe)",
    "L.1.N1": "train_set_fixed (upstream baseline)",
    "L.1.N2": "per_session_fixed",
    "L.1.N3": "train_set_scale_only",
    "L.1.N4": "none",
    "L.1.N5": "per_session_robust_mad",
    "L.1.N6": "train_set_robust_mad (Tier-B)",
    "L.1.N7": "per_session_robust_scale (Tier-B)",
    "L.1.N8": "per_channel_train_set_z (Tier-B)",
}

TRANSDUCTIVE_CELLS: set[str] = {"L.1.N2", "L.1.N5", "L.1.N7"}


def main() -> None:
    args = _parse_args()
    sweep_root = args.sweep_root.resolve()
    diag_path = sweep_root / "aggregate_diagnostics.csv"
    if not diag_path.exists():
        raise SystemExit(
            f"Missing {diag_path} — run collect_l1_normalization_sweep.py first."
        )
    df = pd.read_csv(diag_path)

    cells = sorted(df["cell"].unique())
    tasks = sorted(df["task"].unique())

    cell_task = build_cell_task(df, cells, tasks)
    ci = bootstrap_cis(df, cells)
    paired = paired_tests(df, cells)

    write_heatmap(sweep_root / "per_task_heatmap.png", cell_task, cells, tasks)
    write_forest(sweep_root / "cell_ci_forest.png", ci)
    paired.to_csv(sweep_root / "paired_tests.csv", index=False)

    payload = build_payload(cells, ci, paired, cell_task)
    (sweep_root / "freeze_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (sweep_root / "freeze_analysis.md").write_text(render_markdown(payload))
    print(f"wrote {sweep_root / 'per_task_heatmap.png'}")
    print(f"wrote {sweep_root / 'cell_ci_forest.png'}")
    print(f"wrote {sweep_root / 'paired_tests.csv'}")
    print(f"wrote {sweep_root / 'freeze_analysis.json'}")
    print(f"wrote {sweep_root / 'freeze_analysis.md'}")


# ---------- per-task cell × task matrix -----------------------------------------

def build_cell_task(df: pd.DataFrame, cells: list[str], tasks: list[str]) -> pd.DataFrame:
    pivot = (
        df.groupby(["cell", "task"], observed=True)["test_roc_auc"]
        .mean()
        .reset_index()
        .pivot(index="cell", columns="task", values="test_roc_auc")
        .reindex(index=cells, columns=tasks)
    )
    return pivot


def write_heatmap(path: Path, cell_task: pd.DataFrame, cells: list[str], tasks: list[str]) -> None:
    arr = cell_task.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(0.7 * len(tasks) + 3, 0.45 * len(cells) + 2.5))
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([f"{c}  {CELL_RECIPE.get(c, c)}" for c in cells], fontsize=8.5)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title("L.1 cell × task AUROC (test, mean across sessions)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="AUROC")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------- bootstrap CIs --------------------------------------------------------

def bootstrap_cis(df: pd.DataFrame, cells: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for cell in cells:
        v = df.loc[df["cell"] == cell, "test_roc_auc"].to_numpy()
        if v.size == 0:
            continue
        means = np.empty(N_BOOTSTRAP)
        for k in range(N_BOOTSTRAP):
            idx = RNG.integers(0, v.size, v.size)
            means[k] = v[idx].mean()
        out[cell] = {
            "n": int(v.size),
            "mean": float(v.mean()),
            "ci_lo": float(np.quantile(means, 0.025)),
            "ci_hi": float(np.quantile(means, 0.975)),
            "se_bootstrap": float(means.std(ddof=1)),
        }
    return out


def write_forest(path: Path, ci: dict[str, dict[str, float]]) -> None:
    cells = list(ci)
    means = np.array([ci[c]["mean"] for c in cells])
    los = np.array([ci[c]["ci_lo"] for c in cells])
    his = np.array([ci[c]["ci_hi"] for c in cells])
    err = np.vstack([means - los, his - means])
    order = np.argsort(means)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(cells) + 2))
    y = np.arange(len(cells))
    ax.errorbar(means[order], y, xerr=err[:, order], fmt="o",
                capsize=4, color="#1f4e8a")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{cells[i]}  {CELL_RECIPE.get(cells[i], cells[i])}"
                        for i in order])
    ax.set_xlabel("mean test AUROC (bootstrap 95% CI, N=2000 resamples)")
    ax.set_title("L.1 cell ranking with bootstrap CIs")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------- paired Wilcoxon ------------------------------------------------------

def paired_tests(df: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for c1, c2 in combinations(cells, 2):
        v1, v2 = paired_vectors(df, c1, c2)
        if v1.size == 0:
            continue
        diff = v1 - v2
        if np.allclose(diff, 0):
            stat, p = float("nan"), 1.0
        else:
            res = cast(Any, wilcoxon(v1, v2, zero_method="wilcox", alternative="two-sided"))
            stat, p = float(res.statistic), float(res.pvalue)
        rows.append({
            "cell_a": c1,
            "cell_b": c2,
            "n_paired_obs": int(v1.size),
            "mean_a": float(v1.mean()),
            "mean_b": float(v2.mean()),
            "mean_diff_a_minus_b": float(v1.mean() - v2.mean()),
            "wilcoxon_stat": stat,
            "wilcoxon_p": p,
        })
    return pd.DataFrame(rows).sort_values("wilcoxon_p")


def paired_vectors(df: pd.DataFrame, c1: str, c2: str) -> tuple[np.ndarray, np.ndarray]:
    a = df[df["cell"] == c1].set_index(["subject_id", "trial_id", "task"])["test_roc_auc"]
    b = df[df["cell"] == c2].set_index(["subject_id", "trial_id", "task"])["test_roc_auc"]
    common = a.index.intersection(b.index)
    return a.loc[common].to_numpy(), b.loc[common].to_numpy()


# ---------- payload + markdown ---------------------------------------------------

def build_payload(
    cells: list[str],
    ci: dict[str, dict[str, float]],
    paired: pd.DataFrame,
    cell_task: pd.DataFrame,
) -> dict:
    cell_task_dict = {
        c: {t: (None if pd.isna(cell_task.at[c, t]) else float(cell_task.at[c, t]))
            for t in cell_task.columns}
        for c in cell_task.index
    }
    return {
        "n_cells": len(cells),
        "cells": cells,
        "bootstrap_ci": ci,
        "paired_wilcoxon": paired.to_dict(orient="records"),
        "cell_task_auroc": cell_task_dict,
        "rank_by_mean": sorted(
            [(c, ci[c]["mean"]) for c in ci], key=lambda x: -x[1]
        ),
    }


def render_markdown(payload: dict) -> str:
    lines: list[str] = ["# L.1 Freeze Analysis (Tier A)", ""]

    lines.append("## Cell ranking (mean AUROC, bootstrap 95% CI)")
    lines.append("")
    lines.append("| rank | cell | recipe | n | mean | CI_lo | CI_hi |")
    lines.append("|---|---|---|---|---|---|---|")
    for rank, (cell, mean) in enumerate(payload["rank_by_mean"], start=1):
        ci = payload["bootstrap_ci"][cell]
        lines.append(
            f"| {rank} | {cell} | {CELL_RECIPE.get(cell, cell)} | "
            f"{ci['n']} | {mean:.4f} | {ci['ci_lo']:.4f} | {ci['ci_hi']:.4f} |"
        )

    lines.append("")
    lines.append("## Paired Wilcoxon (top significance, all comparisons in `paired_tests.csv`)")
    lines.append("")
    lines.append("| cell A | cell B | n pairs | Δ (A − B) | p |")
    lines.append("|---|---|---|---|---|")
    for row in payload["paired_wilcoxon"][:15]:
        lines.append(
            f"| {row['cell_a']} | {row['cell_b']} | {row['n_paired_obs']} | "
            f"{row['mean_diff_a_minus_b']:+.4f} | {row['wilcoxon_p']:.2e} |"
        )

    lines.append("")
    lines.append("## Per-task winners")
    lines.append("")
    lines.append("Best cell per task by mean AUROC:")
    lines.append("")
    cell_task = payload["cell_task_auroc"]
    by_task: dict[str, tuple[str, float]] = {}
    for cell, task_dict in cell_task.items():
        for task, val in task_dict.items():
            if val is None:
                continue
            cur = by_task.get(task)
            if cur is None or val > cur[1]:
                by_task[task] = (cell, val)
    lines.append("| task | best cell | best AUROC |")
    lines.append("|---|---|---|")
    for task in sorted(by_task):
        cell, val = by_task[task]
        lines.append(f"| {task} | {cell} | {val:.4f} |")

    lines.append("")
    lines.append("## Transductive-leakage audit")
    lines.append("")
    lines.append(
        "**Finding**: cells in `{N2, N5, N7}` use the **test session's own**\n"
        "feature distribution to compute their normalization statistics:\n"
        "\n"
        "```python\n"
        "elif normalization == \"per_session_fixed\":\n"
        "    train_scaler = StandardScaler(copy=False).fit(X_train)\n"
        "    test_scaler  = StandardScaler(copy=False).fit(X_test)  # ← uses test features\n"
        "    X_train = train_scaler.transform(X_train)\n"
        "    X_test  = test_scaler.transform(X_test)\n"
        "```\n"
        "\n"
        "Same shape for `_mad_normalize(X_test)` and `_mad_scale_only(X_test)`.\n"
        "\n"
        "- **Test labels are never used.** This is not label leakage.\n"
        "- **Test feature distribution IS used.** This is *transductive normalization*\n"
        "  (test-time adaptation) — common practice and arguably valid in deployment\n"
        "  because the same operation would happen on a freshly-recorded session.\n"
        "- **It is, however, an unfair comparison axis** with the inductive cells\n"
        "  (N0, N1, N3, N4, N6, N8) which never see the test distribution.\n"
        "\n"
        "**Implications for freeze**:\n"
        "\n"
        "1. If a transductive cell wins, document explicitly that v14 must replicate\n"
        "   the same test-time normalization at inference (per-session stats fit on\n"
        "   the held-out session's own data).\n"
        "2. The cleanest A/B for 'is per-session scope load-bearing?' is\n"
        "   `train_set_robust_mad (N6, inductive)` vs `per_session_robust_mad (N5, transductive)`.\n"
        "   If N5 ≫ N6, the win is mostly the transductive trick. If N5 ≈ N6, it's\n"
        "   the robust statistic doing the work and per-session scope is decorative.\n"
        "3. Even if a transductive cell wins L.1's CrossSession protocol, freezing it\n"
        "   creates a constraint on Stage-1 v14: the model's preprocessing must\n"
        "   accept a test session as a whole (no streaming inference per single\n"
        "   trial). Worth flagging in `docs/strategy/stage_1.md`.\n"
        "\n"
        "**Cells classified inductive vs transductive**:\n"
        "\n"
        "| inductive (no test feature access) | transductive (uses test feature stats) |\n"
        "|---|---|\n"
        "| N0 per_window_z (per-sample, no fit) | **N2 per_session_fixed** |\n"
        "| N1 train_set_fixed | **N5 per_session_robust_mad** |\n"
        "| N3 train_set_scale_only | **N7 per_session_robust_scale** (new) |\n"
        "| N4 none | |\n"
        "| N6 train_set_robust_mad (new) | |\n"
        "| N8 per_channel_train_set_z (new) | |"
    )
    lines.append("")

    lines.append("## Tier-B decomposition logic (once N6/N7/N8 land)")
    lines.append("")
    lines.append(
        "| Question | Comparison | If yes |\n"
        "|---|---|---|\n"
        "| Is N5's win the **robust statistic** or the **per-session scope**? | N5 vs N6 | N5 ≈ N6 → robust stat is the win; per-session is moot (and freeze N6, the inductive analog) |\n"
        "| Does **centering** matter at any robust scope? | N5 vs N7 | N5 ≈ N7 → demean is decorative everywhere; freeze the simpler scale-only recipe |\n"
        "| Is the standard iEEG **per-channel** recipe competitive? | N8 vs N1 | N8 > N1 → freeze per-channel scope; matters for v14 token-level normalization |\n"
        "| Is per-window z's −5.4 pp tax driven by **a few amplitude-sensitive tasks** (volume/pitch)? | per-task heatmap row N0 | yes → frame the headline as task-conditional, not universal |"
    )
    lines.append("")

    lines.append("## Recommended freeze decision tree")
    lines.append("")
    lines.append(
        "1. **If any cell's CI fully strictly dominates all others' CIs**: freeze it.\n"
        "2. **Else, if top-2 CIs overlap**: prefer the inductive cell (N1/N6/N8) over\n"
        "   the transductive one (N2/N5/N7). Document the equivalence and the\n"
        "   reason for choosing the simpler/inductive option.\n"
        "3. **Else, if top-3 CIs overlap**: freeze N1 (upstream baseline) by default.\n"
        "   Note in `docs/strategy/stage_1.md` that L.1 was inconclusive between\n"
        "   `train_set_fixed`, `per_session_*`, and `per_channel`.\n"
        "4. **Always**: re-confirm winner on D.0a CrossSubject binary (Tier C) before\n"
        "   final freeze. Cross-subject distribution shift may swap the winner."
    )
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-root", type=Path, required=True)
    return p.parse_args()


if __name__ == "__main__":
    main()
