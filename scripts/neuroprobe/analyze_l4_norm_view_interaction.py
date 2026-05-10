"""Quick analyzer for L.4 norm × view × ref interaction sweep (2×2×2).

Reads `metrics.json` from each cell folder under `--sweep-root`, computes:
- per-cell mean AUROC + bootstrap CI
- 2×2×2 factor decomposition (ref × view × norm main effects + interactions)
- rank table

Outputs `interaction_analysis.{md,csv}` to the sweep root.

Usage:
    .venv/bin/python scripts/neuroprobe/analyze_l4_norm_view_interaction.py \
        --sweep-root reports/neuroprobe_stage0_l4_norm_view_interaction_2026_05_09/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


CELL_RE = re.compile(r"L\.4i\.(R\d+)x(I\d+[A-Z]*)x(N\d+)")
N_BOOT = 2000
RNG = np.random.default_rng(20260510)


def load_diagnostics(sweep_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for diag_path in sorted(sweep_root.rglob("diagnostics.csv")):
        cell = diag_path.parents[1].name
        m = CELL_RE.match(cell)
        if not m:
            continue
        ref, view, norm = m.group(1), m.group(2), m.group(3)
        df = pd.read_csv(diag_path)
        df["cell"] = cell
        df["ref_code"] = ref
        df["view_code"] = view
        df["norm_code"] = norm
        rows.append(df)
    if not rows:
        raise SystemExit(f"no diagnostics under {sweep_root}")
    return pd.concat(rows, ignore_index=True)


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    means = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = RNG.integers(0, values.size, values.size)
        means[k] = values[idx].mean()
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-root", type=Path, required=True)
    args = p.parse_args()
    root = args.sweep_root.resolve()

    df = load_diagnostics(root)
    cells = sorted(df["cell"].unique())

    rows: list[dict] = []
    for cell in cells:
        v = df.loc[df["cell"] == cell, "test_roc_auc"].to_numpy(dtype=float)
        m, lo, hi = bootstrap_mean_ci(v)
        sub = df.loc[df["cell"] == cell].iloc[0]
        rows.append({
            "cell": cell,
            "ref": sub["ref_code"],
            "view": sub["view_code"],
            "norm": sub["norm_code"],
            "n": int(v.size),
            "mean": m,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    cell_tab = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    cell_tab.to_csv(root / "interaction_cell_table.csv", index=False)

    grand = df["test_roc_auc"].mean()
    ref_main = df.groupby("ref_code")["test_roc_auc"].mean() - grand
    view_main = df.groupby("view_code")["test_roc_auc"].mean() - grand
    norm_main = df.groupby("norm_code")["test_roc_auc"].mean() - grand

    interactions: list[dict] = []
    for ref in df["ref_code"].unique():
        for view in df["view_code"].unique():
            for norm in df["norm_code"].unique():
                obs = df.loc[
                    (df["ref_code"] == ref) & (df["view_code"] == view) & (df["norm_code"] == norm),
                    "test_roc_auc",
                ].mean()
                additive = grand + ref_main[ref] + view_main[view] + norm_main[norm]
                interactions.append({
                    "ref": ref, "view": view, "norm": norm,
                    "observed": obs, "additive_pred": additive,
                    "residual": obs - additive,
                })
    inter_tab = pd.DataFrame(interactions)
    inter_tab["abs_residual"] = inter_tab["residual"].abs()
    inter_tab = inter_tab.sort_values("abs_residual", ascending=False).reset_index(drop=True)
    inter_tab.to_csv(root / "interaction_residuals.csv", index=False)

    lines: list[str] = ["# L.4 Norm × View × Ref Interaction Analysis", ""]
    lines.append(f"Grid: 2 × 2 × 2 = {len(cells)} cells. n_rows total = {len(df)}.")
    lines.append("")
    lines.append("## Cell ranking (mean AUROC, bootstrap 95% CI)")
    lines.append("")
    lines.append("| rank | cell | ref | view | norm | n | mean | CI |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, r in cell_tab.iterrows():
        lines.append(
            f"| {i + 1} | {r['cell']} | {r['ref']} | {r['view']} | {r['norm']} | "
            f"{r['n']} | {r['mean']:.4f} | [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] |"
        )
    lines.append("")
    lines.append(f"Grand mean = {grand:.4f}")
    lines.append("")
    lines.append("## Main effects (deviation from grand mean)")
    lines.append("")
    lines.append("| factor | level | Δ |")
    lines.append("|---|---|---|")
    for level, val in ref_main.items():
        lines.append(f"| ref | {level} | {val:+.4f} |")
    for level, val in view_main.items():
        lines.append(f"| view | {level} | {val:+.4f} |")
    for level, val in norm_main.items():
        lines.append(f"| norm | {level} | {val:+.4f} |")
    lines.append("")
    lines.append("## Interaction residuals (observed − additive prediction)")
    lines.append("")
    lines.append(f"max |residual| = {inter_tab['abs_residual'].max():.4f}")
    lines.append("")
    lines.append("| ref | view | norm | observed | additive | residual |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in inter_tab.iterrows():
        lines.append(
            f"| {r['ref']} | {r['view']} | {r['norm']} | "
            f"{r['observed']:.4f} | {r['additive_pred']:.4f} | {r['residual']:+.4f} |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    max_res = float(inter_tab["abs_residual"].max())
    if max_res < 0.005:
        lines.append(f"**Greedy hill-climb safe.** Max interaction |residual| = {max_res:.4f} < 0.005 — refs, views, and norm act as nearly independent factors. The L.1 → L.2 → L.3 sequential freezes do not lose meaningful AUROC.")
    elif max_res < 0.01:
        lines.append(f"**Greedy hill-climb borderline.** Max interaction |residual| = {max_res:.4f} ∈ [0.005, 0.01]. Marginal; report both the cell-level table and the marginals in stage_0.md.")
    else:
        lines.append(f"**Greedy hill-climb risky.** Max interaction |residual| = {max_res:.4f} ≥ 0.01. The independence assumption breaks at this scale; future sweeps should hold L.1 + L.2 jointly when sweeping L.3/L.4, not independently.")
    lines.append("")

    (root / "interaction_analysis.md").write_text("\n".join(lines))
    print(f"wrote {root / 'interaction_analysis.md'}")
    print(f"wrote {root / 'interaction_cell_table.csv'}")
    print(f"wrote {root / 'interaction_residuals.csv'}")


if __name__ == "__main__":
    main()
