"""Tier-A freeze analysis for the L.2 reference × input-view sweep.

Reads `aggregate_diagnostics.csv` produced by
`collect_l2_reference_view_sweep.py`, then emits:

- `cell_ci_forest.png` — bootstrap 95% CIs of each cell's mean AUROC.
- `factor_marginals.png` — reference-marginal and view-marginal effects with CIs.
- `freeze_analysis.md` — human-readable summary with freeze decision tree.
- `freeze_analysis.json` — machine-readable summary including bootstrap stats,
  paired tests, and reference/view marginal decomposition.

The collector already writes `cell_task_heatmap.png`, `cell_aggregate_bar.png`,
and `paired_tests.csv` — this script adds bootstrap CIs and the factor
decomposition that turns the 9-cell grid into a "does ref or view matter more"
verdict.

Usage:
    .venv/bin/python scripts/neuroprobe/analyze_l2_freeze.py \\
        --sweep-root reports/neuroprobe_stage0_l2_neuralset_2026_05_06/
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
RNG = np.random.default_rng(20260506)
BASELINE_CELL = "L.2.R4xI2"  # upstream Lap+spec; Neuroprobe rebuttal multiclass = 0.611

CELL_LABEL: dict[str, str] = {
    "L.2.R0xI0": "raw × voltage",
    "L.2.R0xI2": "raw × stft_abs",
    "L.2.R0xI3": "raw × HG envelope",
    "L.2.R1xI0": "globalCAR × voltage",
    "L.2.R1xI2": "globalCAR × stft_abs",
    "L.2.R1xI3": "globalCAR × HG envelope",
    "L.2.R2xI0": "shaftCAR × voltage",
    "L.2.R2xI2": "shaftCAR × stft_abs",
    "L.2.R2xI3": "shaftCAR × HG envelope",
    "L.2.R3xI0": "bipolar × voltage",
    "L.2.R3xI2": "bipolar × stft_abs",
    "L.2.R3xI3": "bipolar × HG envelope",
    "L.2.R4xI0": "shaftLap × voltage",
    "L.2.R4xI1": "shaftLap × low-LFP",
    "L.2.R4xI2": "shaftLap × stft_abs [D.0 baseline]",
    "L.2.R4xI2L": "shaftLap × log-STFT",
    "L.2.R4xI3": "shaftLap × HG envelope (privileged)",
    "L.2.R4xI3W": "shaftLap × wide-HG (70-250)",
    "L.2.R4xI4": "shaftLap × multi-band log-power",
    "L.2.R4xI5": "shaftLap × wavelet",
    "L.2.R4xI6": "shaftLap × theta-phase",
    "L.2.R5xI0": "median × voltage",
    "L.2.R5xI2": "median × stft_abs",
    "L.2.R5xI3": "median × HG envelope",
}

REF_LABEL: dict[str, str] = {
    "R0": "raw",
    "R1": "global CAR",
    "R2": "shaft CAR",
    "R3": "bipolar (adjacent within-shaft)",
    "R4": "shaft Laplacian",
    "R5": "median",
}
VIEW_LABEL: dict[str, str] = {
    "I0": "raw voltage",
    "I1": "low-LFP (<30 Hz)",
    "I2": "STFT magnitude",
    "I2L": "log STFT",
    "I3": "HG envelope (70-150 Hz)",
    "I3W": "wide HG (70-250 Hz)",
    "I4": "multi-band log-power",
    "I5": "wavelet",
    "I6": "theta-band phase",
}


def main() -> None:
    args = _parse_args()
    sweep_root = args.sweep_root.resolve()
    diag_path = sweep_root / "aggregate_diagnostics.csv"
    if not diag_path.exists():
        raise SystemExit(
            f"Missing {diag_path} — run collect_l2_reference_view_sweep.py first."
        )
    df = pd.read_csv(diag_path)
    if "cell" not in df.columns:
        raise SystemExit("aggregate_diagnostics.csv missing 'cell' column")

    if args.tier_a_only:
        tier_a = sorted(c for c in df["cell"].unique() if _is_tier_a(c))
        if not tier_a:
            raise SystemExit("No Tier-A L.2 cells found in diagnostics.")
    else:
        tier_a = sorted(
            c for c in df["cell"].unique() if isinstance(c, str) and c.startswith("L.2.R")
        )
        if not tier_a:
            raise SystemExit("No L.2 cells found in diagnostics.")
    df_a = cast(pd.DataFrame, df[df["cell"].isin(tier_a)]).copy()
    cell_series = df_a["cell"].astype(str)
    df_a["ref_code"] = cell_series.str.extract(r"L\.2\.(R\d+)x")[0]
    df_a["view_code"] = cell_series.str.extract(r"x(I\d+[A-Z]*)$")[0]

    ci = bootstrap_cis(df_a, tier_a)
    paired = paired_tests(df_a, tier_a)
    ref_marginal = marginal_cis(df_a, factor="ref_code")
    view_marginal = marginal_cis(df_a, factor="view_code")
    interaction = factor_interaction(df_a)

    write_forest(sweep_root / "cell_ci_forest.png", ci)
    write_marginals(
        sweep_root / "factor_marginals.png", ref_marginal, view_marginal
    )

    payload = build_payload(
        tier_a, ci, paired, ref_marginal, view_marginal, interaction, df_a
    )
    (sweep_root / "freeze_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (sweep_root / "freeze_analysis.md").write_text(render_markdown(payload))
    print(f"wrote {sweep_root / 'cell_ci_forest.png'}")
    print(f"wrote {sweep_root / 'factor_marginals.png'}")
    print(f"wrote {sweep_root / 'freeze_analysis.json'}")
    print(f"wrote {sweep_root / 'freeze_analysis.md'}")


def _is_tier_a(cell: str) -> bool:
    if not cell.startswith("L.2.R"):
        return False
    ref = cell.split(".")[2].split("x")[0]
    view = cell.split("x")[-1]
    return ref in {"R0", "R3", "R4"} and view in {"I0", "I2", "I3"}


# ---------- bootstrap CIs --------------------------------------------------------

def bootstrap_cis(df: pd.DataFrame, cells: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for cell in cells:
        v = df.loc[df["cell"] == cell, "test_roc_auc"].to_numpy(dtype=float)
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
    if not cells:
        return
    means = np.array([ci[c]["mean"] for c in cells])
    los = np.array([ci[c]["ci_lo"] for c in cells])
    his = np.array([ci[c]["ci_hi"] for c in cells])
    err = np.vstack([means - los, his - means])
    order = np.argsort(means)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(cells) + 2.2))
    y = np.arange(len(cells))
    colors = [
        "#c0392b" if cells[i] == BASELINE_CELL else "#1f4e8a" for i in order
    ]
    for j, idx in enumerate(order):
        ax.errorbar(
            means[idx], y[j], xerr=[[err[0, idx]], [err[1, idx]]],
            fmt="o", capsize=4, color=colors[j],
        )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{cells[i]}  {CELL_LABEL.get(cells[i], cells[i])}" for i in order],
        fontsize=8.5,
    )
    if BASELINE_CELL in ci:
        ax.axvline(
            ci[BASELINE_CELL]["mean"], color="#c0392b", linestyle="--",
            linewidth=1.0, alpha=0.6, label=f"{BASELINE_CELL} mean",
        )
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("mean test AUROC (bootstrap 95% CI, N=2000 resamples)")
    ax.set_title("L.2 Tier-A cell ranking with bootstrap CIs")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------- factor marginals ----------------------------------------------------

def marginal_cis(df: pd.DataFrame, factor: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for level in sorted(df[factor].unique()):
        v = df.loc[df[factor] == level, "test_roc_auc"].to_numpy(dtype=float)
        if v.size == 0:
            continue
        means = np.empty(N_BOOTSTRAP)
        for k in range(N_BOOTSTRAP):
            idx = RNG.integers(0, v.size, v.size)
            means[k] = v[idx].mean()
        out[level] = {
            "n": int(v.size),
            "mean": float(v.mean()),
            "ci_lo": float(np.quantile(means, 0.025)),
            "ci_hi": float(np.quantile(means, 0.975)),
        }
    return out


def write_marginals(
    path: Path,
    ref_marginal: dict[str, dict[str, float]],
    view_marginal: dict[str, dict[str, float]],
) -> None:
    if not ref_marginal or not view_marginal:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, marg, name, label_map in [
        (axes[0], ref_marginal, "reference", REF_LABEL),
        (axes[1], view_marginal, "input view", VIEW_LABEL),
    ]:
        levels = list(marg.keys())
        means = np.array([marg[l]["mean"] for l in levels])
        los = np.array([marg[l]["ci_lo"] for l in levels])
        his = np.array([marg[l]["ci_hi"] for l in levels])
        err = np.vstack([means - los, his - means])
        order = np.argsort(means)[::-1]
        y = np.arange(len(levels))
        ax.errorbar(
            means[order], y, xerr=err[:, order],
            fmt="o", capsize=4, color="#1f4e8a",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{levels[i]}  {label_map.get(levels[i], levels[i])}" for i in order],
            fontsize=9,
        )
        ax.set_xlabel("mean test AUROC (95% CI)")
        ax.set_title(f"L.2 marginal effect: {name}")
        ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def factor_interaction(df: pd.DataFrame) -> dict[str, Any]:
    """Quantify ref × view interaction. If the marginal effects fully explain
    the cell means, the additive prediction matches; large deviations imply
    interaction (e.g., HG only helps with shaft-Laplacian)."""
    grand = float(df["test_roc_auc"].mean())
    ref_means: dict[str, float] = {
        str(k): float(v)
        for k, v in df.groupby("ref_code", observed=True)["test_roc_auc"].mean().items()
    }
    view_means: dict[str, float] = {
        str(k): float(v)
        for k, v in df.groupby("view_code", observed=True)["test_roc_auc"].mean().items()
    }
    ref_eff = {k: v - grand for k, v in ref_means.items()}
    view_eff = {k: v - grand for k, v in view_means.items()}
    cell_means_df = (
        df.groupby(["ref_code", "view_code"], observed=True)["test_roc_auc"]
        .mean()
        .reset_index()
    )
    cell_obs: dict[tuple[str, str], float] = {
        (str(row["ref_code"]), str(row["view_code"])): float(row["test_roc_auc"])
        for _, row in cell_means_df.iterrows()
    }
    rows: list[dict[str, object]] = []
    for (rc, vc), obs in cell_obs.items():
        pred = grand + ref_eff[rc] + view_eff[vc]
        residual = obs - pred
        rows.append({
            "ref_code": rc, "view_code": vc,
            "observed": obs, "additive_pred": pred,
            "interaction_residual": residual,
        })
    interaction_df = pd.DataFrame(rows).sort_values(
        "interaction_residual", key=lambda s: s.abs(), ascending=False
    )
    return {
        "grand_mean": grand,
        "ref_main_effect": ref_eff,
        "view_main_effect": view_eff,
        "interaction_residuals": interaction_df.to_dict(orient="records"),
        "max_abs_residual": float(interaction_df["interaction_residual"].abs().max()),
    }


# ---------- paired Wilcoxon ------------------------------------------------------

def paired_tests(df: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    keys = ["subject_id", "trial_id", "task", "fold"]
    available = [k for k in keys if k in df.columns]
    if not available:
        return pd.DataFrame(columns=pd.Index(
            ["cell_a", "cell_b", "n_paired_obs", "mean_a", "mean_b",
             "mean_diff_a_minus_b", "wilcoxon_stat", "wilcoxon_p"]
        ))
    rows: list[dict[str, object]] = []
    for c1, c2 in combinations(cells, 2):
        a = df[df["cell"] == c1].set_index(available)["test_roc_auc"]
        b = df[df["cell"] == c2].set_index(available)["test_roc_auc"]
        common = a.index.intersection(b.index)
        if len(common) < 5:
            continue
        v1 = a.loc[common].astype(float).to_numpy()
        v2 = b.loc[common].astype(float).to_numpy()
        diff = v1 - v2
        if np.allclose(diff, 0):
            stat, p = float("nan"), 1.0
        else:
            res = cast(Any, wilcoxon(v1, v2, zero_method="wilcox", alternative="two-sided"))
            stat, p = float(res.statistic), float(res.pvalue)
        rows.append({
            "cell_a": c1, "cell_b": c2,
            "n_paired_obs": int(v1.size),
            "mean_a": float(v1.mean()), "mean_b": float(v2.mean()),
            "mean_diff_a_minus_b": float(v1.mean() - v2.mean()),
            "wilcoxon_stat": stat, "wilcoxon_p": p,
        })
    if not rows:
        return pd.DataFrame(columns=pd.Index(
            ["cell_a", "cell_b", "n_paired_obs", "mean_a", "mean_b",
             "mean_diff_a_minus_b", "wilcoxon_stat", "wilcoxon_p"]
        ))
    return pd.DataFrame(rows).sort_values("wilcoxon_p")


# ---------- payload + markdown ---------------------------------------------------

def build_payload(
    cells: list[str],
    ci: dict[str, dict[str, float]],
    paired: pd.DataFrame,
    ref_marginal: dict[str, dict[str, float]],
    view_marginal: dict[str, dict[str, float]],
    interaction: dict[str, Any],
    df_a: pd.DataFrame,
) -> dict[str, Any]:
    n_sessions = int(df_a.groupby(["subject_id", "trial_id"]).ngroups) if (
        "subject_id" in df_a.columns and "trial_id" in df_a.columns
    ) else None
    return {
        "n_cells": len(cells),
        "cells": cells,
        "n_sessions": n_sessions,
        "baseline_cell": BASELINE_CELL,
        "baseline_mean": ci.get(BASELINE_CELL, {}).get("mean"),
        "bootstrap_ci": ci,
        "paired_wilcoxon": paired.to_dict(orient="records"),
        "rank_by_mean": sorted(
            [(c, ci[c]["mean"]) for c in ci], key=lambda x: -x[1]
        ),
        "reference_marginal": ref_marginal,
        "view_marginal": view_marginal,
        "factor_interaction": interaction,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    n_cells = payload.get("n_cells")
    title = (
        "# L.2 Freeze Analysis (Tier A)" if n_cells == 9
        else f"# L.2 Freeze Analysis ({n_cells} cells, exhaustive)"
    )
    lines: list[str] = [title, ""]

    n_sess = payload.get("n_sessions")
    lines.append(
        f"Grid: {n_cells} cells. "
        f"{n_sess if n_sess is not None else 'N/A'} sessions × 15 tasks per cell."
    )
    lines.append("")
    lines.append(
        f"Baseline (upstream Lap+spec): **{payload['baseline_cell']}** "
        f"= {payload['baseline_mean']:.4f} mean AUROC.\n"
    )

    lines.append("## Cell ranking (mean AUROC, bootstrap 95% CI)")
    lines.append("")
    lines.append("| rank | cell | recipe | n | mean | CI_lo | CI_hi | Δ vs baseline |")
    lines.append("|---|---|---|---|---|---|---|---|")
    base = payload.get("baseline_mean") or 0.0
    for rank, (cell, mean) in enumerate(payload["rank_by_mean"], start=1):
        ci = payload["bootstrap_ci"][cell]
        marker = " ★" if cell == payload["baseline_cell"] else ""
        lines.append(
            f"| {rank} | {cell}{marker} | {CELL_LABEL.get(cell, cell)} | "
            f"{ci['n']} | {mean:.4f} | {ci['ci_lo']:.4f} | {ci['ci_hi']:.4f} | "
            f"{(mean - base):+.4f} |"
        )

    lines.append("")
    lines.append("## Reference marginal (averaged over views)")
    lines.append("")
    lines.append("| ref | recipe | n | mean | CI_lo | CI_hi |")
    lines.append("|---|---|---|---|---|---|")
    for level in sorted(payload["reference_marginal"], key=lambda l: -payload["reference_marginal"][l]["mean"]):
        rm = payload["reference_marginal"][level]
        lines.append(
            f"| {level} | {REF_LABEL.get(level, level)} | {rm['n']} | "
            f"{rm['mean']:.4f} | {rm['ci_lo']:.4f} | {rm['ci_hi']:.4f} |"
        )

    lines.append("")
    lines.append("## View marginal (averaged over references)")
    lines.append("")
    lines.append("| view | recipe | n | mean | CI_lo | CI_hi |")
    lines.append("|---|---|---|---|---|---|")
    for level in sorted(payload["view_marginal"], key=lambda l: -payload["view_marginal"][l]["mean"]):
        vm = payload["view_marginal"][level]
        lines.append(
            f"| {level} | {VIEW_LABEL.get(level, level)} | {vm['n']} | "
            f"{vm['mean']:.4f} | {vm['ci_lo']:.4f} | {vm['ci_hi']:.4f} |"
        )

    inter = payload["factor_interaction"]
    lines.append("")
    lines.append(
        f"## Factor interaction (max |residual| = {inter['max_abs_residual']:.4f})"
    )
    lines.append("")
    lines.append(
        "Residual = observed cell mean − (grand + ref_main + view_main). "
        "Large residuals = the ref × view choice cannot be reduced to two "
        "independent main effects. Top 3:"
    )
    lines.append("")
    lines.append("| ref | view | observed | additive_pred | residual |")
    lines.append("|---|---|---|---|---|")
    for row in inter["interaction_residuals"][:3]:
        lines.append(
            f"| {row['ref_code']} | {row['view_code']} | {row['observed']:.4f} | "
            f"{row['additive_pred']:.4f} | {row['interaction_residual']:+.4f} |"
        )

    lines.append("")
    lines.append("## Paired Wilcoxon (top significance)")
    lines.append("")
    lines.append("| cell A | cell B | n pairs | Δ (A − B) | p |")
    lines.append("|---|---|---|---|---|")
    for row in payload["paired_wilcoxon"][:15]:
        lines.append(
            f"| {row['cell_a']} | {row['cell_b']} | {row['n_paired_obs']} | "
            f"{row['mean_diff_a_minus_b']:+.4f} | {row['wilcoxon_p']:.2e} |"
        )

    lines.append("")
    lines.append("## Freeze decision tree")
    lines.append("")
    lines.append(
        "1. **If a cell's CI strictly dominates the baseline CI AND paired Wilcoxon\n"
        "   vs baseline gives p < 0.001 with Δ ≥ 0.005**: freeze that cell.\n"
        "2. **Else, if the top cell ties baseline within CI overlap**: freeze\n"
        "   baseline (`L.2.R4xI2`, upstream parity). The view/reference choice is\n"
        "   not load-bearing at linear-readout scope.\n"
        "3. **If view marginal swamps reference marginal** (|Δ_view| > 2× |Δ_ref|):\n"
        "   the headline is 'view matters; reference does not.' This shifts the\n"
        "   Stage-1 v14 priority from reference design to spectral feature design.\n"
        "4. **If interaction residuals are large** (max |res| ≥ 0.01): a single\n"
        "   main-effect story is misleading; report the 9-cell heatmap rather\n"
        "   than just the marginals.\n"
        "5. **Always**: confirm winner survives D.0a CrossSubject binary (Tier C)\n"
        "   before final freeze. Cross-subject distribution shift may swap.\n"
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(
        "- `cell_ci_forest.png` — bootstrap-CI ranking (this analyzer)\n"
        "- `factor_marginals.png` — ref/view marginal effects (this analyzer)\n"
        "- `cell_task_heatmap.png`, `cell_aggregate_bar.png` — collector\n"
        "- `paired_tests.csv` — collector (also embedded above)\n"
        "- `aggregate_diagnostics.csv` — collector (per-row source)\n"
        "- `freeze_analysis.{md,json}` — this analyzer"
    )
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-root", type=Path, required=True)
    p.add_argument(
        "--tier-a-only", action="store_true",
        help="Restrict analysis to Tier-A cells (R0/R3/R4 × I0/I2/I3). "
             "Default: analyze all L.2.R*xI* cells present.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
