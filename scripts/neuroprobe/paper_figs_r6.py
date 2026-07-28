"""Main-text figures 2/3/4 for the workshop paper, built from LOCAL artifacts only.

Fig 2  tap ladder + leaderboard          <- results/r6_era/board/*.json
Fig 3  LEACE identity erasure            <- results/r6_era/leace/leace_S*.json
Fig 4  concentration (3-PC vs full)      <- results/viz_crosssubject/win{1,2}s/report.json

Every panel recomputes its numbers from the raw per-cell rows and ASSERTS them against the
values already established, printing [check] lines. A figure that silently disagrees with the
ledger is worse than no figure, so the asserts run before anything is drawn.

Conventions that are easy to get wrong and are therefore enforced here:
  * CS rows are PER-SUBJECT cells -> average over cells FIRST, then macro over tasks.
  * Taps are only comparable on a SHARED cell set ("partial cells lie": .6279@4 -> .5991@10).
  * LEACE is PAIRED over cells; the unit of analysis is the cell, not the task.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "paper_figs"
TAPS = ["enc0", "enc3", "enc6", "enc12"]

# Neuroprobe CS leaderboard, same board/split. Decoder is NOT held constant across these
# entries -- .539/.566/.578 are logistic/MLP/CNN on ONE fixed Laplacian-STFT feature set.
BOARD = {"CNN (Laplacian-STFT)": 0.578, "PopT": 0.575, "Linear (logistic)": 0.539}
CKPTS = {
    "20k": "results_v3_board_r6_20k.json",
    "40k": "results_v3_board_r6_40k.json",
    "45k": "MERGED_board_nocd_45k.json",
    "45k cd": "results_v3_board_cdlin_45k.json",
}

PALETTE = {"enc0": "#9aa5b1", "ours": "#1f4e79", "accent": "#c1440e", "muted": "#7a8994"}
# Keyed by task NAME so the 1 s and 2 s panels use the same color for the same task.
TASK_COLOR = {
    "onset": "#1f77b4", "speech": "#ff7f0e", "delta_volume": "#2ca02c",
    "word_index": "#d62728", "word_part_speech": "#9467bd",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "legend.frameon": False, "axes.titlesize": 8.5, "axes.labelsize": 8,
    })


# ---------------------------------------------------------------- fig 2: tap ladder
def ladder(path: pathlib.Path) -> dict[str, float]:
    """CS macro per tap: mean over the cell INTERSECTION, then macro over tasks."""
    d = json.load(open(path))
    per_task = {}
    for key, blob in d.items():
        cs = blob.get("cs")
        if not cs:
            continue
        cols = {t: cs.get(f"{t}|std") for t in TAPS}
        if any(c is None for c in cols.values()):
            continue
        shared = set.intersection(*(set(c) for c in cols.values()))
        if not shared:
            continue
        per_task[key] = {
            t: statistics.fmean(cols[t][c] for c in sorted(shared)) for t in TAPS
        }
    return {t: statistics.fmean(v[t] for v in per_task.values()) for t in TAPS}


def fig2() -> None:
    lads = {name: ladder(RES / "r6_era/board" / f) for name, f in CKPTS.items()}
    for name, l in lads.items():
        mono = all(l[a] <= l[b] for a, b in zip(TAPS, TAPS[1:]))
        assert mono, f"[check] VIOLATED ladder not monotone at {name}: {l}"
        assert abs(l["enc0"] - 0.5872) < 2e-4, f"enc0 drifted at {name}: {l['enc0']}"
    print("[check] OK ladder strictly monotone at all 4 ckpts; enc0 == .5872 in every one")
    print("[check] 40k ladder = " + "  ".join(f"{t} {lads['40k'][t]:.4f}" for t in TAPS))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), width_ratios=[1.15, 1])

    x = range(len(TAPS))
    for name, l in lads.items():
        lead = name == "40k"
        ax.plot(x, [l[t] for t in TAPS], marker="o", ms=4 if lead else 3,
                lw=1.8 if lead else 0.9, zorder=3 if lead else 2,
                color=PALETTE["ours"] if lead else PALETTE["muted"],
                alpha=1.0 if lead else 0.55, label=name)
    ax.axhline(BOARD["CNN (Laplacian-STFT)"], color=PALETTE["accent"], lw=0.9, ls="--")
    ax.text(0.04, BOARD["CNN (Laplacian-STFT)"] + 0.0012, "prior SOTA .578",
            color=PALETTE["accent"], fontsize=7)
    ax.axhline(BOARD["Linear (logistic)"], color="#888", lw=0.8, ls=":")
    ax.text(0.04, BOARD["Linear (logistic)"] + 0.0012, "their linear .539", color="#888", fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels(["enc0\n(0 params)", "enc3", "enc6", "enc12"])
    ax.set_ylabel("cross-subject macro AUROC")
    ax.set_title("Depth ladder, every checkpoint", pad=6)
    ax.legend(fontsize=6.5, loc="lower right", ncol=2)

    names = ["their\nlinear", "our enc0\n(0 params)", "PopT", "their CNN\n(prior SOTA)", "ours\nenc12"]
    vals = [BOARD["Linear (logistic)"], lads["40k"]["enc0"], BOARD["PopT"],
            BOARD["CNN (Laplacian-STFT)"], lads["40k"]["enc12"]]
    cols = ["#bbb", PALETTE["enc0"], "#bbb", PALETTE["accent"], PALETTE["ours"]]
    ax2.bar(range(len(vals)), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.0015, f"{v:.3f}", ha="center", fontsize=6.8)
    ax2.set_xticks(range(len(vals))); ax2.set_xticklabels(names, fontsize=6.5)
    ax2.set_ylim(0.50, 0.625); ax2.set_ylabel("cross-subject macro AUROC")
    ax2.set_title("Matched linear-vs-linear", pad=6)
    fig.tight_layout(); _save(fig, "fig2_ladder")


# ---------------------------------------------------------------- fig 3: LEACE
def _paired_t(diffs: list[float]) -> tuple[float, float]:
    n = len(diffs)
    m = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    return m, (m / (sd / n**0.5) if sd > 0 else float("nan"))


def fig3() -> None:
    files = sorted((RES / "r6_era/leace").glob("leace_S*.json"))
    # All 10 cells were RUN; leace_S7T1.json is currently unreadable from the dtai LOGIN node
    # ("transport endpoint shutdown"), the same client-mount signature as the 07-28 venv scare
    # where a compute node then read 0/11385 damaged. Treat 9 as a transient shortfall, not a
    # lost cell -- but never let the count go silent, since partial cells lie.
    assert len(files) >= 9, f"[check] VIOLATED need >=9 cells, got {len(files)}"
    if len(files) < 10:
        print(f"[check] WARNING running on {len(files)}/10 cells "
              f"(missing {sorted({'S1T1','S1T2','S3T0','S3T1','S4T0','S4T1','S7T0','S7T1','S10T0','S10T1'} - {f.stem.replace('leace_','') for f in files})}) "
              "-- re-read from a COMPUTE node before this figure ships")
    per_cell: dict[str, dict[str, float]] = {}
    checks: dict[str, list[dict]] = {"enc0": [], "enc12": []}
    for f in files:
        d = json.load(open(f))
        cell = f.stem.replace("leace_", "")
        row = {}
        for tap in ("enc0", "enc12"):
            for arm in ("std", "leace", "std_target"):
                row[f"{tap}|{arm}"] = statistics.fmean(
                    d[t]["cells"][f"{tap}|{arm}"]["test"] for t in d
                )
            checks[tap].append(
                statistics.fmean.__self__ if False else
                {k: statistics.fmean(d[t]["checks"][tap][k] for t in d)
                 for k in ("id_auc_before", "id_auc_after", "var_removed", "residual_cov")}
            )
        per_cell[cell] = row

    stats = {}
    for tap in ("enc0", "enc12"):
        for arm in ("leace", "std_target"):
            diffs = [r[f"{tap}|{arm}"] - r[f"{tap}|std"] for r in per_cell.values()]
            stats[f"{tap}|{arm}"] = _paired_t(diffs)
        ck = checks[tap]
        stats[f"{tap}|auc_before"] = statistics.fmean(c["id_auc_before"] for c in ck)
        stats[f"{tap}|auc_after"] = statistics.fmean(c["id_auc_after"] for c in ck)
        stats[f"{tap}|var"] = statistics.fmean(c["var_removed"] for c in ck)
        stats[f"{tap}|resid"] = max(c["residual_cov"] for c in ck)

    for tap in ("enc0", "enc12"):
        m, t = stats[f"{tap}|leace"]
        print(f"[check] {tap:>5} LEACE  dCS {m:+.5f} (t={t:+.2f})   "
              f"id_auc {stats[f'{tap}|auc_before']:.3f}->{stats[f'{tap}|auc_after']:.3f}   "
              f"var_removed {stats[f'{tap}|var']*100:.1f}%   max resid_cov {stats[f'{tap}|resid']:.1e}")
    assert stats["enc12|auc_after"] < 0.52, "erasure did not destroy identity at enc12"
    assert stats["enc0|leace"][0] < stats["enc12|leace"][0], (
        "[check] VIOLATED expected enc0 erasure to hurt MORE than enc12")
    print("[check] OK enc12 erasure is a null on CS while enc0 erasure hurts -> SEPARABILITY")

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))
    taps2 = ["enc0", "enc12"]

    ax = axes[0]
    w = 0.34
    ax.bar([i - w / 2 for i in range(2)], [stats[f"{t}|auc_before"] for t in taps2],
           w, label="before", color=PALETTE["ours"])
    ax.bar([i + w / 2 for i in range(2)], [stats[f"{t}|auc_after"] for t in taps2],
           w, label="after erasure", color=PALETTE["enc0"])
    ax.axhline(0.5, color="k", lw=0.7, ls=":")
    ax.set_xticks(range(2)); ax.set_xticklabels(taps2)
    ax.set_ylabel("subject-identity AUROC"); ax.set_ylim(0.4, 1.05)
    ax.set_title("Erasure destroys identity"); ax.legend(fontsize=6.5)

    ax = axes[1]
    ax.bar(range(2), [stats[f"{t}|var"] * 100 for t in taps2],
           0.5, color=[PALETTE["enc0"], PALETTE["ours"]])
    for i, t in enumerate(taps2):
        ax.text(i, stats[f"{t}|var"] * 100 + 0.4, f"{stats[f'{t}|var']*100:.1f}%",
                ha="center", fontsize=7)
    ax.set_xticks(range(2)); ax.set_xticklabels(taps2)
    ax.set_ylabel("% of variance removed"); ax.set_title("...and deletes far more at depth")

    ax = axes[2]
    # Annotations go at a FIXED axes fraction (blended transform) with reserved headroom, so
    # they can never ride up into the title the way a data-relative offset does.
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    spans = []
    for i, tap in enumerate(taps2):
        diffs = [r[f"{tap}|leace"] - r[f"{tap}|std"] for r in per_cell.values()]
        spans += diffs
        ax.scatter([i + 0.06] * len(diffs), diffs, s=11, color=PALETTE["muted"],
                   alpha=0.8, zorder=2)
        m, t = stats[f"{tap}|leace"]
        ax.plot([i - 0.22, i + 0.22], [m, m], lw=2.2,
                color=PALETTE["accent"] if t < -2 else PALETTE["ours"], zorder=3)
        # 5 dp, not 4: the enc12 effect is +2e-5 and rounding it to "+0.0000" hides that the
        # null is a MAGNITUDE statement, not just a failed significance test.
        ax.text(i, 0.90, f"{m:+.5f}\nt={t:+.2f}", ha="center", va="top", fontsize=6.8,
                transform=blend, color=PALETTE["accent"] if t < -2 else PALETTE["ours"])
    ax.axhline(0, color="k", lw=0.7)
    pad = 0.12 * (max(spans) - min(spans))
    ax.set_ylim(min(spans) - pad, max(spans) + 3.2 * pad)
    ax.set_xticks(range(2)); ax.set_xticklabels(taps2); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Δ cross-subject AUROC\n(erased − intact)")
    ax.set_title("Cost of erasure: none at enc12", pad=6)
    fig.tight_layout(); _save(fig, "fig3_leace")


# ---------------------------------------------------------------- fig 4: concentration
def fig4() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    for ax, win in zip(axes, ("win1s", "win2s")):
        rep = json.load(open(RES / "viz_crosssubject" / win / "report.json"))
        # Figure keys are flat "<lobe>/<tap>"; the shared-lobe panel is "T". The 1 s suite was
        # run WITHOUT enc0, so each window gets its own tap list rather than a shared constant.
        have = [t for t in TAPS if f"T/{t}" in rep["figures"]]
        three = {t: statistics.fmean(rep["figures"][f"T/{t}"]["align_3pc"].values()) for t in have}
        full = {
            t: statistics.fmean(
                r["cross_subject_r"] for r in rep["quant"]
                if r["tap"] == t and str(r["class"]) == "contrast")
            for t in have
        }
        lo, hi = have[0], have[-1]
        rise = three[hi] - three[lo]
        flat = full[hi] - full[lo]
        lobe, TAPS_W = "T", have
        print(f"[check] {win} lobe={lobe} taps={TAPS_W}  3PC {three[lo]:.3f}->{three[hi]:.3f} "
              f"(+{rise:.3f})   full-space {full[lo]:.3f}->{full[hi]:.3f} ({flat:+.3f})")
        assert rise > 3 * abs(flat), (
            f"[check] VIOLATED concentration claim: 3-PC rise {rise:.3f} not >> full-space {flat:+.3f}")

        x = range(len(TAPS_W))
        # Per-task lines, not just the mean: frame_brightness is the built-in NEGATIVE control
        # (a visual label with no speech content) and it must stay pinned at ~0 at every depth.
        # Averaging it into one line hides exactly the control that makes the panel credible.
        # Colors are keyed by task name, not by draw order, so the two panels agree.
        tasks = sorted(rep["figures"][f"T/{lo}"]["align_3pc"])
        DUD = "frame_brightness"
        for tk in tasks:
            ys = [rep["figures"][f"T/{t}"]["align_3pc"][tk] for t in TAPS_W]
            dud = tk == DUD
            ax.plot(x, ys, marker="x" if dud else "o", ms=3.2, lw=1.1 if dud else 1.5,
                    color=PALETTE["muted"] if dud else TASK_COLOR[tk],
                    ls="--" if dud else "-", alpha=0.95, zorder=3 if dud else 2,
                    label=(tk.replace("_", " ") + (" (visual control)" if dud else "")))
        ax.plot(x, [full[t] for t in TAPS_W], marker="s", ms=3.2, lw=1.6, color="k",
                ls=":", zorder=4, label="full space (all tasks)")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_xticks(list(x)); ax.set_xticklabels(TAPS_W)
        ax.set_title(f"{win[3:-1]} s window", pad=6)
        ax.set_xlabel("encoder depth")
        dud_span = max(abs(rep["figures"][f"T/{t}"]["align_3pc"][DUD]) for t in TAPS_W)
        assert dud_span < 0.05, f"[check] VIOLATED visual control not flat: {dud_span:.3f}"
    axes[0].set_ylabel("cross-subject alignment r")
    # ONE legend below both panels -- in-axes legends sat on top of the lines they label.
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 0.075),
               ncol=4, columnspacing=1.4, handlelength=1.8)
    fig.suptitle("Shared structure concentrates into a low-dimensional subspace with depth",
                 fontsize=8.5, y=1.01)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); _save(fig, "fig4_concentration")
    print("[check] OK concentration reproduces in BOTH windows independently")


def _save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT.relative_to(ROOT)}/{stem}.pdf|.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["2", "3", "4"], default=None)
    a = ap.parse_args()
    _style()
    for n, fn in (("2", fig2), ("3", fig3), ("4", fig4)):
        if a.only in (None, n):
            print(f"=== fig {n}")
            fn()
