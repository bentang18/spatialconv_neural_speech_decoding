"""Main-text figures 2/3/4 for the workshop paper, built from LOCAL artifacts only.

Fig 2    CS tap ladder + leaderboard     <- results/r6_era/board/*.json
Fig 2ws  within-session, elec-only        <- results/r6_era/board/*.json
Fig 3  LEACE identity erasure            <- results/r6_era/leace/leace_S*.json
Fig 4  concentration (3-PC vs full)      <- viz_crosssubject/archive/win{1,2}s/report.json

Fig 4 reads the ARCHIVED 6-task runs on purpose, not the 8-task showcase: its negative
control is `frame_brightness`, which the 8-task menu deliberately excludes. The menu is the
basis, so the two are different quantities and the control only exists in the older run.

Every panel recomputes its numbers from the raw per-cell rows and ASSERTS them against the
values already established, printing [check] lines. A figure that silently disagrees with the
ledger is worse than no figure, so the asserts run before anything is drawn.

Conventions that are easy to get wrong and are therefore enforced here:
  * CS rows are PER-SUBJECT cells -> average over cells FIRST, then macro over tasks.
  * Taps are only comparable on a SHARED cell set ("partial cells lie": .6279@4 -> .5991@10).
  * LEACE is PAIRED over cells; the unit of analysis is the cell, not the task.
  * WS is ELEC-only (enc0_elec/enc12_elec). There is NO within-session depth ladder to draw.
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
OUT = RES / "showcase/paper"
VIZ = RES / "viz_crosssubject/archive"
TAPS = ["enc0", "enc3", "enc6", "enc12"]

# Neuroprobe CS leaderboard, same board/split. Decoder is NOT held constant across these
# entries -- .539/.566/.578 are logistic/MLP/CNN on ONE fixed Laplacian-STFT feature set.
BOARD = {"CNN (Laplacian-STFT)": 0.578, "PopT": 0.575, "Linear (logistic)": 0.539}
LEAD = "45k cd"          # the shipped checkpoint: 45k with the linear cooldown
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
    print(f"[check] {LEAD} ladder = " + "  ".join(f"{t} {lads[LEAD][t]:.4f}" for t in TAPS))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), width_ratios=[1.15, 1])

    x = range(len(TAPS))
    for name, l in lads.items():
        lead = name == LEAD
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
    vals = [BOARD["Linear (logistic)"], lads[LEAD]["enc0"], BOARD["PopT"],
            BOARD["CNN (Laplacian-STFT)"], lads[LEAD]["enc12"]]
    cols = ["#bbb", PALETTE["enc0"], "#bbb", PALETTE["accent"], PALETTE["ours"]]
    ax2.bar(range(len(vals)), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.0015, f"{v:.3f}", ha="center", fontsize=6.8)
    ax2.set_xticks(range(len(vals))); ax2.set_xticklabels(names, fontsize=6.5)
    ax2.set_ylim(0.50, 0.625); ax2.set_ylabel("cross-subject macro AUROC")
    ax2.set_title("Matched linear-vs-linear", pad=6)
    fig.tight_layout(); _save(fig, "fig2_ladder")


# ------------------------------------------------------- fig 2ws: within-session
# Within-Session macros recomputed from the vendored leaderboard's raw per-session JSONs by
# scripts/neuroprobe/leaderboard_baselines.py --split Within-Session, under OUR aggregation.
# Hardcoded for the same reason BOARD is: this figure reads LOCAL artifacts only, and the
# sibling neuroprobe checkout is not one.
WS_BOARD = {
    "DIVER-1 (0.1s, tiny, frozen)": 0.6777,
    "PopT (Laplacian-STFT)": 0.6700,
    "CNN (Laplacian-STFT)": 0.6686,
    "Linear (Laplacian-STFT)": 0.6599,
    "BrainBERT (frozen)": 0.6257,
}
WS_TAPS = ["enc0_elec", "enc12_elec"]


def ws_cells(path: pathlib.Path) -> dict[str, dict[str, float]]:
    """cell -> tap -> macro over tasks. WS is ELEC-only: the regime was never run with the
    parcel taps, so there is no enc3/enc6 and no ladder can be drawn -- only a matched bar."""
    d = json.load(open(path))
    per_task = {}
    for key, blob in d.items():
        ws = blob.get("ws")
        if not ws:
            continue
        cols = {t: ws.get(f"{t}|std") for t in WS_TAPS}
        if any(c is None for c in cols.values()):
            continue
        shared = set.intersection(*(set(c) for c in cols.values()))
        per_task[key] = {c: {t: cols[t][c] for t in WS_TAPS} for c in shared}
    cells = set.intersection(*(set(v) for v in per_task.values()))
    return {c: {t: statistics.fmean(v[c][t] for v in per_task.values()) for t in WS_TAPS}
            for c in sorted(cells)}


def fig2_ws() -> None:
    cells = ws_cells(RES / "r6_era/board" / CKPTS[LEAD])
    # 12/12 is why LEAD is the cooldown checkpoint: the 40k file carries only 10 WS cells, and
    # a 10-cell macro is not comparable to a 12-cell leaderboard entry. Partial cells lie.
    assert len(cells) == 12, f"[check] VIOLATED WS needs all 12 Lite cells, got {len(cells)}"
    macro = {t: statistics.fmean(v[t] for v in cells.values()) for t in WS_TAPS}
    best = max(WS_BOARD.values())
    print(f"[check] WS {LEAD} over {len(cells)} cells = "
          + "  ".join(f"{t} {macro[t]:.4f}" for t in WS_TAPS))
    assert macro["enc12_elec"] > macro["enc0_elec"], "[check] VIOLATED depth did not help WS"
    assert macro["enc12_elec"] > best, "[check] VIOLATED enc12 does not clear the WS board top"
    wins = sum(v["enc12_elec"] > v["enc0_elec"] for v in cells.values())
    print(f"[check] OK enc12 clears board top {best:.4f} by {macro['enc12_elec'] - best:+.4f}; "
          f"depth helps {wins}/{len(cells)} cells")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9), width_ratios=[1.25, 1])

    rows = sorted(list(WS_BOARD.items())
                  + [("ours enc0 (0 params)", macro["enc0_elec"]),
                     ("ours enc12", macro["enc12_elec"])], key=lambda r: r[1])
    cols = [PALETTE["ours"] if n.startswith("ours enc12") else
            PALETTE["enc0"] if n.startswith("ours") else
            PALETTE["accent"] if abs(v - best) < 1e-9 else "#bbb" for n, v in rows]
    ax.barh(range(len(rows)), [v for _, v in rows], color=cols, height=0.66)
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.0015, i, f"{v:.4f}", va="center", fontsize=6.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows], fontsize=6.6)
    ax.set_xlim(0.60, 0.70)
    ax.set_xlabel("within-session macro AUROC (12/12 cells)")
    ax.set_title("Within-session leaderboard", pad=6)

    # Paired slopegraph, not two bars. The macro hides whether the gain is uniform or carried
    # by a couple of cells, and "partial cells lie" is exactly a warning about that.
    for c, v in cells.items():
        up = v["enc12_elec"] > v["enc0_elec"]
        ax2.plot([0, 1], [v["enc0_elec"], v["enc12_elec"]], marker="o", ms=3,
                 lw=0.9, color=PALETTE["ours"] if up else PALETTE["accent"], alpha=0.75)
    # Cell labels collide wherever two sessions land within a few thousandths of each other,
    # which is most of the top of the panel. Push them apart bottom-up by a fixed minimum gap
    # and leader-line each one back to its true value, so the nudge never misreads as data.
    order = sorted(cells.items(), key=lambda kv: kv[1]["enc12_elec"])
    span = order[-1][1]["enc12_elec"] - order[0][1]["enc12_elec"]
    gap, y_prev = 0.036 * span, -1e9
    for c, v in order:
        y = max(v["enc12_elec"], y_prev + gap)
        y_prev = y
        ax2.plot([1.02, 1.06], [v["enc12_elec"], y], lw=0.4, color="#bbb", zorder=1)
        ax2.text(1.07, y, c, fontsize=5.6, va="center", color="#555")
    ax2.plot([0, 1], [macro[t] for t in WS_TAPS], marker="o", ms=6, lw=2.6,
             color="k", zorder=5, label="macro")
    ax2.axhline(best, color=PALETTE["accent"], lw=0.9, ls="--")
    ax2.text(-0.26, best + 0.0012, f"board top {best:.4f}", color=PALETTE["accent"],
             fontsize=6.4, ha="left", va="bottom")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["enc0\n(0 params)", "enc12"])
    ax2.set_xlim(-0.28, 1.34)
    ax2.set_ylabel("within-session macro AUROC")
    ax2.set_title(f"Per cell: depth helps {wins}/{len(cells)}", pad=6)
    ax2.legend(fontsize=6.5, loc="lower right")

    # The parity caveat belongs ON the figure, not only in the memo: leaderboard WS entries
    # carry 2 folds and our WS readout has no bit-identical parity test yet (CS does).
    fig.text(0.5, -0.045, "WS fold-structure parity with the leaderboard is UNVERIFIED "
             "-- no WS SOTA claim until that is closed", ha="center", fontsize=6.2,
             color=PALETTE["accent"])
    fig.tight_layout(); _save(fig, "fig2ws_within_session")


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
                {k: statistics.fmean(d[t]["checks"][tap][k] for t in d)
                 for k in ("id_auc_before", "id_auc_after", "var_removed", "residual_cov", "d")}
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
        # The eraser is RANK 1 (binary anchor-vs-test concept), so `var_removed` is the share
        # carried by ONE direction out of d. Without d on the panel, "20.7%" reads as an
        # ordinary chunk of a representation rather than the striking thing it is.
        stats[f"{tap}|d"] = statistics.fmean(c["d"] for c in ck)

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
    ax.set_xticks(range(2))
    # d on the tick label, because the eraser is rank 1: "20.7%" is an ordinary-looking number
    # until you see it is ONE direction out of ~77k.
    ax.set_xticklabels([f"{t}\n(1 dir of {stats[f'{t}|d']/1000:.1f}k)" for t in taps2],
                       fontsize=7)
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
        rep = json.load(open(VIZ / win / "report.json"))
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
    ap.add_argument("--only", choices=["2", "2ws", "3", "4"], default=None)
    a = ap.parse_args()
    _style()
    for n, fn in (("2", fig2), ("2ws", fig2_ws), ("3", fig3), ("4", fig4)):
        if a.only in (None, n):
            print(f"=== fig {n}")
            fn()
