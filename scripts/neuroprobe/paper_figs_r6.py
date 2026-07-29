"""Main-text figures 2/3/4 for the workshop paper, built from LOCAL artifacts only.

Fig 2    CS tap ladder + leaderboard     <- results/r6_era/board/*.json
Fig 2ws  within-session, elec-only        <- results/r6_era/board/*.json
Fig 2cs  cross-session, elec-only         <- results/r6_era/board/*.json
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
import re
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

# SELECTION RULE, applied identically to BOTH splits: every model FAMILY on the leaderboard,
# represented by its BEST-scoring variant. Nothing is dropped for being weak and nothing is
# kept for being flattering -- picking a subset by eye is how a comparison figure starts lying.
# (Families with several submissions -- Linear x3, BrainBERT x2, PopT x2, DIVER-1 x2 -- would
# otherwise pad the chart with the same model tuned differently. "BrainBERT (untrained)" is
# kept SEPARATE because it is a random-init control, not a tuning variant of BrainBERT.)
#
# Macros are 4 dp recomputed from the vendored leaderboard's raw per-session JSONs under OUR
# aggregation (leaderboard_baselines.py), not 3 dp off a webpage.
#
# Decoder is NOT held constant across these entries: they are logistic/MLP/CNN on ONE fixed
# Laplacian-STFT feature set, which is why the matched comparison is enc0-ridge vs their linear.
# NOTE: DIVER-1 is the Within-Session board top but has NO Cross-Subject submission, so it
# appears in one chart and not the other. That asymmetry is theirs, not a filtering choice.
BOARD = {
    "CNN (Laplacian-STFT)": 0.5777,
    "PopT (Laplacian-STFT)": 0.5750,
    "MLP (Laplacian-STFT)": 0.5659,
    "BrainBERT (frozen)": 0.5471,
    "Linear (Laplacian-STFT)": 0.5392,
    "BrainBERT (untrained)": 0.5266,
    "GLIS-GNN": 0.5152,
}
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


def cs_cells(path: pathlib.Path) -> dict[str, dict[str, float]]:
    """cell -> tap -> macro over tasks, on the cell x task INTERSECTION shared by all taps.

    `ladder` collapses straight to the macro, which cannot say whether the +.0164 enc0->enc12
    gain is uniform across subjects or carried by two cells. That is exactly the question
    "partial cells lie" is a warning about, so the per-cell rows get their own panel.
    """
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
        if shared:
            per_task[key] = {c: {t: cols[t][c] for t in TAPS} for c in shared}
    cells = set.intersection(*(set(v) for v in per_task.values()))
    return {c: {t: statistics.fmean(v[c][t] for v in per_task.values()) for t in TAPS}
            for c in sorted(cells)}


def by_subject(cells: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Collapse session cells (S1T1, S1T2) onto their subject (S1), averaging the taps.

    Ten session lines is more clutter than argument -- the reader is asked whether depth helps
    ACROSS PEOPLE, and two trials of one person are not two pieces of evidence for that. The
    macro on the panel is still the unweighted mean over SESSIONS (it is the board's unit), so
    the black line is unchanged by this collapse; only the grey lines merge.
    """
    groups: dict[str, list[dict[str, float]]] = {}
    for cell, v in cells.items():
        m = re.fullmatch(r"S(\d+)T\d+", cell)
        assert m, f"cannot read a subject out of cell name {cell!r}"
        groups.setdefault(f"S{m.group(1)}", []).append(v)
    return {s: {tap: statistics.fmean(v[tap] for v in vs) for tap in vs[0]}
            for s, vs in sorted(groups.items(), key=lambda kv: int(kv[0][1:]))}


def _slopegraph(ax, cells: dict[str, dict[str, float]], taps: list[str], macro: dict[str, float],
                board: float, board_label: str) -> int:
    """Per-cell lines across taps + the macro in black. Returns the monotone-cell count.

    Labels are pushed apart bottom-up by a fixed gap and leader-lined back to their true value,
    so the de-collision nudge can never be misread as data.
    """
    mono = 0
    for v in cells.values():
        ys = [v[t] for t in taps]
        up = all(a <= b for a, b in zip(ys, ys[1:]))
        mono += up
        ax.plot(range(len(taps)), ys, marker="o", ms=3, lw=0.9,
                color=PALETTE["ours"] if up else PALETTE["accent"], alpha=0.7)
    order = sorted(cells.items(), key=lambda kv: kv[1][taps[-1]])
    span = order[-1][1][taps[-1]] - order[0][1][taps[-1]]
    gap, y_prev, x = 0.036 * span, -1e9, len(taps) - 1
    for c, v in order:
        y = max(v[taps[-1]], y_prev + gap)
        y_prev = y
        ax.plot([x + 0.02, x + 0.06], [v[taps[-1]], y], lw=0.4, color="#bbb", zorder=1)
        ax.text(x + 0.07, y, c, fontsize=5.6, va="center", color="#555")
    ax.plot(range(len(taps)), [macro[t] for t in taps], marker="o", ms=6, lw=2.6,
            color="k", zorder=5, label="macro")
    ax.axhline(board, color=PALETTE["accent"], lw=0.9, ls="--")
    # Whichever side of the rule the macro line is NOT on at the left edge.
    below = macro[taps[0]] > board
    ax.text(-0.26, board + (-0.0012 if below else 0.0012), board_label,
            color=PALETTE["accent"], fontsize=6.4, ha="left",
            va="top" if below else "bottom")
    ax.set_xticks(range(len(taps)))
    ax.set_xticklabels([t.replace("_elec", "") for t in taps], fontsize=7)
    ax.set_xlim(-0.28, x + 0.34)
    return mono


def fig2() -> None:
    lads = {name: ladder(RES / "r6_era/board" / f) for name, f in CKPTS.items()}
    for name, l in lads.items():
        mono = all(l[a] <= l[b] for a, b in zip(TAPS, TAPS[1:]))
        assert mono, f"[check] VIOLATED ladder not monotone at {name}: {l}"
        assert abs(l["enc0"] - 0.5872) < 2e-4, f"enc0 drifted at {name}: {l['enc0']}"
    print("[check] OK ladder strictly monotone at all 4 ckpts; enc0 == .5872 in every one")
    print(f"[check] {LEAD} ladder = " + "  ".join(f"{t} {lads[LEAD][t]:.4f}" for t in TAPS))

    cells = cs_cells(RES / "r6_era/board" / CKPTS[LEAD])
    assert len(cells) == 10, f"[check] VIOLATED CS needs all 10 test cells, got {len(cells)}"
    macro, best = lads[LEAD], max(BOARD.values())
    assert macro["enc12"] > best, "[check] VIOLATED enc12 does not clear the CS board top"

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9), width_ratios=[1.25, 1])

    rows = sorted(list(BOARD.items())
                  + [("ours enc0 (0 params)", macro["enc0"]), ("ours enc12", macro["enc12"])],
                  key=lambda r: r[1])
    cols = [PALETTE["ours"] if n.startswith("ours enc12") else
            PALETTE["enc0"] if n.startswith("ours") else
            PALETTE["accent"] if abs(v - best) < 1e-9 else "#bbb" for n, v in rows]
    ax.barh(range(len(rows)), [v for _, v in rows], color=cols, height=0.66)
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.0015, i, f"{v:.4f}", va="center", fontsize=6.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows], fontsize=6.6)
    ax.set_xlim(0.50, 0.625)
    ax.set_xlabel("cross-subject macro AUROC (10/10 cells)")
    ax.set_title("Cross-subject leaderboard", pad=6)

    subj = by_subject(cells)
    mono = _slopegraph(ax2, subj, TAPS, macro, best, f"board top {best:.4f}")
    beat = sum(v["enc12"] > v["enc0"] for v in subj.values())
    print(f"[check] CS {len(cells)} sessions -> {len(subj)} subjects: monotone in "
          f"{mono}/{len(subj)}, enc12 > enc0 in {beat}/{len(subj)}")
    ax2.set_ylabel("cross-subject macro AUROC")
    # Two DIFFERENT counts, and conflating them would overclaim. Every cell ends higher than it
    # started, but only 4 climb at every rung -- the macro ladder is monotone, the cells are not.
    ax2.set_title(f"Per subject: enc12 > enc0 in {beat}/{len(subj)}\n"
                  f"(monotone at every tap in {mono})", pad=6, fontsize=8)
    ax2.legend(fontsize=6.5, loc="upper left")
    fig.text(0.5, -0.04, "one line per SUBJECT (trials averaged)  ·  orange = not monotone across taps (still ends above enc0)",
             ha="center", fontsize=6.2, color=PALETTE["accent"])
    fig.tight_layout(); _save(fig, "fig2_ladder")


# ------------------------------------------------------- fig 2ws: within-session
# Same selection rule as BOARD: every family, best variant. Hardcoded for the same reason --
# this figure reads LOCAL artifacts only and the sibling neuroprobe checkout is not one.
WS_BOARD = {
    "DIVER-1 (0.1s, tiny, frozen)": 0.6777,
    "PopT (Laplacian-STFT)": 0.6700,
    "CNN (Laplacian-STFT)": 0.6686,
    "Linear (Laplacian-STFT)": 0.6599,
    "MLP (Laplacian-STFT)": 0.6563,
    "BrainBERT (frozen)": 0.6257,
    "BrainBERT (untrained)": 0.5808,
    "GLIS-GNN": 0.5526,
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
    ax.set_xlim(0.54, 0.71)
    ax.set_xlabel("within-session macro AUROC (12/12 cells)")
    ax.set_title("Within-session leaderboard", pad=6)

    # Paired slopegraph, not two bars. The macro hides whether the gain is uniform or carried
    # by a couple of cells, and "partial cells lie" is exactly a warning about that.
    subj = by_subject(cells)
    swins = sum(v["enc12_elec"] > v["enc0_elec"] for v in subj.values())
    print(f"[check] WS {len(cells)} sessions -> {len(subj)} subjects: depth helps {swins}/{len(subj)}")
    _slopegraph(ax2, subj, WS_TAPS, macro, best, f"board top {best:.4f}")
    ax2.set_ylabel("within-session macro AUROC")
    ax2.set_title(f"Per subject: depth helps {swins}/{len(subj)}", pad=6)
    ax2.legend(fontsize=6.5, loc="upper left")

    # Only TWO points, and that is an encode limitation, not a choice: the board encode ran
    # `--elec-taps 0,12` (v3_board_encode_r6.sbatch:57) and WS is per-ELECTRODE, so enc3_elec
    # and enc6_elec were never written. The parcel enc3/enc6 that fig2 uses are a different
    # feature unit and are NOT substitutable here. A 4-tap WS panel needs a GPU re-encode.
    fig.text(0.5, -0.045, "WS is per-electrode and the board encode wrote only elec taps 0 and 12"
             " -- enc3/enc6 need a re-encode, not a replot", ha="center", fontsize=6.2,
             color=PALETTE["muted"])
    fig.tight_layout(); _save(fig, "fig2ws_within_session")


# ------------------------------------------------------- fig 2cs: cross-session
# Same selection rule again: every family, best variant, recomputed from the vendored raw JSONs
# (`leaderboard_baselines.py --split Cross-Session`). Two asymmetries here are theirs, not ours:
# DIVER-1 has no Cross-Session submission so it drops out, and RNN (GRU) appears ONLY here --
# it has no Cross-Subject entry and its Within-Session directory fails task/cell coverage.
CSESSION_BOARD = {
    "CNN (Laplacian-STFT)": 0.6704,
    "PopT (Laplacian-STFT)": 0.6627,
    "MLP (Laplacian-STFT)": 0.6549,
    "Linear (Laplacian-STFT)": 0.6511,
    "BrainBERT (frozen)": 0.6326,
    "BrainBERT (untrained)": 0.5725,
    "GLIS-GNN": 0.5499,
    "RNN (GRU)": 0.5100,
}
# NOT `LEAD`. The cooldown checkpoints never had csession scored -- the regime is per-electrode
# and `board_readout_lean.sbatch` omitted the elec-labels sidecar on those runs -- so the only
# checkpoints carrying it are 45k-no-cooldown and 20k. Since the cooldown is a REAL gain on
# csession, using the no-cooldown file understates us; that direction is the safe one.
CSESSION_CKPT = "MERGED_board_nocd_45k.json"
CSESSION_TAP = "enc12_elec"


def csession_cells(path: pathlib.Path) -> dict[str, dict[str, float]]:
    """cell -> tap -> macro over tasks. ONE tap: the board wrote only `enc12_elec` here."""
    d = json.load(open(path))
    per_task = {}
    for key, blob in d.items():
        cse = blob.get("csession")
        if not cse or f"{CSESSION_TAP}|std" not in cse:
            continue
        per_task[key] = cse[f"{CSESSION_TAP}|std"]
    if not per_task:
        return {}
    cells = set.intersection(*(set(v) for v in per_task.values()))
    return {c: {CSESSION_TAP: statistics.fmean(v[c] for v in per_task.values())}
            for c in sorted(cells)}


def fig2_cs() -> None:
    cells = csession_cells(RES / "r6_era/board" / CSESSION_CKPT)
    assert len(cells) == 12, f"[check] VIOLATED csession needs all 12 Lite cells, got {len(cells)}"
    macro = statistics.fmean(v[CSESSION_TAP] for v in cells.values())
    best = max(CSESSION_BOARD.values())
    print(f"[check] CSession {CSESSION_CKPT} over {len(cells)} cells = {macro:.4f}")
    assert macro > best, f"[check] VIOLATED csession {macro:.4f} does not clear board top {best:.4f}"

    # The claim must not rest on one checkpoint. 20k is the only other file carrying csession;
    # if it did not also clear the board the result would be a checkpoint artifact.
    alt = csession_cells(RES / "r6_era/board" / "results_v3_board_r6_20k.json")
    alt_macro = statistics.fmean(v[CSESSION_TAP] for v in alt.values())
    assert len(alt) == 12 and alt_macro > best, \
        f"[check] VIOLATED 20k csession {alt_macro:.4f} on {len(alt)} cells does not clear {best:.4f}"
    print(f"[check] OK clears board top {best:.4f} by {macro - best:+.4f}; "
          f"20k replicates at {alt_macro:.4f} ({alt_macro - best:+.4f})")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9), width_ratios=[1.25, 1])

    rows = sorted(list(CSESSION_BOARD.items()) + [("ours enc12", macro)], key=lambda r: r[1])
    cols = [PALETTE["ours"] if n.startswith("ours") else
            PALETTE["accent"] if abs(v - best) < 1e-9 else "#bbb" for n, v in rows]
    ax.barh(range(len(rows)), [v for _, v in rows], color=cols, height=0.66)
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.0015, i, f"{v:.4f}", va="center", fontsize=6.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows], fontsize=6.6)
    ax.set_xlim(0.50, 0.71)
    ax.set_xlabel("cross-session macro AUROC (12/12 cells)")
    ax.set_title("Cross-session leaderboard", pad=6)

    # One tap means no slopegraph and no ladder, so the right panel spends its space on the
    # DISTRIBUTION instead: a macro above the board top is compatible with most cells below it.
    subj = by_subject(cells)
    ys = sorted(subj.items(), key=lambda kv: kv[1][CSESSION_TAP])
    lo = min(v[CSESSION_TAP] for _, v in ys)
    hi = max(v[CSESSION_TAP] for _, v in ys)
    pad = 0.055 * (hi - lo)
    # The two reference lines are named in the legend, not inline: at 4/6 above and 2/6 below,
    # inline labels land on top of the rows they are supposed to be read against.
    ax2.axvline(best, color=PALETTE["accent"], lw=0.9, ls="--",
                label=f"board top {best:.4f}")
    ax2.axvline(macro, color="k", lw=1.6, label=f"our macro {macro:.4f}")
    for i, (_, v) in enumerate(ys):
        x = v[CSESSION_TAP]
        col = PALETTE["ours"] if x > best else PALETTE["accent"]
        ax2.plot([best, x], [i, i], lw=0.8, color=col, alpha=0.55, zorder=2)
        ax2.plot([x], [i], marker="o", ms=5, color=col, zorder=3)
        # Labels flip to the outside of the dot so they never straddle the macro rule.
        left = abs(x - macro) < pad * 2.2 and x < macro
        ax2.text(x + (-pad * 0.6 if left else pad * 0.6), i, f"{x:.4f}", fontsize=6.2,
                 va="center", ha="right" if left else "left", zorder=4)
    ax2.set_yticks(range(len(ys)))
    ax2.set_yticklabels([s for s, _ in ys], fontsize=7)
    ax2.set_xlim(lo - pad * 5.0, hi + pad * 4.0)
    ax2.set_ylim(-0.9, len(ys) - 0.15)
    ax2.legend(fontsize=6.3, loc="lower right")
    ax2.set_xlabel("cross-session macro AUROC")
    cell_win = sum(v[CSESSION_TAP] > best for v in cells.values())
    subj_win = sum(v[CSESSION_TAP] > best for v in subj.values())
    print(f"[check] CSession {len(cells)} sessions -> {len(subj)} subjects: clears board top in "
          f"{subj_win}/{len(subj)} subjects, {cell_win}/{len(cells)} sessions")
    ax2.set_title(f"Per subject: clears board top in {subj_win}/{len(subj)}\n"
                  f"({cell_win}/{len(cells)} sessions individually)", pad=6, fontsize=8)

    # Both caveats belong on the figure, not only in the caption: a single tap is not a ladder,
    # and this is the one panel in the set that does not use the shipped checkpoint.
    fig.text(0.5, -0.055, "csession carries only enc12_elec -- no enc0 tap, so no zero-param bar "
             f"and no ladder  ·  checkpoint = 45k NO-cooldown (the shipped {LEAD} was never "
             "scored on this regime; cooldown helps csession, so this understates us)",
             ha="center", fontsize=6.0, color=PALETTE["muted"])
    fig.tight_layout(); _save(fig, "fig2cs_cross_session")


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
    """3-PC r vs FULL-SPACE r. The shaded gap between them IS the claim.

    Two earlier versions failed for opposite reasons. The first drew per-task 3-PC lines against
    a single POOLED full-space line -- different units, so the comparison the figure exists to
    make could not be made. The second paired every task with its own full-space line: correct,
    but twelve lines and six fills, and the claim drowned.

    So: ONE aggregate pair (mean over the five speech tasks) plus the control pair, four lines.
    Per-task detail does not vanish -- the count of tasks whose gap widens is computed over all
    five and printed ON the figure, which answers "does it hold for every task" with a number
    instead of ten more lines. Full per-task values stay in report.json and the [check] output.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    DUD = "frame_brightness"
    holds = []
    for ax, win in zip(axes, ("win1s", "win2s")):
        rep = json.load(open(VIZ / win / "report.json"))
        have = [t for t in TAPS if f"T/{t}" in rep["figures"]]
        three = {t: rep["figures"][f"T/{t}"]["align_3pc"] for t in have}
        full = {t: {r["task"]: r["cross_subject_r"] for r in rep["quant"]
                    if r["tap"] == t and str(r["class"]) == "contrast"} for t in have}
        speech = [k for k in sorted(three[have[0]]) if k != DUD]
        x = list(range(len(have)))
        lo, hi = have[0], have[-1]

        # Per-task, over ALL five: does the gap widen? Reported as a count, not as lines.
        n_hold = sum((three[hi][k] - full[hi][k]) > (three[lo][k] - full[lo][k]) for k in speech)
        holds.append((win, n_hold, len(speech)))

        series = [("mean of 5 speech tasks", speech, PALETTE["ours"], 2.0),
                  (DUD.replace("_", " ") + " (visual control)", [DUD], PALETTE["muted"], 1.4)]
        for lab, keys, col, lw in series:
            y3 = [statistics.fmean(three[t][k] for k in keys) for t in have]
            yf = [statistics.fmean(full[t][k] for k in keys) for t in have]
            ax.fill_between(x, yf, y3, color=col, alpha=0.18, lw=0, zorder=1)
            ax.plot(x, y3, marker="o", ms=4.2, lw=lw, color=col, zorder=3)
            ax.plot(x, yf, marker="s", ms=3.4, lw=lw * 0.6, ls=":", color=col, zorder=2)

        g = [statistics.fmean(three[t][k] - full[t][k] for k in speech) for t in have]
        print(f"[check] {win} taps={have}  gap {g[0]:+.3f} at {lo} -> {g[-1]:+.3f} at {hi}"
              f"   widens in {n_hold}/{len(speech)} tasks   |  {DUD} gap"
              f" {three[hi][DUD] - full[hi][DUD]:+.3f}")
        assert g[-1] > g[0], f"[check] VIOLATED gap did not widen with depth in {win}"
        assert abs(three[hi][DUD]) < 0.05, f"[check] VIOLATED visual control not flat in {win}"

        # Label the two ends of the claim in the panel, so the reader does not have to
        # measure the shaded band against the axis by eye.
        for k, tp in ((0, lo), (len(have) - 1, hi)):
            y3 = statistics.fmean(three[tp][t] for t in speech)
            yf = statistics.fmean(full[tp][t] for t in speech)
            ax.annotate("", xy=(k, y3), xytext=(k, yf),
                        arrowprops=dict(arrowstyle="<->", lw=0.9, color="k"))
            ax.text(k + 0.06, (y3 + yf) / 2, f"{y3 - yf:+.3f}", fontsize=7,
                    ha="left", va="center", fontweight="bold")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels(have)
        ax.set_xlim(-0.3, len(have) - 0.5)
        ax.set_title(f"{win[3:-1]} s window  ·  gap widens in {n_hold}/{len(speech)} tasks",
                     pad=6, fontsize=8)
        ax.set_xlabel("encoder depth")
    axes[0].set_ylabel("cross-subject alignment r")

    handles = [plt.Line2D([], [], color=PALETTE["ours"], lw=2.0, marker="o", ms=4.2,
                          label="3-PC subspace  ·  speech tasks"),
               plt.Line2D([], [], color=PALETTE["ours"], lw=1.2, ls=":", marker="s", ms=3.4,
                          label="full feature space  ·  speech tasks"),
               plt.Line2D([], [], color=PALETTE["muted"], lw=1.4, marker="o", ms=4.2,
                          label="3-PC  ·  visual control"),
               plt.Line2D([], [], color=PALETTE["muted"], lw=0.9, ls=":", marker="s", ms=3.4,
                          label="full space  ·  visual control")]
    fig.legend(handles=handles, fontsize=6.6, loc="upper center",
               bbox_to_anchor=(0.5, 0.10), ncol=2, columnspacing=1.6, handlelength=2.4)
    fig.suptitle("Cross-subject agreement barely grows -- it CONCENTRATES into 3 dimensions",
                 fontsize=8.5, y=1.01)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); _save(fig, "fig4_concentration")
    print("[check] OK gap widens in both windows; per-task " +
          ", ".join(f"{w} {n}/{d}" for w, n, d in holds))


def _save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT.relative_to(ROOT)}/{stem}.pdf|.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["2", "2ws", "2cs", "3", "4"], default=None)
    a = ap.parse_args()
    _style()
    for n, fn in (("2", fig2), ("2ws", fig2_ws), ("2cs", fig2_cs), ("3", fig3), ("4", fig4)):
        if a.only in (None, n):
            print(f"=== fig {n}")
            fn()
