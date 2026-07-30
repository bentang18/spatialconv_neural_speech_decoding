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
OUT = RES / "showcase"
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
    fig.tight_layout(); _save(fig, "fig2_ladder", sub="1_beats_the_board")


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
    fig.tight_layout(); _save(fig, "fig2ws_within_session", sub="1_beats_the_board")


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
    fig.tight_layout(); _save(fig, "fig2cs_cross_session", sub="1_beats_the_board")


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
    # The numeric relation above holds, but its old reading does NOT: a rank-1 deletion at
    # d~93k is free by construction (a matched shuffled-label control costs -0.000056 against
    # identity's +0.000005), and rank <= C-1 makes rank 1 the maximum available in CS. So this
    # null has no power and cannot speak to separability -- a null is what orthogonality
    # predicts. var_removed is likewise between-cloud mean separation, 0.099% within-session.
    print("[check] OK enc0 erasure hurts, enc12 is a null -- but this null has NO POWER "
          "(matched shuffle is equally free) so it is NOT evidence about separability")

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
    fig.tight_layout(); _save(fig, "fig3_leace", sub="retracted")


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
    fig.tight_layout(rect=(0, 0.12, 1, 1)); _save(fig, "fig4_concentration", sub="4_why_it_transfers")
    print("[check] OK gap widens in both windows; per-task " +
          ", ".join(f"{w} {n}/{d}" for w, n, d in holds))


# ------------------------------------------------- fig 5/6: the per-task breakdown
# Why these two figures exist: the macro ladder in fig 2 is a single number, and a single
# number cannot answer "are you only improving the trivially easy perceptual tasks". That is a
# per-TASK question, so it gets per-task panels. CS is the only regime that can carry them --
# it is the one with all four taps (WS/CSession wrote elec taps 0 and 12 only).
#
# FAMILY LABELS ARE THE LEDGER'S, NOT REFIT HERE. event/level is the established split
# (`project-what-transfers-is-change-coded-not-speech-2026-07-29`): does the label track a
# CHANGE/EVENT or a SUSTAINED LEVEL. It is transcribed, so these figures cannot quietly
# invent a grouping that flatters the ladder.
EVENT = ("onset", "speech", "delta_volume", "word_index", "word_head_pos",
         "word_length", "gpt2_surprisal", "word_gap", "word_part_speech")
LEVEL = ("volume", "pitch", "local_flow", "global_flow", "face_num", "frame_brightness")
# Modality is a SEPARATE axis from family and is tagged, not coloured: `volume`/`pitch` are
# acoustic yet group with the visual tasks because they are levels. Collapsing the two axes
# into one colour is exactly the confusion the ledger warns about.
VISUAL = ("local_flow", "global_flow", "face_num", "frame_brightness")
FAM_COLOR = {"event": "#1f4e79", "level": "#d98324"}


def _cells_by_task(path: pathlib.Path) -> dict[str, dict[str, dict[str, float]]]:
    """task -> tap -> cell -> AUROC, on the cell set shared by all four taps.

    Same intersection discipline as `ladder`: "partial cells lie" applies per task as hard as
    it does to the macro.
    """
    d = json.load(open(path))
    out: dict[str, dict[str, dict[str, float]]] = {}
    for key, blob in d.items():
        cs = blob.get("cs")
        if not cs:
            continue
        cols = {t: cs.get(f"{t}|std") for t in TAPS}
        if any(c is None for c in cols.values()):
            continue
        shared = sorted(set.intersection(*(set(c) for c in cols.values())))
        if shared:
            out[key.split("|")[1]] = {t: {c: cols[t][c] for c in shared} for t in TAPS}
    return out


def _k(cells: dict[str, dict[str, float]], tap: str) -> float:
    """Scale-free gain at `tap`: no-intercept slope of (tap - .5) on (enc0 - .5) over cells.

    THIS IS THE UNIT THE PER-TASK COMPARISON MUST USE, and the reason is the whole point of
    fig 6. A task's raw AUROC gain is bounded by its headroom: `onset` sits ~.245 above chance
    and `word_gap` ~.032, so the SAME multiplicative improvement prints ~8x larger on `onset`.
    Plotting raw AUROC points per task therefore manufactures "it only helps the easy tasks"
    out of the axis choice alone.

    A per-cell ratio would be the naive fix and it is a trap -- dividing by an enc0 that sits
    .003 above chance explodes (`face_num` -> 2.8, `frame_brightness` -> negative). The
    no-intercept slope is the same quantity weighted by (enc0 - .5)^2, so near-chance cells
    contribute in proportion to how much signal they actually had. k = 1 means "depth changed
    nothing", k > 1 a multiplicative gain, k < 1 active degradation. k(enc0) == 1 by identity.
    """
    x = cells["enc0"]
    y = cells[tap]
    num = sum((x[c] - 0.5) * (y[c] - 0.5) for c in x)
    den = sum((x[c] - 0.5) ** 2 for c in x)
    return num / den


# The ledger's per-task k table, transcribed from
# `project-what-transfers-is-change-coded-not-speech-2026-07-29`. Asserted, not recomputed
# into agreement: if the estimator here ever drifts from the one that produced the finding,
# this figure must fail loudly rather than publish a second, quietly different number.
LEDGER_K = {
    "word_length": 1.557, "word_head_pos": 1.330, "word_gap": 1.320, "gpt2_surprisal": 1.274,
    "word_index": 1.259, "onset": 1.252, "speech": 1.229, "delta_volume": 1.146,
    "word_part_speech": 1.116, "volume": 1.068, "local_flow": 1.032, "global_flow": 0.932,
    "face_num": 0.827, "frame_brightness": 0.746, "pitch": 0.642,
}


def per_task_cs() -> dict[str, dict]:
    """task -> raw macros per tap, k per tap, paired raw delta, family. Averaged over 4 ckpts.

    Four checkpoints, because a per-task number off ONE checkpoint is a coin flip at this
    effect size. They are one training TRAJECTORY, not four seeds -- so this controls for
    step/cooldown and for nothing else, which is why no seed claim is made anywhere here.
    """
    boards = {name: _cells_by_task(RES / "r6_era/board" / f) for name, f in CKPTS.items()}
    tasks = sorted(set.intersection(*(set(b) for b in boards.values())))
    out: dict[str, dict] = {}
    for t in tasks:
        per_ck = [boards[n][t] for n in CKPTS]
        ncell = {len(c["enc0"]) for c in per_ck}
        assert ncell == {10}, f"[check] VIOLATED {t} cell counts {ncell}, want 10 everywhere"
        row: dict = {"cells": 10, "family": "event" if t in EVENT else "level"}
        for tap in TAPS:
            row[tap] = statistics.fmean(statistics.fmean(c[tap].values()) for c in per_ck)
            row[f"k_{tap}"] = statistics.fmean(_k(c, tap) for c in per_ck)
        # Paired per cell: the SEM of the per-cell DIFFERENCE, not a combination of two
        # marginal SEMs. Unpaired bars on a paired contrast overstate the uncertainty.
        diffs = [statistics.fmean(c["enc12"][x] - c["enc0"][x] for c in per_ck)
                 for x in per_ck[0]["enc0"]]
        row["delta"] = statistics.fmean(diffs)
        row["delta_sem"] = statistics.stdev(diffs) / len(diffs) ** 0.5
        ks = [_k(c, "enc12") for c in per_ck]
        row["k_spread"] = max(ks) - min(ks)
        out[t] = row
    return out


def board_per_task() -> tuple[dict[str, float], dict[str, float], str]:
    """(best-of-any-entry per task, top-macro entry per task, top-macro entry name).

    Two different bars, and the difference matters. `best` is a max over 10 submissions taken
    per task INDEPENDENTLY, so it is a hypothetical model nobody submitted and it is biased in
    the board's favour -- that is the strict bar. `top` is the single highest-macro entry (the
    CNN), which is the honest like-for-like opponent. Reporting only one of them would be
    picking the comparison after seeing the answer.
    """
    lb = json.load(open(RES / "neuroprobe_leaderboard_cs.json"))
    for name, v in lb.items():
        assert v["n_cells"] == 10 and v["n_tasks"] == 15, \
            f"[check] VIOLATED leaderboard entry {name} is not 10 cells x 15 tasks"
    top_name, top = max(lb.items(), key=lambda kv: kv[1]["macro"])
    tasks = top["per_task"].keys()
    best = {t: max(v["per_task"][t] for v in lb.values()) for t in tasks}
    return best, dict(top["per_task"]), top_name


def _pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den


def _spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    return _pearson(rank(xs), rank(ys))


def fig5() -> None:
    """Per-task depth ladder in SCALE-FREE units. The figure Greg's objection asks for.

    The ladder is drawn in k, not AUROC points, and that is the one design decision in this
    figure that matters -- see `_k`. In AUROC points the panels would be ordered almost
    perfectly by how much headroom each task started with, which is a fact about the axis and
    not about the encoder. Raw AUROC is still printed inside every panel, so the absolute
    scale is available and the normalisation hides nothing.
    """
    rows = per_task_cs()
    assert len(rows) == 15, f"[check] VIOLATED need all 15 Lite tasks, got {len(rows)}"
    assert set(rows) == set(LEDGER_K), "[check] VIOLATED task menu != the ledger's k table"
    for t, r in rows.items():
        assert abs(r["k_enc12"] - LEDGER_K[t]) < 5e-4, (
            f"[check] VIOLATED k({t}) = {r['k_enc12']:.4f} but the ledger says "
            f"{LEDGER_K[t]:.3f} -- the estimator has drifted from the finding")
        assert abs(r["k_enc0"] - 1.0) < 1e-9, f"[check] VIOLATED k(enc0) != 1 for {t}"
    print(f"[check] OK all 15 per-task k reproduce the ledger table to <5e-4")

    ev = sorted(rows[t]["k_enc12"] for t in EVENT)
    lv = sorted(rows[t]["k_enc12"] for t in LEVEL)
    assert ev[0] > lv[-1], "[check] VIOLATED the 9/9 event-over-level split does not hold"
    print(f"[check] OK 9/9 split, ZERO interleaving: event k >= {ev[0]:.3f} "
          f"(mean {statistics.fmean(ev):.3f}) > level k <= {lv[-1]:.3f} "
          f"(mean {statistics.fmean(lv):.3f})")

    order = sorted(rows, key=lambda t: -rows[t]["k_enc12"])
    rank = {t: i + 1 for i, t in enumerate(order)}
    print(f"[check] in k units onset ranks {rank['onset']}/15 and speech {rank['speech']}/15 "
          f"-- MID-PACK, beaten by " + ", ".join(order[:rank['onset'] - 1]))
    assert rank["onset"] > 4 and rank["speech"] > 4, (
        "[check] VIOLATED onset/speech are top-4 in k units, so the objection would stand")

    fig, axes = plt.subplots(3, 5, figsize=(9.8, 5.6), sharey=True)
    # SHARED y-axis, which only becomes possible in k units -- the whole menu lives in
    # 0.6..1.6. In AUROC points a shared axis was impossible (tasks span .49 to .82), which is
    # itself a sign that AUROC points are the wrong unit for a 15-panel comparison.
    lo = min(r[f"k_{tp}"] for r in rows.values() for tp in TAPS)
    hi = max(r[f"k_{tp}"] for r in rows.values() for tp in TAPS)
    pad = 0.09 * (hi - lo)
    for ax, t in zip(axes.ravel(), order):
        r = rows[t]
        col = FAM_COLOR[r["family"]]
        ys = [r[f"k_{tp}"] for tp in TAPS]
        ax.axhline(1.0, color="k", lw=0.7, ls=":", alpha=0.6, zorder=1)
        ax.plot(range(4), ys, marker="o", ms=3.6, lw=1.6, color=col, zorder=3)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.35, 3.35)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["0", "3", "6", "12"], fontsize=6)
        ax.tick_params(labelsize=6)
        tag = " ·visual" if t in VISUAL else ""
        ax.set_title(f"{rank[t]}. {t.replace('_', ' ')}{tag}", fontsize=7.2, color=col, pad=3)
        ax.text(0.04, 0.955, f"k {r['k_enc12']:.3f}", fontsize=6.4, fontweight="bold",
                transform=ax.transAxes, va="top", ha="left", color=col)
        # The raw numbers stay ON the panel. Normalising the axis is a fix for a misleading
        # comparison, not a licence to withhold the absolute effect: a reader must be able to
        # see that word_length's k of 1.56 is +.026 AUROC on a task that starts at .529.
        # Placed in whichever vertical half the trace is NOT in -- the declining level panels
        # (pitch, frame_brightness, face_num) run straight through a fixed bottom-right slot.
        high = statistics.fmean(ys) > (lo + hi) / 2
        ax.text(0.96, 0.05 if high else 0.95,
                f"{r['enc0']:.3f}→{r['enc12']:.3f}\nΔ{r['delta']:+.4f}", fontsize=5.5,
                transform=ax.transAxes, va="bottom" if high else "top", ha="right",
                color="#555")
    for ax in axes[:, 0]:
        ax.set_ylabel("k  (scale-free gain)", fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("encoder depth", fontsize=7)

    handles = [plt.Line2D([], [], color=FAM_COLOR["event"], lw=1.6, marker="o", ms=3.6,
                          label="event / change-coded label (9)"),
               plt.Line2D([], [], color=FAM_COLOR["level"], lw=1.6, marker="o", ms=3.6,
                          label="sustained-level label (6)"),
               plt.Line2D([], [], color="k", lw=0.7, ls=":", alpha=0.6,
                          label="k = 1: depth changed nothing")]
    fig.legend(handles=handles, fontsize=6.7, loc="upper center", bbox_to_anchor=(0.5, 0.055),
               ncol=3, columnspacing=1.8, handlelength=2.6)
    fig.suptitle("Cross-subject depth ladder, per task, in scale-free units  ·  panels ranked "
                 f"by k(enc12)  ·  10/10 held-out cells  ·  mean of {len(CKPTS)} checkpoints",
                 fontsize=8.2, y=1.0)
    fig.text(0.5, 0.012, f"k = no-intercept slope of (AUROC−.5) on (enc0−.5) over the 10 cells, "
             f"so k(enc0)=1 by construction  ·  all 9 event tasks rank above all 6 level tasks "
             f"(k ≥ {ev[0]:.3f} vs ≤ {lv[-1]:.3f}), and onset/speech sit "
             f"{rank['onset']}th/{rank['speech']}th", ha="center", fontsize=6.3,
             color=PALETTE["muted"])
    fig.tight_layout(rect=(0, 0.075, 1, 0.985))
    _save(fig, "fig5cs_per_task_ladder", sub="2_what_pretraining_does")


def fig6() -> None:
    """"It only helps the easy tasks" -- three panels: concede, dissolve, then independent check.

    A concedes the objection in the units it was raised in. B shows it is an artifact of those
    units. C stops arguing about our internal ladder altogether and asks the question the paper
    actually rests on -- where do we beat the BOARD -- which needs no normalisation at all,
    because it compares two models on the same task with the same headroom.
    """
    rows = per_task_cs()
    best, top, top_name = board_per_task()
    assert set(best) == set(rows), "[check] VIOLATED leaderboard task menu != ours"
    tasks = sorted(rows)
    x = [rows[t]["enc0"] for t in tasks]

    rho_raw = _spearman(x, [rows[t]["delta"] for t in tasks])
    r_raw = _pearson(x, [rows[t]["delta"] for t in tasks])
    rho_k = _spearman(x, [rows[t]["k_enc12"] for t in tasks])
    r_k = _pearson(x, [rows[t]["k_enc12"] for t in tasks])
    print(f"[check] easiness vs RAW Δ: pearson {r_raw:+.3f} spearman {rho_raw:+.3f}  "
          f"-> his read is CORRECT in AUROC points")
    print(f"[check] easiness vs k     : pearson {r_k:+.3f} spearman {rho_k:+.3f}  "
          f"-> the correlation DISSOLVES once the units are scale-free")
    assert r_raw > 0.6, "[check] VIOLATED panel A must CONCEDE a strong raw correlation"
    assert abs(r_k) < 0.45 and abs(r_k) < r_raw / 2, \
        "[check] VIOLATED panel B claims the correlation dissolves; it did not"

    marg = {t: rows[t]["enc12"] - best[t] for t in tasks}
    rho_m = _spearman(x, [marg[t] for t in tasks])
    ranks = sorted(tasks, key=lambda t: -marg[t])
    n_win = sum(v > 0 for v in marg.values())
    print(f"[check] margin over per-task board best: wins {n_win}/15; "
          f"onset {ranks.index('onset') + 1}/15 ({marg['onset']:+.4f}), "
          f"speech {ranks.index('speech') + 1}/15 ({marg['speech']:+.4f}); "
          f"biggest = {ranks[0]} ({marg[ranks[0]]:+.4f}); spearman(easiness, margin) {rho_m:+.3f}")
    assert marg["volume"] > marg["speech"], \
        "[check] VIOLATED panel C rests on volume out-margining speech"

    fig, (ax, axk, ax2) = plt.subplots(1, 3, figsize=(11.4, 3.6),
                                       width_ratios=[1, 1, 1.15])

    def _scatter(a, ys, ylab, title):
        for t in tasks:
            a.plot(rows[t]["enc0"], ys[t], marker="o", ms=4.8,
                   color=FAM_COLOR[rows[t]["family"]], zorder=3)
        # 15 labels in a crowded lower-left corner overlap into illegibility at a fixed offset,
        # and an unreadable label is the same as a missing one. Candidates are tried in order and
        # the first that does not collide with an already-placed box wins; measured against the
        # real renderer, so it holds at whatever dpi the PDF is written at.
        placed: list = []
        for t in sorted(tasks, key=lambda t: -ys[t]):
            for dx, dy, ha in ((3.4, 2.6, "left"), (-3.4, 2.6, "right"),
                               (3.4, -7.4, "left"), (-3.4, -7.4, "right")):
                txt = a.annotate(t.replace("_", " "), (rows[t]["enc0"], ys[t]), fontsize=5.6,
                                 xytext=(dx, dy), textcoords="offset points", color="#444",
                                 ha=ha)
                a.figure.canvas.draw()
                box = txt.get_window_extent()
                if not any(box.overlaps(b) for b in placed):
                    placed.append(box)
                    break
                txt.remove()
            else:
                placed.append(a.annotate(t.replace("_", " "), (rows[t]["enc0"], ys[t]),
                                         fontsize=5.6, xytext=(3.4, 2.6),
                                         textcoords="offset points",
                                         color="#444").get_window_extent())
        # The minimal pair is the within-difficulty control: the SAME physical signal read as a
        # level and as a rate of change, at nearly matched enc0. Pure difficulty predicts these
        # two move together.
        a.annotate("", xy=(rows["delta_volume"]["enc0"], ys["delta_volume"]),
                   xytext=(rows["volume"]["enc0"], ys["volume"]),
                   arrowprops=dict(arrowstyle="->", lw=1.1, color="k", shrinkA=6, shrinkB=6))
        a.set_xlabel("enc0 cross-subject AUROC  (how easy the task already is)")
        a.set_ylabel(ylab)
        a.set_title(title, fontsize=8, pad=6)

    _scatter(ax, {t: rows[t]["delta"] for t in tasks}, "Δ AUROC  (enc12 − enc0)",
             f"A. Conceded, in AUROC points\npearson {r_raw:+.2f}, spearman {rho_raw:+.2f}"
             " — gain tracks easiness")
    ax.axhline(0, color="k", lw=0.7)
    _scatter(axk, {t: rows[t]["k_enc12"] for t in tasks}, "k  (scale-free gain)",
             f"B. The same data, scale-free\npearson {r_k:+.2f}, spearman {rho_k:+.2f}"
             " — it dissolves")
    axk.axhline(1.0, color="k", lw=0.7, ls=":")

    ys = sorted(tasks, key=lambda t: marg[t])
    for i, t in enumerate(ys):
        named = t in ("onset", "speech")
        ax2.barh(i, marg[t], color=FAM_COLOR[rows[t]["family"]], height=0.66,
                 alpha=1.0 if named else 0.45)
        # Negative bars grow LEFT toward the tick labels, so their value goes on the free right
        # side of zero rather than on top of the task name.
        ax2.text(marg[t] + 0.0016 if marg[t] >= 0 else 0.0016, i, f"{marg[t]:+.4f}",
                 va="center", ha="left", fontsize=6.2)
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_yticks(range(len(ys)))
    ax2.set_yticklabels([f"{t.replace('_', ' ')}{'  ←' if t in ('onset', 'speech') else ''}"
                         for t in ys], fontsize=6.8)
    ax2.set_xlim(-0.032, 0.084)
    ax2.set_xlabel("our enc12 − best Neuroprobe entry for that task")
    # ρ is stated even though it is INCONVENIENT: margin still tracks easiness at +.62, so this
    # panel is not a clean refutation and must not be captioned as one. What it does show is
    # that the ordering is not the objection's ordering -- `speech` ranks 11th of 15 and the top
    # margin is a task the board leaves near chance.
    ax2.set_title(f"C. Independent: where we beat the BOARD\nwins {n_win}/15  ·  onset "
                  f"{ranks.index('onset') + 1}/15, speech {ranks.index('speech') + 1}/15 by "
                  f"margin  ·  ρ(easiness) {rho_m:+.2f}", fontsize=8, pad=6)

    handles = [plt.Line2D([], [], color=FAM_COLOR["event"], lw=0, marker="o", ms=5,
                          label="event / change-coded (9)"),
               plt.Line2D([], [], color=FAM_COLOR["level"], lw=0, marker="o", ms=5,
                          label="sustained-level (6)"),
               plt.Line2D([], [], color="k", lw=1.0,
                          label="arrow = volume → delta_volume (same signal, level vs change)"),
               plt.Line2D([], [], color="#888", lw=0, marker="s", ms=5,
                          label="solid bar = the two tasks the objection names")]
    fig.legend(handles=handles, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, 0.075),
               ncol=4, columnspacing=1.5, handlelength=2.2)
    fig.text(0.5, 0.008, "panel C needs no normalisation: it compares two models on the SAME "
             "task, so headroom cancels  ·  board bar is the strict one (per-task max over all "
             "10 submissions, a model nobody submitted)", ha="center", fontsize=6.2,
             color=PALETTE["muted"])
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _save(fig, "fig6cs_gain_vs_difficulty", sub="2_what_pretraining_does")


def _save(fig, stem: str, sub: str) -> None:
    """Write to ``showcase/<sub>/``, the chapter this figure argues for.

    ``sub`` is required, with no default, on purpose. The folders are the paper's argument in
    order, so placing a figure is an editorial decision, not a filesystem detail -- and fig3 is
    retracted, so a default would silently reinstate it beside live figures on the next run."""
    d = OUT / sub
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {d.relative_to(ROOT)}/{stem}.pdf|.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["2", "2ws", "2cs", "3", "4", "5", "6"], default=None)
    a = ap.parse_args()
    _style()
    for n, fn in (("2", fig2), ("2ws", fig2_ws), ("2cs", fig2_cs), ("3", fig3), ("4", fig4),
                  ("5", fig5), ("6", fig6)):
        if a.only in (None, n):
            print(f"=== fig {n}")
            fn()
