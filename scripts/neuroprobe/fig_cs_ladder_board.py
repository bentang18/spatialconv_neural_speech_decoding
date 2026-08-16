"""CROSS-SUBJECT: where the model lands on the board, and whether depth carries every subject.

Left  the cross-subject leaderboard, our enc12 and our zero-parameter enc0 floor placed in it.
Right the depth ladder, one line per SUBJECT, with the board top drawn as a rule.

WHAT CHANGED FROM THE r6-ERA VERSION (`paper_figs_r6.fig2`), and why this is a separate file:

  FIVE TAPS, NOT FOUR.  The cd55k cache carries enc0/enc3/enc6/enc9/enc12, so the ladder is drawn
      at every tap that exists.  enc9 was never encoded in the r6 era, which is the only reason
      that figure stops at four.
  SHARDS, NOT A MERGED JSON.  `paper_figs_r6` reads `results_v3_board_cdlin_45k.json`, and that
      file is NOT the d=256 canonical -- it reports cs .6036 where the ledger says .6094.  Reading
      the shards is the standing rule for every r6-era number and there is no reason to relax it
      for a new checkpoint.
  THE BOARD IS RECOMPUTED, NOT HARDCODED.  Every published entry is averaged out of the vendored
      leaderboard JSON exactly the way `Overall` is, so a subset macro and the drawn macro are the
      same arithmetic.  The r6 file hardcodes seven floats with no way to re-derive them.

SELECTION RULE for the leaderboard bars, applied before any number is looked at: every model
FAMILY, represented by its best-scoring variant.  `BrainBERT (untrained)` is kept SEPARATE from
`BrainBERT` because it is a random-init control, not a tuning variant.  Nothing is dropped for
being weak.

Null on the right panel: the two counts printed in the title are DIFFERENT questions and are
reported separately on purpose.  "enc12 > enc0" asks whether depth pays at all; "monotone at every
tap" asks whether it pays at every rung.  The macro can be monotone while most subjects are not,
and conflating the two would overclaim.
"""
import argparse
import collections
import json
import pathlib
import re
import statistics

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # NeurIPS rejects Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

BOARD_JSON = pathlib.Path("../neuroprobe/analyses/figures/neuroprobe_eval_leaderboard_CrossSubject.json")
OUT = pathlib.Path("results/showcase/1_beats_the_board")
TAPS = ["enc0", "enc3", "enc6", "enc9", "enc12"]
WANT_CELLS, WANT_TASKS = 10, 15
OURS, FLOOR, ACCENT = "#1f4e79", "#9aa5b1", "#c1440e"

FAMILIES = (
    ("DIVER-1", lambda k: "DIVER-1" in k),
    ("PopT", lambda k: "opulation" in k),
    ("CNN", lambda k: k.startswith("CNN_")),
    ("MLP", lambda k: k.startswith("MLP_")),
    ("Linear", lambda k: k.startswith("Linear (")),
    ("BrainBERT", lambda k: k.startswith("BrainBERT") and "untrained" not in k),
    ("BrainBERT (untrained)", lambda k: "untrained" in k),
    ("GLIS-GNN", lambda k: k.startswith("GLIS-GNN")),
)
PRETTY = {"CNN": "CNN (Laplacian-STFT)", "PopT": "PopT (Laplacian-STFT)",
          "MLP": "MLP (Laplacian-STFT)", "Linear": "Linear (Laplacian-STFT)",
          "BrainBERT": "BrainBERT (frozen)", "DIVER-1": "DIVER-1"}


def board() -> dict[str, float]:
    """Best variant of each family, averaged the way the leaderboard's own `Overall` is."""
    bd = json.loads(BOARD_JSON.read_text())
    tasks = [k for k in bd if k != "Overall"]
    assert len(tasks) == WANT_TASKS, f"board exposes {len(tasks)} tasks, want {WANT_TASKS}"
    macro = {}
    for model in bd["Overall"]:
        per = [bd[t][model]["mean"] for t in tasks if model in bd[t]]
        assert len(per) == WANT_TASKS, f"{model} is on {len(per)}/{WANT_TASKS} tasks"
        macro[model] = statistics.fmean(per)
    out = {}
    for name, match in FAMILIES:
        hit = [(k, v) for k, v in macro.items() if match(k)]
        if hit:
            out[PRETTY.get(name, name)] = max(v for _, v in hit)
    return out


def load(shards: pathlib.Path, arm: str) -> dict[str, dict[str, float]]:
    """cell -> tap -> macro over the 15 tasks, straight from the shards.

    The arm tag inside every cell key is checked against the one asked for. Arm mixing is the
    failure mode this pipeline is most exposed to: a shard from another arm would otherwise be
    averaged in silently and the figure would still look fine.
    """
    per: dict[str, dict[str, dict[str, float]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict))
    for f in sorted(shards.glob("cs_*.json")):
        d = json.loads(f.read_text())
        cell = d["name"]
        for key, blk in d["cells"].items():
            tag, task = key.split("|", 1)
            assert tag == arm, f"ARM MIXING: {f} carries tag {tag}, want {arm}"
            for tapnorm, v in blk["cells"].items():
                tap, norm = tapnorm.split("|")
                if norm == "std" and tap in TAPS:
                    per[cell][tap][task] = v["test"]
    assert len(per) == WANT_CELLS, f"{len(per)} cells, want {WANT_CELLS} — a partial grid lies"
    out = {}
    for cell, taps in per.items():
        assert set(taps) == set(TAPS), f"{cell} has taps {sorted(taps)}, want {TAPS}"
        for tap, byt in taps.items():
            assert len(byt) == WANT_TASKS, f"{cell}/{tap} on {len(byt)}/{WANT_TASKS} tasks"
        out[cell] = {t: statistics.fmean(taps[t].values()) for t in TAPS}
    return dict(sorted(out.items()))


def by_subject(cells: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Collapse session cells (S1T1, S1T2) onto their subject, averaging the taps.

    Ten session lines is more clutter than argument: the reader is asked whether depth helps
    ACROSS PEOPLE, and two trials of one person are not two pieces of evidence for that. The
    black macro stays the unweighted mean over SESSIONS, which is the board's own unit, so this
    collapse moves only the grey lines.
    """
    groups: dict[str, list[dict[str, float]]] = {}
    for cell, v in cells.items():
        m = re.fullmatch(r"S(\d+)T\d+", cell)
        assert m, f"cannot read a subject out of cell name {cell!r}"
        groups.setdefault(f"S{m.group(1)}", []).append(v)
    return {s: {t: statistics.fmean(v[t] for v in vs) for t in TAPS}
            for s, vs in sorted(groups.items(), key=lambda kv: int(kv[0][1:]))}


def slopegraph(ax, subj, macro, top, top_label) -> int:
    """Per-subject lines across taps + the macro in black. Returns the monotone count.

    Labels are pushed apart bottom-up by a fixed gap and leader-lined back to their true value,
    so the de-collision nudge can never be misread as data.
    """
    mono = 0
    for v in subj.values():
        ys = [v[t] for t in TAPS]
        up = all(a <= b for a, b in zip(ys, ys[1:]))
        mono += up
        ax.plot(range(len(TAPS)), ys, marker="o", ms=3, lw=.9,
                color=OURS if up else ACCENT, alpha=.7)
    order = sorted(subj.items(), key=lambda kv: kv[1][TAPS[-1]])
    span = order[-1][1][TAPS[-1]] - order[0][1][TAPS[-1]]
    gap, y_prev, x = .036 * span, -1e9, len(TAPS) - 1
    for s, v in order:
        y = max(v[TAPS[-1]], y_prev + gap)
        y_prev = y
        ax.plot([x + .02, x + .06], [v[TAPS[-1]], y], lw=.4, color="#bbb", zorder=1)
        ax.text(x + .07, y, s, fontsize=5.6, va="center", color="#555")
    ax.plot(range(len(TAPS)), [macro[t] for t in TAPS], marker="o", ms=6, lw=2.6,
            color="k", zorder=5, label="macro")
    ax.axhline(top, color=ACCENT, lw=.9, ls="--")
    below = macro[TAPS[0]] > top
    ax.text(-.26, top + (-.0012 if below else .0012), top_label, color=ACCENT, fontsize=6.4,
            ha="left", va="top" if below else "bottom")
    ax.set_xticks(range(len(TAPS)))
    ax.set_xticklabels(TAPS, fontsize=7)
    ax.set_xlim(-.28, x + .34)
    return mono


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="board_vits384_cd55k")
    ap.add_argument("--shards", default="results/board/shards_board_vits384_cd55k")
    ap.add_argument("--stem", default="fig_cs_ladder_board_vits384_cd55k")
    args = ap.parse_args()

    cells = load(pathlib.Path(args.shards), args.arm)
    macro = {t: statistics.fmean(v[t] for v in cells.values()) for t in TAPS}
    bd = board()
    top = max(bd.values())
    top_name = max(bd, key=lambda k: bd[k])

    print(f"[check] OK {len(cells)} cells x {WANT_TASKS} tasks x {len(TAPS)} taps, arm {args.arm}")
    print("[macro] " + "  ".join(f"{t} {macro[t]:.4f}" for t in TAPS))
    # A ladder that is monotone in the macro is the claim; a break is a result, not a crash.
    breaks = [(a, b) for a, b in zip(TAPS, TAPS[1:]) if macro[a] > macro[b]]
    print(f"[macro] monotone across all five taps: {not breaks}"
          + (f"  🔴 BREAKS AT {breaks}" if breaks else ""))
    print(f"[board] top = {top_name} {top:.4f}; our enc12 clears it: {macro['enc12'] > top}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9), width_ratios=[1.25, 1])
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    rows = sorted(list(bd.items())
                  + [("ours enc0 (0 params)", macro["enc0"]), ("ours enc12", macro["enc12"])],
                  key=lambda r: r[1])
    cols = [OURS if n == "ours enc12" else FLOOR if n.startswith("ours") else
            ACCENT if abs(v - top) < 1e-9 else "#bbb" for n, v in rows]
    ax.barh(range(len(rows)), [v for _, v in rows], color=cols, height=.66)
    for i, (_, v) in enumerate(rows):
        ax.text(v + .0015, i, f"{v:.4f}", va="center", fontsize=6.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows], fontsize=6.6)
    ax.set_xlim(.50, .635)
    ax.set_xlabel(f"cross-subject macro AUROC ({len(cells)}/{WANT_CELLS} cells)", fontsize=8)
    ax.set_title("Cross-subject leaderboard", pad=6, fontsize=8.5)

    subj = by_subject(cells)
    mono = slopegraph(ax2, subj, macro, top, f"board top {top:.4f}")
    beat = sum(v["enc12"] > v["enc0"] for v in subj.values())
    print(f"[check] {len(cells)} sessions -> {len(subj)} subjects: enc12 > enc0 in "
          f"{beat}/{len(subj)}, monotone at every tap in {mono}/{len(subj)}")
    ax2.set_ylabel("cross-subject macro AUROC", fontsize=8)
    ax2.set_title(f"Per subject: enc12 > enc0 in {beat}/{len(subj)}\n"
                  f"(monotone at every tap in {mono})", pad=6, fontsize=8)
    ax2.legend(fontsize=6.5, loc="upper left")
    for a in (ax, ax2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(labelsize=7)
    fig.text(.5, -.05,
             f"one line per SUBJECT (sessions averaged)  ·  orange = not monotone across taps "
             f"(still ends above enc0)  ·  {args.arm}  ·  15 Lite tasks  ·  recomputed from shards",
             ha="center", fontsize=6.2, color=ACCENT)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{args.stem}.{ext}", bbox_inches="tight", dpi=300)
    print(f"\nwrote {OUT}/{args.stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
