"""Retrieval similarity matrices as one grid: rows are tasks, columns are encoder depth.

The single-panel `figR_*` images answer "does this task retrieve at this depth". The claim
that actually matters is the DERIVATIVE -- the diagonal sharpening from enc0 to enc12 -- and
that claim lives across sixteen separate files where nobody can see it. This draws them in
one frame.

Two things make it an argument rather than a montage:

* **One shared colour scale for every cell.** Per-panel normalisation would sharpen every
  diagonal, dud included, purely by rescaling. The progression has to survive a fixed scale
  or it is not a progression. The single-panel figures normalise per panel, which is why they
  cannot be cropped and pasted into a grid.
* **A null task as the bottom row.** Retrieval is basis-free, so a task outside the trajectory
  menu costs nothing to include -- no basis is refit, nothing else on the page moves. If depth
  sharpened diagonals as a generic side effect of smoother deep features, the null row would
  sharpen too. It is the falsifier, and it belongs in the same frame as the effect.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.neuroprobe.viz_common import load_all, shared_lobes  # noqa: E402
from scripts.neuroprobe.viz_figures import retrieval_sims  # noqa: E402


def _time_ticks(t_len: int, hz: float, offset: float, step: float = 0.5):
    """Frame indices and second-labels at `step`-second marks inside the window.

    Anchored on t=0 rather than on frame 0, so the onset tick is always present and always
    labelled '0' -- the reader is looking for where the word starts, not where the array does.
    """
    if not hz:
        return [], []
    k0, k1 = np.ceil(offset / step), np.floor((offset + t_len / hz) / step)
    secs = [round(k * step, 3) for k in np.arange(k0, k1 + 1)]
    ticks = [(s - offset) * hz for s in secs]
    keep = [(t, s) for t, s in zip(ticks, secs) if -0.5 <= t <= t_len - 0.5]
    return [t for t, _ in keep], [f"{s:g}" for _, s in keep]


def _cells(sessions, lobes, taps, tasks, n_pre):
    """(task, tap) -> (mean similarity matrix, stats). Missing combinations are dropped."""
    out = {}
    for task in tasks:
        for tap in taps:
            sims, stats = retrieval_sims(sessions, lobes, tap, task, n_pre=n_pre)
            if stats:
                out[(task, tap)] = (np.mean(sims, axis=0), stats)
    return out


def grid(sessions, lobes, taps, tasks, out_path: str, *, n_pre: int | None = None,
         hz: float = 32.0, offset: float = 0.0, null_task: str | None = None,
         cmap: str = "RdBu_r", arm: str = "") -> dict:
    cells = _cells(sessions, lobes, taps, tasks, n_pre)
    assert cells, "no (task, tap) combination produced a similarity matrix"
    tasks = [t for t in tasks if any((t, tp) in cells for tp in taps)]

    # ONE scale for the whole grid. Symmetric about zero so the colour map's midpoint is
    # "unrelated", which is the reference the eye should be reading against.
    hi = max(float(np.abs(m).max()) for m, _ in cells.values())
    nrow, ncol = len(tasks), len(taps)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.65 * ncol + 1.1, 2.65 * nrow + 0.9),
                             squeeze=False)
    im = None
    for i, task in enumerate(tasks):
        for j, tap in enumerate(taps):
            ax = axes[i][j]
            got = cells.get((task, tap))
            if got is None:
                ax.axis("off")
                continue
            m, st = got
            im = ax.imshow(m, cmap=cmap, vmin=-hi, vmax=hi, interpolation="nearest")
            t_len = st["n_frames"]
            # Where a perfect match would sit. Without it the eye has no reference for which
            # way "tighter" is, and a bright off-diagonal band reads the same as a good one.
            ax.plot([0, t_len - 1], [0, t_len - 1], color="#111", lw=0.8, ls=":", alpha=0.6)
            if n_pre:
                # Dark, not white: a diverging map is white at zero, so white rules vanish
                # exactly where the contrast is weakest.
                ax.axhline(n_pre - 0.5, color="#111", lw=0.6, alpha=0.5)
                ax.axvline(n_pre - 0.5, color="#111", lw=0.6, alpha=0.5)
            xr = st["top1"] / st["chance"]
            ax.set_title(f"{xr:.1f}x chance · rank {st['median_rank']:.0f}/{t_len}",
                         fontsize=7.5, pad=3)
            # Ticks in SECONDS, not frame index. Without them the t=0 rule is just a white
            # line and the reader cannot tell whether the bright block sits before or after
            # the word -- which is the whole question the panel is being asked.
            ticks, labels = _time_ticks(t_len, hz, offset)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(labels if i == nrow - 1 else [], fontsize=6.5)
            ax.set_yticklabels(labels if j == 0 else [], fontsize=6.5)
            ax.tick_params(length=2, width=0.5, pad=1.5)
            if i == nrow - 1:
                ax.set_xlabel("subject B time (s)", fontsize=7.5, labelpad=1)
            if i == 0:
                ax.text(0.5, 1.30, tap, transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel("subject A time (s)", fontsize=7.5, labelpad=1)
                # Task name goes OUTSIDE the ylabel, which now belongs to the tick units.
                lab = task + ("\n(null control)" if task == null_task else "")
                ax.text(-0.42, 0.5, lab, transform=ax.transAxes, ha="center", va="center",
                        rotation=90, fontsize=9,
                        fontweight="bold" if task == null_task else "normal")
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.015)
        cb.set_label("mean cross-subject cosine  ·  white = unrelated  ·  "
                     "one scale for every cell", fontsize=8)
    fig.suptitle("Cross-subject token retrieval, depth left to right"
                 "  ·  dotted line = the correct match"
                 "  ·  solid rules = word onset (t=0)", fontsize=10.5)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    # The matrices go to disk beside the figure so a cosmetic re-render never has to
    # recompute them, and so the arm that produced them travels with the numbers.
    mat_path = os.path.splitext(out_path)[0] + "_mats.npz"
    np.savez_compressed(mat_path, **{f"{t}|{tp}": m for (t, tp), (m, _) in cells.items()})
    return {"tasks": tasks, "taps": list(taps), "vmax": hi, "null_task": null_task,
            "arm": arm, "cmap": cmap, "hz": hz, "offset": offset, "n_pre": n_pre,
            "matrices": os.path.basename(mat_path),
            "cells": {f"{t}|{tp}": {k: v for k, v in st.items() if k != "n_pairs"}
                      for (t, tp), (_, st) in cells.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--red-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--taps", default="enc0,enc3,enc6,enc12")
    ap.add_argument("--tasks", default="onset,delta_volume,word_index,frame_brightness",
                    help="rows, top to bottom; put the null control last")
    ap.add_argument("--null-task", default="frame_brightness",
                    help="row labelled as the control; '' to label none")
    ap.add_argument("--n-pre", type=int, default=0)
    ap.add_argument("--hz", type=float, default=32.0)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--cmap", default="RdBu_r",
                    help="diverging, centred on zero: the sign of the cosine IS the story")
    ap.add_argument("--arm", default="",
                    help="training arm / checkpoint tag, recorded in the sidecar")
    args = ap.parse_args()

    sessions = load_all(args.red_dir)
    lobes = shared_lobes(sessions)
    assert lobes, "no lobe shared by every subject"
    taps = [t for t in args.taps.split(",") if t and any(t in s.shapes for s in sessions)]
    tasks = [t for t in args.tasks.split(",") if t]
    print(f"[load] {len(sessions)} sessions, taps {taps}, lobes {lobes}", flush=True)

    info = grid(sessions, lobes, taps, tasks, args.out, n_pre=args.n_pre or None,
                hz=args.hz, offset=args.offset, null_task=args.null_task or None,
                cmap=args.cmap, arm=args.arm or os.path.basename(args.red_dir.rstrip("/")))
    for t in info["tasks"]:
        row = [info["cells"].get(f"{t}|{tp}") for tp in taps]
        print(f"[grid] {t:18s} " + "  ".join(
            "   --   " if c is None else f"{c['top1'] / c['chance']:4.1f}x r{c['median_rank']:.0f}"
            for c in row))
    print(f"[write] {args.out} (shared scale +/-{info['vmax']:.3f})")
    side = os.path.splitext(args.out)[0] + ".json"
    with open(side, "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"[write] {side}")


if __name__ == "__main__":
    main()
