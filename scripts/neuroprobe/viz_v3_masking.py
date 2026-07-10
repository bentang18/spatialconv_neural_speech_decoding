"""Visualize the v3 electrode-tube mask on a REAL BrainTreebank session clip.

Loads one session's real surviving electrode labels + DK support, builds the v3
sidecar/geometry, samples the actual `sample_contact_mask`, and renders electrodes
(grouped by shaft, sorted by depth) x time. Red = masked. Because masking is
time-tubed (a masked electrode is hidden for the WHOLE clip), masked electrodes
appear as full-width red bands.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from speech_decoding.extractors.reference import parse_shaft
from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import (
    V3MaskConfig,
    sample_contact_mask,
)
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar
from speech_decoding.studies.braintreebank.anatomy import (
    aligned_voltage_support,
    voltage_electrode_order,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt-root", default="/work/ht203/data/braintreebank")
    ap.add_argument("--subject", type=int, default=3)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--t-slots", type=int, default=128)  # 4 s @ 32 Hz
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-lo", type=int, default=None)  # override cfg block_w_lo
    ap.add_argument("--block-hi", type=int, default=None)  # override cfg block_w_hi
    ap.add_argument("--whole-frac", type=float, default=None)  # override whole_shaft_frac
    ap.add_argument("--r-lo", type=float, default=None)  # override per-shaft r_lo
    ap.add_argument("--r-hi", type=float, default=None)  # override per-shaft r_hi
    ap.add_argument("--out", default="/work/ht203/v3_mask_viz.png")
    args = ap.parse_args()

    labels = voltage_electrode_order(args.bt_root, args.subject, args.trial)
    sup = aligned_voltage_support(
        args.bt_root, args.subject, trial_id=args.trial, unmapped_policy="zero"
    )
    support = torch.from_numpy(np.asarray(sup.support))
    sc = build_sidecar(labels, support=support)
    geom = build_l1_geometry(sc)
    n = len(labels)

    base = V3MaskConfig()
    cfg = V3MaskConfig(
        mask_frac=base.mask_frac,
        block_w_lo=args.block_lo if args.block_lo is not None else base.block_w_lo,
        block_w_hi=args.block_hi if args.block_hi is not None else base.block_w_hi,
        whole_shaft_frac=args.whole_frac if args.whole_frac is not None else base.whole_shaft_frac,
        r_lo=args.r_lo if args.r_lo is not None else base.r_lo,
        r_hi=args.r_hi if args.r_hi is not None else base.r_hi,
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    mask = sample_contact_mask(geom, n, n_rows=1, generator=g, cfg=cfg)[0]  # (N,)

    # grid: column = shaft, row = contact depth-rank (shallowest at top).
    # cell code: 0 = absent (shaft shorter than max), 1 = visible, 2 = masked.
    prefixes: list[str] = []
    wholes: list[bool] = []
    cols: list[np.ndarray] = []
    max_c = int(geom.max_c)
    for s in range(sc.n_shafts):
        idx = (sc.shaft_id == s).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(sc.depth[idx])]
        col = np.zeros(max_c)
        col[: len(idx)] = np.where(mask[idx].numpy(), 2.0, 1.0)
        cols.append(col)
        prefixes.append(parse_shaft(str(labels[idx[0].item()]))[0])
        wholes.append(bool(mask[idx].all()))
    grid = np.stack(cols, axis=1)  # (max_c, n_shafts)

    n_masked = int(mask.sum())
    frac = n_masked / n
    n_whole = sum(wholes)
    # melec = isolated masked contact (both same-shaft depth-neighbors visible)
    melec = 0
    for col in cols:
        present = col[col > 0]
        for j in range(len(present)):
            if present[j] == 2 and (j == 0 or present[j - 1] == 1) and (
                j == len(present) - 1 or present[j + 1] == 1
            ):
                melec += 1

    fig, ax = plt.subplots(
        figsize=(max(7, sc.n_shafts * 0.62), max(5, max_c * 0.32))
    )
    cmap = ListedColormap(["#ffffff", "#e4ebf1", "#d62728"])  # absent, visible, masked
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")

    # thin cell borders so individual contacts read as cells
    ax.set_xticks(np.arange(-0.5, sc.n_shafts, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, max_c, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(range(sc.n_shafts))
    ax.set_xticklabels(
        [f"{p}{'  ●' if w else ''}" for p, w in zip(prefixes, wholes)],
        rotation=90, fontsize=9,
    )
    for tick, w in zip(ax.get_xticklabels(), wholes):
        tick.set_color("#d62728" if w else "#333333")
        if w:
            tick.set_fontweight("bold")
    ax.set_yticks(range(0, max_c, max(1, max_c // 12)))
    ax.set_xlabel("shaft  (● = whole shaft masked)", fontsize=10)
    ax.set_ylabel("contact depth-rank  (shallow → deep)", fontsize=10)
    ax.set_title(
        f"v3 mask — subject {args.subject} trial {args.trial}, seed {args.seed}\n"
        f"N={n} contacts, {sc.n_shafts} shafts  |  masked {n_masked}/{n} "
        f"= {frac:.0%}  |  whole-shaft {n_whole} (100%)  |  melec {melec}  |  "
        f"cfg frac={cfg.mask_frac:.0%} whole={cfg.whole_shaft_frac:.0%} "
        f"r~[{cfg.r_lo:.2f},{cfg.r_hi:.2f}] block≥{cfg.block_w_lo}",
        fontsize=10,
    )
    ax.legend(
        handles=[
            Patch(facecolor="#d62728", label="masked (target)"),
            Patch(facecolor="#e4ebf1", label="visible (context)"),
        ],
        loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, framealpha=1.0,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(
        f"WROTE {args.out}  N={n} shafts={sc.n_shafts} masked={n_masked} "
        f"({frac:.1%}) whole_shaft={n_whole} melec={melec}"
    )


if __name__ == "__main__":
    main()
