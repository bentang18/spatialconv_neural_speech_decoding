"""LAUNCH GATE: draw the mask the production sampler actually emits, on a real montage.

No neural data. This renders `sample_masks_r6` output directly, so what you see is what the
objective will hide, and it verifies the 2026-08-14 contract in the picture rather than in prose:

  SPACE   50% of each shaft's contacts, hidden for the WHOLE clip in that band (a tube along
          time). Independent per band under `--per-band-space`, so a contact can be gone in
          HGA and present in SLOW.
  TIME    contiguous width-4 blocks covering 50% of each band's tokens, drawn PER SHAFT and
          shared by every contact of that shaft (a tube across contacts). Shafts are drawn
          INDEPENDENTLY of each other.

The two assertions the picture has to satisfy are checked in code and printed:
  1. within a shaft, every contact's band time mask is identical  -> the tube is closed
  2. across shafts, the masks differ                              -> not r4's single global draw

`--band-time-unit contact` renders the 07-23 behaviour that this replaces, for comparison.

Montage comes from the BrainTreebank electrode labels shipped with the dataset, so the shaft
sizes and the per-shaft counts are the real ones, not a toy.
"""

import argparse
import json
import pathlib

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import ListedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry  # noqa: E402
from speech_decoding.models.v14_converged_v3.masking import (  # noqa: E402
    V3MaskConfig,
    sample_masks_r6,
)
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar  # noqa: E402
from speech_decoding.studies.braintreebank.anatomy import clean_bt_electrode_label  # noqa: E402

BT = pathlib.Path(".cache/braintreebank")
OUT = pathlib.Path("results/showcase/2_what_pretraining_does")

BAND_COLOR = {"slow": "#1f4e79", "mid": "#b8860b", "hga": "#c1440e"}
BAND_ORDER = ("slow", "mid", "hga")
BAND_STRIDE = {"slow": 8, "mid": 2, "hga": 1}
BAND_HZ = {"slow": "4 Hz", "mid": "16 Hz", "hga": "32 Hz"}
INK, TIME_GREY, SPACE_GREY = "#333333", "#9aa7b4", "#e8ebee"


def load_labels(subject):
    p = BT / "electrode_labels" / f"sub_{subject}" / "electrode_labels.json"
    return [clean_bt_electrode_label(str(x)) for x in json.load(open(p))]


def shaft_name(label):
    i = len(label)
    while i > 0 and label[i - 1].isdigit():
        i -= 1
    return label[:i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--n-time", type=int, default=64)  # 2 s at 32 Hz
    ap.add_argument("--shafts", type=int, default=4, help="how many shafts to DRAW")
    ap.add_argument("--band-time-unit", choices=("shaft", "contact"), default="shaft")
    ap.add_argument("--stem", default=None)
    args = ap.parse_args()
    stem = args.stem or f"fig_mask_check_{args.band_time_unit}"

    labels = load_labels(args.subject)
    sidecar = build_sidecar(labels, parcel_id=torch.zeros(len(labels), dtype=torch.long))
    geom = build_l1_geometry(sidecar)
    n = len(labels)
    cfg = V3MaskConfig(per_band_space=True, block_w_space_bands=(1, 1, 1),
                       band_time_unit=args.band_time_unit)
    g = torch.Generator().manual_seed(args.seed)
    m = sample_masks_r6(geom, n, n_time=args.n_time, n_rows=1, generator=g, cfg=cfg)
    space = m.contact_mask[0].numpy()  # (N, 3) True = spatially masked
    time_m = {"slow": m.slow_mask[0].numpy(), "mid": m.mid_mask[0].numpy(),
              "hga": m.hga_mask[0].numpy()}
    shaft_of = geom.shaft_of_contact.numpy()
    print(f"subject {args.subject}: {n} contacts, {geom.n_shafts} shafts, "
          f"band_time_unit={args.band_time_unit}")

    # ── the two assertions the picture has to satisfy ────────────────────────────────────
    for b in BAND_ORDER:
        tm = time_m[b]
        tubed = all(
            bool((tm[members] == tm[members[0]]).all())
            for s in range(geom.n_shafts)
            if len(members := np.flatnonzero(shaft_of == s)) > 0
        )
        heads = [np.flatnonzero(shaft_of == s)[0] for s in range(geom.n_shafts)
                 if (shaft_of == s).any()]
        independent = len({tm[h].tobytes() for h in heads}) > 1
        print(f"  {b:<5} tubed within shaft = {tubed}   shafts differ = {independent}   "
              f"masked {tm[0].mean() * 100:.0f}% of {tm.shape[1]} tokens")
        if args.band_time_unit == "shaft":
            assert tubed, f"{b}: contacts of a shaft do not share their time mask"
            assert independent, f"{b}: every shaft shares one mask — that is r4's global draw"

    # ── draw ─────────────────────────────────────────────────────────────────────────────
    order = sorted(range(geom.n_shafts), key=lambda s: -(shaft_of == s).sum())[: args.shafts]
    order = sorted(order)
    rows, bounds, names = [], [], []
    for s in order:
        members = np.flatnonzero(shaft_of == s)
        bounds.append((len(rows), len(rows) + len(members)))
        names.append(shaft_name(labels[members[0]]))
        rows.extend(members.tolist())
    rows = np.asarray(rows)
    print(f"drawing shafts {names} ({len(rows)} contacts)")

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.4))
    for ax, b in zip(axes, BAND_ORDER):
        bi = BAND_ORDER.index(b)
        tm, st = time_m[b], BAND_STRIDE[b]
        img = np.zeros((len(rows), tm.shape[1]), dtype=int)  # 0 visible
        img[tm[rows] > 0] = 1  # 1 hidden by TIME
        img[space[rows, bi] > 0, :] = 2  # 2 hidden by SPACE, whole row
        cmap = ListedColormap([to_rgb(BAND_COLOR[b]), to_rgb(TIME_GREY), to_rgb(SPACE_GREY)])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest",
                  extent=(0.0, args.n_time / 32.0, len(rows), 0.0))
        ax.add_collection(LineCollection(
            [[(0.0, y), (args.n_time / 32.0, y)] for _, y in bounds[:-1]],
            colors=INK, linewidths=1.2, zorder=3))
        for (y0, y1), nm in zip(bounds, names):
            ax.text(-0.012, (y0 + y1) / 2, nm, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=7.0, color=INK)
        ax.set_title(f"{b.upper()}   {BAND_HZ[b]}   {tm.shape[1]} tokens",
                     fontsize=9.4, color=BAND_COLOR[b], pad=6)
        ax.set_yticks([])
        ax.set_xlabel("time (s)", fontsize=8.4, color=INK)
        ax.tick_params(axis="x", labelsize=7.4)
        for sp in ax.spines.values():
            sp.set_color("#8b959f")
    axes[0].legend(
        handles=[Patch(facecolor=BAND_COLOR["slow"], label="visible"),
                 Patch(facecolor=TIME_GREY, label="hidden by TIME (width-4 blocks)"),
                 Patch(facecolor=SPACE_GREY, label="hidden by SPACE (whole clip)")],
        loc="upper left", bbox_to_anchor=(0.0, -0.14), ncol=3, frameon=False, fontsize=7.6)
    tube = ("time blocks TUBED across the contacts of a shaft, shafts independent"
            if args.band_time_unit == "shaft"
            else "time blocks drawn PER CONTACT (the 07-23 draw this replaces)")
    fig.suptitle(f"r6 mask as sampled  ·  btbank{args.subject} montage, {len(rows)} of {n} "
                 f"contacts  ·  seed {args.seed}  ·  {tube}",
                 fontsize=9.0, color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=210)
    plt.close(fig)
    print(f"wrote {OUT}/{stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
