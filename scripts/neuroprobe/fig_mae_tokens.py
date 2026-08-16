"""The MAE input/target picture, drawn on a real clip of what the v3 encoder actually reads.

Same idea as the canonical video-MAE figure -- input with the masked tokens removed, full
target beside it -- but flat rather than isometric. The band axis is drawn as three stacked
panels instead of three tiers of one cube, because in projection a far MID cell lands at the
same screen height as a near HGA cell and the band a cell belongs to stops being readable.

    panel (vertical)   band     SLOW blue, MID gold, HGA red
    rows within panel  shaft    one block per contact, the band's freq bins inside the block
    columns            time     each band hops at its OWN rate, so a SLOW token is 8x the
                                width of an HGA token: the multirate front-end, drawn

Cells are the |STFT| BINS, one row each, at their own robust-z. The box drawn around each
group of bins is one TOKEN: the stem collapses a band's bins into one channel group, so the
box is the collapse made visible. Token count follows PerBandStem, which decimates the
shared 32 Hz cache by stride 8 / 2 / 1 (`stem.py:104`), giving 8 SLOW, 32 MID and 64 HGA
tokens per contact over 2 s. One shared colour scale for all three bands.

The holes are the REAL mask, sampled from `sample_masks_r6` with the canonical launcher's
config (`v3_r6_vits384_cd55k.sbatch:135`): space_frac 0.50 with `--per-band-space` and
per-band depth-block widths 1,1,1, plus per-SENSOR width-4 time blocks at 0.50 on each
band's own grid. A token is visible iff its contact survives the space draw IN THAT BAND
and its band-token is not time-masked, which is the objective's outer product verbatim.
The space draw is the tube: a contact masked in a band is gone for the whole clip, so it
shows as an empty block of rows. The time draw is per-sensor and NOT tubed across contacts
(`masking.py:437`, changed 2026-07-23 because the encoder is L1-within-shaft only), so the
width-4 gaps sit at different places on different contacts. Both are drawn as they are.

Data: `results/showcase/_data/v3_clip_btbank1_t0.npz`, pulled from the production spec
cache on Delta (CAR + notch + |STFT| baked in, frozen robust-z + per-band winsor applied).
Clip selection was fixed before rendering: among temporal-lobe shafts with at least 8
contacts, the one with the largest sentence-onset HGA response over ALL onsets, then the
onset at the MEDIAN of that shaft's response distribution. `--quantile` renders the 25th
or 75th instead, same clip machinery.
"""

import argparse
import pathlib

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # NeurIPS rejects Type 3
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry  # noqa: E402
from speech_decoding.models.v14_converged_v3.masking import (  # noqa: E402
    V3MaskConfig,
    sample_masks_r6,
)
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar  # noqa: E402

DATA = pathlib.Path("results/showcase/_data/v3_clip_btbank1_t0.npz")
OUT = pathlib.Path("results/showcase/2_what_pretraining_does")

# House palette: SLOW keeps the arm blue, HGA the floor red, MID the gold already used in
# fig_correspondence_priors. Ben: hga red, mid yellow, slow blue.
BAND_COLOR = {"slow": "#1f4e79", "mid": "#b8860b", "hga": "#c1440e"}
BAND_ORDER = ("slow", "mid", "hga")  # R4Grid.band order; drawn top-down as hga, mid, slow
BAND_HZ = {"slow": "4 Hz", "mid": "16 Hz", "hga": "32 Hz"}
BAND_RANGE = {"slow": "2-14 Hz", "mid": "16-56 Hz", "hga": "64-160 Hz"}
INK, GRID, CAGE = "#333333", "#dfe3e7", "#8b959f"


def band_cmap(hexcolor):
    """White -> band hue. Masked tokens are drawn white, so visible cells get a tint floor."""
    return LinearSegmentedColormap.from_list("b", [(1, 1, 1), to_rgb(hexcolor)])


def darker(hexcolor, f=0.55):
    return tuple(np.clip(np.array(to_rgb(hexcolor)) * f, 0, 1))


def load_clip(quantile):
    d = np.load(DATA, allow_pickle=True)
    tag = f"q{quantile}"
    bands = {}
    for b in BAND_ORDER:
        z = d[f"{tag}_{b}"]  # (C, F, T_b)
        bands[b] = dict(bins=z, n_bins=z.shape[1], n_tok=z.shape[2],
                        stride=int(d[f"{b}_stride"]), freqs=d[f"{b}_freqs"])
    return d, bands, tag


def sample_real_mask(contacts, n_time, seed, band_time_unit="shaft",
                     space_frac=0.50, time_frac=0.50):
    """The launcher's r6 mask for these contacts: (C, 3) space, plus the three band time masks."""
    labels = [str(c) for c in contacts]
    sidecar = build_sidecar(labels, parcel_id=torch.zeros(len(labels), dtype=torch.long))
    geom = build_l1_geometry(sidecar)
    cfg = V3MaskConfig(per_band_space=True, block_w_space_bands=(1, 1, 1),
                       band_time_unit=band_time_unit, space_frac=space_frac,
                       hga_mask_frac=time_frac, mid_mask_frac=time_frac,
                       slow_mask_frac=time_frac)
    g = torch.Generator().manual_seed(seed)
    m = sample_masks_r6(geom, len(labels), n_time=n_time, n_rows=1, generator=g, cfg=cfg)
    return (
        m.contact_mask[0].numpy(),  # (C, 3) True = spatially masked, band order SLOW/MID/HGA
        {"slow": m.slow_mask[0].numpy(), "mid": m.mid_mask[0].numpy(),
         "hga": m.hga_mask[0].numpy()},
    )


def panel_image(band, vis):
    """(n_contacts * n_bins, n_frames) image at the shared 32 Hz grid, NaN where removed.

    A token occupies `stride` frames, so a SLOW token is repeated across 8 columns: the
    repetition is the hop, and the box drawn on top marks where one token ends.
    """
    z, nb, st = band["bins"], band["n_bins"], band["stride"]
    n_c = z.shape[0]
    a = z.copy()
    if vis is not None:
        a = np.where(vis[:, None, :], a, np.nan)
    a = np.repeat(a, st, axis=2)  # (C, F, n_tok*stride)
    img = np.full((n_c * nb, a.shape[2]), np.nan)
    for i in range(n_c):
        img[i * nb:(i + 1) * nb] = a[i, ::-1, :]  # freq ascending upward inside the block
    return img, n_c * nb


def token_boxes(band, n_c, vis, clip_s):
    """Outlines of every token, split into removed and kept, in data coords (s, row)."""
    nb, st, n_tok = band["n_bins"], band["stride"], band["n_tok"]
    dt = clip_s / (n_tok * st)
    kept, gone = [], []
    for i in range(n_c):
        y0, y1 = i * nb, (i + 1) * nb
        for t in range(n_tok):
            x0, x1 = t * st * dt, (t + 1) * st * dt
            r = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            (kept if (vis is None or vis[i, t]) else gone).append(r)
    return kept, gone


def draw_column(fig, gs, col, bands, vis, clip_s, vmin, vmax, title, onset_s, labels):
    axes = {}
    for r, b in enumerate(reversed(BAND_ORDER)):  # hga on top, slow at the bottom
        ax = fig.add_subplot(gs[r, col])
        axes[b] = ax
        v = None if vis is None else vis[b]
        img, n_rows = panel_image(bands[b], v)
        cmap = band_cmap(BAND_COLOR[b])
        cmap.set_bad("#ffffff")
        ax.imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=vmin, vmax=vmax,
                  extent=(0.0, clip_s, n_rows, 0.0), aspect="auto",
                  interpolation="nearest", zorder=1)
        n_c, nb = bands[b]["bins"].shape[0], bands[b]["n_bins"]
        kept, gone = token_boxes(bands[b], n_c, v, clip_s)
        if gone:
            ax.add_collection(LineCollection(gone, colors=GRID, linewidths=0.30, zorder=2))
        ax.add_collection(
            LineCollection(kept, colors=(darker(BAND_COLOR[b]), ), linewidths=0.32,
                           alpha=0.75, zorder=3))
        # contact rules: without them a block of bins reads as one wide row rather than as
        # one contact, and the space tube stops being countable
        ax.add_collection(LineCollection(
            [[(0.0, i * nb), (clip_s, i * nb)] for i in range(1, n_c)],
            colors=CAGE, linewidths=0.55, zorder=4))
        for s in ax.spines.values():
            s.set_color(CAGE)
            s.set_linewidth(0.7)
        ax.set_xlim(0.0, clip_s)
        ax.set_ylim(n_rows, 0.0)
        ax.set_yticks([])
        ax.axvline(onset_s, color=INK, lw=0.7, ls=(0, (3, 2)), alpha=0.55, zorder=4)
        if b == "slow":
            ax.set_xticks(np.arange(0, clip_s + 0.01, 0.5))
            ax.tick_params(axis="x", labelsize=7.4, length=2.5, colors=INK)
            ax.set_xlabel("time (s)", fontsize=8.4, color=INK, labelpad=2)
        else:
            ax.set_xticks([])
        if col == 0:
            ax.text(-0.012, 0.5,
                    f"{b.upper()}\n{BAND_RANGE[b]}\n{nb} bins per token\n{BAND_HZ[b]}",
                    transform=ax.transAxes, ha="right", va="center", fontsize=7.2,
                    color=BAND_COLOR[b], linespacing=1.45)
        if r == 0:
            ax.set_title(title, fontsize=10.5, color=INK, pad=5)
    if labels:
        # the shaft axis, named at both ends of the top panel so the row blocks are contacts
        ax = axes["hga"]
        nb = bands["hga"]["n_bins"]
        n_c = bands["hga"]["bins"].shape[0]
        for i, name in ((0, labels[0]), (n_c - 1, labels[-1])):
            ax.text(clip_s * 1.012, (i + 0.5) * nb, name, ha="left", va="center",
                    fontsize=6.6, color="#77808a", clip_on=False)
    return axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantile", type=int, default=50, choices=(25, 50, 75))
    ap.add_argument("--seed", type=int, default=33)  # the locked training seed
    ap.add_argument("--stem", default="fig_mae_tokens")
    ap.add_argument("--width", type=float, default=7.4)
    ap.add_argument("--contacts", type=int, default=8,
                    help="how many contiguous contacts to DRAW; the mask is always sampled on "
                         "the whole shaft, so the drawn masked fraction is a subsample of a real "
                         "50%% draw and is printed alongside the full-shaft one")
    ap.add_argument("--contact-start", type=int, default=0)
    ap.add_argument("--pool", action="store_true",
                    help="fill each token with its band-mean instead of its bins, so a token "
                         "box is one flat value: the collapse the stem performs, drawn")
    ap.add_argument("--time-tube", dest="time_tube", choices=("shaft", "contact"),
                    default="shaft",
                    help="passed straight to V3MaskConfig.band_time_unit. shaft (default, the "
                         "2026-08-14 contract) tubes the width-4 time blocks across the contacts "
                         "of a shaft; contact is the 07-23 per-sensor draw it replaces")
    ap.add_argument("--space-frac", dest="space_frac", type=float, default=0.50,
                    help="V3MaskConfig.space_frac. 0.0 draws the no-space-tier arm, where "
                         "the whole mask budget is spent on the time axis.")
    ap.add_argument("--time-frac", dest="time_frac", type=float, default=0.50,
                    help="set on all three bands. The tiers COMPOSE, so the masked fraction "
                         "is 1-(1-space)(1-time): 0.50/0.50 and 0.0/0.75 both give 75%%.")
    args = ap.parse_args()

    d, bands, tag = load_clip(args.quantile)
    contacts = [str(c) for c in d["contacts"]]
    n_c = len(contacts)
    n_time = bands["hga"]["n_tok"]
    clip_s = float(d["clip_s"])
    onset_s = float(d[f"{tag}_onset_s"]) - float(d[f"{tag}_clip_start_s"])

    space, band_time = sample_real_mask(contacts, n_time, args.seed, args.time_tube,
                                        args.space_frac, args.time_frac)
    if args.pool:
        for b in BAND_ORDER:
            z = bands[b]["bins"]
            bands[b]["bins"] = np.repeat(z.mean(1, keepdims=True), z.shape[1], axis=1)
    vis = {}
    for bi, b in enumerate(BAND_ORDER):
        keep_c = ~space[:, bi][:, None]  # (C, 1)
        vis[b] = keep_c & ~band_time[b]  # (C, T_b) outer product, the objective's rule
        print(f"{b:<5} {bands[b]['n_bins']} bins  space-masked {int(space[:, bi].sum())}/{n_c} "
              f"contacts (tube)   time-masked {int(band_time[b][0].sum())}/{bands[b]['n_tok']} "
              f"per contact   visible {vis[b].mean() * 100:5.1f}%")
    tot = sum(vis[b].size for b in BAND_ORDER)
    seen = sum(int(vis[b].sum()) for b in BAND_ORDER)
    print(f"whole shaft: tokens {tot}  visible {seen} ({seen / tot * 100:.1f}%)")

    # Draw a window of the shaft. Fifteen contacts times 64 HGA tokens is more cells than a
    # 5.5 in column can resolve, and the picture stops being readable as tokens. The mask is
    # sampled on the whole shaft either way, so what is drawn is a real draw, subsampled.
    sl = slice(args.contact_start, args.contact_start + args.contacts)
    for b in BAND_ORDER:
        bands[b]["bins"] = bands[b]["bins"][sl]
        vis[b] = vis[b][sl]
    contacts = contacts[sl]
    n_c = len(contacts)
    tot = sum(vis[b].size for b in BAND_ORDER)
    seen = sum(int(vis[b].sum()) for b in BAND_ORDER)
    print(f"drawn {n_c} contacts {contacts[0]}..{contacts[-1]}: tokens {tot}  "
          f"visible {seen} ({seen / tot * 100:.1f}%)")

    allz = np.concatenate([bands[b]["bins"].ravel() for b in BAND_ORDER])
    vmin, vmax = float(np.percentile(allz, 5)), float(np.percentile(allz, 95))
    print(f"colour scale: robust-z bins in [{vmin:+.2f}, {vmax:+.2f}] (5th-95th pct, shared)")
    print(f"clip start {float(d[f'{tag}_clip_start_s']):.3f}s  onset at +{onset_s:.3f}s")

    ratios = [bands[b]["n_bins"] * n_c for b in reversed(BAND_ORDER)]
    fig = plt.figure(figsize=(args.width, args.width * 0.62))
    gs = fig.add_gridspec(3, 2, height_ratios=ratios, hspace=0.10, wspace=0.10,
                          left=0.115, right=0.925, top=0.90, bottom=0.150)
    draw_column(fig, gs, 0, bands, vis, clip_s, vmin, vmax,
                f"input   {100 - seen / tot * 100:.0f}% of tokens removed", onset_s, None)
    draw_column(fig, gs, 1, bands, None, clip_s, vmin, vmax, "target   all tokens",
                onset_s, contacts)

    how = ("width-4 time blocks tubed across the contacts of a shaft"
           if args.time_tube == "shaft"
           else "width-4 time blocks drawn per contact (the 07-23 draw)")
    fig.text(.5, .052,
             f"{d['session']}, shaft {d['shaft']}, {n_c} of {len(d['contacts'])} contacts  ·  "
             f"{clip_s:.0f} s at a sentence onset (dashed)  ·  "
             + ("colour is the token's band-mean robust-z" if args.pool else
                "rows are |STFT| bins, colour is robust-z") + ", one shared scale",
             ha="center", fontsize=6.9, color="#666")
    fig.text(.5, .014,
             f"mask seed {args.seed}: " + (
                 "no space tier, every contact stays present"
                 if args.space_frac == 0.0 else
                 f"{args.space_frac * 100:.0f}% of contacts hidden per band for the whole clip")
             + f", {args.time_frac * 100:.0f}% of each band's tokens hidden in time  ·  {how}",
             ha="center", fontsize=6.9, color="#666")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{args.stem}.{ext}", dpi=220)
    plt.close(fig)
    print(f"wrote {OUT}/{args.stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
