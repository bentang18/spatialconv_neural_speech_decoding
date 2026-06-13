"""2STFT dual-band M2 mask showcase - render what the masks look like.

Static (non-interactive) sibling of ``fe_stft_bin_designer_gui.py``: same STFT
"what the model sees" visual language (magma spectrogram, low at bottom, colored
band side-bars, freq-patch + time gridlines), but instead of designing bins it
overlays the ACTUAL production dual-band M2 mask
(:func:`speech_decoding.ssl.mask.sample_m2_dual_band_mask`) on a real STFT
backdrop, for several example parcels so you can SEE the structure:

  * LOW band  (slow STFT, spectral resolution) - a freq-block TUBE: one
    contiguous freq-block masked across ALL time (a horizontal stripe).
  * HIGH band (fast STFT, temporal resolution) - an anchor-dilate TIME tube:
    a fraction of time positions sampled as anchors, each dilated `width`-wide
    (overlaps allowed), masked across ALL freq (vertical stripes that merge).

Each parcel draws its own random placement, so the panels show the variety. The
mask is the real one (imported from the SSL code), not a reimplementation.

Agg backend - writes a PNG, no GUI. Run:
    .venv/bin/python scripts/neuroprobe/fe_2stft_mask_showcase.py \
        --high-anchor-frac 0.10 --high-anchor-width 3 --out fe_2stft_mask_showcase.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")  # static PNG, no interactive backend needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from speech_decoding.ssl.mask import sample_m2_dual_band_mask  # noqa: E402

FS = 2048
# lr6e4 5 s clip dual-band patch grid (what the mask indexes).
F_P_LOW, T_LOW_P = 7, 40      # low: 7 freq-patches x 40 time-patches  (S_low=280)
F_P_HIGH, T_HIGH_P = 3, 80    # high: 3 freq-patches x 80 time-patches (S_high=240)
S_LOW = F_P_LOW * T_LOW_P
# Per-band native STFT for the backdrop (documented 2STFT config).
LOW_N, LOW_HOP, LOW_KBINS = 512, 256, (1, 14)    # df=4 Hz -> 14 bins (4..56 Hz)
HIGH_N, HIGH_HOP, HIGH_KBINS = 128, 64, (2, 10)  # df=16 Hz -> 9 bins (32..160 Hz)
LOW_COL, HIGH_COL = "#4C72B0", "#C44E52"          # blue / red band side-bars
CLIP_DUR = 5.0


# ----------------------------- backdrop signal ----------------------------- #
def make_test_signal(fs=FS, dur=CLIP_DUR, seed=0):
    """Low-freq tones (6/11 Hz) + two high-gamma (80-180 Hz) bursts + pink floor
    - the same flavor as the bin designer, so the spectrogram has real structure
    in BOTH bands for the mask to cut across."""
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 6 * t) + 0.5 * np.sin(2 * np.pi * 11 * t)
    f = np.fft.rfftfreq(n, 1 / fs)
    hg = rng.standard_normal(n)
    H = np.fft.rfft(hg)
    H[(f < 80) | (f > 180)] = 0
    hg = np.fft.irfft(H, n)
    hg = hg / (np.std(hg) + 1e-9)
    for onset in (1.8, 3.1, 3.9):
        x += 3.0 * np.exp(-0.5 * ((t - onset) / 0.010) ** 2) * hg
    P = np.fft.rfft(rng.standard_normal(n))
    P[1:] /= f[1:]
    pink = np.fft.irfft(P, n)
    x += 0.5 * pink / (np.std(pink) + 1e-9)
    return x.astype(np.float32)


def stft_mag(x, nperseg, hop):
    xt = torch.from_numpy(x).float()
    win = torch.hann_window(nperseg)
    spec = torch.stft(xt, n_fft=nperseg, hop_length=hop, win_length=nperseg,
                      window=win, return_complex=True, center=True)
    return spec.abs().numpy()  # (n_freq, n_frames)


def _pool_freq(mag, fk):
    """Mean-pool contiguous fk freq-bins -> freq-patches (drops the remainder)."""
    nb = (mag.shape[0] // fk) * fk
    return mag[:nb].reshape(nb // fk, fk, -1).mean(axis=1)


def _resize_time(mag, n_out):
    """Linear-resample the time axis to exactly n_out columns."""
    src = np.linspace(0.0, 1.0, mag.shape[1])
    dst = np.linspace(0.0, 1.0, n_out)
    return np.stack([np.interp(dst, src, row) for row in mag], axis=0)


def band_backdrops():
    """Return normalized (F_P_LOW, T_LOW_P) and (F_P_HIGH, T_HIGH_P) magnitudes."""
    sig = make_test_signal()
    lo = stft_mag(sig, LOW_N, LOW_HOP)[LOW_KBINS[0]:LOW_KBINS[1] + 1]   # (14, ~41)
    hi = stft_mag(sig, HIGH_N, HIGH_HOP)[HIGH_KBINS[0]:HIGH_KBINS[1] + 1]  # (9, ~161)
    lo = _resize_time(_pool_freq(lo, 2), T_LOW_P)    # (7, 40)
    hi = _resize_time(_pool_freq(hi, 3), T_HIGH_P)   # (3, 80)
    norm = lambda a: (a - a.min()) / (a.max() - a.min() + 1e-9)
    return norm(lo), norm(hi)


# ----------------------------- mask -> grids ------------------------------- #
def parcel_grids(mask_row):
    """(S,) bool flat mask -> (low (F_P_LOW,T_LOW_P), high (F_P_HIGH,T_HIGH_P))
    bool, freq-major for display (low freq at row 0)."""
    low = mask_row[:S_LOW].reshape(T_LOW_P, F_P_LOW).T       # (F_P_LOW, T_LOW_P)
    high = mask_row[S_LOW:].reshape(T_HIGH_P, F_P_HIGH).T    # (F_P_HIGH, T_HIGH_P)
    return low.numpy(), high.numpy()


def _upsample_cols(grid, factor):
    return np.repeat(grid, factor, axis=1)


def assemble(low_mag, high_mag, low_mask, high_mask, tcols=160):
    """Stack low (bottom) + high (top) onto a common tcols time axis. Returns the
    magnitude image, the mask image, and the y-row where high starts."""
    lo_f, hi_f = tcols // T_LOW_P, tcols // T_HIGH_P  # 4, 2
    mag = np.vstack([_upsample_cols(low_mag, lo_f), _upsample_cols(high_mag, hi_f)])
    msk = np.vstack([_upsample_cols(low_mask, lo_f), _upsample_cols(high_mask, hi_f)])
    return mag, msk.astype(bool), F_P_LOW  # high starts at row F_P_LOW


def red_overlay(mask_img, alpha=0.6):
    """RGBA overlay: translucent red where masked, transparent elsewhere."""
    ov = np.zeros((*mask_img.shape, 4), dtype=float)
    ov[mask_img] = (1.0, 0.10, 0.10, alpha)
    return ov


# ----------------------------- figure -------------------------------------- #
def render(masks, low_mag, high_mag, *, cfg_line, out_path, tcols=160):
    n = masks.shape[0]
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.4 * nrow + 1.0),
                             squeeze=False)
    F_tot = F_P_LOW + F_P_HIGH
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i >= n:
            ax.axis("off")
            continue
        low_mask, high_mask = parcel_grids(masks[i])
        mag, msk, hi_row = assemble(low_mag, high_mag, low_mask, high_mask, tcols)
        ax.imshow(mag, aspect="auto", origin="lower", cmap="magma",
                  extent=[0, CLIP_DUR, 0, F_tot], interpolation="nearest")
        ax.imshow(red_overlay(msk), aspect="auto", origin="lower",
                  extent=[0, CLIP_DUR, 0, F_tot], interpolation="nearest")
        # band side-bars (low blue at bottom, high red on top)
        ax.add_patch(Rectangle((0, 0), 0.06, hi_row, color=LOW_COL,
                               alpha=0.95, clip_on=False))
        ax.add_patch(Rectangle((0, hi_row), 0.06, F_P_HIGH, color=HIGH_COL,
                               alpha=0.95, clip_on=False))
        ax.text(-0.14, hi_row / 2, "LOW\nfreq", color=LOW_COL, fontsize=8,
                ha="right", va="center", weight="bold")
        ax.text(-0.14, hi_row + F_P_HIGH / 2, "HIGH\nfreq", color=HIGH_COL,
                fontsize=8, ha="right", va="center", weight="bold")
        # freq-patch + band gridlines
        for y in range(1, F_tot):
            ax.axhline(y, color="w", lw=0.3, alpha=0.25)
        ax.axhline(hi_row, color="w", lw=1.2, alpha=0.8)  # band divider
        # realized masked fractions
        lo_pct = 100.0 * low_mask.mean()
        hi_pct = 100.0 * high_mask.mean()
        all_pct = 100.0 * masks[i].float().mean().item()
        ax.set_title(f"parcel {i}   low {lo_pct:.0f}%  |  high {hi_pct:.0f}%  "
                     f"|  overall {all_pct:.0f}%", fontsize=9)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_yticks([])  # bands are labeled by the colored LOW/HIGH side bars
        ax.tick_params(labelsize=7)

    legend = [
        Patch(facecolor=(1.0, 0.10, 0.10, 0.6), label="MASKED (prediction target)"),
        Patch(facecolor="#999999", label="visible (encoder input)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("2STFT dual-band M2 mask  -  " + cfg_line, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--low-freq-width", type=int, default=2,
                    help="LOW band freq-tube width (patches). Default 2.")
    ap.add_argument("--low-freq-nbands", type=int, default=1,
                    help="LOW band number of freq blocks. Default 1.")
    ap.add_argument("--high-anchor-frac", type=float, default=0.10,
                    help="HIGH band: fraction of time positions sampled as "
                         "anchors. Default 0.10.")
    ap.add_argument("--high-anchor-width", type=int, default=3,
                    help="HIGH band: anchor dilation width (overlaps allowed). "
                         "Default 3.")
    ap.add_argument("--out", default="fe_2stft_mask_showcase.png")
    args = ap.parse_args()

    low_mag, high_mag = band_backdrops()
    gen = torch.Generator().manual_seed(args.seed)
    masks = sample_m2_dual_band_mask(
        1, args.n_examples,
        F_p_low=F_P_LOW, T_low_p=T_LOW_P, F_p_high=F_P_HIGH, T_high_p=T_HIGH_P,
        low_freq_floor=args.low_freq_width,
        low_freq_frac=(args.low_freq_width * args.low_freq_nbands) / F_P_LOW,
        high_time_anchor_frac=args.high_anchor_frac,
        high_time_anchor_width=args.high_anchor_width,
        generator=gen,
    )[0]  # (n_examples, S)

    cfg_line = (
        f"low: {args.low_freq_width}-wide freq tube x{args.low_freq_nbands}  |  "
        f"high: anchor frac={args.high_anchor_frac} width={args.high_anchor_width} "
        f"(overlaps)   [grid F_p_low={F_P_LOW} T_low={T_LOW_P} / "
        f"F_p_high={F_P_HIGH} T_high={T_HIGH_P}]"
    )
    render(masks, low_mag, high_mag, cfg_line=cfg_line, out_path=args.out)


if __name__ == "__main__":
    main()
