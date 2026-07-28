"""Draw the masked-reconstruction strip from one dump of ``v3_mae_recon.py``.

This is the figure that shows what the model was actually trained to do, with nothing
downstream of it: 75% of the tokens are hidden and it predicts each hidden token's own
|STFT| bins. Three rows per contact -- ground truth, what the encoder was shown, what came
back -- so a reader can see which columns were holes.

Two things this figure must not quietly do:

  * Do not draw the prediction at VISIBLE tokens as if it were a reconstruction. The model
    is not scored there and an autoencoder-style copy at visible positions would pad the
    figure with easy wins. Visible columns are shown as the truth in the "shown" row and
    the prediction is drawn everywhere, but the mask overlay marks exactly which columns
    the loss actually saw.
  * Do not stretch each panel separately. One shared colour scale per band, computed on the
    truth, so a flat prediction cannot be rescaled into looking like structure.

Colour scale is per BAND because the bands carry genuinely different amplitudes (r6 drops
norm_pix precisely so that ratio survives), and one scale across bands would crush the
quiet ones to a flat wash.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

BANDS = ("slow", "mid", "hga")


def token_index(band: np.ndarray, contact: np.ndarray, k_full: int, band_lengths):
    """(contact block, band) -> the token rows of that band in TIME order.

    ``build_r4_grid`` lays tokens contact-major: one ``k_full`` block per contact, band-major
    [SLOW; MID; HGA] inside it. Rather than trust that, the layout is READ back off the
    stored band vector and checked against band_lengths -- a silently transposed strip is
    exactly the kind of figure that survives review.
    """
    n_tok = len(band)
    assert n_tok % k_full == 0, f"{n_tok} tokens is not a whole number of {k_full}-blocks"
    n_contacts = n_tok // k_full
    blocks = band.reshape(n_contacts, k_full)
    assert (blocks == blocks[0]).all(), "band layout differs between contact blocks"
    for b, t_b in enumerate(band_lengths):
        got = int((blocks[0] == b).sum())
        assert got == int(t_b), f"band {b}: {got} tokens in the block, band_lengths says {t_b}"
    rows = {}
    for c in range(n_contacts):
        for b in range(len(band_lengths)):
            rows[(c, b)] = np.where(blocks[c] == b)[0] + c * k_full
    contacts = contact.reshape(n_contacts, k_full)[:, 0]
    return rows, contacts


def panels(z, clip: int, contact_block: int, band: int):
    """(F, T) truth / prediction / masked-out flag for one contact and band."""
    band_lengths = z["band_lengths"]
    fdims = z["band_fdims"]
    rows, contacts = token_index(z["band"], z["contact"], int(z["k_full"]), band_lengths)
    ix = rows[(contact_block, band)]
    f_b = int(fdims[band])
    truth = z["target"][clip][ix][:, :f_b].T            # (F_b, T_b)
    pred = z["pred"][clip][ix][:, :f_b].T
    masked = z["in_loss"][clip][ix].astype(bool)        # (T_b,)
    # feat_count is the per-token bin count; if it disagrees with band_fdims the slice above
    # is reading pad columns as data
    fc = z["feat_count"][ix]
    assert (fc == f_b).all(), f"band {band}: feat_count {sorted(set(fc.tolist()))} != {f_b}"
    return truth, pred, masked, int(contacts[contact_block])


def figure(path: str, out_path: str, *, clip: int, band: int, n_contacts: int,
           rate: float, offset: float) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(path, allow_pickle=True)
    band_lengths = z["band_lengths"]
    rows, contacts = token_index(z["band"], z["contact"], int(z["k_full"]), band_lengths)
    n_blocks = len(contacts)

    # pick the contacts with the most masked tokens in this band: a contact that happened to
    # be almost fully visible shows nothing about reconstruction
    hidden = []
    for c in range(n_blocks):
        m = z["in_loss"][clip][rows[(c, band)]].astype(bool)
        hidden.append(m.sum())
    pick = list(np.argsort(-np.asarray(hidden))[:n_contacts])

    # the three bands run at DIFFERENT rates (4/16/32 Hz), so the time axis comes from this
    # band's own token count over the clip duration, never from a single assumed rate
    t_b = int(band_lengths[band])
    times = offset + np.arange(t_b) * (float(z["clip_frames"]) / rate) / max(t_b, 1)

    fig, axes = plt.subplots(len(pick) * 3, 1,
                             figsize=(9.0, 1.05 * len(pick) * 3), squeeze=False)
    axes = axes[:, 0]
    frac = []
    for i, c in enumerate(pick):
        truth, pred, masked, label = panels(z, clip, c, band)
        frac.append(float(masked.mean()))
        shown = truth.copy()
        shown[:, masked] = np.nan                        # holes, not zeros: zero is a value
        vmin, vmax = np.percentile(truth, [2, 98])
        for j, (img, name) in enumerate(((truth, "truth"), (shown, "encoder input"),
                                         (pred, "prediction"))):
            ax = axes[i * 3 + j]
            ax.imshow(img, aspect="auto", origin="lower", cmap="magma",
                      vmin=vmin, vmax=vmax, interpolation="nearest",
                      extent=(times[0], times[-1], 0, img.shape[0]))
            ax.set_yticks([])
            ax.set_ylabel(f"c{label}\n{name}" if j == 0 else name, fontsize=6)
            if j != 2 or i != len(pick) - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("time (s)", fontsize=8)
            if j == 2:
                for k in np.where(masked)[0]:
                    ax.axvspan(times[k], times[min(k + 1, t_b - 1)], color="w", alpha=0.10,
                               lw=0)
    fig.suptitle(f"r6 masked reconstruction · {BANDS[band]} band · clip {clip} · "
                 f"S{int(z['subject_id'])}T{int(z['trial_id'])} · "
                 f"{np.mean(frac):.0%} of shown tokens hidden", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"[write] {out_path}", flush=True)
    return {"clip": clip, "band": BANDS[band], "contacts": [int(contacts[c]) for c in pick],
            "masked_frac": frac}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True, help="npz from v3_mae_recon.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--clip", type=int, default=0)
    ap.add_argument("--band", default="hga", choices=BANDS)
    ap.add_argument("--n-contacts", type=int, default=4)
    ap.add_argument("--rate", type=float, default=32.0)
    ap.add_argument("--offset", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"figM_mae_recon_{args.band}_clip{args.clip}.png")
    info = figure(args.recon, out, clip=args.clip, band=BANDS.index(args.band),
                  n_contacts=args.n_contacts, rate=args.rate, offset=args.offset)
    print(f"[check] masked fraction per contact: "
          f"{[round(v, 3) for v in info['masked_frac']]}", flush=True)


if __name__ == "__main__":
    main()
