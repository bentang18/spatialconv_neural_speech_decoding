"""Board/viz encode cache -> trial-averaged condition tensors, one small .npz per session.

The cache is ~50-90 GB per session and the figures need condition means, so this is the
step that makes everything downstream cheap and local. Trial averaging is not cosmetic:
single-clip iEEG features are dominated by noise, and every claim here is about the mean
response to a condition. |STFT| magnitude survives averaging in a way raw voltage does not
-- induced (non-phase-locked) power adds rather than cancelling -- so the average is a real
response, not an ERP artifact.

ONE streaming pass per tap accumulates everything:

  * per-column sum / sumsq over ALL trials. Pooling those over parcels and time gives exact
    per-channel mean/var, so the figure script can standardize per-channel or per-column
    without another pass over 90 GB.
  * per (task, class) sums for the grand average AND for two interleaved half-averages.
    The halves are the noise ceiling: no cross-subject alignment score means anything
    without knowing what the same subject's own two halves score.

Standardization is deliberately NOT applied here. It is a figure-level choice, it is linear
so it commutes with averaging ((mean(x) - mu)/sigma), and applying it early would bake one
choice into a 12-session reduction.

Bands: the HGA slice is the default. Encoder taps are (k_full, d) with k band-major
[SLOW; MID; HGA]; enc0 is the raw spectrogram, per band time-major (T_b, F_b). Both end up
(T, C) after slicing, which is what every figure consumes.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

BANDS = ("slow", "mid", "hga")

# Picked off the 45k linear-cooldown board table, not by intuition: onset is the strongest
# transferring task (CS .8153, enc0->enc12 uplift +.0706), word_part_speech is flat (.5355,
# uplift -.0010) and frame_brightness sits below chance in BOTH regimes (.4971 CS / .4972
# WS). A figure that only shows the winners is not evidence, so the duds are not optional.
DEFAULT_TASKS = ("onset", "speech", "delta_volume", "word_index",
                 "word_part_speech", "frame_brightness")
DEFAULT_TAPS = ("enc0", "enc3", "enc6", "enc12")


def _band_slice(tap: str, band: str, band_lengths, band_fdims, n_feat: int):
    """Column indices of one band inside a tap's flattened feature axis, plus (T, C).

    Encoder taps store (k_full, d): k is band-major so a band is a contiguous run of k, and
    every band shares the same d. enc0 stores the spectrogram, each band (T_b, F_b)
    time-major, so its blocks have DIFFERENT widths and the offset is a running sum.
    """
    b = BANDS.index(band)
    t_b = int(band_lengths[b])
    k_full = int(sum(band_lengths))
    if tap.startswith("enc0"):
        if band_fdims is None:
            raise KeyError(
                "enc0 needs band_fdims, which this record predates. Either reduce the encoder "
                "taps only (--taps enc3,enc6,enc12) or re-encode -- F_b is NOT recoverable "
                "from the flattened width alone.")
        fd = [int(f) for f in band_fdims]
        total = sum(t * f for t, f in zip(band_lengths, fd))
        if total != n_feat:
            raise ValueError(f"{tap}: sum_b T_b*F_b = {total} != stored width {n_feat}")
        off = sum(int(band_lengths[i]) * fd[i] for i in range(b))
        f_b = fd[b]
        return np.arange(off, off + t_b * f_b), t_b, f_b
    if n_feat % k_full:
        raise ValueError(f"{tap}: width {n_feat} not divisible by k_full {k_full}")
    d = n_feat // k_full
    k0 = int(sum(band_lengths[:b]))
    cols = (np.arange(k0, k0 + t_b)[:, None] * d + np.arange(d)[None, :]).ravel()
    return cols, t_b, d


def _halves(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interleave, don't randomize. Session drift is slow, so alternating trials splits it
    evenly between the halves; a random split leaves the ceiling sensitive to the seed."""
    order = np.sort(idx)
    return order[0::2], order[1::2]


def reduce_session(path: str, *, taps, tasks, band: str, chunk: int, verbose: bool = True,
                   band_fdims_override=None):
    rec = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    present = np.asarray(rec["present_parcels"], dtype=np.int64)
    canon = np.asarray(rec["parcel_canon"], dtype=np.int64)
    band_lengths = tuple(int(x) for x in rec["band_lengths"])
    band_fdims = rec.get("band_fdims")
    if band_fdims is None and band_fdims_override is not None:
        # Records written before band_fdims existed. The width check in _band_slice is
        # necessary but NOT sufficient: sum_b T_b*F_b does not determine F_b uniquely --
        # (7,6,7) and (3,5,8) both total 348 at the 1 s window and 696 at 2 s. So only ever
        # pass a value READ off a record the same frontend wrote, never one back-solved from
        # the width.
        band_fdims = tuple(int(f) for f in band_fdims_override)
    n = int(rec["n_windows"])
    subj, trial = int(rec["subject_id"]), int(rec["trial_id"])

    counts = np.array([int((canon == p).sum()) for p in present], dtype=np.int64)
    assert counts.sum() == len(canon), "parcel electrode counts must partition the montage"
    assert (counts > 0).all(), "present_parcels lists a parcel with no electrodes"

    out: dict = {
        "subject_id": np.int64(subj), "trial_id": np.int64(trial),
        "present_parcels": present, "parcel_counts": counts,
        "band_lengths": np.asarray(band_lengths, dtype=np.int64),
        "n_windows": np.int64(n),
    }
    if band_fdims is not None:
        out["band_fdims"] = np.asarray([int(f) for f in band_fdims], dtype=np.int64)

    # trial index per (task, class), and the interleaved halves, shared by every tap
    groups: dict[tuple[str, int], np.ndarray] = {}
    for task in tasks:
        y = np.asarray(rec["labels"][task], dtype=float)
        for cls in (0, 1):
            sel = np.where(np.isfinite(y) & (y == cls))[0]
            groups[(task, cls)] = sel
            if verbose:
                print(f"[trials] s{subj}_t{trial} {task} class{cls}: {len(sel)}", flush=True)
            out[f"count/{task}/c{cls}"] = np.int64(len(sel))

    for tap in taps:
        if tap not in rec["feats"]:
            print(f"[skip] {tap} not in record", flush=True)
            continue
        feat = rec["feats"][tap]["raw"]
        n_p, n_feat = int(feat.shape[1]), int(feat.shape[2])
        cols, t_len, c_len = _band_slice(tap, band, band_lengths, band_fdims, n_feat)
        assert n_p == len(present), f"{tap}: |P| {n_p} != present {len(present)}"
        assert len(cols) == t_len * c_len

        col_sum = np.zeros((n_p, t_len * c_len), dtype=np.float64)
        col_sq = np.zeros((n_p, t_len * c_len), dtype=np.float64)
        acc: dict[str, np.ndarray] = {}
        member: dict[str, np.ndarray] = {}
        for task in tasks:
            for cls in (0, 1):
                sel = groups[(task, cls)]
                h0, h1 = _halves(sel)
                for name, ix in (("all", sel), ("h0", h0), ("h1", h1)):
                    key = f"{tap}/{task}/c{cls}/{name}"
                    acc[key] = np.zeros((n_p, t_len * c_len), dtype=np.float64)
                    mask = np.zeros(n, dtype=bool)
                    mask[ix] = True
                    member[key] = mask
                    out[f"n/{task}/c{cls}/{name}"] = np.int64(len(ix))

        # Both layouts put a band in ONE contiguous run of columns (encoder taps: a k block
        # times d; enc0: a (T_b, F_b) block). Slicing instead of fancy-indexing avoids pulling
        # the other bands' pages off disk at all, which is most of the cost at this size.
        contiguous = len(cols) > 0 and cols[-1] - cols[0] == len(cols) - 1
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            if contiguous:
                x = feat[i0:i1, :, int(cols[0]):int(cols[-1]) + 1].to(torch.float32).numpy()
            else:
                x = feat[i0:i1, :, :].to(torch.float32).numpy()[:, :, cols]  # (c, |P|, T*C)
            col_sum += x.sum(0)
            col_sq += np.square(x, dtype=np.float64).sum(0)
            for key, mask in member.items():
                m = mask[i0:i1]
                if m.any():
                    acc[key] += x[m].sum(0)
            del x

        out[f"{tap}/col_sum"] = (col_sum / n).astype(np.float32)
        out[f"{tap}/col_sq"] = (col_sq / n).astype(np.float32)
        out[f"{tap}/shape"] = np.asarray([n_p, t_len, c_len], dtype=np.int64)
        for key, s in acc.items():
            k = int(out[f"n/{key.split('/', 1)[1]}"])
            out[key] = (s / max(k, 1)).reshape(n_p, t_len, c_len).astype(np.float32)
        if verbose:
            print(f"[tap] s{subj}_t{trial} {tap}: |P|={n_p} T={t_len} C={c_len} "
                  f"band={band}", flush=True)
        del feat, col_sum, col_sq, acc, member
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--session-index", type=int, default=None,
                    help="reduce only sorted(cache)[i] — one Slurm array task per session")
    ap.add_argument("--taps", default=",".join(DEFAULT_TAPS))
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--band", default="hga", choices=BANDS)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--band-fdims", default=None,
                    help="per-band frequency bins for records written before band_fdims "
                         "was stored. The v3 3-band frontend is 7,6,7 -- READ off a record "
                         "the same frontend wrote. Do NOT back-solve it from the enc0 "
                         "width: that total has more than one decomposition.")
    args = ap.parse_args()
    fdims = tuple(int(x) for x in args.band_fdims.split(",")) if args.band_fdims else None

    import glob
    paths = sorted(glob.glob(os.path.join(args.cache, "enc_s*_t*.pt")))
    assert paths, f"no encode shards in {args.cache}"
    if args.session_index is not None:
        paths = [paths[args.session_index]]
    os.makedirs(args.out_dir, exist_ok=True)

    taps = tuple(t for t in args.taps.split(",") if t)
    tasks = tuple(t for t in args.tasks.split(",") if t)
    for p in paths:
        out = reduce_session(p, taps=taps, tasks=tasks, band=args.band, chunk=args.chunk,
                             band_fdims_override=fdims)
        name = f"red_s{int(out['subject_id'])}_t{int(out['trial_id'])}_{args.band}.npz"
        dst = os.path.join(args.out_dir, name)
        np.savez_compressed(dst, **out)
        print(f"[write] {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
