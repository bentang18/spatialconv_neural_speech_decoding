"""M5 — what the dual-axis mask sampler ACTUALLY produces. Pure simulation, no data.

`block_w_time = 7` / `block_w_space = 4` are FLOORS, not realized run lengths: `_cover_rank`
places blocks at random start ranks with OVERLAPS ALLOWED and takes the union of the
lowest-ranked overlapping blocks (masking.py:115-139). Overlapping blocks MERGE, so the
realized contiguous-run distribution has mean and tail above the floor. Nobody has ever
looked at it.

Two questions, both answered by sampling the real `sample_masks`:

  RUN LENGTHS  Histogram of realized contiguous masked-run lengths, time axis and space
               axis. Tells us what the floor actually buys.

  LEAK MARGIN  THE important one. The SLOW band's 1024-sample window spans +-8 slots
               (500 ms @ hop 64). A masked HGA cell is slow-protected ONLY if the nearest
               VISIBLE frame on its own contact is > 8 slots away — otherwise a visible
               slow frame literally contains that cell's raw samples. So the quantity that
               matters is not the run length but the distribution of
               `distance from each masked cell to the nearest visible frame`.
               Report the fraction of masked cells with margin > 8 (slow-protected),
               > 2 (mid, 256-sample window), > 1 (HGA, 128-sample window).

               NOTE this bounds the leak under the CURRENT all-band JEPA target. r4's
               HGA-only scoring closes the slow leak by construction; this measures how
               big the thing we are closing actually was, and it sets `block_w_time` if we
               ever want the slow band protected on its own terms.

Whole-shaft-dropped contacts are EXCLUDED from the time-axis stats: every frame of theirs
is masked, so their margin is infinite by construction and would flatter the numbers.

Local self-check (synthetic montage, no data needed):
    .venv/bin/python -m scripts.neuroprobe.probe_v3_mask_geometry --synthetic

Real 13-session geometry (Delta CPU; reads only cache ch_names + stats, not the .npy):
    .venv/bin/python -m scripts.neuroprobe.probe_v3_mask_geometry \
        --band-root /work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain \
        --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
        --bt-root /projects/bhqk/htang13/braintreebank
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import (
    V3MaskConfig,
    _cover_rank,
    sample_masks,
)
from speech_decoding.models.v14_converged_v3.sidecar import SensorSidecar

# SLOW nperseg=1024 -> +-8 slots; MID 256 -> +-2; HGA 128 -> +-1. (view.py:236-241, hop 64.)
BAND_HALF_SPAN = {"slow": 8, "mid": 2, "hga": 1}


def _runs(mask_row: np.ndarray) -> list[int]:
    """Lengths of contiguous True runs in a 1-D bool array."""
    out, n = [], 0
    for v in mask_row:
        if v:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def _margins(frame_mask: np.ndarray) -> np.ndarray:
    """Distance from each masked frame to the nearest VISIBLE frame, along time.

    frame_mask: (T,) bool. Returns the margin for each masked position (>=1).
    A run of length L has margins 1,2,...,ceil(L/2),...,2,1 from its two edges — except
    at a sequence boundary, where that side does not bound it. Rows that are entirely
    masked return an empty array (caller excludes them)."""
    t = len(frame_mask)
    vis = np.where(~frame_mask)[0]
    if len(vis) == 0:
        return np.empty(0, dtype=int)
    masked = np.where(frame_mask)[0]
    # nearest visible index, either side
    j = np.searchsorted(vis, masked)
    left = np.where(j > 0, masked - vis[np.clip(j - 1, 0, len(vis) - 1)], t + 1)
    right = np.where(j < len(vis), vis[np.clip(j, 0, len(vis) - 1)] - masked, t + 1)
    return np.minimum(left, right)


def _stats(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {}
    return {
        "n": int(len(x)),
        "mean": round(float(x.mean()), 2),
        "median": int(np.median(x)),
        "p90": int(np.percentile(x, 90)),
        "max": int(x.max()),
    }


def guardian_ablation(n_shafts: int, n_time: int, n_rows: int, cfg: V3MaskConfig, seed: int) -> dict:
    """Counterfactual: the SAME time-block cover-rank, with the guardian hold-out REMOVED.

    Isolates how much of the realized run-length / leak-margin distribution is set by the
    block floor versus destroyed by the balanced-random guardian (masking.py:209-212),
    which forces ~T/V randomly-scattered frames per shaft to stay visible and therefore
    SLICES the contiguous blocks. Mirrors the time branch of `sample_masks` only — it is a
    probe, not a second implementation of the sampler."""
    g = torch.Generator().manual_seed(seed)
    valid_t = torch.ones(n_shafts, n_time, dtype=torch.bool)
    cover = _cover_rank(valid_t, cfg.block_w_time, n_rows, g)
    rank = cover.argsort(-1).argsort(-1)
    fmask = (rank < round(cfg.time_frac * n_time)).numpy()
    rl, mg = [], []
    for r in range(n_rows):
        for s in range(n_shafts):
            rl.extend(_runs(fmask[r, s]))
            mg.append(_margins(fmask[r, s]))
    mgs = np.concatenate(mg)
    return {
        "time_run_lengths": _stats(np.asarray(rl)),
        "leak_margin_slots": _stats(mgs),
        "frac_protected": {b: round(float((mgs > h).mean()), 4) for b, h in BAND_HALF_SPAN.items()},
    }


def simulate(geom, n_contacts: int, n_time: int, n_rows: int, cfg: V3MaskConfig, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    m = sample_masks(geom, n_contacts, n_time=n_time, n_rows=n_rows, generator=g, cfg=cfg)
    contact_mask = m.contact_mask.numpy()  # (R, N)
    frame_mask = m.frame_mask.numpy()  # (R, S, T)
    whole = m.whole_contact.numpy()  # (R, N)
    shaft_of = geom.shaft_of_contact.numpy()  # (N,)

    time_runs: list[int] = []
    space_runs: list[int] = []
    margins: list[np.ndarray] = []
    n_masked_cells = 0

    for r in range(n_rows):
        # --- space axis: contiguous runs of masked contacts WITHIN a shaft, by depth order
        for s in range(geom.n_shafts):
            idx = np.where(shaft_of == s)[0]  # already in depth order (build_l1_geometry)
            if len(idx) == 0:
                continue
            space_runs.extend(_runs(contact_mask[r, idx]))

        # --- time axis: only for contacts that are spatially masked but NOT whole-shaft
        sel = np.where(contact_mask[r] & ~whole[r])[0]
        for c in sel:
            fm = frame_mask[r, shaft_of[c]]  # (T,) — time mask is per SHAFT
            time_runs.extend(_runs(fm))
            mg = _margins(fm)
            margins.append(mg)
            n_masked_cells += int(fm.sum())
        # NOTE time_runs above double-counts a shaft once per masked contact on it; that is
        # the right weighting for "what does a masked CELL see", which is the question.

    mgs = np.concatenate(margins) if margins else np.empty(0, dtype=int)
    prot = {
        band: (round(float((mgs > h).mean()), 4) if len(mgs) else None)
        for band, h in BAND_HALF_SPAN.items()
    }
    return {
        "n_contacts": int(n_contacts),
        "n_shafts": int(geom.n_shafts),
        "n_time": int(n_time),
        "masked_contacts_per_row": int(contact_mask[0].sum()),
        "masked_frames_per_shaft": int(frame_mask[0, 0].sum()),
        "time_run_lengths": _stats(np.asarray(time_runs)),
        "space_run_lengths": _stats(np.asarray(space_runs)),
        "leak_margin_slots": _stats(mgs),
        "frac_protected": prot,  # fraction of masked cells with margin > band half-span
        "time_run_hist": dict(sorted(Counter(time_runs).items())),
        "space_run_hist": dict(sorted(Counter(space_runs).items())),
    }


def _synthetic_geom(shaft_sizes: list[int]):
    """A stand-in montage for the local self-check. NOT a substitute for the real one."""
    shaft_id, depth = [], []
    for s, c in enumerate(shaft_sizes):
        shaft_id += [s] * c
        depth += list(range(1, c + 1))
    n = len(shaft_id)
    sc = SensorSidecar(
        shaft_id=torch.tensor(shaft_id),
        depth=torch.tensor(depth),
        parcel_id=torch.zeros(n, dtype=torch.long),
        n_shafts=len(shaft_sizes),
        labels=tuple(f"S{s}-{d}" for s, d in zip(shaft_id, depth)),
    )
    return build_l1_geometry(sc), n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--band-root")
    p.add_argument("--span-dir")
    p.add_argument("--bt-root", default=os.environ.get("ROOT_DIR_BRAINTREEBANK", ""))
    p.add_argument("--out")
    p.add_argument("--n-rows", type=int, default=256)
    p.add_argument("--clip-frames", type=int, default=96)  # --clip-len 3.0 @ 32 Hz
    p.add_argument("--seed", type=int, default=33)
    # B6 sweep: the shipped floor 7 is justified in masking.py's docstring as "the HGA
    # |STFT| support", but HGA is nperseg=128 @ hop 64 => its support is 2 slots, not 7.
    # The realized geometry at a candidate width is what decides it, so make it sweepable
    # instead of re-deriving from the same wrong premise.
    p.add_argument("--block-w-time", type=int, default=None,
                   help="override V3MaskConfig.block_w_time (B6 sweep). None = shipped.")
    args = p.parse_args()

    cfg = V3MaskConfig()  # r2's config, unchanged — this measures what we ALREADY run
    if args.block_w_time is not None:
        cfg = dataclasses.replace(cfg, block_w_time=args.block_w_time)
    print(f"cfg: space_frac={cfg.space_frac} time_frac={cfg.time_frac} "
          f"whole_shaft_frac={cfg.whole_shaft_frac} "
          f"block_w_space={cfg.block_w_space} block_w_time={cfg.block_w_time}\n")

    results = []
    if args.synthetic:
        geom, n = _synthetic_geom([10, 12, 8, 14, 10, 12, 10, 8, 12, 10])
        results.append({"session": "synthetic", **simulate(
            geom, n, args.clip_frames, args.n_rows, cfg, args.seed)})
    else:
        from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
        from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
        from scripts.neuroprobe.probe_v3_field_stats import BAND_DIRS, V3_SESSIONS, WINSOR

        specs = load_v3_sessions(
            sessions=V3_SESSIONS,
            band_cache_dirs=[os.path.join(args.band_root, b) for b in BAND_DIRS],
            span_dir=args.span_dir,
            parcel_fn=make_bt_parcel_fn(args.bt_root),
            lof_report_path=None,
            winsor=WINSOR,
        )
        for spec in specs:
            sid, tid = spec.session_key
            geom = spec.setup.geom
            n = int(spec.keep_idx.numel())
            results.append({"session": f"{sid}/{tid}", **simulate(
                geom, n, args.clip_frames, args.n_rows, cfg, args.seed)})

    for r in results:
        print(f"=== {r['session']} — {r['n_contacts']} contacts / {r['n_shafts']} shafts, "
              f"T={r['n_time']} ===")
        print(f"  masked: {r['masked_contacts_per_row']}/{r['n_contacts']} contacts, "
              f"{r['masked_frames_per_shaft']}/{r['n_time']} frames per shaft")
        tr, sr = r["time_run_lengths"], r["space_run_lengths"]
        print(f"  TIME  run len (floor {cfg.block_w_time}): mean {tr['mean']} "
              f"median {tr['median']} p90 {tr['p90']} max {tr['max']}")
        print(f"  SPACE run len (floor {cfg.block_w_space}): mean {sr['mean']} "
              f"median {sr['median']} p90 {sr['p90']} max {sr['max']}")
        lm = r["leak_margin_slots"]
        print(f"  LEAK MARGIN (masked cell -> nearest visible frame): mean {lm['mean']} "
              f"median {lm['median']} p90 {lm['p90']}")
        fp = r["frac_protected"]
        print(f"    protected  HGA(>1): {fp['hga']:.1%}   MID(>2): {fp['mid']:.1%}   "
              f"SLOW(>8): {fp['slow']:.1%}   <-- the slow leak\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
