"""M8 — how much of the masked-prediction gradient is spent on QUIET cells?

Ben's point (2026-07-15): HGA is normally quiet. It is sparse and event-like — most cells
are baseline. A masked L1 loss weights every masked cell equally, so the quiet majority
dominates the gradient, and "predict quiet" is solved by emitting the mean. If that is what
is happening, the objective is largely PAYING THE ENCODER TO OUTPUT A SMOOTH, LOW-VARIANCE,
CORRECT-ON-AVERAGE SIGNAL — i.e. the smear — by a route entirely independent of L2.

This sizes it, so the weighting scheme (focal vs thresholded vs none) is chosen from a
number instead of the armchair.

WHAT IS MEASURED
  The real scored set. `objective.py:233` defines
      cell_masked = contact_mask[:, :, None] | frame_mask[:, shaft_of, :]     # (B, N, T)
  — a UNION (a spatially-masked contact is scored at every t; a masked frame scores every
  contact on that shaft). Masks come from the real `sample_masks` with the shipped
  `V3MaskConfig`, on the real 13 pretrain montages, on the real 3-band cache with r2's exact
  normalization (robust-z + winsor 15/15/20). So `|z|` is already in units of robust sigma
  about the per-(electrode,bin) median => "|z| <= 0.5" IS "within half a robust sigma of this
  electrode's own baseline", which is the definition we want.

  Per band, over masked cells:
    * fraction of cells that are QUIET (|z| <= 0.5)
    * the share of the total L1 loss mass  sum|z|  those quiet cells hold
    * the share held by the most active decile
    * the |z| percentile ladder

HONEST SCOPE — read this before quoting the number.
  The v3 JEPA loss is scored in LATENT space (masked-L1 against the EMA teacher's
  layer-normed latent), not on the input |STFT|. This probe measures the INPUT-side activity
  and its L1 mass against a mean/zero predictor. It is a PROXY, and it is deliberately the
  right proxy for the decision at hand for two reasons:
    1. Any activity weighting we ship must be computed from something the model can see
       WITHOUT the teacher forward — i.e. from the input band power. Weighting by the
       teacher's own latent norm is circular. So the input activity distribution IS the
       quantity that parameterizes the weight function.
    2. sum|z| against a constant predictor is exactly the gradient share a variance-weighted
       reconstruction objective assigns, which is the mechanism under suspicion.
  It does NOT prove the latent-space loss has the same mass split. Do not claim that.

Precedent for the fix: BrainBERT (2302.14367) ships a content-aware loss on iEEG for
exactly this reason.

Delta / DeltaAI login node (CPU, model-free, no checkpoint):
    .venv/bin/python -m scripts.neuroprobe.probe_hga_activity_loss_mass \
        --band-root /work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain \
        --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
        --bt-root /projects/bhqk/htang13/braintreebank \
        --out /projects/bhqk/htang13/probe_out_v3/field_stats/hga_activity.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import V3MaskConfig, sample_masks
from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
from scripts.neuroprobe.probe_v3_field_stats import (
    BAND_DIRS,
    BAND_NAMES,
    V3_SESSIONS,
    WINSOR,
    _read_frames,
)

QUIET_SIGMA = 0.5  # |z| <= this == "at baseline"


def _cell_masked(masks, shaft_of: torch.Tensor) -> np.ndarray:
    """The exact scored set — objective.py:233, verbatim. (R, N, T) bool."""
    cm = masks.contact_mask[:, :, None]              # (R, N, 1)
    fm = masks.frame_mask[:, shaft_of, :]            # (R, N, T)
    return (cm | fm).numpy()


def _summarize(z: np.ndarray) -> dict:
    """z = |robust-z| of every MASKED cell, flat. Loss mass = sum|z| (L1 vs a mean predictor)."""
    total = float(z.sum())
    quiet = z <= QUIET_SIGMA
    n = z.size
    order = np.sort(z)
    top10_cut = float(np.percentile(z, 90))
    top10_mass = float(z[z >= top10_cut].sum())
    return {
        "n_masked_cells": int(n),
        "frac_quiet": round(float(quiet.mean()), 4),
        "quiet_loss_mass_share": round(float(z[quiet].sum() / max(total, 1e-9)), 4),
        "top_decile_loss_mass_share": round(float(top10_mass / max(total, 1e-9)), 4),
        "pct": {f"p{q}": round(float(np.percentile(z, q)), 3)
                for q in (10, 25, 50, 75, 90, 95, 99)},
        "mean_abs_z": round(float(z.mean()), 4),
        "max_abs_z": round(float(order[-1]), 3),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--band-root", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", default=os.environ.get("ROOT_DIR_BRAINTREEBANK", ""))
    p.add_argument("--n-clips", type=int, default=128)
    p.add_argument("--clip-frames", type=int, default=96)
    p.add_argument("--mask-rows", type=int, default=8, help="mask draws per clip batch")
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--out")
    args = p.parse_args()

    specs = load_v3_sessions(
        sessions=V3_SESSIONS,
        band_cache_dirs=[os.path.join(args.band_root, b) for b in BAND_DIRS],
        span_dir=args.span_dir,
        parcel_fn=make_bt_parcel_fn(args.bt_root),
        lof_report_path=None,
        winsor=WINSOR,
    )
    cfg = V3MaskConfig()
    print(f"mask cfg: space {cfg.space_frac} time {cfg.time_frac} "
          f"whole_shaft {cfg.whole_shaft_frac} | quiet := |z| <= {QUIET_SIGMA}\n", flush=True)

    per_band: dict[str, list[np.ndarray]] = {b: [] for b in BAND_NAMES}
    results = []

    for spec in specs:
        sid, tid = spec.session_key
        sc = spec.setup.sidecar
        geom = build_l1_geometry(sc)
        n_contacts = int(sc.shaft_id.shape[0])
        shaft_of = geom.shaft_of_contact

        # Same reader the field-stats probe used: the MODEL'S OWN clip sampler (guard-2 spans
        # excluded exactly as in training), robust-z + winsor already applied.
        bands = _read_frames(spec, args.n_clips, args.clip_frames, args.seed)
        envs = [b.mean(2) for b in bands]              # per band: (n_clips, N, T)
        n_time = envs[0].shape[-1]

        g = torch.Generator().manual_seed(args.seed + sid)
        masks = sample_masks(geom, n_contacts, n_time=n_time,
                             n_rows=args.mask_rows, generator=g, cfg=cfg)
        cm = _cell_masked(masks, shaft_of)            # (R, N, T)

        rec = {"subject_id": sid, "trial_id": tid, "n_contacts": n_contacts}
        for bi, name in enumerate(BAND_NAMES):
            env = envs[bi]                             # (n_clips, N, T)
            # every (clip, mask-row) pair: take |z| at the masked cells
            z = np.concatenate([np.abs(env[:, cm[r]]).ravel() for r in range(args.mask_rows)])
            per_band[name].append(z)
            rec[name] = _summarize(z)
        results.append(rec)

        h = rec["hga"]
        print(f"[s{sid}t{tid}] N={n_contacts}  HGA: quiet {h['frac_quiet']:.1%} of masked cells, "
              f"holding {h['quiet_loss_mass_share']:.1%} of the L1 mass "
              f"(top decile {h['top_decile_loss_mass_share']:.1%})", flush=True)

    print("\n" + "=" * 78)
    print("M8 — POOLED over all 13 pretrain sessions (the number that sets the weighting)")
    print("=" * 78)
    summary = {}
    for name in BAND_NAMES:
        z = np.concatenate(per_band[name])
        s = _summarize(z)
        summary[name] = s
        print(f"\n[{name}]  {s['n_masked_cells']:,} masked cells")
        print(f"  QUIET (|z| <= {QUIET_SIGMA}): {s['frac_quiet']:.1%} of masked cells, "
              f"holding {s['quiet_loss_mass_share']:.1%} of the total L1 loss mass")
        print(f"  most-active decile:      10.0% of masked cells, "
              f"holding {s['top_decile_loss_mass_share']:.1%} of the total L1 loss mass")
        print(f"  |z| ladder: " + "  ".join(f"{k} {v}" for k, v in s["pct"].items()))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"quiet_sigma": QUIET_SIGMA, "mask_cfg": repr(cfg),
                       "per_session": results, "pooled": summary}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
