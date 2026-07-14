"""M11 — per-BAND, per-parcel split-half reliability. Sets K and the soft-bin kernel width.

M10c measured parcel reliability on HGA ONLY: 0.558 with the common mode removed, 0.652
raw (pooled, 13 montages). But B3's secondary target is the parcel state across ALL THREE
BANDS — that is the whole point of moving SLOW/MID out of the masked loss and into the
global head. Their reliability is plausibly HIGHER than HGA's (slower, more shared), but
that is a GUESS, and a loss should not be specced on a guess.

WHAT IT SETS — three frozen numbers B3 cannot be written without.

  1. WHETHER SLOW/MID BELONG IN THE TARGET AT ALL.
     If a band's parcel summary is unreliable, predicting it is predicting noise — and a
     KL head answering noise emits the MARGINAL, exactly as a regression head answering
     noise emits the conditional mean. Same disease, one level up. Reliability is the
     admission test.

  2. K, the number of bins per (parcel, band).
     Derivation, not a default. Write the z-scored parcel summary as signal + noise:
         var = 1,  signal sd = sqrt(r),  noise sd = sqrt(1 - r)
     A bin narrower than the noise sd is not resolvable — the target would flip bins on
     measurement noise alone, and we would be asking the head to predict a coin flip. So
         K  =  (observed dynamic range) / (noise sd)
     with the range taken EMPIRICALLY as p1..p99 of the summary (not assumed Gaussian —
     HGA is sparse and event-like, so its range is wider than +/-2.5 sd).
     At HGA's r = 0.558: noise sd = sqrt(0.442) = 0.665, and a ~5 sd range gives K ~ 7.5.
     That arithmetic is an ILLUSTRATION. This probe computes it per (parcel, band).

  3. THE SOFT-ASSIGNMENT KERNEL WIDTH = the measured noise sd.
     This is the load-bearing one. Set the kernel to the noise sd and the target's OWN
     ENTROPY becomes the IRREDUCIBLE AMBIGUITY of the measurement — a cell that genuinely
     sits between two activity levels gets a genuinely bimodal target, and a cell that is
     unambiguous gets a sharp one. That is precisely what S-JEPA's soft-GMM posteriors are
     trying to preserve (their hard-cluster critique: hard IDs "collapse acoustic ambiguity
     at category boundaries"). We get it BY CONSTRUCTION, from a physical noise estimate,
     instead of by hoping an online GMM discovers it — because unlike MFCC features, our
     target has a natural scale (robust-z, already frozen) and natural coordinates (the
     atlas label).

  Also reports the CEILING: r bounds what any head can achieve on this target. Without it
  we cannot tell a working head from a degenerate one — which is exactly the trap M9 caught
  us in, after 40k steps.

COMMON-MODE CONVENTION
  Reliability is reported RAW and with the COMMON MODE PROJECTED OUT, where the common mode
  is the mean of contacts OUTSIDE the parcel (leak-free — parcel p is nowhere in its own
  common-mode regressor). The common-mode-removed number is the one that matters: a head
  that only learns the common mode has learned the part that is ALREADY FREE to the
  downstream readout (input-linear scores 0.6462 largely on it).

Model-FREE (no checkpoint). CPU. DeltaAI/Delta login node:

  ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
  .venv/bin/python -m scripts.neuroprobe.probe_v3_parcel_reliability_bands \
      --band-root $ROOT \
      --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
      --bt-root /projects/bhqk/htang13/braintreebank \
      --out /projects/bhqk/htang13/probe_out_v3/field_stats/parcel_reliability_bands.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
from scripts.neuroprobe.probe_v3_field_stats import BAND_DIRS, V3_SESSIONS, WINSOR
from scripts.neuroprobe.probe_v3_global_structure import (
    MIN_PARCEL_CONTACTS,
    _read_clips,
)

BANDS = ("slow", "mid", "hga")
N_SPLITS = 40          # split-half draws per parcel; the estimate is a mean over these
RANGE_PCT = (1.0, 99.0)  # empirical dynamic range — NOT +/-k sd (HGA is not Gaussian)


def _spearman_brown(r: float) -> float:
    """half-length correlation -> full-length reliability."""
    return 2.0 * r / (1.0 + r) if r > -1.0 else 0.0


def _reliability(summary_halves) -> float | None:
    rs = []
    for h1, h2 in summary_halves:
        if h1.std() < 1e-8 or h2.std() < 1e-8:
            continue
        rs.append(float(np.corrcoef(h1, h2)[0, 1]))
    if not rs:
        return None
    return _spearman_brown(float(np.mean(rs)))


def _project_out(y: np.ndarray, cm: np.ndarray) -> np.ndarray:
    A = np.column_stack([cm, np.ones(len(cm))])
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def _band_parcel_stats(env: np.ndarray, parcel_id: np.ndarray, parcels, rng) -> dict:
    """For ONE band: split-half reliability per parcel (raw + common-mode-removed), and
    the derived K / kernel width. env: (n_clips, N, T)."""
    per_parcel = []
    for p in parcels:
        idx = np.where(parcel_id == p)[0]
        out_idx = np.where(parcel_id != p)[0]
        cm = env[:, out_idx].mean(1).reshape(-1)            # leak-free: excludes parcel p

        halves_raw, halves_cm = [], []
        for _ in range(N_SPLITS):
            perm = rng.permutation(idx)
            a, b = perm[: len(perm) // 2], perm[len(perm) // 2 :]
            h1 = env[:, a].mean(1).reshape(-1)
            h2 = env[:, b].mean(1).reshape(-1)
            halves_raw.append((h1, h2))
            halves_cm.append((_project_out(h1, cm), _project_out(h2, cm)))

        r_raw = _reliability(halves_raw)
        r_cm = _reliability(halves_cm)
        if r_cm is None:
            continue

        # The target the head actually sees: the FULL-parcel summary, z-scored (this is
        # what makes each parcel contribute equally to the KL regardless of its variance).
        full = env[:, idx].mean(1).reshape(-1)
        z = (full - full.mean()) / max(full.std(), 1e-8)
        lo, hi = np.percentile(z, RANGE_PCT)
        rng_z = float(hi - lo)

        # noise sd in z units; bins narrower than this are not resolvable
        r_use = float(np.clip(r_cm, 1e-3, 0.999))
        noise_sd = float(np.sqrt(1.0 - r_use))
        per_parcel.append({
            "parcel": int(p),
            "n_contacts": int(len(idx)),
            "reliability_raw": round(float(r_raw), 4) if r_raw is not None else None,
            "reliability_cm_removed": round(float(r_cm), 4),
            "range_p1_p99_in_sd": round(rng_z, 3),
            "noise_sd_in_sd": round(noise_sd, 4),
            "K_resolvable": int(max(2, round(rng_z / noise_sd))),
        })

    if not per_parcel:
        return {"n_parcels": 0}

    def _m(k):
        v = [d[k] for d in per_parcel if d[k] is not None]
        return round(float(np.mean(v)), 4) if v else None

    ks = [d["K_resolvable"] for d in per_parcel]
    return {
        "n_parcels": len(per_parcel),
        "reliability_raw": _m("reliability_raw"),
        "reliability_cm_removed": _m("reliability_cm_removed"),
        "range_p1_p99_in_sd": _m("range_p1_p99_in_sd"),
        "kernel_width_in_sd": _m("noise_sd_in_sd"),   # <- the soft-assignment sigma
        "K_resolvable_mean": _m("noise_sd_in_sd") and int(round(float(np.mean(ks)))),
        "K_resolvable_median": int(np.median(ks)),
        "per_parcel": per_parcel,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--band-root", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", default=os.environ.get("ROOT_DIR_BRAINTREEBANK", ""))
    p.add_argument("--n-clips", type=int, default=128)
    p.add_argument("--clip-frames", type=int, default=96)
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--out")
    a = p.parse_args()

    specs = load_v3_sessions(
        sessions=V3_SESSIONS,
        band_cache_dirs=[os.path.join(a.band_root, b) for b in BAND_DIRS],
        span_dir=a.span_dir,
        parcel_fn=make_bt_parcel_fn(a.bt_root),
        lof_report_path=None,
        winsor=WINSOR,
    )
    print(f"M11 — per-band parcel reliability | {len(specs)} sessions | {N_SPLITS} "
          f"split-half draws/parcel | Spearman-Brown corrected\n"
          f"     HGA is the only band measured so far (M10c: 0.558 cm-removed). SLOW/MID "
          f"are the open question.\n", flush=True)

    rows = []
    for spec in specs:
        sid, tid = spec.session_key
        parcel_id = spec.setup.parcel_id.cpu().numpy()
        parcels = [q for q in np.unique(parcel_id)
                   if int((parcel_id == q).sum()) >= MIN_PARCEL_CONTACTS]
        if len(parcels) < 3:
            print(f"[s{sid}t{tid}] SKIP — {len(parcels)} usable parcels", flush=True)
            continue

        bands, _ = _read_clips(spec, a.n_clips, a.clip_frames, a.seed)
        env_raw = [b.mean(2) for b in bands]                # per band (n_clips, N, T)

        rng = np.random.default_rng(a.seed + sid * 100 + tid)
        rec = {"subject_id": sid, "trial_id": tid, "n_parcels": len(parcels)}
        for bi, name in enumerate(BANDS):
            rec[name] = _band_parcel_stats(env_raw[bi], parcel_id, parcels, rng)
        rows.append(rec)

        print(f"[s{sid}t{tid}] parcels {len(parcels)}", flush=True)
        for name in BANDS:
            s = rec[name]
            print(f"    {name:<5} reliability cm-removed {s['reliability_cm_removed']:.4f} "
                  f"(raw {s['reliability_raw']:.4f})  kernel sigma {s['kernel_width_in_sd']:.3f} sd  "
                  f"K~{s['K_resolvable_median']}", flush=True)

    print("\n" + "=" * 82)
    print("M11 — POOLED. These numbers freeze B3's target: which bands, K, kernel width.")
    print("=" * 82)
    pooled: dict = {}
    for name in BANDS:
        vals = [r[name] for r in rows if r[name].get("n_parcels")]
        if not vals:
            continue
        pooled[name] = {
            "reliability_cm_removed": round(
                float(np.mean([v["reliability_cm_removed"] for v in vals])), 4),
            "reliability_raw": round(float(np.mean([v["reliability_raw"] for v in vals])), 4),
            "kernel_width_in_sd": round(
                float(np.mean([v["kernel_width_in_sd"] for v in vals])), 4),
            "range_p1_p99_in_sd": round(
                float(np.mean([v["range_p1_p99_in_sd"] for v in vals])), 3),
            "K_resolvable": int(round(float(np.mean([v["K_resolvable_median"] for v in vals])))),
        }
        v = pooled[name]
        print(f"\n[{name}]")
        print(f"  split-half reliability, common mode REMOVED : {v['reliability_cm_removed']:.4f}"
              f"   <- CEILING on the B3 head for this band")
        print(f"  split-half reliability, raw                 : {v['reliability_raw']:.4f}")
        print(f"  dynamic range (p1..p99)                     : {v['range_p1_p99_in_sd']:.2f} sd")
        print(f"  SOFT-BIN KERNEL WIDTH  = sqrt(1 - r)        : {v['kernel_width_in_sd']:.4f} sd")
        print(f"  K (resolvable bins)    = range / kernel     : {v['K_resolvable']}")

    if pooled:
        print("\n" + "-" * 82)
        ks = [pooled[b]["K_resolvable"] for b in pooled]
        print(f"  => a single shared K across bands: {int(round(float(np.mean(ks))))} "
              f"(per-band {dict((b, pooled[b]['K_resolvable']) for b in pooled)})")
        print(f"  => bands admissible as B3 targets (reliability > 0.3): "
              f"{[b for b in pooled if pooled[b]['reliability_cm_removed'] > 0.3]}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"n_splits": N_SPLITS, "range_pct": RANGE_PCT,
                       "per_session": rows, "pooled": pooled}, fh, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
