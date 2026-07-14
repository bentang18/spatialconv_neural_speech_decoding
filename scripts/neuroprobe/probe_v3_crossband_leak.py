"""M13 — how much masked HGA is recoverable from the contact's OWN visible SLOW/MID?

THE QUESTION THAT GATES THE r4 MASK DESIGN, and nobody has asked it.

B6 shortened block_w_time 7 -> 4 by reasoning about HGA's OWN temporal autocorrelation
(measured half-life 18 ms = 0.6 slots; Greg says 20 ms). That reasoning is correct AND
INCOMPLETE, because it only considers one channel by which a masked cell can leak.

The other channel: the three bands share hop=64 (32 Hz) but NOT nperseg.
    HGA   nperseg 128   =  62.5 ms =  2 slots
    MID   nperseg 256   = 125   ms =  4 slots
    SLOW  nperseg 1024  = 500   ms = 16 slots
A v3 token folds all three bands, so masking contact c at frames [t0, t0+4) hides c's
SLOW token there too — but SLOW's STFT WINDOW is 16 slots wide. A 4-slot hole inside a
16-slot window HIDES NOTHING: c's own SLOW at t0-1 and t0+4, which remain VISIBLE, are
computed from raw voltage that INCLUDES the masked interval. If HGA and SLOW/MID are
correlated within a contact (they are both driven by the same local population), the
model can read c's own visible SLOW/MID and reconstruct the masked HGA.

That is a WITHIN-CONTACT CROSS-BAND LEAK. It is invisible to every measurement we have
made so far — M5 counted which CELLS are hidden, not whether the hidden cell's VALUE is
recoverable, and B6 only looked at HGA's own ACF.

WHAT IS MEASURED
  Take the deepest-buried masked cells and ask what the model can still see:
    TARGET      |HGA|(c, t) for t inside a width-w masked block
    BASELINE    c's OWN visible HGA in a +/-G neighbourhood of the block   [the B6 channel]
    FULL        BASELINE + c's OWN visible SLOW and MID in that same neighbourhood
    dR2 = FULL - BASELINE   ==   THE CROSS-BAND LEAK
  Plus SLOW-only and MID-only, so we know WHICH band leaks, and per MARGIN (distance from
  the masked cell to the nearest visible frame) so we can see whether depth helps at all.

  Only contact c's own bands are used as predictors. Other shafts are deliberately
  EXCLUDED: M10a already established a global summary adds dR2 ~ +0.0125 to local HGA, so
  the cross-shaft path is not where the danger is. This probe isolates the within-contact
  channel, which is the one no mask width can close.

WHY THE ANSWER FORCES THE DESIGN
  IF dR2 IS LARGE — the frame mask is broken AT EVERY WIDTH, and:
    * B1's multi-rate token split becomes MANDATORY, and its rates are FORCED: each band
      needs its own token rate matched to its support, so that masking a band's token
      actually hides that band. SLOW at 32 Hz is 16x oversampled and unmaskable in
      principle; SLOW at ~4 Hz (1 token = 250 ms) is maskable.
    * the mask must be applied in PHYSICAL (contact, time-interval) units and projected
      into every band's own grid, rounding OUTWARD.
    * B7 FLIPS. The SPATIAL mask (contact hidden at ALL t) is the one part of the scheme
      with NO cross-band leak — the contact's SLOW and MID are gone too. It should be
      weighted UP, not dropped.
  IF dR2 IS SMALL — SLOW/MID can stay visible through masked intervals, B1's rates are a
  free efficiency choice, and B7 stays a live option.

  Either way this is the number that decides, and it is cheap to get.

GEOMETRY — the block is simulated, not sampled, and that is deliberate.
  sample_masks draws a stochastic, guardian-shredded mask; realized run lengths at floor 4
  are 4.10 (measured, 13 montages). To ask "how recoverable is a cell at margin m", the
  block must be a FIXED, KNOWN width, otherwise margin is confounded with run length. So
  we impose the canonical block [t0, t0+w) at the SHIPPED width and read off each cell's
  margin exactly. The realized-geometry question is M5's; this is the value question.

RIGOR (same bar as M10)
  * held out BY CLIP, with test clips whose spans overlap train dropped
  * ridge lambda swept on VAL only, refit on train+val, scored once on test
  * NULL = within-clip circular shift of the predictor block (preserves each predictor's
    autocorrelation, destroys only the cross-relation; a plain shuffle would inflate it)
  * train AND test R2 both reported
  * per-session AND pooled over the 13 pretrain montages

Model-FREE (no checkpoint). CPU. DeltaAI/Delta login node:

  ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
  .venv/bin/python -m scripts.neuroprobe.probe_v3_crossband_leak \
      --band-root $ROOT \
      --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
      --bt-root /projects/bhqk/htang13/braintreebank \
      --out /projects/bhqk/htang13/probe_out_v3/field_stats/crossband_leak.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
from speech_decoding.models.v14_converged_v3.masking import V3MaskConfig
from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
from scripts.neuroprobe.probe_v3_field_stats import BAND_DIRS, V3_SESSIONS, WINSOR
from scripts.neuroprobe.probe_v3_global_structure import (
    LAMBDAS,
    N_PERM,
    _fit_eval,
    _read_clips,
    _split_clips,
)

HGA = 2

# Visible context on EACH side of the masked block, in 32 Hz slots. 8 slots = 250 ms,
# which is half of SLOW's 500 ms window — far enough out that a SLOW frame at the edge of
# this neighbourhood still shares raw voltage with the masked interval. Shorter would
# under-state the leak by hiding the very frames that carry it.
G_CONTEXT = 8


def _neighbourhood(env: np.ndarray, clips: np.ndarray, c: int, t0: int, w: int, g: int):
    """c's own band values at the VISIBLE frames flanking the block [t0, t0+w).
    -> (n_clips, 2g). These are exactly the frames the model can still read."""
    left = env[clips, c, t0 - g : t0]                 # (n, g)
    right = env[clips, c, t0 + w : t0 + w + g]        # (n, g)
    return np.concatenate([left, right], axis=1)


def _blocks(T: int, w: int, g: int, stride: int) -> list[int]:
    """Canonical block starts with a full +/-g visible neighbourhood inside the clip."""
    return list(range(g, T - w - g, stride))


def _circshift(X: np.ndarray, rng) -> np.ndarray:
    """Row-wise circular shift ACROSS CLIPS. Each column keeps its marginal distribution
    and the design keeps its Gram matrix; only the row-alignment to the target dies."""
    return np.roll(X, int(rng.integers(1, len(X))), axis=0)


def _selfband(env_raw, tr, va, te, T, targets, w, g, stride) -> dict:
    """M13b — THE COPY-ABILITY OF EACH BAND. This is the number B2 actually rests on, and
    it is not the same question as the cross-band leak.

    For each band b: predict the MASKED cell of band b from that contact's OWN VISIBLE
    neighbourhood of THE SAME BAND. That is exactly what a masked-prediction loss asks the
    model to do, per band.

    The B2 argument — "the loss scores 3 bands but the mask can only hide 1, so ~2/3 of the
    scored target is a COPY task" — predicts a wide spread here: masked SLOW should be
    highly predictable (its 500 ms window bleeds far outside a 125 ms hole) and masked HGA
    should not (18 ms ACF half-life). If the spread is NOT there, B2's premise is wrong and
    I should not have written the task.
    """
    starts = _blocks(T, w, g, stride)
    out = {}
    for b, name in enumerate(("slow", "mid", "hga")):
        per_margin: dict[int, list[float]] = {}
        for c in targets:
            def _stack(ix):
                return np.concatenate(
                    [_neighbourhood(env_raw[b], ix, c, t0, w, g) for t0 in starts], axis=0)

            Xtr, Xva, Xte = _stack(tr), _stack(va), _stack(te)
            for j in range(w):
                margin = min(j + 1, w - j)

                def _y(ix):
                    return np.concatenate(
                        [env_raw[b][ix, c, t0 + j] for t0 in starts], axis=0)

                r2, _ = _fit_eval(Xtr, _y(tr), Xva, _y(va), Xte, _y(te))
                per_margin.setdefault(margin, []).append(r2)
        out[name] = {
            f"margin{m}": round(float(np.mean(v)), 5) for m, v in sorted(per_margin.items())
        }
    return out


def _leak(env_raw, tr, va, te, T, targets, w, g, stride, rng) -> dict:
    """For every (target contact, canonical block, cell-in-block): can the model read the
    masked HGA off the contact's own visible bands?"""
    # accumulate per MARGIN (distance to nearest visible frame) — margin is the whole point
    acc: dict[int, dict[str, list[float]]] = {}

    starts = _blocks(T, w, g, stride)
    for c in targets:
        # The visible flanks are the same for every cell inside a given block, so the
        # design is built once per (contact, band) and reused across the w cells.
        def _stack(ix, band):
            return np.concatenate(
                [_neighbourhood(env_raw[band], ix, c, t0, w, g) for t0 in starts], axis=0)

        Xtr = {b: _stack(tr, b) for b in range(3)}
        Xva = {b: _stack(va, b) for b in range(3)}
        Xte = {b: _stack(te, b) for b in range(3)}

        for j in range(w):
            margin = min(j + 1, w - j)               # slots to the nearest visible frame
            acc.setdefault(margin, {k: [] for k in
                                    ("base", "full", "slow", "mid", "null", "base_train",
                                     "full_train")})

            def _y(ix):
                return np.concatenate(
                    [env_raw[HGA][ix, c, t0 + j] for t0 in starts], axis=0)

            ytr, yva, yte = _y(tr), _y(va), _y(te)

            # BASELINE — the channel B6 reasoned about: c's own visible HGA
            r2_b, r2_b_tr = _fit_eval(Xtr[HGA], ytr, Xva[HGA], yva, Xte[HGA], yte)
            # FULL — + c's own visible SLOW and MID. The increment IS the cross-band leak.
            ctr = np.column_stack([Xtr[0], Xtr[1], Xtr[2]])
            cva = np.column_stack([Xva[0], Xva[1], Xva[2]])
            cte = np.column_stack([Xte[0], Xte[1], Xte[2]])
            r2_f, r2_f_tr = _fit_eval(ctr, ytr, cva, yva, cte, yte)
            # which band carries it?
            r2_s, _ = _fit_eval(Xtr[0], ytr, Xva[0], yva, Xte[0], yte)
            r2_m, _ = _fit_eval(Xtr[1], ytr, Xva[1], yva, Xte[1], yte)

            a = acc[margin]
            a["base"].append(r2_b)
            a["full"].append(r2_f)
            a["slow"].append(r2_s)
            a["mid"].append(r2_m)
            a["base_train"].append(r2_b_tr)
            a["full_train"].append(r2_f_tr)

            # NULL: shift ONLY the SLOW/MID columns, keep HGA and y aligned. This asks
            # how much dR2 the same number of USELESS extra columns buys by chance.
            for _ in range(max(N_PERM // 4, 3)):
                ntr = np.column_stack([_circshift(Xtr[0], rng), _circshift(Xtr[1], rng), Xtr[2]])
                nva = np.column_stack([_circshift(Xva[0], rng), _circshift(Xva[1], rng), Xva[2]])
                nte = np.column_stack([_circshift(Xte[0], rng), _circshift(Xte[1], rng), Xte[2]])
                r2_n, _ = _fit_eval(ntr, ytr, nva, yva, nte, yte)
                a["null"].append(r2_n - r2_b)

    out = {}
    for margin, a in sorted(acc.items()):
        if not a["base"]:
            continue
        out[f"margin{margin}"] = {
            "n_fits": len(a["base"]),
            "R2_base_ownHGA_only": round(float(np.mean(a["base"])), 5),
            "R2_full_plus_own_SLOW_MID": round(float(np.mean(a["full"])), 5),
            "dR2_CROSSBAND_LEAK": round(float(np.mean(a["full"]) - np.mean(a["base"])), 5),
            "dR2_null_p95": round(float(np.percentile(a["null"], 95)), 5) if a["null"] else None,
            "R2_SLOW_alone": round(float(np.mean(a["slow"])), 5),
            "R2_MID_alone": round(float(np.mean(a["mid"])), 5),
            "R2_train_base": round(float(np.mean(a["base_train"])), 5),
            "R2_train_full": round(float(np.mean(a["full_train"])), 5),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--band-root", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", default=os.environ.get("ROOT_DIR_BRAINTREEBANK", ""))
    p.add_argument("--n-clips", type=int, default=128)
    p.add_argument("--clip-frames", type=int, default=96)
    p.add_argument("--n-targets", type=int, default=12, help="target contacts per session")
    p.add_argument("--block-w", type=int, default=None,
                   help="masked block width in slots. None = the SHIPPED V3MaskConfig value.")
    p.add_argument("--g-context", type=int, default=G_CONTEXT)
    p.add_argument("--stride", type=int, default=6, help="spacing between canonical blocks")
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--out")
    a = p.parse_args()

    w = a.block_w if a.block_w is not None else int(V3MaskConfig().block_w_time)

    specs = load_v3_sessions(
        sessions=V3_SESSIONS,
        band_cache_dirs=[os.path.join(a.band_root, b) for b in BAND_DIRS],
        span_dir=a.span_dir,
        parcel_fn=make_bt_parcel_fn(a.bt_root),
        lof_report_path=None,
        winsor=WINSOR,
    )
    print(f"M13 — cross-band leak | {len(specs)} sessions | block w={w} slots "
          f"({w * 31.25:.0f} ms) | visible context +/-{a.g_context} slots "
          f"({a.g_context * 31.25:.0f} ms)\n"
          f"     SLOW window = 16 slots (500 ms) — WIDER THAN THE BLOCK. That is the "
          f"hypothesis under test.\n", flush=True)

    rows = []
    for spec in specs:
        sid, tid = spec.session_key
        bands, starts = _read_clips(spec, a.n_clips, a.clip_frames, a.seed)
        env_raw = [b.mean(2) for b in bands]                  # per band (n_clips, N, T)
        T = env_raw[0].shape[-1]

        tr, va, te = _split_clips(starts, a.clip_frames)
        if len(va) < 4 or len(te) < 4:
            print(f"[s{sid}t{tid}] SKIP — val/test too small after overlap pruning", flush=True)
            continue

        rng = np.random.default_rng(a.seed + sid * 100 + tid)
        n_c = env_raw[0].shape[1]
        targets = rng.choice(n_c, size=min(a.n_targets, n_c), replace=False)

        res = _leak(env_raw, tr, va, te, T, targets, w, a.g_context, a.stride, rng)
        sb = _selfband(env_raw, tr, va, te, T, targets, w, a.g_context, a.stride)
        rows.append({"subject_id": sid, "trial_id": tid, "n_contacts": n_c,
                     "leak": res, "selfband": sb})

        print(f"[s{sid}t{tid}] N={n_c} clips {len(tr)}/{len(va)}/{len(te)}", flush=True)
        for k, v in res.items():
            print(f"    {k}  base(own HGA) {v['R2_base_ownHGA_only']:.4f}  "
                  f"+own SLOW/MID {v['R2_full_plus_own_SLOW_MID']:.4f}  "
                  f"LEAK dR2 {v['dR2_CROSSBAND_LEAK']:+.4f} (null p95 {v['dR2_null_p95']:+.4f})"
                  f"  | SLOW alone {v['R2_SLOW_alone']:.4f}  MID alone {v['R2_MID_alone']:.4f}",
                  flush=True)
        print(f"    M13b COPY-ABILITY (predict masked band b from own visible band b): "
              + "  ".join(f"{n} {sb[n]}" for n in ("slow", "mid", "hga")), flush=True)

    print("\n" + "=" * 82)
    print("M13 — POOLED. dR2_CROSSBAND_LEAK is the number that decides B1's rates and B7.")
    print("=" * 82)
    pooled: dict = {}
    margins = sorted({m for r in rows for m in r["leak"]})
    for m in margins:
        vals = [r["leak"][m] for r in rows if m in r["leak"]]
        pooled[m] = {
            k: round(float(np.mean([v[k] for v in vals])), 5)
            for k in ("R2_base_ownHGA_only", "R2_full_plus_own_SLOW_MID",
                      "dR2_CROSSBAND_LEAK", "dR2_null_p95", "R2_SLOW_alone",
                      "R2_MID_alone", "R2_train_base", "R2_train_full")
        }
        v = pooled[m]
        real = v["dR2_CROSSBAND_LEAK"] > v["dR2_null_p95"]
        print(f"\n[{m}]  (margin = slots from the masked cell to the nearest VISIBLE frame)")
        print(f"  R2 from c's own visible HGA alone      : {v['R2_base_ownHGA_only']:+.5f}"
              f"   <- the channel B6 addressed")
        print(f"  R2 adding c's own visible SLOW + MID   : {v['R2_full_plus_own_SLOW_MID']:+.5f}")
        print(f"  ==> CROSS-BAND LEAK  dR2               : {v['dR2_CROSSBAND_LEAK']:+.5f}"
              f"   (null p95 {v['dR2_null_p95']:+.5f})  -> {'REAL' if real else 'null'}")
        print(f"      SLOW alone {v['R2_SLOW_alone']:+.5f}   MID alone {v['R2_MID_alone']:+.5f}")
        print(f"      train R2: base {v['R2_train_base']:+.5f}  full {v['R2_train_full']:+.5f}")

    # ---- M13b: the number B2 actually rests on ----
    print("\n" + "=" * 82)
    print("M13b — COPY-ABILITY per band: predict the MASKED cell of band b from that")
    print("       contact's OWN VISIBLE neighbourhood of THE SAME band. This is exactly")
    print("       what the masked loss asks, per band. B2's premise needs a WIDE SPREAD.")
    print("=" * 82)
    pooled_sb: dict = {}
    for name in ("slow", "mid", "hga"):
        margins_sb = sorted({m for r in rows for m in r["selfband"][name]})
        pooled_sb[name] = {
            m: round(float(np.mean([r["selfband"][name][m] for r in rows
                                    if m in r["selfband"][name]])), 5)
            for m in margins_sb
        }
        cells = "  ".join(f"{m} {v:+.4f}" for m, v in pooled_sb[name].items())
        print(f"  {name:<5} {cells}")
    if pooled_sb.get("slow") and pooled_sb.get("hga"):
        m_deep = max(pooled_sb["hga"])
        s, h = pooled_sb["slow"][m_deep], pooled_sb["hga"][m_deep]
        print(f"\n  At the DEEPEST cell ({m_deep}): SLOW is {s:+.4f} predictable, "
              f"HGA is {h:+.4f}.")
        print(f"  B2's premise ('scoring SLOW is scoring a copy task') needs SLOW >> HGA "
              f"here. Ratio: {s / h:.1f}x" if h > 1e-6 else "")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({
                "block_w": w, "g_context": a.g_context, "stride": a.stride,
                "n_perm": max(N_PERM // 4, 3), "lambdas": list(LAMBDAS),
                "per_session": rows, "pooled_crossband": pooled,
                "pooled_selfband_copyability": pooled_sb,
            }, fh, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
