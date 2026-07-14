"""M14 — LOSS-WEIGHTED predictability of the masked cells the sampler ACTUALLY produces.

WHY THIS ISN'T JUST "M13 WITH A WIDTH SWEEP" (which is what I was asked for, and it is
not enough — this is the correction, stated up front)

M13b measured ONE channel: predict a masked cell from that contact's OWN visible frames
of the same band. It found masked SLOW is 98.8% self-copyable, MID 0.077, HGA 0.097 at
margin 2, and I reported that as "the pretext task is a photocopy."

Then I read the objective:

    objective.py:233
    cell_masked = contact_mask[:, :, None] | frame_mask[:, shaft_of, :]   # (B, N, T)

A SPATIALLY-masked contact has ALL T frames masked. It therefore has NO visible frames of
its own, and M13b's self-copy channel DOES NOT EXIST for it. With space_frac 0.60 and
time_frac 0.5, only (1-0.60)*0.5 / (1 - 0.40*0.5) = 0.25 of masked cells are of the kind
M13b measured. So M13b's 0.988 covers a QUARTER of the loss, not all of it, and the
"photocopy" headline was scoped wrong. The fix is not a bigger sweep on the same channel;
it is to measure every channel, weighted by how often the real sampler produces it.

THE THREE CLASSES OF MASKED CELL, and the context each one actually leaves behind

  W  whole-shaft dropped contact  (contact_mask & whole_contact)
       context: OTHER SHAFTS ONLY. No same-shaft path of any kind. The pure montage task.
  S  intra-spatial masked contact (contact_mask & ~whole_contact)
       context: own-shaft VISIBLE contacts (at the frames where the shaft is not time-
       masked) + other shafts.  NO own frames — the contact is gone at every t.
  T  temporal-only masked cell    (~contact_mask & frame_mask[shaft])
       context: the contact's OWN visible frames (= M13b) + other shafts at t.
       NOTE: NOT its own-shaft neighbours at t — the frame mask is PER SHAFT, so when
       frame t is masked for shaft s it is masked for EVERY contact on s. This is easy to
       get wrong and it changes the answer.

WHAT DECIDES WHAT

  1. block_w_time, PER BAND  (Ben: "without it MID goes in half-leaky")
     The maskability rule: a band is fully hidden only >= (STFT support) frames from any
     visible frame of itself. support = HGA 2 (nperseg 128), MID 4 (256), SLOW 16 (1024).
     A width-w block's deepest cell sits at margin ceil(w/2). So:
         HGA  needs margin >= 2  ->  w_t >= 4     <- the shipped 4 is EXACTLY right
         MID  needs margin >= 4  ->  w_t >= 8     <- the shipped 4 is HALF what MID needs
         SLOW needs margin >= 16 ->  w_t >= 32    <- absurd; SLOW is off the masked loss
     THIS IS A PREDICTION, MADE BEFORE THE RUN. Part B tests it: MID's self-copy R2 should
     fall off a cliff between margin 3 and margin 4 and HGA's between 1 and 2. If it does
     not, the rule is wrong and so is B1's rate design.

  2. block_w_space  (Ben: "is our 4 wide spatial block still justified?")
     Its docstring says "floor 4 = HGA along-shaft autocorr". That is FALSE — M2 measured
     HGA depth-lag ACF at 0.34 / 0.21 / 0.18 for lags 1/2/3, i.e. gone by lag 2. Same
     shape of error as the old block_w_time = 7.
     BUT THE SPATIAL AXIS IS NOT THE TEMPORAL AXIS, and conflating them would be the real
     mistake:
         a TEMPORAL leak is a TRANSFORM ARTIFACT. The visible STFT frame is COMPUTED FROM
         the masked samples. Copying it teaches nothing. It must be defeated.
         a SPATIAL correlation is PHYSICS. A neighbouring contact genuinely sees a
         correlated population. Predicting a hidden sensor from visible sensors IS THE
         TASK — it is literally what cross-sensor generalization means.
     So the spatial question is NOT "how wide until it is unpredictable". It is: does the
     INTRA tier earn anything the WHOLE-SHAFT tier does not already provide? If the
     own-shaft path is dead by margin 2, then a width-4 intra block's deepest cells are
     already a pure cross-shaft task — which is what whole-shaft masking gives for free,
     at 100%. Part C measures where the own-shaft path dies, per band.

  3. THE GUARDIAN  (Ben: "I think kill the guardian? but make sure it doesn't ruin the
     static path")
     STATIC PATH: SAFE, and this is provable from the code, not the docstring. The
     guardian is ONE line -- cover_time = where(forbid, inf, cover_time) -- placed BEFORE
     the snap `frame_mask = time_rank < t_mask`. It only pushes guarded frames to the back
     of the sort. Exactly t_mask frames are still masked per (row, shaft). T_kept, D,
     online tokens (N-D)*T_kept and the loss denominator are ALL unchanged. Deleting it
     also deletes a feasibility constraint (V >= 2), so the sampler gets strictly more
     robust.
     WHAT IT COSTS: it forbids ~T/V frames per shaft from ever being masked, which shreds
     the blocks (M5: floor 7 -> realized run 5.16). At w=4 that is nearly free. At the
     w_t=8 that MID is predicted to need, it would shred exactly the width we widened to
     defeat MID's STFT support. Self-defeating.
     WHAT IT BUYS: it guarantees >= 1 live shaft per t. Part A MEASURES how often that is
     violated without it (frames are exact-t_mask per shaft and independent ACROSS shafts,
     so the rate should be ~time_frac^V -- but blocks correlate frames WITHIN a shaft, so
     measure it, do not trust the arithmetic).

  4. B3's PARCEL GUARD (Ben: "does union transfer to a new subject?")
     The 7-way parcel intersection is EMPTY, so the head is f(latents, parcel_emb) scored
     only on the parcels present in that montage. The failure mode is a parcel attested in
     ONE subject: its embedding is fit on that subject alone, so its gradient teaches a
     subject-specific decoding. Part E prints the coverage curve (how many parcels survive
     a ">= k distinct subjects" rule) so the threshold is READ OFF A CURVE, not guessed.

RIGOR (same bar as M10/M13, and the invariants are ASSERTED AND PRINTED)
  * held out BY CLIP, test clips whose spans overlap train dropped
  * ridge lambda swept on VAL only, refit on train+val, scored once on test
  * Part A checks its own cell count against the objective's OWN denominator formula
    N*T - (N-D)*T_kept. If the audit disagrees with the loss, the audit is wrong.
  * Part B/C R2 must be non-increasing in margin; a violation is printed as a WARNING.
  * Part B at w=4 must reproduce M13b (slow .988 / mid .077 / hga .097 at margin 2).
    Two probes, one number, or something is broken.

Model-FREE (no checkpoint). CPU. DeltaAI/Delta login node:

  ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
  .venv/bin/python -m scripts.neuroprobe.probe_v3_masked_cell_audit \
      --band-root $ROOT \
      --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
      --bt-root /projects/bhqk/htang13/braintreebank \
      --out /projects/bhqk/htang13/probe_out_v3/field_stats/masked_cell_audit.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import torch

from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import V3MaskConfig, sample_masks
from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
from scripts.neuroprobe.probe_v3_field_stats import BAND_DIRS, BAND_NAMES, V3_SESSIONS, WINSOR
from scripts.neuroprobe.probe_v3_global_structure import (
    _fit_eval,
    _read_clips,
    _split_clips,
)

# STFT support per band, in 32 Hz slots (hop 64 @ 2048 Hz). nperseg / hop.
# THE maskability rule: a cell is fully hidden only at margin >= its band's support.
SUPPORT = {"slow": 16, "mid": 4, "hga": 2}

# Widths to sweep. Time: the shipped 4 plus the widths the support rule predicts MID and
# (hypothetically) SLOW would need. Space: down to 1 (no block at all) and up past 4,
# because the spatial answer may well be "narrower", not "wider".
W_TIME = (4, 6, 8, 12)
W_SPACE = (1, 2, 4, 6, 8)

G_TIME = 8      # visible temporal context each side, slots (250 ms; half of SLOW's window)
G_SPACE = 2     # visible along-shaft context each side, contacts
DT_SPACE = 1    # spatial predictors are read at t-1, t, t+1 (be GENEROUS: an "unpredictable"
                # verdict is only meaningful if the predictor set was not starved)


# ---------------------------------------------------------------------------
# Part A — the geometry the REAL sampler produces
# ---------------------------------------------------------------------------
def _sample_no_guardian(geom, n_contacts, *, n_time, n_rows, generator, cfg):
    """``sample_masks`` with the guardian line deleted, and NOTHING else changed.

    Probe-local, so the shipped sampler is untouched until the number says to change it.
    Re-derives the TIME axis only: the SPACE axis is reused verbatim by calling the real
    sampler and keeping its contact_mask / whole_contact."""
    m = sample_masks(geom, n_contacts, n_time=n_time, n_rows=n_rows,
                     generator=generator, cfg=cfg)
    from speech_decoding.models.v14_converged_v3.masking import _cover_rank

    r, s, t = n_rows, geom.n_shafts, n_time
    t_mask = round(cfg.time_frac * t)
    valid_time = torch.ones(s, t, dtype=torch.bool)
    cover_time = _cover_rank(valid_time, cfg.block_w_time, r, generator)  # NO forbid mask
    time_rank = cover_time.argsort(-1).argsort(-1)
    frame_mask = time_rank < t_mask  # still EXACTLY t_mask per (row, shaft) — static-safe
    return dataclasses.replace(m, frame_mask=frame_mask)


def _margins(mask_1d: np.ndarray) -> np.ndarray:
    """Distance from each masked position to the nearest VISIBLE position, along one axis.
    -> int array, 0 on visible positions, >=1 on masked ones (inf if the whole axis is
    masked, encoded as -1)."""
    n = len(mask_1d)
    vis = np.where(~mask_1d)[0]
    if len(vis) == 0:
        return np.where(mask_1d, -1, 0)
    idx = np.arange(n)
    d = np.abs(idx[:, None] - vis[None, :]).min(1)
    return np.where(mask_1d, d, 0)


def _geometry(geom, n_contacts, n_time, n_rows, cfg, seed, guardian: bool) -> dict:
    """Classify EVERY masked cell of n_rows real masks into W / S / T, with its margin."""
    g = torch.Generator().manual_seed(seed)
    fn = sample_masks if guardian else _sample_no_guardian
    m = fn(geom, n_contacts, n_time=n_time, n_rows=n_rows, generator=g, cfg=cfg)

    cmask = m.contact_mask.numpy()          # (R, N)
    whole = m.whole_contact.numpy()         # (R, N)
    fmask = m.frame_mask.numpy()            # (R, S, T)
    shaft_of = geom.shaft_of_contact.numpy()  # (N,)
    r, n = cmask.shape
    t = n_time

    cell_masked = cmask[:, :, None] | fmask[:, shaft_of, :]  # (R, N, T) — objective.py:233

    # --- INVARIANT: the audit's cell count must equal the loss's OWN denominator ---
    d = round(cfg.space_frac * n)
    t_kept = t - round(cfg.time_frac * t)
    expect = n * t - (n - d) * t_kept
    got = int(cell_masked[0].sum())

    cls_w = whole[:, :, None] & np.ones((1, 1, t), bool)
    cls_s = (cmask & ~whole)[:, :, None] & np.ones((1, 1, t), bool)
    cls_t = (~cmask)[:, :, None] & fmask[:, shaft_of, :]
    counts = {"W": int(cls_w.sum()), "S": int(cls_s.sum()), "T": int(cls_t.sum())}
    tot = sum(counts.values())

    # --- temporal margin, for T cells (the ONLY class with own visible frames) ---
    t_marg: dict[int, int] = {}
    for ri in range(r):
        for si in range(geom.n_shafts):
            mm = _margins(fmask[ri, si])
            n_kept_contacts = int((~cmask[ri]) [shaft_of == si].sum())
            if n_kept_contacts == 0:
                continue  # dead shaft: its cells are W or S, not T
            for tt in np.where(fmask[ri, si])[0]:
                t_marg[int(mm[tt])] = t_marg.get(int(mm[tt]), 0) + n_kept_contacts

    # --- spatial margin, for S cells (own-shaft visible contacts, by DEPTH INDEX) ---
    s_marg: dict[int, int] = {}
    for ri in range(r):
        for si in range(geom.n_shafts):
            members = np.where(shaft_of == si)[0]           # contact ids, depth order
            if whole[ri, members].any():
                continue                                     # whole drop -> class W
            sm = _margins(cmask[ri, members])
            for j, c in enumerate(members):
                if cmask[ri, c]:
                    s_marg[int(sm[j])] = s_marg.get(int(sm[j]), 0) + t

    # --- guardian: how many (row, t) have NO live shaft visible at all? ---
    vis_per_shaft = np.zeros((r, geom.n_shafts), int)
    for si in range(geom.n_shafts):
        vis_per_shaft[:, si] = (~cmask[:, shaft_of == si]).sum(1)
    live = vis_per_shaft > 0                                  # (R, S)
    shaft_vis_at_t = live[:, :, None] & ~fmask                # (R, S, T)
    dead_t = ~shaft_vis_at_t.any(1)                           # (R, T) no live shaft at t
    # cells with NO predictive path at all: spatially masked AND at a dead frame
    pathless = (cmask[:, :, None] & dead_t[:, None, :]).sum()

    # --- realized run length of the time blocks (the guardian's real cost) ---
    runs = []
    for ri in range(r):
        for si in range(geom.n_shafts):
            row = fmask[ri, si]
            cur = 0
            for v in row:
                if v:
                    cur += 1
                elif cur:
                    runs.append(cur)
                    cur = 0
            if cur:
                runs.append(cur)

    return {
        "guardian": guardian,
        "block_w_time": cfg.block_w_time,
        "block_w_space": cfg.block_w_space,
        "n_masked_cells": got,
        "n_masked_cells_expected": expect,
        "cellcount_invariant_OK": bool(got == expect),
        "class_frac": {k: round(v / tot, 5) for k, v in counts.items()},
        "time_margin_frac": {str(k): round(v / max(counts["T"], 1), 5)
                             for k, v in sorted(t_marg.items())},
        "space_margin_frac": {str(k): round(v / max(counts["S"], 1), 5)
                              for k, v in sorted(s_marg.items())},
        "dead_frame_frac": round(float(dead_t.mean()), 6),
        "pathless_cell_frac": round(float(pathless / max(tot, 1)), 6),
        "realized_run_mean": round(float(np.mean(runs)), 3) if runs else None,
    }


# ---------------------------------------------------------------------------
# Part B — TEMPORAL copy-ability (class T only), per band, per width, per margin
# ---------------------------------------------------------------------------
def _temporal(env, tr, va, te, T, targets, w, g, stride) -> dict:
    """Predict the masked cell of band b from that contact's OWN VISIBLE frames of band b.
    Exactly M13b, but at an arbitrary width. This is the ONLY channel a class-T cell has
    to itself; everything else it gets is cross-shaft."""
    starts = list(range(g, T - w - g, stride))
    out: dict[str, dict[str, float]] = {}
    for b, name in enumerate(BAND_NAMES):
        per_margin: dict[int, list[float]] = {}
        for c in targets:
            def _X(ix):
                return np.concatenate(
                    [np.concatenate([env[b][ix, c, t0 - g:t0],
                                     env[b][ix, c, t0 + w:t0 + w + g]], axis=1)
                     for t0 in starts], axis=0)

            Xtr, Xva, Xte = _X(tr), _X(va), _X(te)
            for j in range(w):
                margin = min(j + 1, w - j)

                def _y(ix):
                    return np.concatenate([env[b][ix, c, t0 + j] for t0 in starts], axis=0)

                r2, _ = _fit_eval(Xtr, _y(tr), Xva, _y(va), Xte, _y(te))
                per_margin.setdefault(margin, []).append(r2)
        out[name] = {str(m): round(float(np.mean(v)), 5) for m, v in sorted(per_margin.items())}
    return out


# ---------------------------------------------------------------------------
# Part C — SPATIAL copy-ability (class S), per band, per width, per margin
# ---------------------------------------------------------------------------
def _spatial(env, geom, tr, va, te, T, w, g, dt, max_blocks, rng) -> dict:
    """Predict a masked contact's cell at time t from the OWN-SHAFT visible contacts, read
    at t-dt..t+dt. A spatially masked contact has NO frames of its own (objective.py:233),
    so this — plus the cross-shaft path — is all it has.

    The predictor set is deliberately GENEROUS (3 time points x 2g contacts): a verdict of
    "unpredictable" only means something if the model was given a fair shot."""
    shaft_of = geom.shaft_of_contact.numpy()
    blocks = []
    for si in range(geom.n_shafts):
        members = np.where(shaft_of == si)[0]
        for d0 in range(g, len(members) - w - g + 1):
            blocks.append((members, d0))
    if not blocks:
        return {}
    pick = rng.choice(len(blocks), size=min(max_blocks, len(blocks)), replace=False)

    out: dict[str, dict[str, float]] = {}
    for b, name in enumerate(BAND_NAMES):
        per_margin: dict[int, list[float]] = {}
        for bi in pick:
            members, d0 = blocks[bi]
            left = members[d0 - g:d0]
            right = members[d0 + w:d0 + w + g]
            nb = np.concatenate([left, right])

            def _X(ix):
                # (n_clips, |nb|, T) -> stack the dt time offsets -> rows = clips x times
                cols = [env[b][ix][:, nb, dt + k:T - dt + k] for k in range(-dt, dt + 1)]
                x = np.stack(cols, axis=-1)                       # (n, |nb|, T', 2dt+1)
                x = x.transpose(0, 2, 1, 3).reshape(-1, len(nb) * (2 * dt + 1))
                return x

            Xtr, Xva, Xte = _X(tr), _X(va), _X(te)
            for j in range(w):
                margin = min(j + 1, w - j)
                c = members[d0 + j]

                def _y(ix):
                    return env[b][ix][:, c, dt:T - dt].reshape(-1)

                r2, _ = _fit_eval(Xtr, _y(tr), Xva, _y(va), Xte, _y(te))
                per_margin.setdefault(margin, []).append(r2)
        out[name] = {str(m): round(float(np.mean(v)), 5) for m, v in sorted(per_margin.items())}
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--band-root", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", default=os.environ.get("ROOT_DIR_BRAINTREEBANK", ""))
    p.add_argument("--n-clips", type=int, default=128)
    p.add_argument("--clip-frames", type=int, default=96)
    p.add_argument("--n-targets", type=int, default=12)
    p.add_argument("--n-space-blocks", type=int, default=8)
    p.add_argument("--mask-rows", type=int, default=64, help="mask draws for the geometry")
    p.add_argument("--stride", type=int, default=6)
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
    cfg0 = V3MaskConfig()
    print(f"M14 — masked-cell audit | {len(specs)} sessions | shipped cfg "
          f"space {cfg0.space_frac} time {cfg0.time_frac} w_s {cfg0.block_w_space} "
          f"w_t {cfg0.block_w_time} whole {cfg0.whole_shaft_frac}\n"
          f"     PREDICTION (the maskability rule, made BEFORE the run): a band is hidden "
          f"only at margin >= its STFT support.\n"
          f"     support: HGA {SUPPORT['hga']}  MID {SUPPORT['mid']}  SLOW {SUPPORT['slow']} "
          f"slots  =>  w_t: HGA 4, MID 8, SLOW 32 (absurd — SLOW leaves the masked loss)\n",
          flush=True)

    rows = []
    for spec in specs:
        sid, tid = spec.session_key
        bands, starts = _read_clips(spec, a.n_clips, a.clip_frames, a.seed)
        env = [b.mean(2) for b in bands]                       # per band (n_clips, N, T)
        T = env[0].shape[-1]
        tr, va, te = _split_clips(starts, a.clip_frames)
        if len(va) < 4 or len(te) < 4:
            print(f"[s{sid}t{tid}] SKIP — val/test too small after overlap pruning", flush=True)
            continue

        geom = build_l1_geometry(spec.setup.sidecar)
        n_c = env[0].shape[1]
        rng = np.random.default_rng(a.seed + sid * 100 + tid)
        targets = rng.choice(n_c, size=min(a.n_targets, n_c), replace=False)

        # ---- A: geometry. class_frac depends on NEITHER width, the time margins only on
        # w_t and the space margins only on w_s, so the cross product would be wasted work.
        geo = []
        for wt in W_TIME:
            for guard in (True, False):
                cfg = dataclasses.replace(cfg0, block_w_time=wt)
                geo.append(_geometry(geom, n_c, T, a.mask_rows, cfg, a.seed, guard))
        for ws in W_SPACE:
            cfg = dataclasses.replace(cfg0, block_w_space=ws)
            geo.append(_geometry(geom, n_c, T, a.mask_rows, cfg, a.seed, True))

        # ---- B: temporal copy-ability, per width ----
        tmp = {str(w): _temporal(env, tr, va, te, T, targets, w, G_TIME, a.stride)
               for w in W_TIME}
        # ---- C: spatial copy-ability, per width ----
        spa = {str(w): _spatial(env, geom, tr, va, te, T, w, G_SPACE, DT_SPACE,
                                a.n_space_blocks, rng) for w in W_SPACE}

        rows.append({"subject_id": sid, "trial_id": tid, "n_contacts": n_c,
                     "n_shafts": geom.n_shafts, "geometry": geo,
                     "temporal": tmp, "spatial": spa,
                     "parcels": [int(q) for q in spec.setup.parcel_id.cpu().numpy()]})
        bad = [g for g in geo if not g["cellcount_invariant_OK"]]
        print(f"[s{sid}t{tid}] N={n_c} S={geom.n_shafts} clips {len(tr)}/{len(va)}/{len(te)}"
              f"  cellcount invariant: {'OK' if not bad else '*** VIOLATED x%d ***' % len(bad)}",
              flush=True)

    if not rows:
        raise SystemExit("no usable sessions")

    def _pool(sel):
        """mean over sessions of a per-session dict-of-float."""
        keys = sorted({k for r in rows for k in sel(r)})
        return {k: round(float(np.mean([sel(r)[k] for r in rows if k in sel(r)])), 5)
                for k in keys}

    print("\n" + "=" * 86)
    print("PART A — WHAT THE SAMPLER ACTUALLY MAKES (pooled). The loss weights.")
    print("=" * 86)
    base = [g for r in rows for g in r["geometry"]
            if g["block_w_time"] == cfg0.block_w_time
            and g["block_w_space"] == cfg0.block_w_space and g["guardian"]]
    cf = {k: float(np.mean([g["class_frac"][k] for g in base])) for k in ("W", "S", "T")}
    print(f"  class W (whole-shaft, cross-shaft only)     : {cf['W']:.3f}")
    print(f"  class S (intra-spatial, no own frames)      : {cf['S']:.3f}")
    print(f"  class T (temporal, HAS own frames = M13b)   : {cf['T']:.3f}   "
          f"<- M13b's 0.988 applies to THIS SLICE ONLY")
    inv_ok = all(g["cellcount_invariant_OK"] for r in rows for g in r["geometry"])
    print(f"  [check] masked-cell count == objective's own denominator N*T-(N-D)*T_kept : "
          f"{'OK' if inv_ok else '*** VIOLATED — THE AUDIT IS WRONG ***'}")

    print("\n  GUARDIAN — what it buys, what it costs (pooled over sessions):")
    print(f"  {'w_t':>4} {'guard':>6} {'dead-frame frac':>16} {'pathless cells':>15} "
          f"{'realized run':>13}")
    for wt in W_TIME:
        for guard in (True, False):
            sel = [g for r in rows for g in r["geometry"]
                   if g["block_w_time"] == wt and g["block_w_space"] == cfg0.block_w_space
                   and g["guardian"] is guard]
            print(f"  {wt:>4} {str(guard):>6} {np.mean([g['dead_frame_frac'] for g in sel]):>16.6f} "
                  f"{np.mean([g['pathless_cell_frac'] for g in sel]):>15.6f} "
                  f"{np.mean([g['realized_run_mean'] for g in sel]):>13.2f}  (target {wt})")
    print("  -> KILL THE GUARDIAN iff pathless-cell frac (guard=False) is negligible AND")
    print("     the realized run recovers toward the target width. It is static-safe either")
    print("     way: the snap `time_rank < t_mask` is untouched.")

    print("\n" + "=" * 86)
    print("PART B — TEMPORAL copy-ability. Predict the masked cell from the contact's OWN")
    print("         visible frames of the SAME band. THE LEAK. Must die at margin >= support.")
    print("=" * 86)
    for name in BAND_NAMES:
        print(f"\n  {name.upper()}  (STFT support {SUPPORT[name]} slots -> needs margin "
              f">= {SUPPORT[name]}, i.e. w_t >= {2 * SUPPORT[name]})")
        for w in W_TIME:
            pm = _pool(lambda r, w=w, n=name: r["temporal"][str(w)][n])
            marg = "  ".join(f"m{k} {v:+.3f}" for k, v in pm.items())
            # loss-weighted over the margin mix this width actually produces
            geo_w = [g for r in rows for g in r["geometry"]
                     if g["block_w_time"] == w and g["block_w_space"] == cfg0.block_w_space
                     and g["guardian"]]
            wts: dict[str, float] = {}
            for g in geo_w:
                for k, v in g["time_margin_frac"].items():
                    wts[k] = wts.get(k, 0.0) + v / len(geo_w)
            lw = sum(pm.get(k, 0.0) * v for k, v in wts.items() if k in pm)
            tot_w = sum(v for k, v in wts.items() if k in pm)
            lw = lw / tot_w if tot_w > 0 else float("nan")
            flag = "LEAKY" if lw > 0.15 else "ok"
            print(f"    w_t={w:>2}  {marg}   |  LOSS-WEIGHTED {lw:+.4f}  [{flag}]")

    print("\n" + "=" * 86)
    print("PART C — SPATIAL copy-ability. Predict the masked contact from its OWN-SHAFT")
    print("         visible neighbours. NOT A LEAK — this is PHYSICS, and it IS the task.")
    print("         The question is whether the INTRA tier earns anything WHOLE-SHAFT doesn't.")
    print("=" * 86)
    for name in BAND_NAMES:
        print(f"\n  {name.upper()}  (M2 measured along-shaft ACF: 0.34 / 0.21 / 0.18 at "
              f"depth-lag 1/2/3 — HGA)")
        for w in W_SPACE:
            pm = _pool(lambda r, w=w, n=name: r["spatial"][str(w)].get(n, {}))
            if not pm:
                continue
            marg = "  ".join(f"m{k} {v:+.3f}" for k, v in pm.items())
            geo_w = [g for r in rows for g in r["geometry"]
                     if g["block_w_space"] == w and g["block_w_time"] == cfg0.block_w_time
                     and g["guardian"]]
            wts: dict[str, float] = {}
            for g in geo_w:
                for k, v in g["space_margin_frac"].items():
                    wts[k] = wts.get(k, 0.0) + v / max(len(geo_w), 1)
            tot_w = sum(v for k, v in wts.items() if k in pm)
            lw = (sum(pm[k] * v for k, v in wts.items() if k in pm) / tot_w
                  if tot_w > 0 else float("nan"))
            print(f"    w_s={w:>2}  {marg}   |  LOSS-WEIGHTED (class S) {lw:+.4f}")
    print("\n  READ THIS AS: how much of the INTRA-spatial task is local interpolation the")
    print("  WHOLE-SHAFT tier already subsumes. If own-shaft R2 dies by margin 2, a width-4")
    print("  intra block's deep cells ARE the cross-shaft task -> move budget to whole-shaft.")

    print("\n" + "=" * 86)
    print("PART E — B3 PARCEL COVERAGE. How many parcels survive a '>= k distinct subjects'")
    print("         rule? A 1-subject parcel's embedding is fit on one subject = a subject code.")
    print("=" * 86)
    subj_of_parcel: dict[int, set] = {}
    for r in rows:
        for q in set(r["parcels"]):
            subj_of_parcel.setdefault(int(q), set()).add(r["subject_id"])
    n_subj = len({r["subject_id"] for r in rows})
    cov = {k: sum(1 for v in subj_of_parcel.values() if len(v) >= k)
           for k in range(1, n_subj + 1)}
    print(f"  {n_subj} distinct subjects, {len(subj_of_parcel)} distinct parcels in the union")
    for k, c in cov.items():
        print(f"    parcels attested in >= {k} subjects : {c:>3}")
    print(f"  [check] 7-way intersection empty (known): {cov.get(n_subj, 0) == 0}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"w_time": list(W_TIME), "w_space": list(W_SPACE),
                       "g_time": G_TIME, "g_space": G_SPACE, "dt_space": DT_SPACE,
                       "support": SUPPORT, "shipped_cfg": dataclasses.asdict(cfg0),
                       "per_session": rows, "parcel_coverage": cov}, fh, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
