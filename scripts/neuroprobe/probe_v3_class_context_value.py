"""M15 — is class S (64% of the masked loss) a SHORTCUT, and is class W worth more of it?

THE QUESTION, AND THE CLAIM I HAD TO RETRACT TO GET TO IT

M14 found the class mix is a DERIVED RESIDUAL: with space .60 / time .50 / whole .10 the
sampler makes W .108 / S .641 / T .250, and S is simply `space_frac - whole_eff` — whatever
is left after the whole-shaft tier takes its cut. Nobody chose 64% intra-spatial.

I then argued "so raise whole_shaft_frac, W is the cross-SUBJECT task." That was an
over-claim and Ben caught it. Cross-SHAFT is not cross-SUBJECT: predicting shaft F from
shaft G still happens inside ONE montage, and the model is free to memorise "in subject 2,
F tracks G". Montage-invariance is bought by the parcel embedding being the only channel
position enters through — an ARCHITECTURE property, not a masking one. And there is a
counter-argument in the other direction: if W is barely predictable at all, a loss dominated
by W trains the encoder to emit the conditional mean — the smear, which is the M13 disease
wearing a different hat.

So the question is NOT "is W more cross-subject". It is:

    Does class S let the model SKIP LEARNING ANYTHING, by copying a contact 3.6 mm away?

and the trap (M14, and it is written into masking.py) is that a SPATIAL correlation is
PHYSICS: predicting a hidden sensor from visible sensors IS the cross-sensor task. So what
separates "shortcut" from "the task"? This:

  * an OWN-SHAFT interpolation is a FIXED DEPTH-LAG KERNEL. Same weights on every shaft of
    every subject. It does not need to know WHERE IN THE BRAIN IT IS, so it trains nothing
    that routes through the parcel embedding, so it cannot transfer to a new montage.
  * a CROSS-SHAFT prediction is PARCEL-CONDITIONAL. Whether shaft F predicts shaft G depends
    on which regions they sit in. It MUST route through the parcel embedding — the exact
    pathway cross-subject transfer rides on.

That is a measurable difference, and it is what Part C measures. Model-free, CPU.

WHAT IS MEASURED — three NESTED models for the same held-out target contact c on shaft s

    cm       common mode only: the MEAN of every contact NOT on shaft s, at t-1, t, t+1.
    cross    every contact NOT on shaft s, individually, at t-1, t, t+1.
             == EXACTLY the context a class-W cell has. Nothing else survives a whole-shaft
             drop. So R2(cross) IS class W's achievable ceiling.
    cross+own(m)  + the own-shaft contacts that survive a block burying c at MARGIN m, i.e.
             depths d+-m and d+-(m+1), at t-1, t, t+1.
             == EXACTLY the context a class-S cell at margin m has (objective.py:233 — a
             spatially masked contact has NO frames of its own, so own-SHAFT is all it adds).

  => dR2(cross | cm)   = the genuine cross-shaft structure, ABOVE the global gain state.
                         This is what class W can actually teach. If it is ~0, W is a
                         common-mode-fitting trap and whole_shaft_frac MUST NOT go up.
  => dR2(own | cross)  = THE CRUTCH. Exactly what class S gets that class W does not.
                         If it is ~0, S already IS the cross-shaft task, in bulk (64% of the
                         loss), and whole_shaft_frac is a nothing-knob — leave it alone.

MARGIN COVERAGE — the silent truncation that M15-v1 AND M14 both had (fixed here)

A width-4 block only ever realizes margins 1 and 2, which is **61.5%** of real class-S cells.
The other 38.5% (margins >=3, plus the 4.2% of shafts that saturate to ZERO visible own-shaft
contacts) were dropped and the average silently renormalized over what was left — i.e. over
the EASY cells, since own-shaft help decays with margin. So both probes OVERSTATED the crutch.

Here the margin is swept DIRECTLY, 1..M_MAX, and the A/B is weighted by the REAL class-S
margin distribution M14 measured from the shipped sampler (S_MARGIN_FRAC below). Margin -1
(no visible own-shaft contact at all) is folded in EXACTLY, not estimated: such a cell's own
model IS its cross model, so its dR2(own|cross) is 0 by construction. The dropped mass is
PRINTED, and it sits at the DEEP margins where own-shaft help is weakest, so what remains is
an UPPER bound on the crutch. State it that way.

  Part C: IS THE OWN-SHAFT KERNEL POSITION-FREE?
    Same design matrix (the flanking contacts at margin-relative depth offsets
    -(m+1), -m, +m, +(m+1), so the column semantics are IDENTICAL on every shaft of every
    subject), fit three ways and all three SCORED ON THE SAME POOLED TEST ROWS against the
    SAME pooled-train-mean baseline:
       global      ONE ridge, pooled over every shaft of every session.
       per-parcel  one ridge per DK parcel of the target contact.
       per-contact one ridge per (session, contact).   <- this is M14 Part C, exactly.
    global ~= per-contact  => the own-shaft map is a UNIVERSAL FILTER. 64% of the masked loss
                              trains something that needs no position and cannot transfer.
    global <<  per-contact => the own-shaft map is position-specific after all, and the
                              "shortcut" framing is wrong. Say so.
    (Targets are z-scored per contact, so pooling R2 across contacts is well-posed: every
    target has unit variance and the pooled baseline is not dominated by one loud contact.
    The pooled score is printed ALONGSIDE the per-contact-mean score so the reader can see
    that the two agree and the pooling is not doing any work.)

RIGOR (feedback-build-the-invariant-into-the-probe)
  * NO CAR of any kind. M7 died by CAR'ing away the very inter-shaft coupling it measured.
  * NESTING INVARIANT, asserted and PRINTED WITH MAGNITUDES: cm subset cross subset cross+own
    are strictly nested feature sets, so TRAIN R2 must be non-decreasing. Violations are
    reported with their size, because lambda is picked per-model on VAL and a larger model may
    select a heavier lambda and score LOWER on train — that is selection noise (tiny), not an
    estimator bug (large). The number decides, not the adjective.
  * SELF-LEAK ASSERT: the target contact is in neither predictor set, and no cross-shaft
    predictor sits on the target's own shaft. Checked per fit, printed once.
  * CROSS-VALIDATION AGAINST M14, PER MARGIN (not through a weighted average that would hide a
    disagreement): Part C's per-contact own-only fit must reproduce M14's own-only R2 at
    margins 1 and 2. Printed as a delta per band per margin.
  * held out BY CLIP; test clips overlapping a train clip's span are dropped; ridge lambda
    swept on VAL only, refit on train+val, scored once on TEST.

DeltaAI/Delta login node, CPU, no checkpoint:

  ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
  .venv/bin/python -m scripts.neuroprobe.probe_v3_class_context_value \
      --band-root $ROOT \
      --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
      --bt-root /projects/bhqk/htang13/braintreebank \
      --out /projects/bhqk/htang13/probe_out_v3/field_stats/class_context_value.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
from scripts.neuroprobe.probe_v3_field_stats import BAND_DIRS, BAND_NAMES, V3_SESSIONS, WINSOR
from scripts.neuroprobe.probe_v3_global_structure import (
    LAMBDAS,
    _r2,
    _read_clips,
    _split_clips,
    _standardize,
)

G_SPACE = 2   # flanking VISIBLE own-shaft contacts each side of the block (M14 Part C parity)
DT = 1        # every predictor is read at t-1, t, t+1 (M14 parity; deliberately generous)

# The REAL spatial-margin distribution of class-S masked cells under the SHIPPED sampler
# (space .60 / time .50 / whole .10, block_w_space=4), measured by M14 —
# probe_v3_masked_cell_audit.py, `space_margin_frac` in
# /projects/bhqk/htang13/probe_out_v3/field_stats/masked_cell_audit.json.
# -1 == the shaft saturated: the contact has NO visible own-shaft mate at any depth.
S_MARGIN_FRAC: dict[int, float] = {
    -1: 0.0419, 1: 0.3428, 2: 0.2719, 3: 0.1498, 4: 0.0909, 5: 0.0474,
    6: 0.0277, 7: 0.0153, 8: 0.0070, 9: 0.0033, 10: 0.0014, 11: 0.0010,
    12: 0.0006, 13: 0.0002,
}

# M14 Part C, own-shaft-ONLY R2 per margin (w_s=4, 13 montages). The per-margin cross-check.
M14_OWN_ONLY: dict[str, dict[int, float]] = {
    "slow": {1: 0.576, 2: 0.397},
    "mid": {1: 0.332, 2: 0.171},
    "hga": {1: 0.221, 2: 0.155},
}


def _solve_path(Xtr, ytr, Xva, yva, Xte, yte=None):
    """Ridge with the lambda swept on VAL and refit on train+val — IDENTICAL math to
    ``_fit_eval``, but the Gram matrix is built ONCE instead of once per lambda.

    ``_fit_eval`` calls ``_ridge_fit`` inside the lambda loop, and ``_ridge_fit`` starts
    with ``G = X.T @ X``. With the 528 cross-shaft columns that Gram is ~2.2 GFLOP and
    the 15-lambda sweep pays it 15 times over; the solve itself is p^3/3 ~ 5e7, i.e.
    noise. Hoisting it is an ~11x speedup and changes no number. (M15's first run was
    silently SIGKILLed by the Delta login-node reaper for exactly this waste.)

    -> (test R2, train R2 at MIN lambda) if yte is given, else the test PREDICTIONS.

    The train R2 returned for the nesting check is at the FIXED SMALLEST lambda, NOT the
    VAL-selected one. That matters: the nesting invariant (a strictly larger feature set
    cannot lower TRAIN R2) is a linear-algebra fact only AT A FIXED lambda. Scored at the
    per-model VAL-selected lambda it is NOT monotone — a larger model that picks a heavier
    lambda legitimately fits train worse — so a check on the selected-lambda train R2 cries
    wolf (verified: 0.01-0.05 non-monotonicity arises naturally). At the fixed min lambda,
    with n_train >> n_features here, any violation is a genuine estimator bug.

    Predictions are what Part C needs: its three models must be scored on ONE pooled test
    set against ONE baseline, and a mean of per-group R2s is not a comparable number."""
    Xtr, Xva, Xte = _standardize(Xtr, Xva, Xte)
    Xtr = np.column_stack([Xtr, np.ones(len(Xtr), dtype=Xtr.dtype)])
    Xva = np.column_stack([Xva, np.ones(len(Xva), dtype=Xva.dtype)])
    Xte = np.column_stack([Xte, np.ones(len(Xte), dtype=Xte.dtype)])
    mu_tr = float(ytr.mean())

    Xtr64, Xva64 = Xtr.astype(np.float64), Xva.astype(np.float64)
    G, b = Xtr64.T @ Xtr64, Xtr64.T @ ytr.astype(np.float64)   # ONCE
    eye = np.eye(G.shape[0])
    best_lam, best = LAMBDAS[0], -np.inf
    for lam in LAMBDAS:
        w = np.linalg.solve(G + lam * eye, b)
        s = _r2(yva, Xva64 @ w, mu_tr)
        if s > best:
            best, best_lam = s, lam

    Gf = G + Xva64.T @ Xva64                                    # reuse the train Gram
    w = np.linalg.solve(Gf + best_lam * eye, b + Xva64.T @ yva.astype(np.float64))
    if yte is None:
        return Xte.astype(np.float64) @ w
    w_min = np.linalg.solve(G + LAMBDAS[0] * eye, b)            # near-OLS, for the nesting check
    return _r2(yte, Xte.astype(np.float64) @ w, mu_tr), _r2(ytr, Xtr64 @ w_min, mu_tr)


def _zscore(env, tr):
    """z-score each (band, contact) on TRAIN clips only. Required for Part C: a GLOBAL kernel
    must not be penalised for per-contact scale differences it was never asked to model, and
    pooling R2 across contacts is only well-posed if every target has the same variance."""
    out = []
    for e in env:
        mu = e[tr].mean(axis=(0, 2), keepdims=True)
        sd = e[tr].std(axis=(0, 2), keepdims=True) + 1e-6
        out.append((e - mu) / sd)
    return out


def _cols(e, ix, contacts, T):
    """(clips, |contacts|, T) -> rows = clips x times, cols = contact x {t-1, t, t+1}."""
    x = np.stack([e[ix][:, contacts, DT + k:T - DT + k] for k in range(-DT, DT + 1)], axis=-1)
    return x.transpose(0, 2, 1, 3).reshape(-1, len(contacts) * (2 * DT + 1)).astype(np.float32)


def _target(e, ix, c, T):
    return e[ix][:, c, DT:T - DT].reshape(-1).astype(np.float32)


def _flank(members, d, m):
    """Own-shaft contacts that survive a block burying depth `d` at margin `m`: the two
    nearest visible mates on each side, at depth offsets -(m+1), -m, +m, +(m+1). Column
    order is MARGIN-RELATIVE and therefore identical on every shaft of every subject —
    that is what makes the Part C global fit a well-posed question. None if the shaft is
    too short on either side (a one-sided flank would break the column semantics)."""
    lo = [d - m - k for k in range(G_SPACE - 1, -1, -1)]        # d-(m+1), d-m
    hi = [d + m + k for k in range(G_SPACE)]                    # d+m,     d+(m+1)
    if lo[0] < 0 or hi[-1] >= len(members):
        return None
    return members[np.array(lo + hi)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--band-root", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", default=os.environ.get("ROOT_DIR_BRAINTREEBANK", ""))
    p.add_argument("--n-clips", type=int, default=128)
    p.add_argument("--clip-frames", type=int, default=96)
    p.add_argument("--n-targets", type=int, default=20, help="target contacts sampled per session")
    p.add_argument("--m-max", type=int, default=4,
                   help="deepest spatial margin measured (needs shaft length > 2*(m_max+G_SPACE-1)); "
                        "4 covers 90%% of class-S cells, 5 covers 94%% but starves short shafts")
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--out")
    a = p.parse_args()
    M_MAX = a.m_max

    specs = load_v3_sessions(
        sessions=V3_SESSIONS,
        band_cache_dirs=[os.path.join(a.band_root, b) for b in BAND_DIRS],
        span_dir=a.span_dir,
        parcel_fn=make_bt_parcel_fn(a.bt_root),
        lof_report_path=None,
        winsor=WINSOR,
    )

    margins = list(range(1, M_MAX + 1))
    tot = sum(S_MARGIN_FRAC.values())
    cov = (S_MARGIN_FRAC[-1] + sum(S_MARGIN_FRAC[m] for m in margins)) / tot
    mw = {m: S_MARGIN_FRAC[m] / tot / cov for m in margins}     # renormalized over covered
    mw_none = S_MARGIN_FRAC[-1] / tot / cov                     # margin -1: own == cross, exactly
    mw_own = {m: v / (1.0 - mw_none) for m, v in mw.items()}    # weights among cells that HAVE a flank

    print(f"M15 — class context value | {len(specs)} sessions | margins 1..{M_MAX} "
          f"| G_SPACE={G_SPACE} DT={DT} | NO CAR\n"
          f"  cm     = mean of all OTHER-shaft contacts        (the global gain state)\n"
          f"  cross  = all OTHER-shaft contacts, individually  == CLASS W's ENTIRE CONTEXT\n"
          f"  +own(m)= + the own-shaft mates that survive a block burying c at margin m\n"
          f"                                                   == CLASS S's ENTIRE CONTEXT\n"
          f"  dR2(cross|cm)  = what W can teach beyond the common mode\n"
          f"  dR2(own|cross) = THE CRUTCH: exactly what S gets that W does not\n")
    print(f"  MARGIN COVERAGE (real class-S distribution, M14): margins 1..{M_MAX} + the "
          f"{S_MARGIN_FRAC[-1] / tot:.3f} at margin -1 (no visible own-shaft mate at all, where\n"
          f"  own == cross BY CONSTRUCTION) = {cov:.3f} of class-S cells. "
          f"DROPPED: {1 - cov:.3f} at margins >{M_MAX}.\n"
          f"  The dropped mass is where own-shaft help is WEAKEST, so the loss-weighted crutch "
          f"below is an UPPER BOUND.\n", flush=True)

    rows, targets_C = [], []
    envs: dict[tuple[int, int], dict] = {}
    leak_ok, n_fits = True, 0
    nest_viol: list[float] = []

    for spec in specs:
        sid, tid = spec.session_key
        bands, starts = _read_clips(spec, a.n_clips, a.clip_frames, a.seed)
        tr, va, te = _split_clips(starts, a.clip_frames)
        if len(va) < 4 or len(te) < 4:
            print(f"[s{sid}t{tid}] SKIP — val/test too small after overlap pruning", flush=True)
            continue
        env = _zscore([b.mean(2) for b in bands], tr)          # per band (n_clips, N, T)
        T = env[0].shape[-1]
        geom = build_l1_geometry(spec.setup.sidecar)
        shaft_of = geom.shaft_of_contact.numpy()
        parcel = spec.setup.parcel_id.cpu().numpy()
        rng = np.random.default_rng(a.seed + sid * 100 + tid)

        # a target must have room for the DEEPEST margin's flank on BOTH sides, so every
        # margin 1..M_MAX is measured on the SAME contacts — otherwise the margin curve
        # would confound depth-in-shaft with margin.
        cand = []
        for si in range(geom.n_shafts):
            members = np.where(shaft_of == si)[0]
            need = M_MAX + G_SPACE - 1
            for d in range(need, len(members) - need):
                cand.append((si, members, d))
        if not cand:
            print(f"[s{sid}t{tid}] SKIP — no shaft long enough for a margin-{M_MAX} flank",
                  flush=True)
            continue
        pick = rng.choice(len(cand), size=min(a.n_targets, len(cand)), replace=False)
        envs[(sid, tid)] = {"env": env, "tr": tr, "va": va, "te": te, "T": T}

        rec: dict[str, dict[str, list[float]]] = {n: {} for n in BAND_NAMES}
        for bi in pick:
            si, members, d = cand[bi]
            c = int(members[d])
            other = np.where(shaft_of != si)[0]                # class W's whole world
            if len(other) < 4:
                continue
            flanks = {m: _flank(members, d, m) for m in margins}

            for b, name in enumerate(BAND_NAMES):
                e = env[b]
                # cross / cm depend ONLY on the target's shaft, not on the block. Build once.
                Xc = {k: _cols(e, ix, other, T) for k, ix in (("tr", tr), ("va", va), ("te", te))}
                Xm = {k: v.reshape(len(v), len(other), 2 * DT + 1).mean(1) for k, v in Xc.items()}
                y = {k: _target(e, ix, c, T) for k, ix in (("tr", tr), ("va", va), ("te", te))}

                r_cm, tr_cm = _solve_path(Xm["tr"], y["tr"], Xm["va"], y["va"], Xm["te"], y["te"])
                r_cr, tr_cr = _solve_path(Xc["tr"], y["tr"], Xc["va"], y["va"], Xc["te"], y["te"])
                rec[name].setdefault("cm", []).append(r_cm)
                rec[name].setdefault("cross", []).append(r_cr)
                n_fits += 1
                if tr_cm > tr_cr + 1e-9:
                    nest_viol.append(tr_cm - tr_cr)

                for m in margins:
                    fl = flanks[m]
                    if fl is None:
                        continue
                    leak_ok &= (c not in other) and (c not in fl) \
                        and bool((shaft_of[other] != si).all()) \
                        and bool((shaft_of[fl] == si).all())
                    Xf = {k: _cols(e, ix, fl, T) for k, ix in (("tr", tr), ("va", va), ("te", te))}
                    Xo = {k: np.concatenate([Xc[k], Xf[k]], axis=1) for k in Xc}
                    r_ow, tr_ow = _solve_path(Xo["tr"], y["tr"], Xo["va"], y["va"],
                                              Xo["te"], y["te"])
                    rec[name].setdefault(f"m{m}_own", []).append(r_ow)
                    if tr_cr > tr_ow + 1e-9:
                        nest_viol.append(tr_cr - tr_ow)

            # Part C reuses this target's flank recipe across all bands, so record it ONCE.
            flank_lists = {m: (None if fl is None else fl.tolist()) for m, fl in flanks.items()}
            targets_C.append({"sid": sid, "tid": tid, "contact": c, "parcel": int(parcel[c]),
                              "flanks": flank_lists})

        rows.append({"subject_id": sid, "trial_id": tid, "n_contacts": int(len(shaft_of)),
                     "n_shafts": int(geom.n_shafts),
                     "r2": {n: {k: float(np.mean(v)) for k, v in d.items()}
                            for n, d in rec.items()}})
        print(f"[s{sid}t{tid}] N={len(shaft_of)} S={geom.n_shafts} "
              f"clips {len(tr)}/{len(va)}/{len(te)} targets {len(pick)}", flush=True)

    if not rows:
        raise SystemExit("no usable sessions")

    print(f"\n  [check] no self-leak (target in neither predictor set; cross excludes its own "
          f"shaft; flank is ON its shaft) : {'OK' if leak_ok else '*** VIOLATED ***'}")
    if not nest_viol:
        print(f"  [check] nesting  cm <= cross <= cross+own  on TRAIN R2 (at fixed min lambda) "
              f": OK ({n_fits} targets)")
    else:
        v = np.array(nest_viol)
        print(f"  [check] nesting on TRAIN R2 at FIXED MIN LAMBDA : {len(v)} violations, "
              f"max {v.max():.5f}, median {np.median(v):.5f}")
        print(f"          At a FIXED lambda a strictly larger feature set CANNOT lower train R2 "
              f"(n_train >> n_feat here), so any\n          violation is a genuine estimator bug "
              f"— NOT lambda-selection noise. *** INVESTIGATE ***")

    def _mean(name, key):
        vals = [r["r2"][name][key] for r in rows if key in r["r2"][name]]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n" + "=" * 92)
    print("A/B — CLASS W's CEILING, AND CLASS S's CRUTCH")
    print("=" * 92)
    print(f"  {'band':>5} {'R2 cm':>8} {'R2 cross':>9} | " +
          " ".join(f"{'own m' + str(m):>9}" for m in margins) +
          f" | {'own (lw)':>9} {'dR2(cross|cm)':>14} {'dR2(own|cross)':>15}")
    summary = {}
    for name in BAND_NAMES:
        cm, cr = _mean(name, "cm"), _mean(name, "cross")
        own_m = {m: _mean(name, f"m{m}_own") for m in margins}
        # margin -1 contributes EXACTLY cr (no own-shaft mate exists -> own model IS cross)
        ow = mw_none * cr + sum(mw[m] * own_m[m] for m in margins)
        summary[name] = {"cm": cm, "cross": cr, "own_per_margin": own_m, "own_lw": ow,
                         "dR2_cross_given_cm": cr - cm, "dR2_own_given_cross": ow - cr}
        print(f"  {name:>5} {cm:>8.4f} {cr:>9.4f} | " +
              " ".join(f"{own_m[m]:>9.4f}" for m in margins) +
              f" | {ow:>9.4f} {cr - cm:>+14.4f} {ow - cr:>+15.4f}")
    print("\n  READ:")
    print("   dR2(cross|cm) ~ 0   -> class W is a COMMON-MODE-FITTING TRAP. Do NOT raise")
    print("                          whole_shaft_frac: W would train the encoder to emit the")
    print("                          global gain and nothing else (the M13 smear, one level up).")
    print("   dR2(own|cross) ~ 0  -> class S ALREADY IS the cross-shaft task, at 64% of the")
    print("                          loss. whole_shaft_frac is a NOTHING-KNOB; leave the fracs")
    print("                          alone — the 64% was never the problem.")
    print("   both large          -> S HAS A CRUTCH that W does not, AND W teaches real")
    print("                          structure. Only then is whole_shaft_frac a real lever —")
    print("                          and Part C says whether that crutch is position-free.")

    # ---------------- Part C: is the own-shaft kernel position-free? ----------------
    print("\n" + "=" * 92)
    print("C — IS THE OWN-SHAFT KERNEL POSITION-FREE?  Own-shaft flank ONLY (no cross-shaft),")
    print("    columns = margin-relative depth offsets, so they mean the same thing on every")
    print("    shaft of every subject. Fit GLOBAL / PER-PARCEL / PER-CONTACT, all scored on the")
    print("    SAME pooled test rows against the SAME pooled-train-mean baseline.")
    print("=" * 92)
    partc: dict[str, dict] = {}
    for b, name in enumerate(BAND_NAMES):
        per_m: dict[int, dict] = {}
        for m in margins:
            items = []
            for t in targets_C:
                fl = t["flanks"][m]
                if fl is None:
                    continue
                E = envs[(t["sid"], t["tid"])]
                e, T = E["env"][b], E["T"]
                fl = np.array(fl)
                X = {k: _cols(e, E[k], fl, T) for k in ("tr", "va", "te")}
                y = {k: _target(e, E[k], t["contact"], T) for k in ("tr", "va", "te")}
                items.append((X, y, t))
            if not items:
                continue
            cat = lambda d, k: np.concatenate([x[k] for x in d])  # noqa: E731
            Xtr, Xva, Xte = (cat([i[0] for i in items], k) for k in ("tr", "va", "te"))
            ytr, yva, yte = (cat([i[1] for i in items], k) for k in ("tr", "va", "te"))
            mu = float(ytr.mean())          # ONE baseline for all three models

            yh_g = _solve_path(Xtr, ytr, Xva, yva, Xte)

            spans, pos = [], 0
            for X, y, t in items:
                n = len(X["te"])
                spans.append((X, y, t, pos, pos + n))
                pos += n

            def _grouped(key, spans=spans, yte=yte, mu=mu):
                """Fit per group, predict that group's test rows, scatter into the SAME pooled
                test vector, score ONCE. A mean of per-group R2s would not be comparable to the
                global fit's single pooled R2 — that mismatch is what killed M12 v1."""
                yh = np.zeros_like(yte)
                groups: dict = {}
                for X, y, t, s0, s1 in spans:
                    groups.setdefault(key(t), []).append((X, y, s0, s1))
                per_group = []
                for g in groups.values():
                    gtr, gva, gte = (np.concatenate([X[k] for X, _, _, _ in g])
                                     for k in ("tr", "va", "te"))
                    gytr, gyva = (np.concatenate([y[k] for _, y, _, _ in g])
                                  for k in ("tr", "va"))
                    pred = _solve_path(gtr, gytr, gva, gyva, gte)
                    gyte = np.concatenate([y["te"] for _, y, _, _ in g])
                    per_group.append(_r2(gyte, pred, float(gytr.mean())))
                    off = 0
                    for _, _, s0, s1 in g:
                        n = s1 - s0
                        yh[s0:s1] = pred[off:off + n]
                        off += n
                return _r2(yte, yh, mu), float(np.mean(per_group)), len(groups)

            p_r2, _, n_par = _grouped(lambda t: t["parcel"])
            c_r2, c_mean, n_con = _grouped(lambda t: (t["sid"], t["contact"]))
            per_m[m] = {"global": _r2(yte, yh_g, mu), "per_parcel": p_r2, "per_contact": c_r2,
                        "per_contact_own_baseline_mean": c_mean,
                        "n_parcels": n_par, "n_contacts": n_con, "n_test_rows": int(len(yte))}

        gl = sum(per_m[m]["global"] * mw_own[m] for m in per_m)
        pa = sum(per_m[m]["per_parcel"] * mw_own[m] for m in per_m)
        co = sum(per_m[m]["per_contact"] * mw_own[m] for m in per_m)
        ratio = gl / co if co > 1e-6 else float("nan")
        print(f"\n  {name}")
        print(f"    {'margin':>7} {'global':>9} {'per-parcel':>11} {'per-contact':>12} "
              f"{'(pc, own base)':>15} {'M14':>7} {'delta':>8}")
        for m in per_m:
            ref = M14_OWN_ONLY[name].get(m)
            d = "" if ref is None else f"{per_m[m]['per_contact'] - ref:>+8.3f}"
            print(f"    {m:>7} {per_m[m]['global']:>+9.4f} {per_m[m]['per_parcel']:>+11.4f} "
                  f"{per_m[m]['per_contact']:>+12.4f} "
                  f"{per_m[m]['per_contact_own_baseline_mean']:>+15.4f} "
                  f"{'—' if ref is None else f'{ref:>7.3f}'} {d}")
        print(f"    loss-weighted (over cells that HAVE a flank): global {gl:+.4f}  "
              f"per-parcel {pa:+.4f}  per-contact {co:+.4f}  |  global/per-contact = {ratio:.2f}")
        partc[name] = {"per_margin": {str(k): v for k, v in per_m.items()},
                       "loss_weighted": {"global": gl, "per_parcel": pa, "per_contact": co,
                                         "global_over_per_contact": ratio}}

    print("\n  [check] the M14 column is M14 Part C's own-only R2 at the same margin. The two")
    print("          probes must agree. M15 z-scores its targets and M14 did not, so expect")
    print("          CLOSE, not identical. A LARGE delta means one of them is wrong — say which.")
    print("  [check] 'pc, own base' scores each contact against ITS OWN train mean and averages;")
    print("          'per-contact' pools. If the two differ, the pooled baseline is doing work")
    print("          it should not and the global-vs-per-contact comparison is not clean.")
    print("\n  READ: global ~= per-contact -> the own-shaft map is a UNIVERSAL DEPTH-LAG FILTER.")
    print("        It needs no parcel, no position, no subject. 64% of the masked loss then")
    print("        trains something that CANNOT transfer to a new montage — and THAT, not")
    print("        'W is the cross-subject task', is the argument for moving budget to W.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"margins": margins, "g_space": G_SPACE, "dt": DT,
                       "s_margin_frac": {str(k): v for k, v in S_MARGIN_FRAC.items()},
                       "margin_coverage": cov, "margin_weights": {str(k): v for k, v in mw.items()},
                       "margin_weight_none": mw_none,
                       "summary": summary, "kernel_universality": partc, "per_session": rows,
                       "checks": {"self_leak_ok": bool(leak_ok),
                                  "nesting_violations": len(nest_viol),
                                  "nesting_violation_max": float(max(nest_viol))
                                  if nest_viol else 0.0,
                                  "n_targets": int(n_fits)}}, fh, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
