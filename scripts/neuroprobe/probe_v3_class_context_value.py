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
    cross+own  + the own-shaft contacts that SURVIVE a width-w_s block around c, at t-1,t,t+1.
             == EXACTLY the context a class-S cell has (objective.py:233 — a spatially
             masked contact has NO frames of its own, so own-SHAFT is all it adds).

  => dR2(cross | cm)   = the genuine cross-shaft structure, ABOVE the global gain state.
                         This is what class W can actually teach. If it is ~0, W is a
                         common-mode-fitting trap and whole_shaft_frac MUST NOT go up.
  => dR2(own | cross)  = THE CRUTCH. Exactly what class S gets that class W does not.
                         If it is ~0, S already IS the cross-shaft task, in bulk (64% of the
                         loss), and whole_shaft_frac is a nothing-knob — leave it alone.

  Part C: IS THE OWN-SHAFT KERNEL POSITION-FREE?
    Same design matrix (the block's flanking contacts at block-relative depth offsets, so the
    column semantics are IDENTICAL on every shaft of every subject), fit three ways and all
    three SCORED ON THE SAME POOLED TEST ROWS against the SAME pooled-train-mean baseline:
       global      ONE ridge, pooled over every shaft of every session.
       per-parcel  one ridge per DK parcel of the target contact.
       per-contact one ridge per (session, contact).   <- this is M14 Part C, exactly.
    global ~= per-contact  => the own-shaft map is a UNIVERSAL FILTER. 64% of the masked loss
                              trains something that needs no position and cannot transfer.
    global <<  per-contact => the own-shaft map is position-specific after all, and the
                              "shortcut" framing is wrong. Say so.
    (Targets are z-scored per contact, so pooling R2 across contacts is well-posed: every
    target has unit variance and the pooled baseline is not dominated by one loud contact.)

RIGOR (feedback-build-the-invariant-into-the-probe)
  * NO CAR of any kind. M7 died by CAR'ing away the very inter-shaft coupling it measured.
  * NESTING INVARIANT, asserted and PRINTED: cm subset cross subset cross+own are strictly
    nested feature sets, so TRAIN R2 must be non-decreasing across the three. A violation is
    an estimator bug, not a finding. (TEST R2 need not be — that is what a held-out set is
    for — so the assert is on TRAIN.)
  * SELF-LEAK ASSERT: the target contact is in neither predictor set, and no cross-shaft
    predictor sits on the target's own shaft. Checked per fit, printed once.
  * CROSS-VALIDATION AGAINST M14: Part C's per-contact fit must reproduce M14 Part C at
    w_s=4 (loss-weighted slow .497 / mid .261 / hga .192). Two probes, one number.
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
from speech_decoding.models.v14_converged_v3.masking import V3MaskConfig
from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
from scripts.neuroprobe.probe_v3_field_stats import BAND_DIRS, BAND_NAMES, V3_SESSIONS, WINSOR
from scripts.neuroprobe.probe_v3_global_structure import (
    LAMBDAS,
    _fit_eval,
    _r2,
    _read_clips,
    _ridge_fit,
    _split_clips,
    _standardize,
)

G_SPACE = 2   # flanking VISIBLE own-shaft contacts each side of the block (M14 Part C parity)
DT = 1        # every predictor is read at t-1, t, t+1 (M14 parity; deliberately generous)


def _fit_predict(Xtr, ytr, Xva, yva, Xte) -> np.ndarray:
    """Same protocol as _fit_eval (standardize, lambda on VAL, refit on train+val) but returns
    the TEST PREDICTIONS. Part C needs predictions, not R2s, because the three models must be
    scored on ONE pooled test set against ONE baseline — a mean of per-group R2s is not a
    comparable number (each group would carry its own denominator)."""
    Xtr, Xva, Xte = _standardize(Xtr, Xva, Xte)
    Xtr = np.column_stack([Xtr, np.ones(len(Xtr))])
    Xva = np.column_stack([Xva, np.ones(len(Xva))])
    Xte = np.column_stack([Xte, np.ones(len(Xte))])
    mu_tr = float(ytr.mean())
    best_lam, best = LAMBDAS[0], -np.inf
    for lam in LAMBDAS:
        s = _r2(yva, Xva @ _ridge_fit(Xtr, ytr, lam), mu_tr)
        if s > best:
            best, best_lam = s, lam
    w = _ridge_fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]), best_lam)
    return Xte @ w


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--band-root", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", default=os.environ.get("ROOT_DIR_BRAINTREEBANK", ""))
    p.add_argument("--n-clips", type=int, default=128)
    p.add_argument("--clip-frames", type=int, default=96)
    p.add_argument("--n-blocks", type=int, default=6, help="width-w_s blocks sampled per session")
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--out")
    a = p.parse_args()

    cfg = V3MaskConfig()
    w = cfg.block_w_space
    specs = load_v3_sessions(
        sessions=V3_SESSIONS,
        band_cache_dirs=[os.path.join(a.band_root, b) for b in BAND_DIRS],
        span_dir=a.span_dir,
        parcel_fn=make_bt_parcel_fn(a.bt_root),
        lof_report_path=None,
        winsor=WINSOR,
    )
    print(f"M15 — class context value | {len(specs)} sessions | w_s={w} (shipped) "
          f"G_SPACE={G_SPACE} DT={DT} | NO CAR\n"
          f"  cm     = mean of all OTHER-shaft contacts        (the global gain state)\n"
          f"  cross  = all OTHER-shaft contacts, individually  == CLASS W's ENTIRE CONTEXT\n"
          f"  +own   = + own-shaft survivors of a w_s block    == CLASS S's ENTIRE CONTEXT\n"
          f"  dR2(cross|cm)  = what W can teach beyond the common mode\n"
          f"  dR2(own|cross) = THE CRUTCH: exactly what S gets that W does not\n", flush=True)

    rows, blocks_C = [], []
    leak_ok, nest_bad, n_fits = True, 0, 0

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

        cand = []
        for si in range(geom.n_shafts):
            members = np.where(shaft_of == si)[0]              # contact ids, depth order
            for d0 in range(G_SPACE, len(members) - w - G_SPACE + 1):
                cand.append((si, members, d0))
        if not cand:
            print(f"[s{sid}t{tid}] SKIP — no shaft long enough for a w_s={w} block", flush=True)
            continue
        pick = rng.choice(len(cand), size=min(a.n_blocks, len(cand)), replace=False)

        rec: dict[str, dict[str, list[float]]] = {n: {} for n in BAND_NAMES}
        for bi in pick:
            si, members, d0 = cand[bi]
            flank = np.concatenate([members[d0 - G_SPACE:d0], members[d0 + w:d0 + w + G_SPACE]])
            other = np.where(shaft_of != si)[0]                # class W's whole world
            if len(other) < 4:
                continue

            for b, name in enumerate(BAND_NAMES):
                e = env[b]
                Xc = {k: _cols(e, ix, other, T) for k, ix in (("tr", tr), ("va", va), ("te", te))}
                Xm = {k: v.reshape(len(v), len(other), 2 * DT + 1).mean(1) for k, v in Xc.items()}
                Xf = {k: _cols(e, ix, flank, T) for k, ix in (("tr", tr), ("va", va), ("te", te))}
                Xo = {k: np.concatenate([Xc[k], Xf[k]], axis=1) for k in Xc}

                # Part C stores the FLANK design ONCE per (session, block, band) — it does not
                # depend on j. Only the target moves.
                blk = {"band": name, "X": Xf, "targets": []}
                for j in range(w):
                    c = int(members[d0 + j])
                    margin = min(j + 1, w - j)
                    leak_ok &= (c not in other) and (c not in flank) \
                        and bool((shaft_of[other] != si).all())
                    y = {k: _target(e, ix, c, T) for k, ix in
                         (("tr", tr), ("va", va), ("te", te))}

                    r_cm, tr_cm = _fit_eval(Xm["tr"], y["tr"], Xm["va"], y["va"],
                                            Xm["te"], y["te"])
                    r_cr, tr_cr = _fit_eval(Xc["tr"], y["tr"], Xc["va"], y["va"],
                                            Xc["te"], y["te"])
                    r_ow, tr_ow = _fit_eval(Xo["tr"], y["tr"], Xo["va"], y["va"],
                                            Xo["te"], y["te"])
                    n_fits += 1
                    # NESTING INVARIANT, on TRAIN (strictly nested feature sets)
                    if not (tr_cm <= tr_cr + 2e-3 and tr_cr <= tr_ow + 4e-3):
                        nest_bad += 1

                    for k, v in (("cm", r_cm), ("cross", r_cr), ("own", r_ow)):
                        rec[name].setdefault(f"m{margin}_{k}", []).append(v)
                    blk["targets"].append({"j": j, "contact": c, "parcel": int(parcel[c]),
                                           "sid": sid, "y": y})
                blocks_C.append(blk)

        rows.append({"subject_id": sid, "trial_id": tid, "n_contacts": int(len(shaft_of)),
                     "n_shafts": int(geom.n_shafts),
                     "r2": {n: {k: float(np.mean(v)) for k, v in d.items()}
                            for n, d in rec.items()}})
        print(f"[s{sid}t{tid}] N={len(shaft_of)} S={geom.n_shafts} "
              f"clips {len(tr)}/{len(va)}/{len(te)} blocks {len(pick)}", flush=True)

    if not rows:
        raise SystemExit("no usable sessions")

    print(f"\n  [check] no self-leak (target in neither predictor set; cross excludes its own "
          f"shaft), {n_fits * 3} fits : {'OK' if leak_ok else '*** VIOLATED ***'}")
    print(f"  [check] nesting  cm <= cross <= cross+own  on TRAIN R2, {n_fits} triples : "
          f"{'OK' if nest_bad == 0 else f'*** VIOLATED x{nest_bad} — ESTIMATOR BUG ***'}")

    # loss weights over the block: margin m gets the fraction of positions j that realize it
    mw: dict[int, float] = {}
    for j in range(w):
        m = min(j + 1, w - j)
        mw[m] = mw.get(m, 0.0) + 1.0 / w

    def _lw(name, kind):
        return float(sum(np.mean([r["r2"][name][f"m{m}_{kind}"] for r in rows
                                  if f"m{m}_{kind}" in r["r2"][name]]) * v
                         for m, v in mw.items()))

    print("\n" + "=" * 88)
    print(f"A/B — CLASS W's CEILING, AND CLASS S's CRUTCH  (loss-weighted over the block, w_s={w})")
    print("=" * 88)
    print(f"  {'band':>5} {'R2 cm':>8} {'R2 cross':>9} {'R2 +own':>8} "
          f"{'dR2(cross|cm)':>14} {'dR2(own|cross)':>15}")
    summary = {}
    for name in BAND_NAMES:
        cm, cr, ow = _lw(name, "cm"), _lw(name, "cross"), _lw(name, "own")
        summary[name] = {"cm": cm, "cross": cr, "own": ow,
                         "dR2_cross_given_cm": cr - cm, "dR2_own_given_cross": ow - cr}
        print(f"  {name:>5} {cm:>8.4f} {cr:>9.4f} {ow:>8.4f} {cr - cm:>+14.4f} "
              f"{ow - cr:>+15.4f}")
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
    print("\n" + "=" * 88)
    print("C — IS THE OWN-SHAFT KERNEL POSITION-FREE?  Same columns (block-relative flankers),")
    print("    fit GLOBAL / PER-PARCEL / PER-CONTACT, all scored on the SAME pooled test rows.")
    print("=" * 88)
    partc: dict[str, dict] = {}
    for name in BAND_NAMES:
        sel = [b for b in blocks_C if b["band"] == name]
        per_j: dict[int, dict] = {}
        for j in range(w):
            items = [(b["X"], t) for b in sel for t in b["targets"] if t["j"] == j]
            if not items:
                continue
            Xtr = np.concatenate([X["tr"] for X, _ in items])
            Xva = np.concatenate([X["va"] for X, _ in items])
            Xte = np.concatenate([X["te"] for X, _ in items])
            ytr = np.concatenate([t["y"]["tr"] for _, t in items])
            yva = np.concatenate([t["y"]["va"] for _, t in items])
            yte = np.concatenate([t["y"]["te"] for _, t in items])
            mu = float(ytr.mean())          # ONE baseline for all three models

            yh_g = _fit_predict(Xtr, ytr, Xva, yva, Xte)

            # per-parcel and per-contact: fit on the group, predict that group's test rows,
            # scatter into the SAME pooled test vector, score once.
            def _grouped(key):
                yh = np.zeros_like(yte)
                pos, spans = 0, []
                for X, t in items:
                    n = len(X["te"])
                    spans.append((key(t), X, t, pos, pos + n))
                    pos += n
                groups: dict[int, list] = {}
                for gk, X, t, s0, s1 in spans:
                    groups.setdefault(gk, []).append((X, t, s0, s1))
                for g in groups.values():
                    gtr = np.concatenate([X["tr"] for X, _, _, _ in g])
                    gva = np.concatenate([X["va"] for X, _, _, _ in g])
                    gte = np.concatenate([X["te"] for X, _, _, _ in g])
                    gytr = np.concatenate([t["y"]["tr"] for _, t, _, _ in g])
                    gyva = np.concatenate([t["y"]["va"] for _, t, _, _ in g])
                    pred = _fit_predict(gtr, gytr, gva, gyva, gte)
                    off = 0
                    for X, _, s0, s1 in g:
                        n = s1 - s0
                        yh[s0:s1] = pred[off:off + n]
                        off += n
                return _r2(yte, yh, mu), len(groups)

            p_r2, n_par = _grouped(lambda t: t["parcel"])
            c_r2, n_con = _grouped(lambda t: (t["sid"], t["contact"]))
            per_j[j] = {"global": _r2(yte, yh_g, mu), "per_parcel": p_r2, "per_contact": c_r2,
                        "n_parcels": n_par, "n_contacts": n_con, "n_test_rows": int(len(yte))}
        gl = sum(per_j[j]["global"] * mw[min(j + 1, w - j)] for j in per_j)
        pa = sum(per_j[j]["per_parcel"] * mw[min(j + 1, w - j)] for j in per_j)
        co = sum(per_j[j]["per_contact"] * mw[min(j + 1, w - j)] for j in per_j)
        ratio = gl / co if co > 1e-6 else float("nan")
        print(f"  {name:>5}  global {gl:+.4f}   per-parcel {pa:+.4f}   per-contact {co:+.4f}"
              f"   |  global / per-contact = {ratio:.2f}")
        partc[name] = {"per_j": {str(k): v for k, v in per_j.items()},
                       "loss_weighted": {"global": gl, "per_parcel": pa, "per_contact": co,
                                         "global_over_per_contact": ratio}}
    print("\n  [check] per-contact should reproduce M14 Part C at w_s=4 "
          "(loss-weighted slow .497 / mid .261 / hga .192).")
    print("          M15 z-scores its targets and M14 did not, so expect CLOSE, not identical.")
    print("          A LARGE gap means one of the two probes is wrong — find out which.")
    print("\n  READ: global ~= per-contact -> the own-shaft map is a UNIVERSAL DEPTH-LAG FILTER.")
    print("        It needs no parcel, no position, no subject. 64% of the masked loss then")
    print("        trains something that CANNOT transfer to a new montage — and THAT, not")
    print("        'W is the cross-subject task', is the argument for moving budget to W.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"w_s": w, "g_space": G_SPACE, "dt": DT,
                       "margin_weights": {str(k): v for k, v in mw.items()},
                       "summary": summary, "kernel_universality": partc, "per_session": rows,
                       "checks": {"self_leak_ok": bool(leak_ok),
                                  "nesting_violations": int(nest_bad),
                                  "n_fit_triples": int(n_fits)}}, fh, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
