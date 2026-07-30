"""Price the READOUT FAMILY at FROZEN features — the missing R0 term. NO GPU, NO encoder.

WHY THIS EXISTS. The WS partial-FT pilot reported ``d(D-A) = -.0060`` (D = fine-tuned block 12 +
trained logistic head, A = frozen features + const-lambda dual ridge) and that number is not
decomposable as it stands: it is the sum of a feature gain and a readout cost, which can cancel.
Ben 07-30: *"a ridge regression might be optimal on frozen features - but that might not be true on
fine tuned features? don't be black and white here"* — exactly right, and it is why the readout
cost has to be measured SEPARATELY at each feature set rather than assumed constant:

    R0 = head_frozen - A        readout cost AT FROZEN FEATURES   <- what this script measures
    d(C - A)                    feature gain as the ridge sees it  (the pilot's gate)
    R1 = D - C                  readout cost AT FINE-TUNED FEATURES
    R1 - R0                     how the readout family's value CHANGES with fine-tuning

A ridge is a fixed-capacity closed-form squared-loss estimator. Fine-tuning changes the feature
DISTRIBUTION — it can concentrate task information into fewer directions and sharpen the margin —
and on such features a logistic loss can beat a squared loss whose lambda was calibrated for the old
spectrum. So R0 does NOT predict R1, and R0 <= 0 does not make the pilot's numbers
uninterpretable. It only supplies the offset that makes them decomposable.

⚠️ MEASURED, and it corrects an earlier claim: ``ridge_df`` came back **539 of n_train=1279** on
S1T0 onset (job 20598494). The dual solve does bound capacity by n rather than p, but df at ~42% of
n is NOT "far below n" — do not write that the ridge is barely using its capacity.

🔑 PREPROCESSING IS MATCHED BY CONSTRUCTION, NOT BY AN ARM. MAE's linear-probe recipe is "an extra
BatchNorm layer without affine transformation before the linear classifier" (mae.tex:616), and per-
feature z-scoring on train statistics is exactly that (``RDO._standardize``) — which is also exactly
what the ridge gets. So the ``std`` variant reduces head-vs-ridge to LOSS + SHRINKAGE SELECTION,
with nothing left over. ``ln`` is kept only as the pilot head's incumbent: ``LayerNorm`` normalizes
PER ROW across all 212,992 features, a different preprocessing from the ridge's, and the suspected
defect in that head.

🔑 CONVERGENCE IS NOT A STORY HERE. Both variants are FIXED data transforms, so the model stays
LINEAR in the features and the L2-penalized optimum lies in the row span (representer theorem):
w = Z^T alpha gives logits = G alpha + b and ||w||^2 = alpha^T G alpha, the SAME objective on a
1279x1279 Gram the ridge already builds. A step costs 1.6 MFLOP instead of 0.8 GFLOP, so the fit
runs to convergence for free. This matters because the first primal smoke run stopped at 60 steps
with ``median_step == 60`` — still climbing — and reporting that as "the head loses" would have been
the exact underfit artifact this pilot exists to avoid. ``hit_wall`` flags any cell whose best step
was the last one, so a floor can never be misread as a converged value.

Everything reads the EXISTING encode cache (``v3_probe_cache_<tag>``), so this spends no GPU and no
encoder forward — the cache was already built for the pilot's L0 parity gate.

Usage (Delta CPU, one array task per session):
  python -m scripts.neuroprobe.v3_frozen_readout_sweep \
      --cache-dir /projects/bhqk/htang13/v3_probe_cache_pbs50_20k --tag pbs50_20k \
      --session-index 0 --out /projects/bhqk/htang13/frozen_readout_s0.json
  python -m scripts.neuroprobe.v3_frozen_readout_sweep --merge 'frozen_readout_s*.json'
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch


def _load_sibling(name):
    """Import a sibling script by path — they are scripts, not a package."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


RDO = _load_sibling("v3_probe_readout_r4")
PROBE_TASKS = RDO.PROBE_TASKS
PROBE_COHORT_7 = RDO.PROBE_COHORT_7
RIDGE_LAMS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
TAP = "enc12"


def _sign_p(k: int, n: int) -> float:
    """Two-sided exact binomial sign test at p=0.5."""
    if n == 0:
        return float("nan")
    k = max(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def _standardize_t(z_tr, *others):
    """``RDO._standardize`` in torch, fit on TRAIN only. Constant columns -> 0, never inf."""
    mu = z_tr.mean(0)
    sd = z_tr.std(0, unbiased=False)
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    return [(z_tr - mu) / sd] + [(o - mu) / sd for o in others]


def _ridge_eval(s_tr, s_va, s_te, y_tr, y_va, y_te, lams):
    """val/test AUROC + effective dof from ONE Gram and ONE eigendecomposition.

    Inputs are ALREADY standardized so the caller can reuse them for the head arms — the ridge and
    the heads must see byte-identical features or the contrast is not a readout contrast.
    Returns (val@const, test@const, test@val-selected-lambda, df@const)."""
    g = (s_tr @ s_tr.T).double()
    k_va = (s_va @ s_tr.T).double()
    k_te = (s_te @ s_tr.T).double()
    n = g.shape[0]
    basel = float(torch.diagonal(g).sum() / max(n, 1))
    w, v = torch.linalg.eigh(g)
    vty = v.T @ torch.as_tensor(y_tr, dtype=torch.float64)

    def sc(k, lm):
        return (k @ (v @ (vty / (w + lm * basel)))).numpy()

    c = RDO.CONST_LAM_MULT
    best = max(lams, key=lambda L: RDO.auroc(sc(k_va, L), y_va))
    return (RDO.auroc(sc(k_va, c), y_va), RDO.auroc(sc(k_te, c), y_te),
            RDO.auroc(sc(k_te, best), y_te), float((w / (w + c * basel)).sum()))


def _variants(z_tr, z_va, z_te, which):
    """Build the FEATURE VARIANTS the readouts are compared on. Both are FIXED data transforms —
    neither has parameters — which is what keeps the model linear and lets the dual fit be exact.

      ``std``  per-feature z-score on TRAIN stats (``RDO._standardize``). This is the ridge's own
               preprocessing, and at full batch it is MAE's affine-free BatchNorm (mae.tex:616).
      ``ln``   per-ROW LayerNorm of the RAW features — the pilot head's incumbent. Applied to raw
               z, not to the standardized z, because that is what ``_Head`` actually does. Its
               learnable affine is omitted deliberately: fc(g*n + b) = (w*g).n + (w.b + c), so the
               affine is exactly absorbable into the following Linear and buys no function class.

    🪦 DROPPED: the convex parcel pool. Measured R0 = -.0507 on S1T0 onset (job 20598494) against
    -.0148 for `std` — pooling over parcels LOSES BADLY, and the reason is structural rather than
    tunable: the ridge reads the CONCAT of all 16 parcels and therefore knows which parcel a
    feature came from, while any pool discards that. It is also the one variant with a trainable
    parameter (beta), so it is not kernelizable. Not worth a slot."""
    out = {}
    if "std" in which:
        out["std"] = _standardize_t(z_tr, z_va, z_te)
    if "ln" in which:
        out["ln"] = [torch.nn.functional.layer_norm(x, (x.shape[1],)) for x in (z_tr, z_va, z_te)]
    return out


def _kernels(s_tr, s_va, s_te):
    """(G, K_va, K_te, basel), all float64, normalized by the mean Gram diagonal.

    WHY NORMALIZE. Raw Gram entries scale with the feature count (basel ~ p = 212,992), so an
    un-normalized dual would need alpha ~ 1e-5 and no single Adam lr would work across cells.
    Dividing by basel puts the logits on an O(1) scale, which makes ONE lr grid valid everywhere."""
    g = (s_tr @ s_tr.T).double()
    n = g.shape[0]
    basel = float(torch.diagonal(g).sum() / max(n, 1))
    return (g / basel, (s_va @ s_tr.T).double() / basel, (s_te @ s_tr.T).double() / basel, basel)


def _ridge_from_kernels(g, k_va, k_te, y_tr, y_va, y_te, lams):
    """Cell A. Reads the SAME kernels the logistic arms use, so the contrast is the readout alone.

    ALSO returns the ridge's EFFECTIVE DEGREES OF FREEDOM, df = sum_i s_i/(s_i + lam). MEASURED,
    not argued: on S1T0 onset it came back 539/1279 — the dual solve does bound capacity by n, but
    df sits at ~42% of n, which is NOT "far below n". Do not claim the ridge is barely using its
    capacity."""
    w, v = torch.linalg.eigh(g)
    vty = v.T @ torch.as_tensor(y_tr, dtype=torch.float64)

    def sc(k, lm):
        return (k @ (v @ (vty / (w + lm)))).numpy()

    c = RDO.CONST_LAM_MULT
    best = max(lams, key=lambda L: RDO.auroc(sc(k_va, L), y_va))
    return (RDO.auroc(sc(k_va, c), y_va), RDO.auroc(sc(k_te, c), y_te),
            RDO.auroc(sc(k_te, best), y_te), float((w / (w + c)).sum()))


def _fit_logit_dual(g, k_va, k_te, y_tr, y_va, y_te, *, lam, lr, steps, seed):
    """EXACT L2-regularized logistic regression, fitted in the DUAL.

    WHY THE DUAL. The primal is a 212,992-dim weight against ~1.3k rows, and one SGD step must
    stream a 1.09 GB feature matrix twice — memory-bandwidth-bound at ~0.22 s/step, which is why
    the first smoke run stopped at 60 steps with ``median_step == 60``, i.e. STILL CLIMBING. A head
    that never converged would have been reported as "the head loses", which is the exact artifact
    this pilot exists to avoid. By the representer theorem the L2-penalized optimum lies in the row
    span, w = Z^T alpha, so

        logits = Z w + b = G alpha + b        and       ||w||^2 = alpha^T G alpha

    is the SAME objective on a 1279x1279 Gram — 1.6 MFLOP a step instead of 0.8 GFLOP. The Gram is
    already computed for the ridge, so convergence becomes free rather than expensive.

    Returns the val-selected step's val/test AUROC, plus ``hit_wall``: True if the best step WAS
    the last step, i.e. the fit was still improving and the number is a floor, not a converged
    value."""
    torch.manual_seed(seed)
    n = g.shape[0]
    a = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    yb = torch.as_tensor((np.asarray(y_tr) > 0).astype(np.float64))
    opt = torch.optim.Adam([a, b], lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    best = {"val": -1.0, "test": float("nan"), "step": -1}
    gn = float("nan")
    for st in range(steps + 1):
        if st > 0:
            ga = g @ a
            loss = lossf(ga + b, yb) + 0.5 * lam * (a @ ga)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.sqrt(a.grad.pow(2).sum() + b.grad.pow(2).sum()))
            opt.step()
        if st % 25 and st != steps:
            continue
        with torch.no_grad():
            v = RDO.auroc((k_va @ a + b).numpy(), y_va)
            if np.isfinite(v) and v > best["val"]:
                best = {"val": float(v), "test": float(RDO.auroc((k_te @ a + b).numpy(), y_te)),
                        "step": st}
    return {**best, "grad_norm": gn, "hit_wall": best["step"] == steps}


def _cells(cache_dir, tag, session, args):
    """Every (task, fold) cell for ONE session. The cache is mmap'd: the big ``enc12_elec`` tap is
    never touched, so peak RSS stays near the parcel-pooled tap we actually read."""
    s, t = session
    path = f"{cache_dir}/enc_s{s}_t{t}_{tag}.pt"
    rec = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    skey = f"S{s}T{t}"
    out = []
    for task in PROBE_TASKS:
        y_all = np.asarray(rec["labels"][task], dtype=np.float64)
        for fold, sp in rec["ws_split"][task].items():
            tr = RDO._finite(y_all, sp["train"])
            va = RDO._finite(y_all, sp["val"])
            te = RDO._finite(y_all, sp["test"])
            # Same margin rule as the pilot: val and test are ADJACENT contiguous blocks, so the
            # last val window can overlap the first test window (the M14 STFT hop leak). Trim the
            # VAL side only — `te` must stay byte-identical to the frozen arm's test rows.
            if args.val_margin and len(va) > args.val_margin:
                va = np.sort(va)[: -args.val_margin]
            if min(len(tr), len(va), len(te)) < 2:
                print(f"[cell] {skey} {task} f{fold} SKIP (n<2)", flush=True)
                continue
            z = [torch.from_numpy(RDO._feat(rec, TAP, r)) for r in (tr, va, te)]
            n_parcels = int(rec["feats"][TAP]["raw"].shape[1])
            var = _variants(*z, args.norms)
            dim = int(z[0].shape[1])
            del z
            yv, yt_, ytl = y_all[tr], y_all[va], y_all[te]
            row = {"session": skey, "task": task, "fold": int(fold),
                   "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                   "dim": dim, "n_parcels": n_parcels}
            for norm, (s_tr, s_va, s_te) in var.items():
                g, k_va, k_te, basel = _kernels(s_tr, s_va, s_te)
                if norm == "std":
                    # A is the ridge on the STD variant -- the preprocessing the board quotes.
                    a_val, a_test, a_vallam, df = _ridge_from_kernels(
                        g, k_va, k_te, yv, yt_, ytl, RIDGE_LAMS)
                    row.update(A_test=a_test, A_val=a_val, A_test_vallam=a_vallam, ridge_df=df,
                               basel=basel)
                # Every (lam, lr) is val-selected, exactly as the ridge selects lambda on val:
                # denying the head its own shrinkage selection would handicap it on a
                # hyperparameter the ridge is allowed to tune. Free now that the Gram is reused.
                cand = [(_fit_logit_dual(g, k_va, k_te, yv, yt_, ytl, lam=lm, lr=lr,
                                         steps=args.steps, seed=args.seed), lm, lr)
                        for lm in args.lams for lr in args.lrs]
                bst, lm, lr = max(cand, key=lambda c: c[0]["val"])
                row.update({f"{norm}_test": bst["test"], f"{norm}_val": bst["val"],
                            f"{norm}_step": bst["step"], f"{norm}_lam": lm, f"{norm}_lr": lr,
                            f"{norm}_grad_norm": bst["grad_norm"],
                            f"{norm}_hit_wall": bool(bst["hit_wall"])})
                del g, k_va, k_te
            del var
            out.append(row)
            print(f"[cell] {skey} {task:16s} f{fold} A={row['A_test']:.4f} "
                  + " ".join(f"{n}={row[f'{n}_test']:.4f}"
                             f"(R0={row[f'{n}_test'] - row['A_test']:+.4f}"
                             f"{'!WALL' if row[f'{n}_hit_wall'] else ''})"
                             for n in args.norms)
                  + f" | df={row['ridge_df']:.1f}/{len(tr)} dim={row['dim']} |P|={n_parcels} "
                    f"n_tr={len(tr)} n_va={len(va)} n_te={len(te)}", flush=True)
    return out


def _assert_one_row_per_cell(rows):
    """A cell is (session, task, fold). Merging two rows for the same cell silently POOLS ARMS —
    which is the #1 defect class in this project — and it happened: the merge glob `s*.json` also
    matched `smoke.json`, a deliberately underfit 60-step run over cells the real shards cover, so
    a 56-cell design reported 58 cells with two duplicated cells dragging the mean. Refuse instead
    of averaging, and name the files, because a silent duplicate is indistinguishable from a result."""
    seen = {}
    dupes = []
    for r in rows:
        k = (r["session"], r["task"], int(r["fold"]))
        if k in seen:
            dupes.append((k, seen[k], r.get("_src", "?")))
        else:
            seen[k] = r.get("_src", "?")
    if dupes:
        lines = "\n".join(f"    {k[0]} {k[1]} f{k[2]}: {a} vs {b}" for k, a, b in dupes)
        raise SystemExit(
            f"FATAL: {len(dupes)} duplicated cell(s) in the merge — arms are being pooled.\n"
            f"{lines}\n  Narrow the glob (s[0-9]*.json, not s*.json) or delete the stale shard.")
    return len(seen)


def _report(rows, norms):
    """Paired over CELLS — the board test. R0 = head - A at frozen features."""
    print(f"\n=== FROZEN READOUT SWEEP — {len(rows)} cells ===")
    print(f"{'session':8s} {'task':16s} {'f':>2s} {'A':>8s} "
          + " ".join(f"{n:>8s} {'R0':>8s}" for n in norms))
    for r in sorted(rows, key=lambda x: (x["session"], x["task"], x["fold"])):
        print(f"{r['session']:8s} {r['task']:16s} {r['fold']:2d} {r['A_test']:8.4f} "
              + " ".join(f"{r[f'{n}_test']:8.4f} {r[f'{n}_test'] - r['A_test']:+8.4f}"
                         for n in norms))
    print()
    for n in norms:
        ds = [r[f"{n}_test"] - r["A_test"] for r in rows]
        nz = [x for x in ds if abs(x) > 1e-9]
        k = sum(x > 0 for x in nz)
        wall = sum(bool(r.get(f"{n}_hit_wall")) for r in rows)
        print(f"  R0[{n:5s}] mean={float(np.mean(ds)):+.4f}  {k}/{len(nz)} positive  "
              f"p={_sign_p(k, len(nz)):.4f}  "
              f"median_step={int(np.median([r[f'{n}_step'] for r in rows]))}  "
              f"grad_norm_med={float(np.median([r[f'{n}_grad_norm'] for r in rows])):.2e}  "
              f"hit_wall={wall}/{len(rows)}")
        if wall:
            print(f"    ⚠️ {wall} cell(s) selected the LAST step — those R0 values are FLOORS, "
                  f"not converged. Raise --steps before reading a negative R0 as a verdict.")
    dv = [r["A_test_vallam"] - r["A_test"] for r in rows]
    print(f"  ridge val-lambda vs const-lambda: mean={float(np.mean(dv)):+.4f}  "
          f"(prices the ridge's OWN shrinkage selection)")
    # ⚖️ THE MATCHED BAR. Every head number above is val-selected over (lam, lr), while `A_test` is
    # the ridge at CONST lambda -- so R0 hands the head a model-selection advantage the ridge was
    # never given. `A_test_vallam` is the ridge selected on the SAME val split, which is the only
    # comparison where both estimators paid the same price. Report BOTH, always: R0 answers "does a
    # head beat the column we quote", R0_matched answers "does a head beat a ridge tuned as hard".
    # If they disagree, the win is model selection, not the readout family.
    for n in norms:
        dm = [r[f"{n}_test"] - r["A_test_vallam"] for r in rows]
        nz = [x for x in dm if abs(x) > 1e-9]
        k = sum(x > 0 for x in nz)
        print(f"  R0_matched[{n:5s}] (vs val-selected ridge) mean={float(np.mean(dm)):+.4f}  "
              f"{k}/{len(nz)} positive  p={_sign_p(k, len(nz)):.4f}")
    print(f"  ridge_df median={float(np.median([r['ridge_df'] for r in rows])):.1f} "
          f"of n_train median={float(np.median([r['n_train'] for r in rows])):.0f}")
    best = max(norms, key=lambda n: float(np.mean([r[f"{n}_test"] - r["A_test"] for r in rows])))
    bd = float(np.mean([r[f"{best}_test"] - r["A_test"] for r in rows]))
    print(f"\n  BEST FROZEN HEAD: {best} at R0={bd:+.4f}")
    print("  R0 is the readout cost AT FROZEN FEATURES. It does NOT predict the cost at "
          "fine-tuned\n  features (the spectrum moves), so it cannot by itself condemn or "
          "vindicate the FT arms —\n  it supplies the offset that makes d(D-A) decomposable: "
          "d(D-A) = [feature gain] + R1, and\n  R1-R0 is how much the readout family's value "
          "CHANGES with fine-tuning.")
    bm = float(np.mean([r[f"{best}_test"] - r["A_test_vallam"] for r in rows]))
    if bd > 0:
        print("  ⚠️ A FROZEN head BEATS the const-lambda ridge — that is a win with NO "
              "fine-tuning and\n     it also means our quoted column was leaving readout on the "
              "table. Verify on the board\n     before it is claimed.")
        if bm <= 0:
            print(f"  🚫 BUT IT LOSES TO THE VAL-SELECTED RIDGE (R0_matched={bm:+.4f}) — so the "
                  f"win is\n     MODEL SELECTION, not the readout family. The cheap fix is to "
                  f"val-select the ridge's\n     lambda in the readout, NOT to train a head.")
        else:
            print(f"  ✅ AND it survives the matched bar (R0_matched={bm:+.4f}) — the head beats a "
                  f"ridge\n     tuned on the same val split.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir")
    p.add_argument("--tag", default="pbs50_20k")
    p.add_argument("--session-index", type=int, default=None)
    p.add_argument("--out")
    p.add_argument("--merge", default=None, help="glob of shard JSONs -> report only")
    p.add_argument("--norms", default="std,ln",
                   help="std = per-feature train z-score (== MAE's affine-free BN at full batch, "
                        "== the ridge's own preprocessing); ln = the pilot head's per-ROW "
                        "LayerNorm. `pool` was dropped: measured R0=-.0507 vs -.0148 (job "
                        "20598494) and it is not kernelizable.")
    p.add_argument("--lams", default="1e-4,1e-3,1e-2,1e-1,1.0,10.0",
                   help="L2 penalty on the dual: 0.5*lam*alpha^T G alpha. Val-selected, so the "
                        "head gets the same shrinkage tuning the ridge gets over RIDGE_LAMS.")
    p.add_argument("--lrs", default="0.03,0.1,0.3")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--val-margin", type=int, default=2)
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--tasks", default=None, help="smoke-test escape hatch; default = all 4")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()
    args.norms = [x for x in args.norms.split(",") if x]
    args.lrs = [float(x) for x in args.lrs.split(",")]
    args.lams = [float(x) for x in args.lams.split(",")]
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.merge:
        rows = [dict(r, _src=os.path.basename(f)) for f in sorted(glob.glob(args.merge))
                for r in json.load(open(f))]
        if not rows:
            raise SystemExit(f"no rows matched {args.merge}")
        _assert_one_row_per_cell(rows)
        _report(rows, args.norms)
        return

    if args.tasks:
        global PROBE_TASKS
        PROBE_TASKS = tuple(x for x in args.tasks.split(",") if x)
    cohort = list(PROBE_COHORT_7)
    sessions = ([cohort[args.session_index]] if args.session_index is not None else cohort)
    print(f"[sweep] tag={args.tag} tap={TAP} sessions={sessions} norms={args.norms} "
          f"lrs={args.lrs} lams={args.lams} steps={args.steps} "
          f"threads={torch.get_num_threads()}",
          flush=True)
    rows = []
    for session in sessions:
        rows += _cells(args.cache_dir, args.tag, session, args)
        if args.out:
            json.dump(rows, open(args.out, "w"), indent=1)
    _report(rows, args.norms)


if __name__ == "__main__":
    main()
