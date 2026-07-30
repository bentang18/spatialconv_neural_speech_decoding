#!/usr/bin/env python
"""Is the task axis's overlap with the identity subspace IDENTITY-SPECIFIC, or just anisotropy?

WHY THIS IS REQUIRED. viz_ident_overlap.py scores ||P d||^2 / ||d||^2 against the null k/d. That
null is EXACT only for an ISOTROPIC space. Neural features are strongly anisotropic, so a task
direction lands preferentially in ANY subspace built from high-variance directions -- identity or
not. Reporting ratio ~ 4-10 as "entangled with subject identity" without this control would repeat
precisely the error that produced the +/-.002 erasure result, where a variance-matched top-PC
erasure turned out to be equally free and the identity-specific reading collapsed.

Four rank-matched bases, same data, same math, same rank (keep), only the CONSTRUCTION differs:

  identity  span{mu_session - mu_grand}          the measurement
  perm      same construction, session labels SHUFFLED over rows at matched counts. Preserves
            anisotropy AND the mean-difference construction; destroys only real session identity.
            THE DECISIVE CONTROL: identity >> perm means identity-specific; identity ~ perm means
            the construction plus anisotropy explains everything.
  toppc     top-keep right singular vectors of the pooled rows (total variance). Asks whether "the
            task axis lives in high-variance directions" already accounts for the overlap.
  rand      isotropic random keep-dim subspace. A SANITY CHECK ON THE NULL ITSELF: this must land
            at ratio ~ 1.0. If it does not, k/d is the wrong null and every ratio is misscaled.

Read `identity / perm`. That ratio, not the raw ratio_to_chance, is the identity-specificity claim.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from viz_common import load_all, shared_lobes
from viz_figures import collect

SEED = 33


def _basis_from_means(means: np.ndarray, d: int):
    """Orthonormal basis of the row space of (means - their own mean), rank-truncated."""
    x = means - means.mean(axis=0)
    _, sv, vt = np.linalg.svd(x, full_matrices=False)
    tol = max(1e-6 * float(sv.max(initial=0.0)), 1e-12)
    keep = int((sv > tol).sum())
    if not keep or keep >= d:
        return None, keep
    return vt[:keep].T, keep


def _frac(q: np.ndarray, t: np.ndarray) -> float | None:
    n2 = float(t @ t)
    if n2 <= 0:
        return None
    p = q.T @ t
    return float(p @ p) / n2


def run(sessions, lobes, tap: str, task: str, rng) -> dict | None:
    per: dict = {}
    for cls in (0, 1):
        for s, m in collect(sessions, tap, task, cls, "all", lobes, centered=False):
            per.setdefault(s.key, {})[cls] = m.reshape(-1, m.shape[-1])
    keys = sorted(k for k, v in per.items() if 0 in v and 1 in v)
    if len(keys) < 3:
        return None
    d = int(per[keys[0]][0].shape[1])

    sess_mean = np.stack([np.stack([per[k][0].mean(axis=0), per[k][1].mean(axis=0)]).mean(axis=0)
                          for k in keys])
    q_id, keep = _basis_from_means(sess_mean, d)
    tasks = {k: per[k][1].mean(axis=0) - per[k][0].mean(axis=0) for k in keys}
    out = {"tap": tap, "task": task, "n_sessions": len(keys), "feature_dim": d,
           "identity_rank": keep, "chance_sq": keep / d if keep else float("nan")}
    if q_id is None:
        return {**out, "vacuous": True}
    out["vacuous"] = False

    def score(q):
        v = [f for f in (_frac(q, tasks[k]) for k in keys) if f is not None]
        return float(np.mean(v)) if v else float("nan")

    # pooled rows, per-session row counts preserved for the permutation
    blocks = [np.concatenate([per[k][0], per[k][1]]) for k in keys]
    counts = [b.shape[0] for b in blocks]
    pooled = np.concatenate(blocks)

    # perm: shuffle which rows form each pseudo-session, matched counts, same construction
    perm_scores = []
    for _ in range(20):
        idx = rng.permutation(pooled.shape[0])
        off, pm = 0, []
        for c in counts:
            pm.append(pooled[idx[off:off + c]].mean(axis=0))
            off += c
        qp, kp = _basis_from_means(np.stack(pm), d)
        if qp is not None and kp == keep:
            perm_scores.append(score(qp))

    # toppc: top-keep total-variance directions
    xc = pooled - pooled.mean(axis=0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    q_top = vt[:keep].T

    # rand: isotropic random keep-dim subspace -> validates that chance == keep/d
    rand_scores = []
    for _ in range(20):
        g = rng.standard_normal((d, keep))
        qr, _ = np.linalg.qr(g)
        rand_scores.append(score(qr))

    out.update({
        "identity": score(q_id),
        "perm": float(np.mean(perm_scores)) if perm_scores else float("nan"),
        "perm_n": len(perm_scores),
        "toppc": score(q_top),
        "rand": float(np.mean(rand_scores)),
    })
    for k in ("identity", "perm", "toppc", "rand"):
        out[f"ratio_{k}"] = out[k] / out["chance_sq"]
    out["identity_over_perm"] = out["identity"] / out["perm"] if out["perm"] > 0 else float("nan")
    out["identity_over_toppc"] = out["identity"] / out["toppc"] if out["toppc"] > 0 else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--red-dir", required=True)
    ap.add_argument("--taps", default="enc3,enc6,enc12")
    ap.add_argument("--tasks", default="onset,speech,delta_volume,word_index,"
                                      "word_part_speech,frame_brightness")
    ap.add_argument("--out", default="ident_control.json")
    a = ap.parse_args()

    rng = np.random.default_rng(SEED)
    sessions = load_all(a.red_dir)
    lobes = shared_lobes(sessions)
    taps = [t for t in a.taps.split(",") if t and any(t in s.shapes for s in sessions)]
    tasks = [t for t in a.tasks.split(",") if t]

    print(f"[load]  {len(sessions)} sessions  red_dir={a.red_dir}  taps={taps}")
    print(f"[check] shared lobes: {lobes}   seed={SEED}   20 reps for perm and rand")
    print("[null]  rand MUST land at ratio ~1.0 -- that is the check that k/d is the right null")
    print("[read]  identity/perm is the identity-specificity claim, NOT ratio_to_chance\n")
    print(f"{'tap':6s} {'task':18s} {'ident':>7s} {'perm':>7s} {'toppc':>7s} {'rand':>7s}  "
          f"{'r_id':>6s} {'r_perm':>6s} {'r_rand':>6s}  {'id/perm':>7s} {'id/top':>7s}")

    rows = []
    for tap in taps:
        for task in tasks:
            r = run(sessions, lobes, tap, task, rng)
            if r is None:
                continue
            if r["vacuous"]:
                print(f"{tap:6s} {task:18s}  VACUOUS (complete subspace, rank "
                      f"{r['identity_rank']}/{r['feature_dim']})")
                rows.append(r)
                continue
            rows.append(r)
            print(f"{tap:6s} {task:18s} {r['identity']:7.4f} {r['perm']:7.4f} {r['toppc']:7.4f} "
                  f"{r['rand']:7.4f}  {r['ratio_identity']:6.2f} {r['ratio_perm']:6.2f} "
                  f"{r['ratio_rand']:6.2f}  {r['identity_over_perm']:7.2f} "
                  f"{r['identity_over_toppc']:7.2f}")

    print("\n=== MACRO over tasks ===")
    ok = [r for r in rows if not r["vacuous"]]
    macro = {}
    for tap in taps:
        rs = [r for r in ok if r["tap"] == tap]
        if not rs:
            continue
        m = {k: float(np.mean([r[k] for r in rs])) for k in
             ("ratio_identity", "ratio_perm", "ratio_toppc", "ratio_rand",
              "identity_over_perm", "identity_over_toppc")}
        macro[tap] = m
        wins = sum(1 for r in rs if r["identity"] > r["perm"])
        print(f"  {tap:6s} r_id {m['ratio_identity']:6.2f}  r_perm {m['ratio_perm']:6.2f}  "
              f"r_toppc {m['ratio_toppc']:6.2f}  r_rand {m['ratio_rand']:5.2f}  |  "
              f"id/perm {m['identity_over_perm']:5.2f}  id/toppc {m['identity_over_toppc']:5.2f}  "
              f"| id>perm {wins}/{len(rs)}")

    if macro:
        rr = float(np.mean([m["ratio_rand"] for m in macro.values()]))
        print(f"\n[check] rand ratio across taps = {rr:.3f}  "
              f"{'OK -- k/d is the right null' if 0.9 <= rr <= 1.1 else 'FAIL -- null is misscaled'}")
        print("[read]  id/perm ~1 => NOT identity-specific (anisotropy + construction explain it)")
        print("[read]  id/perm >>1 => identity-specific entanglement is real")

    json.dump({"red_dir": a.red_dir, "seed": SEED, "rows": rows, "macro": macro},
              open(a.out, "w"), indent=1)
    print(f"\n[out] {a.out}  {len(rows)} cells")


if __name__ == "__main__":
    main()
