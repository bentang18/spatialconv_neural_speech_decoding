"""Re-derive the mask 2x2 from per-cell probe shards. All four arms at 20k, new chassis.

Rows   = composition slide at fixed 25% visibility (space .5/time .50  ->  space 0/time .75)
Cols   = band block width (4 -> 1)
Never reads ws_cohort / cs_mean.
"""
import json, math, sys
from itertools import product

D = "/Users/bentang/.claude/jobs/971ec3c5/tmp/results_v3_probe_%s.json"
ARMS = {
    ("pbs50", "w4"): "pbs50w4_384_20k",
    ("pbs50", "w1"): "pbs50w1_384_20k",
    ("nospace", "w4"): "nospacew4_384_20k",
    ("nospace", "w1"): "nospacew1_384_20k",
}
TAP, NORM = "enc12", "std"


def load(tag):
    d = json.load(open(D % tag))
    ws, cs = {}, {}
    for k, leaf in d.items():
        if "|" not in k:
            continue
        t, tap, norm, task = k.split("|")
        if t != tag or norm != NORM or tap != TAP:
            continue
        for s, v in leaf["ws_per_session"].items():
            ws[(task, s)] = float(v)
        for s, v in leaf["cs_per_test"].items():
            cs[(task, s)] = float(v)
    return ws, cs


def sign(deltas, eps=1e-12):
    nz = [x for x in deltas if abs(x) > eps]
    n = len(nz)
    if n == 0:
        return float("nan"), 0, 0
    npos = sum(1 for x in nz if x > 0)
    k = min(npos, n - npos)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return p, npos, n


A = {k: load(v) for k, v in ARMS.items()}
for regime, idx in (("CS", 1), ("WS", 0)):
    print(f"\n===== {regime} {TAP} ({NORM}) — cell means =====")
    for r, c in product(("pbs50", "nospace"), ("w4", "w1")):
        m = A[(r, c)][idx]
        print(f"  {r:>8}/{c:<3} {sum(m.values())/len(m):.4f}   n={len(m)}")

    def contrast(name, a, b):
        X, Y = A[a][idx], A[b][idx]
        keys = sorted(set(X) & set(Y))
        d = [Y[k] - X[k] for k in keys]
        p, npos, n = sign(d)
        print(f"  {name:<44} {sum(d)/len(d):+.4f}  p={p:.3f}  {npos}/{n}")

    print(f"----- contrasts ({regime}) -----")
    contrast("width 4->1  @ pbs50 (clean single knob)", ("pbs50", "w4"), ("pbs50", "w1"))
    contrast("width 4->1  @ nospace", ("nospace", "w4"), ("nospace", "w1"))
    contrast("slide pbs50->nospace @ w4 (fixed 25% vis)", ("pbs50", "w4"), ("nospace", "w4"))
    contrast("slide pbs50->nospace @ w1", ("pbs50", "w1"), ("nospace", "w1"))
    contrast("corner: pbs50/w4 -> nospace/w1", ("pbs50", "w4"), ("nospace", "w1"))

    def mean(k):
        m = A[k][idx]
        return sum(m.values()) / len(m)

    inter = (mean(("nospace", "w1")) - mean(("nospace", "w4"))) - (
        mean(("pbs50", "w1")) - mean(("pbs50", "w4")))
    add = (mean(("pbs50", "w1")) - mean(("pbs50", "w4"))) + (
        mean(("nospace", "w4")) - mean(("pbs50", "w4")))
    print(f"  {'interaction (width x slide)':<44} {inter:+.4f}")
    print(f"  {'additive prediction for corner':<44} {add:+.4f}"
          f"   actual {mean(('nospace','w1')) - mean(('pbs50','w4')):+.4f}")

print("\n----- per-task CS enc12, width 4->1 @ pbs50 -----")
X, Y = A[("pbs50", "w4")][1], A[("pbs50", "w1")][1]
for task in sorted({k[0] for k in X}):
    ks = sorted(k for k in X if k[0] == task and k in Y)
    d = [Y[k] - X[k] for k in ks]
    print(f"  {task:<16} {sum(X[k] for k in ks)/len(ks):.4f} -> "
          f"{sum(Y[k] for k in ks)/len(ks):.4f}  {sum(d)/len(d):+.4f}  "
          f"{sum(1 for x in d if x<0)}/{len(d)} worse")
