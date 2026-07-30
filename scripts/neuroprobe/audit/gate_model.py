# PROVENANCE (see PROVENANCE.md in this directory)
#   reads:   results/r6_era/board/shards_{r6_40k,cdlin_45k}
#   run IDs: board readout arrays; parcel arm ws 20580123 / csession 20580124
#   arm:     LEACE artifacts are r6_40k, NOT canonical cdlin_45k.
#   audited: 2026-07-29 -- memory/project-number-provenance-audit-2026-07-29.md
"""Is transfer a MULTIPLICATIVE GATE on a blind gain, or a UNIFORM PENALTY?

The filter reading says: pretraining produces ONE representational change; each regime
cashes a task-dependent FRACTION of it. That is a concrete model:

    Model G (gate):     gain_far(t) = a * gain_near(t)          1 free param
    Model P (penalty):  gain_far(t) = gain_near(t) - c          1 free param
    Model Gg (grouped): gain_far(t) = a_grp * gain_near(t)      2 free params (event/level)

If G beats P, transfer SCALES the gain rather than taxing it -- i.e. it is a filter, and a
task that gains nothing near gains nothing far. If P wins, transfer is a fixed cost and the
filter framing is wrong.

CRITICAL: ws and csession are BOTH per-electrode taps, so ws->csession is UNIT-CONTROLLED.
ws->cs crosses elec->parcel and is reported but NOT used to decide.

Invariant printed first: a gate model REQUIRES sign agreement. If a task gains near and
loses far, no positive `a` can fit it -- count those before fitting anything.
"""
import json, statistics
from pathlib import Path

SH = Path("results/r6_era/board")
LEVEL = {"volume","pitch","frame_brightness","face_num","global_flow","local_flow"}
UNSTABLE = {"face_num","frame_brightness","word_length"}
CKPTS = ["r6_40k","cdlin_45k"]

def shards(ck, regime):
    """Read shard files directly: merged board JSONs carry csession:{} (stale merge)."""
    out={}
    d = SH/f"shards_{ck}"
    for p in sorted(d.glob(f"{regime}_*.json")):
        blob = json.load(open(p))
        for key, rec in blob.get("cells", {}).items():
            task = key.split("|")[-1]
            cells = rec.get("cells", {})
            out.setdefault(task, {})[p.stem] = cells
    return out

def gains(ck, regime, t0, t12):
    per = shards(ck, regime)
    g={}
    for task, byshard in per.items():
        vals=[]
        for sh, cells in byshard.items():
            a, b = cells.get(f"{t0}|std"), cells.get(f"{t12}|std")
            if a and b: vals.append(b["test"]-a["test"])
        if vals: g[task]=statistics.fmean(vals)
    return g

def fit(near, far, tasks):
    a = sum(near[t]*far[t] for t in tasks)/sum(near[t]**2 for t in tasks)
    c = statistics.fmean(near[t]-far[t] for t in tasks)
    sseG = sum((far[t]-a*near[t])**2 for t in tasks)
    sseP = sum((far[t]-(near[t]-c))**2 for t in tasks)
    ag = {}
    for grp in ("event","level"):
        sel=[t for t in tasks if (t in LEVEL)==(grp=="level")]
        ag[grp] = (sum(near[t]*far[t] for t in sel)/sum(near[t]**2 for t in sel)) if sel else float("nan")
    sseGg = sum((far[t]-ag["level" if t in LEVEL else "event"]*near[t])**2 for t in tasks)
    return a, c, sseG, sseP, ag, sseGg

for ck in CKPTS:
    ws  = gains(ck,"ws","enc0_elec","enc12_elec")
    cse = gains(ck,"csession","enc0_elec","enc12_elec")
    cs  = gains(ck,"cs","enc0","enc12")
    for lbl, near, far, controlled in (("ws -> csession  [UNIT-CONTROLLED]", ws, cse, True),
                                       ("ws -> cs        [unit-confounded]", ws, cs, False)):
        tasks = sorted(set(near)&set(far) - UNSTABLE)
        if not tasks: continue
        print(f"=== {ck}   {lbl}   {len(tasks)} tasks")
        disagree=[t for t in tasks if near[t]*far[t] < 0]
        print(f"  INVARIANT sign agreement: {len(tasks)-len(disagree)}/{len(tasks)} agree"
              + (f"   VIOLATIONS: {disagree}" if disagree else "   (gate model admissible)"))
        a,c,sseG,sseP,ag,sseGg = fit(near,far,tasks)
        print(f"  Model G  gain_far = {a:6.3f} * gain_near        SSE {sseG:.6f}")
        print(f"  Model P  gain_far = gain_near - {c:+.4f}        SSE {sseP:.6f}")
        print(f"  Model Gg a_event {ag['event']:6.3f}  a_level {ag['level']:6.3f}   SSE {sseGg:.6f}")
        win = min((sseG,"GATE"),(sseP,"PENALTY"),(sseGg,"GROUPED-GATE"))[1]
        print(f"  --> {win} fits best"
              f"   (G/P ratio {sseG/sseP:.2f}; grouped buys {100*(1-sseGg/sseG):.0f}% of G's SSE)")
        if not controlled: print("  (do not decide on this row: elec->parcel)")
        print(f"  {'task':20s} {'near':>8s} {'far':>8s} {'far/near':>9s}  grp")
        for t in sorted(tasks, key=lambda t:-near[t]):
            r = far[t]/near[t] if near[t] else float('nan')
            print(f"  {t:20s} {near[t]:+8.4f} {far[t]:+8.4f} {r:9.2f}  "
                  f"{'level' if t in LEVEL else 'event'}")
        print()
