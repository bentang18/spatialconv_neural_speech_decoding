"""CLIP-aggressiveness sensitivity sweep — find the most aggressive bad-window
fences that still keep the cleanest session (the canary) at ~zero drops, from
data instead of argument (Ben 2026-06-18).

The CLIP layer's only non-redundant knob vs WINSOR is the COMMON-MODE count: how
many electrodes must be simultaneously hot in a 5 s window before we drop the
whole pretrain clip. The shipped fences (HOT 5×q / CAT 12×q / FRAC_HOT 0.05 /
N_FLOOR 3) are carried over from the 2STFT knee sweep; this re-measures the knee
on the rebuilt 3STFT cache domain (post-static ``bt_load_raw``).

It reuses ``precompute_bad_windows.compute_ewm_by_band`` to pay the expensive
streaming pass ONCE per session, then re-runs the pure ``_decide_bad_windows``
over a one-factor-at-a-time grid (cheap numpy). The decisive number is the canary
btbank3_t2: the most aggressive value of each lever where it stays at its
baseline (≈0) is the answer.

Two modes (one file = the sweep AND its collector, so the loop is closed):

  # array task: scan + sweep ONE session (picks via SLURM_ARRAY_TASK_ID)
  python scripts/neuroprobe/sweep_clip_aggressiveness.py --out-dir <DIR>

  # collector: aggregate the per-session JSONs, print the canary-gated knee
  python scripts/neuroprobe/sweep_clip_aggressiveness.py --summarize --out-dir <DIR>

The per-session output ``{session}.json`` is sweep-only diagnostics — it is NOT a
clip sidecar and is never read by a training run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# scripts/ is not a package; import the precompute by adding its dir to the path.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precompute_bad_windows import (  # noqa: E402
    Q_PCT,
    V14_PRETRAIN_SESSIONS,
    _FRONTEND_BANDS,
    _decide_bad_windows,
    _make_band_specs,
    compute_ewm_by_band,
)

# Canary = the cleanest session: its baseline drop count is ~0, so the most
# aggressive lever value that keeps it at that floor is the safe answer.
CANARY: str = "btbank3_t2"

# Pre-sweep fences (the OFAT origin — every lever holds these while one varies).
# This is the OLD reference point the sweep measures AGAINST, NOT the locked answer.
BASELINE: dict[str, float] = {
    "hot_mult": 5.0, "cat_mult": 12.0, "frac_hot": 0.05, "n_floor": 3,
}
# LOCKED fences (what the sweep chose, now the production defaults): hot 5→4,
# cat 12→8, + the q-independent abs floor K=200. ``locked_combined`` reports the
# REAL union drop (hot@4 ∪ cat@8 ∪ flat ∪ abs@200) — a union, not the sum of the
# OFAT marginals — so it is the honest "% of clips dropped under the lock".
LOCKED: dict[str, float] = {
    "hot_mult": 4.0, "cat_mult": 8.0, "frac_hot": 0.05, "n_floor": 3,
}
# One-factor-at-a-time grid. Each value is swept with the OTHER three held at
# BASELINE, so the marginal effect of that single lever is isolated. Values reach
# MORE aggressive than baseline (lower fence / lower count) — the question is how
# far we can push before the canary starts dropping clean clips. Extended BELOW the
# current lock (cat 8 / hot 4) to test "can we be a little stricter" (#264).
GRID: dict[str, list[float]] = {
    "hot_mult": [3.0, 3.5, 4.0, 4.5, 5.0],
    "cat_mult": [5.0, 6.0, 7.0, 8.0, 10.0, 12.0],
    "frac_hot": [0.02, 0.03, 0.04, 0.05],
    "n_floor": [2, 3],
}

# Candidate STRICTER locks (#264): the OFAT marginals isolate one lever; these are
# full multi-lever combos whose REAL union drop (hot ∧ cat ∧ flat ∧ abs200) +
# canary count are reported, so the cost of tightening is honest (not the sum of
# marginals). All keep abs_floor=200 (the prior sweep's clean-safe floor). The
# decisive number is each candidate's canary count: it must stay at the canary
# baseline (~0) to be safe.
CANDIDATE_LOCKS: dict[str, dict[str, float]] = {
    "current(hot4/cat8)": {"hot_mult": 4.0, "cat_mult": 8.0, "frac_hot": 0.05, "n_floor": 3},
    "cat6": {"hot_mult": 4.0, "cat_mult": 6.0, "frac_hot": 0.05, "n_floor": 3},
    "hot3.5/cat6": {"hot_mult": 3.5, "cat_mult": 6.0, "frac_hot": 0.05, "n_floor": 3},
    "hot3.5/cat6/frac.03": {"hot_mult": 3.5, "cat_mult": 6.0, "frac_hot": 0.03, "n_floor": 3},
    "hot3/cat5": {"hot_mult": 3.0, "cat_mult": 5.0, "frac_hot": 0.05, "n_floor": 3},
    "hot3/cat5/frac.03": {"hot_mult": 3.0, "cat_mult": 5.0, "frac_hot": 0.03, "n_floor": 3},
}
# Absolute-floor K (in session-normalized MADs): drop a window if ANY band cell
# exceeds K, regardless of the relative q. This is the session-INDEPENDENT
# backstop — on a noisy session q rides up and the relative cat fence (12·q)
# rides with it, so a fixed-MAD floor is the only thing that still bites. It only
# adds windows beyond baseline where K < 12·q_band (else relative cat caught it).
ABS_FLOOR_K: list[float] = [100.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]

# Common-mode histogram depth: report how many windows have ≥ k electrodes hot
# together (at the baseline 5×q fence), so the count distribution near the
# FRAC_HOT threshold is visible — Ben's "how many windows have 4-6 hot".
GE_K_MAX: int = 12


def _key(v: float) -> str:
    """JSON-safe dict key for a lever value (ints stay int-looking)."""
    return str(int(v)) if float(v).is_integer() else str(v)


def sweep_decision(
    ewm_by_band: dict[str, np.ndarray],
    n_flat: np.ndarray,
    n_elec: int,
    *,
    abs_floor_k: list[float] | None = None,
) -> dict:
    """Pure sweep core (testable on synthetic ewm, no BT): given one session's
    per-band per-(elec,window) |z|-max + flat counts, run the OFAT grid + the
    absolute-floor variant + the common-mode count histogram. Returns the bad
    window COUNTS at every grid point (drops, not spans)."""
    abs_floor_k = ABS_FLOOR_K if abs_floor_k is None else abs_floor_k
    n_windows = int(n_flat.shape[0])

    def n_bad(**over: float) -> int:
        # abs_floor_mad=inf disables the NEW production abs backstop so the OFAT grid
        # measures the RELATIVE rules in isolation (the abs floor is swept separately
        # below). Without this pin the relative levers would silently fold in abs@200.
        kw = {**BASELINE, "abs_floor_mad": float("inf"), **over}
        bad_idx, _ = _decide_bad_windows(ewm_by_band, n_flat, n_elec, **kw)  # type: ignore[arg-type]
        return len(bad_idx)

    baseline = n_bad()
    levers = {
        lev: {_key(v): n_bad(**{lev: v}) for v in vals}
        for lev, vals in GRID.items()
    }
    # The honest answer: the REAL union drop at the locked fences (relative AND the
    # abs floor together), not the sum of OFAT marginals.
    locked_idx, _ = _decide_bad_windows(ewm_by_band, n_flat, n_elec, **LOCKED)  # type: ignore[arg-type]
    locked_combined = len(locked_idx)

    # Candidate STRICTER locks (#264): real union drop (abs floor 200 ON) per combo.
    candidate_locks = {}
    for name, fences in CANDIDATE_LOCKS.items():
        idx, _ = _decide_bad_windows(
            ewm_by_band, n_flat, n_elec, abs_floor_mad=200.0, **fences,  # type: ignore[arg-type]
        )
        candidate_locks[name] = len(idx)

    # Absolute floor = baseline-bad UNION (any band cell > K). Union (not replace)
    # because it's an ADDITIONAL backstop on top of the relative rules.
    base_idx, _ = _decide_bad_windows(
        ewm_by_band, n_flat, n_elec, **BASELINE, abs_floor_mad=float("inf"),  # type: ignore[arg-type]
    )
    base_set = set(base_idx)
    allband_cellmax = np.zeros(n_windows, np.float32)
    for ewm in ewm_by_band.values():
        if ewm.shape[0]:
            allband_cellmax = np.maximum(allband_cellmax, ewm.max(axis=0))
    abs_floor = {"baseline": baseline}
    for K in abs_floor_k:
        extra = set(np.nonzero(allband_cellmax > K)[0].tolist())
        abs_floor[_key(K)] = len(base_set | extra)

    # Common-mode count histogram at the baseline hot fence (5×q per band): for
    # each window, the max over bands of how many electrodes are hot. ge_k[k] =
    # #windows with ≥ k hot electrodes. Shows whether windows pile up just under
    # the FRAC_HOT threshold (permissive fence) or the threshold sits in a gap.
    max_n_hot = np.zeros(n_windows, np.int32)
    for ewm in ewm_by_band.values():
        if not ewm.shape[0]:
            continue
        q = float(np.percentile(ewm, Q_PCT))
        n_hot = (ewm > BASELINE["hot_mult"] * q).sum(axis=0).astype(np.int32)
        max_n_hot = np.maximum(max_n_hot, n_hot)
    ge_k = {str(k): int((max_n_hot >= k).sum()) for k in range(1, GE_K_MAX + 1)}

    return {
        "baseline": int(baseline),
        "locked_combined": int(locked_combined),
        "candidate_locks": {k: int(v) for k, v in candidate_locks.items()},
        "n_windows": int(n_windows),
        "n_elec": int(n_elec),
        "hot_count_thresh": int(
            max(BASELINE["n_floor"], int(np.ceil(BASELINE["frac_hot"] * n_elec)))
        ),
        "levers": levers,
        "abs_floor": abs_floor,
        "common_mode_ge_k": ge_k,
    }


def run_session(
    subject_id: int, trial_id: int, out_dir: Path,
    band_specs: list[tuple[str, int, int, int, int]] | None = None,
) -> dict:
    """Scan one session (the expensive streaming pass) and sweep it. Writes
    ``{session}.json`` of bad-window counts at every grid point."""
    tag = f"btbank{subject_id}_t{trial_id}"
    ewm_by_band, n_flat, n_elec, total_s, n_windows = compute_ewm_by_band(
        subject_id, trial_id, band_specs,
    )
    res = sweep_decision(ewm_by_band, n_flat, n_elec)
    res.update(
        {"session": tag, "subject_id": subject_id, "trial_id": trial_id,
         "duration_s": float(total_s)}
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(res, indent=2))
    frac = res["baseline"] / n_windows if n_windows else 0.0
    print(f"[{tag}] baseline_bad={res['baseline']}/{n_windows} "
          f"({frac:.2%})  n_elec={n_elec}  thresh={res['hot_count_thresh']}")
    return res


# --------------------------------------------------------------------- collector
def _aggregate(recs: list[dict]) -> dict:
    """Cohort + canary marginal curves from per-session sweep records. Cohort =
    summed drop counts (windows are absolute, so summing is exact); canary = the
    single canary session's counts (the gating signal)."""
    canary = next((r for r in recs if r["session"] == CANARY), None)
    cohort_windows = sum(r["n_windows"] for r in recs)

    def cohort_lever(lev: str, key: str) -> int:
        return sum(r["levers"][lev][key] for r in recs)

    levers: dict[str, dict] = {}
    for lev, vals in GRID.items():
        levers[lev] = {}
        for v in vals:
            k = _key(v)
            levers[lev][k] = {
                "cohort": cohort_lever(lev, k),
                "canary": (canary["levers"][lev][k] if canary else None),
            }

    abs_keys = ["baseline"] + [_key(K) for K in ABS_FLOOR_K]
    abs_floor = {
        k: {
            "cohort": sum(r["abs_floor"][k] for r in recs),
            "canary": (canary["abs_floor"][k] if canary else None),
        }
        for k in abs_keys
    }

    ge_k = {
        str(k): {
            "cohort": sum(r["common_mode_ge_k"][str(k)] for r in recs),
            "canary": (canary["common_mode_ge_k"][str(k)] if canary else None),
        }
        for k in range(1, GE_K_MAX + 1)
    }
    candidate_locks = {
        name: {
            "cohort": sum(r.get("candidate_locks", {}).get(name, 0) for r in recs),
            "canary": (canary.get("candidate_locks", {}).get(name) if canary else None),
        }
        for name in CANDIDATE_LOCKS
    }
    return {
        "n_sessions": len(recs),
        "cohort_windows": cohort_windows,
        "canary_session": CANARY,
        "canary_baseline": (canary["baseline"] if canary else None),
        "cohort_baseline": sum(r["baseline"] for r in recs),
        "cohort_locked_combined": sum(r.get("locked_combined", 0) for r in recs),
        "canary_locked_combined": (canary.get("locked_combined") if canary else None),
        "candidate_locks": candidate_locks,
        "levers": levers,
        "abs_floor": abs_floor,
        "common_mode_ge_k": ge_k,
    }


def _most_aggressive_safe(curve: dict, vals: list[float], baseline_canary: int) -> str:
    """Most aggressive (smallest) lever value whose canary count stays at the
    canary baseline. Returns the chosen key, or 'none' if even the gentlest pushes
    the canary above baseline."""
    safe = [v for v in vals if curve[_key(v)]["canary"] == baseline_canary]
    return _key(min(safe)) if safe else "none"


def summarize(out_dir: Path) -> dict:
    """Load every per-session JSON, aggregate, print the canary-gated knee."""
    recs = [json.loads(p.read_text())
            for p in sorted(out_dir.glob("btbank*.json"))]
    if not recs:
        raise SystemExit(f"no sweep JSONs in {out_dir}")
    agg = _aggregate(recs)
    base_can = agg["canary_baseline"]

    print(f"\n=== CLIP-aggressiveness sweep — {agg['n_sessions']} sessions, "
          f"{agg['cohort_windows']} windows total ===")
    print(f"canary = {CANARY} (baseline drops = {base_can})\n")

    cw = agg["cohort_windows"]
    cb, cl = agg["cohort_baseline"], agg["cohort_locked_combined"]
    print("COMBINED drop (real union, not OFAT sum):")
    print(f"  pre-sweep  (hot5/cat12, no abs) : {cb:>5}/{cw} "
          f"({cb / cw:.2%})  canary={base_can}")
    print(f"  LOCKED (hot4/cat8/abs200)       : {cl:>5}/{cw} "
          f"({cl / cw:.2%})  canary={agg['canary_locked_combined']}\n")

    print("CANDIDATE STRICTER LOCKS (real union, abs200 ON)  "
          "[canary must stay at baseline to be safe]:")
    print(f"  {'lock':>22}  {'cohort':>8}  {'cohort%':>8}  {'canary':>8}")
    for name in CANDIDATE_LOCKS:
        c = agg["candidate_locks"][name]
        frac = c["cohort"] / cw if cw else 0.0
        flag = "" if c["canary"] == base_can else "  <-canary LEAVES baseline"
        print(f"  {name:>22}  {c['cohort']:>8}  {frac:>7.2%}  {c['canary']:>8}{flag}")
    print()

    print("per-session baseline drops:")
    for r in sorted(recs, key=lambda r: r["session"]):
        frac = r["baseline"] / r["n_windows"] if r["n_windows"] else 0.0
        print(f"  {r['session']:>14}  {r['baseline']:>4}/{r['n_windows']:<4} "
              f"({frac:5.2%})  n_elec={r['n_elec']}  thresh={r['hot_count_thresh']}")

    for lev, vals in GRID.items():
        print(f"\n{lev} (OFAT, others at baseline)  "
              f"[most aggressive canary-safe: "
              f"{_most_aggressive_safe(agg['levers'][lev], vals, base_can)}]")
        print(f"  {'value':>8}  {'cohort':>8}  {'canary':>8}")
        for v in vals:
            c = agg["levers"][lev][_key(v)]
            star = "  <-baseline" if float(v) == BASELINE[lev] else ""
            print(f"  {_key(v):>8}  {c['cohort']:>8}  {c['canary']:>8}{star}")

    print(f"\nabsolute-floor K (MADs; baseline ∪ any-cell>K)  "
          f"[most aggressive canary-safe: "
          f"{_most_aggressive_safe(agg['abs_floor'], ABS_FLOOR_K, base_can)}]")
    print(f"  {'K':>8}  {'cohort':>8}  {'canary':>8}")
    for k in ["baseline"] + [_key(K) for K in ABS_FLOOR_K]:
        c = agg["abs_floor"][k]
        print(f"  {k:>8}  {c['cohort']:>8}  {c['canary']:>8}")

    print("\ncommon-mode count distribution (#windows with >= k electrodes hot "
          "at 5×q):")
    print(f"  {'k':>4}  {'cohort':>8}  {'canary':>8}")
    for k in range(1, GE_K_MAX + 1):
        c = agg["common_mode_ge_k"][str(k)]
        print(f"  {k:>4}  {c['cohort']:>8}  {c['canary']:>8}")
    return agg


# -------------------------------------------------------------------------- main
def _resolve_session(task_id: int, sessions: list[tuple[int, int]]) -> tuple[int, int]:
    if not 0 <= task_id < len(sessions):
        raise SystemExit(f"task id {task_id} out of range 0..{len(sessions) - 1}")
    return sessions[task_id]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--summarize", action="store_true",
                   help="Collector mode: aggregate the per-session JSONs + print the knee.")
    p.add_argument("--task-id", type=int, default=None,
                   help="Session index (defaults to SLURM_ARRAY_TASK_ID).")
    p.add_argument("--sessions", default=None,
                   help='"S:T,S:T,..." override (array indexes into it).')
    p.add_argument("--frontend", choices=("3stft", "2band"), default="3stft",
                   help="Band set for the scan (default 3stft). 2band = converged-v2 LFS/HGA.")
    args = p.parse_args()

    if args.summarize:
        summarize(args.out_dir)
        return

    band_specs = _make_band_specs(_FRONTEND_BANDS[args.frontend])
    sessions = (
        [tuple(int(x) for x in tok.split(":")) for tok in args.sessions.split(",")]
        if args.sessions else list(V14_PRETRAIN_SESSIONS)
    )
    task_id = (args.task_id if args.task_id is not None
               else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    subject_id, trial_id = _resolve_session(task_id, sessions)  # type: ignore[arg-type]
    run_session(subject_id, trial_id, args.out_dir, band_specs)


if __name__ == "__main__":
    main()
