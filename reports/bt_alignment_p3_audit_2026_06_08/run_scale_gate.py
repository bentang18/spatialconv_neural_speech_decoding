"""Scale-aware alignment probe for the two films the per-word RMS gate flagged
(fantastic-mr-fox, lotr-2).

WHY (P3 alignment audit follow-up, 2026-06-08)
---------------------------------------------
``run_audio_gate.py`` searched only a constant per-quarter time SHIFT over a
+-1.0 s grid. That cannot diagnose a global time-SCALE mismatch: a PAL 25/24
speedup (1.0417x) drifts ~53 s *within* one film-quarter, far outside +-1 s, so a
PAL-sped rip would show low r at every quarter even though the audio is the right
content at the wrong clock RATE. A scale mismatch is fixable by resampling (no
new rip); a genuinely wrong cut/source is not.

This sweeps OUR rip clock as ``rip_time = scale * bt_time + shift`` and reports,
for each film, the (scale, shift) that maximizes whole-film log-RMS Pearson r vs
BT ``features.csv['rms']``, plus the per-quarter best at that scale. Interpretation:
  * r recovers to ~0.85+ at scale near 0.96 or 1.04  -> SPEED issue, resample.
  * r recovers at scale 1.0 with a single shift       -> constant offset.
  * r stays low at every scale                         -> wrong source, re-rip.

scipy/numpy only. Reuses loaders from run_audio_gate.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_audio_gate as g  # noqa: E402

OUT_JSON = Path(__file__).resolve().parent / "scale_gate_results.json"


def logsafe(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan)
    m = np.isfinite(x) & (x > 0)
    out[m] = np.log(x[m])
    return out


def sweep(movie: str, scales: np.ndarray, shifts: np.ndarray) -> dict:
    t0 = time.time()
    wav, sr = g.load_wav_mono(g.WAV_DIR / f"{movie}.wav")
    bt = g.load_bt_words(movie)
    starts, ends, bt_log = bt["start"], bt["end"], logsafe(bt["rms"])
    n = len(starts)

    best = {"r": float("-inf"), "scale": None, "shift": None}
    grid = []
    for sc in scales:
        for sh in shifts:
            ours = logsafe(g.word_rms_ours(wav, sr, starts * sc + sh, ends * sc + sh))
            r, used = g.pearson(ours, bt_log)
            if np.isfinite(r):
                grid.append({"scale": round(float(sc), 4), "shift": round(float(sh), 3),
                             "r": round(r, 4), "n": used})
                if r > best["r"]:
                    best = {"r": round(r, 4), "scale": round(float(sc), 4),
                            "shift": round(float(sh), 3), "n": used}

    # per-quarter at the winning scale, fine shift search
    sc = best["scale"]
    per_q = []
    fine = np.round(np.arange(best["shift"] - 2.0, best["shift"] + 2.001, 0.1), 3)
    by_sh = {float(s): logsafe(g.word_rms_ours(wav, sr, starts * sc + s, ends * sc + s))
             for s in fine}
    for q in range(4):
        sl = slice(q * n // 4, (q + 1) * n // 4)
        bq = {"r": float("-inf"), "shift": None}
        for s, ours in by_sh.items():
            r, _ = g.pearson(ours[sl], bt_log[sl])
            if np.isfinite(r) and r > bq["r"]:
                bq = {"r": round(r, 4), "shift": round(s, 3)}
        per_q.append({"quarter": q + 1,
                      "t_lo_min": round(float(starts[q * n // 4]) / 60, 2),
                      "t_hi_min": round(float(starts[min((q + 1) * n // 4, n - 1)]) / 60, 2),
                      **bq})

    top = sorted(grid, key=lambda d: d["r"], reverse=True)[:8]
    return {"film": movie, "n_words": n, "wav_dur_s": round(len(wav) / sr, 1),
            "best": best, "per_quarter_at_best_scale": per_q, "top8": top,
            "elapsed_s": round(time.time() - t0, 1)}


def main() -> int:
    # coarse scale grid covering PAL/NTSC both directions + fine near 1.0
    scales = np.unique(np.round(np.concatenate([
        np.arange(0.94, 1.061, 0.01),
        [24 / 25, 25 / 24, 23.976 / 24, 24 / 23.976],
    ]), 4))
    shifts = np.round(np.arange(-4.0, 4.01, 0.5), 3)
    movies = sys.argv[1:] or ["fantastic-mr-fox", "lotr-2"]
    results = []
    for mv in movies:
        print(f"[{mv}] sweeping {len(scales)} scales x {len(shifts)} shifts ...", flush=True)
        r = sweep(mv, scales, shifts)
        results.append(r)
        b = r["best"]
        print(f"  best: scale={b['scale']} shift={b['shift']}s r={b['r']} "
              f"| per-quarter shifts={[q['shift'] for q in r['per_quarter_at_best_scale']]} "
              f"r={[q['r'] for q in r['per_quarter_at_best_scale']]} ({r['elapsed_s']}s)")
    OUT_JSON.write_text(json.dumps(
        {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results}, indent=2))
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
