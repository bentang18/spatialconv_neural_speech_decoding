"""Collector for the CLIP bad-window scan (#222) — read the per-session sidecars
``precompute_bad_windows.py`` wrote and summarize the per-band knee + drop stats.

This is the paired analyzer the sweep discipline requires: the scan is not "done"
until its output is pulled, summarized, and judged. It answers two questions:

1. **Is the CLIP layer behaving?** cohort frac_bad should sit ~0.1–2% (aggressive
   but not nuking the corpus); a session at ~0 or >5% is a flag.
2. **Are the per-band fences right?** For each band it prints the self-calibrated
   scale ``q_b`` and the realized headroom ``max_cell_max / q_b`` — the ratio the
   ``CAT_MULT`` fence (default 12) must clear. That headroom IS the knee Ben locks
   the per-band ``HOT_MULT_BY_BAND`` / ``CAT_MULT_BY_BAND`` finals against.

Pure ``summarize_bad_windows(records)`` core (no I/O) so it is unit-tested without
DCC sidecars; the CLI just loads ``*.json`` from a dir and prints.

Usage (laptop, after pulling the sidecars, or on DCC in place):
    .venv/bin/python scripts/neuroprobe/analyze_bad_windows.py /work/ht203/v14_bad_windows_3stft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_BANDS = ("slow", "beta", "hg")


def summarize_bad_windows(records: list[dict]) -> dict:
    """Aggregate per-session sidecar payloads into cohort + per-band stats.

    ``records`` = the list of parsed ``{session}.json`` dicts. Returns a dict with
    ``sessions`` (per-session rows) and ``per_band`` (cohort knee per band)."""
    sessions = []
    for r in sorted(records, key=lambda d: d.get("session", "")):
        rule = r.get("rule", {})
        pb = rule.get("per_band", {})
        sessions.append({
            "session": r.get("session", "?"),
            "n_elec": r.get("n_elec", 0),
            "duration_min": round(r.get("duration_s", 0.0) / 60.0, 1),
            "n_windows": r.get("n_windows", 0),
            "n_bad": r.get("n_bad_windows", 0),
            "frac_bad": r.get("frac_bad", 0.0),
            "by_rule": {
                "hot": r.get("n_hot_windows", 0),
                "cat": r.get("n_cat_windows", 0),
                "flat": r.get("n_flat_windows", 0),
            },
            "per_band": {
                b: {
                    "q": pb.get(b, {}).get("q", 0.0),
                    "max_cell_max": pb.get(b, {}).get("max_cell_max", 0.0),
                    "max_n_hot": pb.get(b, {}).get("max_n_hot", 0),
                    "headroom": (
                        pb.get(b, {}).get("max_cell_max", 0.0)
                        / pb.get(b, {}).get("q", 0.0)
                        if pb.get(b, {}).get("q", 0.0) > 0 else 0.0
                    ),
                    "n_cat_windows": pb.get(b, {}).get("n_cat_windows", 0),
                    "n_hot_windows": pb.get(b, {}).get("n_hot_windows", 0),
                }
                for b in _BANDS if b in pb
            },
        })

    n_win_tot = sum(s["n_windows"] for s in sessions)
    n_bad_tot = sum(s["n_bad"] for s in sessions)
    per_band: dict[str, dict] = {}
    for b in _BANDS:
        qs = [s["per_band"][b]["q"] for s in sessions if b in s["per_band"]]
        heads = [s["per_band"][b]["headroom"] for s in sessions if b in s["per_band"]]
        ncat = sum(s["per_band"][b]["n_cat_windows"] for s in sessions if b in s["per_band"])
        nhot = sum(s["per_band"][b]["n_hot_windows"] for s in sessions if b in s["per_band"])
        if not qs:
            continue
        per_band[b] = {
            "q_min": min(qs), "q_med": sorted(qs)[len(qs) // 2], "q_max": max(qs),
            # headroom = worst cell / q; the CAT fence (12·q) must sit above the
            # CLEAN ceiling but below the artifact — this is the per-band knee.
            "headroom_min": min(heads), "headroom_med": sorted(heads)[len(heads) // 2],
            "headroom_max": max(heads),
            "n_cat_windows": ncat, "n_hot_windows": nhot,
        }

    return {
        "n_sessions": len(sessions),
        "n_windows_total": n_win_tot,
        "n_bad_total": n_bad_tot,
        "frac_bad_cohort": (n_bad_tot / n_win_tot) if n_win_tot else 0.0,
        "sessions": sessions,
        "per_band": per_band,
    }


def _load(out_dir: Path) -> list[dict]:
    records = []
    for p in sorted(out_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "session" in d and "rule" in d:  # a sidecar, not some other json
            records.append(d)
    return records


def _fmt(summary: dict) -> str:
    out: list[str] = []
    out.append(
        f"=== {summary['n_sessions']} sessions  "
        f"{summary['n_bad_total']}/{summary['n_windows_total']} windows bad  "
        f"cohort frac_bad={100 * summary['frac_bad_cohort']:.2f}% ==="
    )
    out.append("")
    out.append(f"{'session':16s} {'elec':>4s} {'min':>5s} {'bad':>4s} {'frac%':>6s}  "
               f"{'hot/cat/flat':>12s}   per-band q[headroom]")
    for s in summary["sessions"]:
        bands = "  ".join(
            f"{b}:{s['per_band'][b]['q']:.1f}[{s['per_band'][b]['headroom']:.1f}x]"
            for b in _BANDS if b in s["per_band"]
        )
        br = s["by_rule"]
        out.append(
            f"{s['session']:16s} {s['n_elec']:>4d} {s['duration_min']:>5.1f} "
            f"{s['n_bad']:>4d} {100 * s['frac_bad']:>5.2f}%  "
            f"{br['hot']:>3d}/{br['cat']:>3d}/{br['flat']:>3d}   {bands}"
        )
    out.append("")
    out.append("per-band cohort knee (q = P99 scale; headroom = worst cell / q; "
               "CAT fence = 12·q, HOT fence = 5·q):")
    for b, pb in summary["per_band"].items():
        out.append(
            f"  {b:5s}  q[min/med/max]={pb['q_min']:.1f}/{pb['q_med']:.1f}/{pb['q_max']:.1f}  "
            f"headroom[min/med/max]={pb['headroom_min']:.1f}/{pb['headroom_med']:.1f}/"
            f"{pb['headroom_max']:.1f}x  "
            f"windows[hot={pb['n_hot_windows']},cat={pb['n_cat_windows']}]"
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path, help="dir of {session}.json sidecars")
    args = ap.parse_args()
    records = _load(args.out_dir)
    if not records:
        print(f"no sidecars found in {args.out_dir}")
        return
    print(_fmt(summarize_bad_windows(records)))


if __name__ == "__main__":
    main()
