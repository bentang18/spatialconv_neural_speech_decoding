#!/usr/bin/env python
"""GUARD-1 static-drop collector (#208) — pairs the #207 DCC precompute.

The #207 precompute (DCC) scans each BT (subject, trial)'s pre-CAR notch+HPF
voltage with the three locked detectors (``guard1.static_bad_mask``) and dumps a
per-session signature JSON. This collector reads those signatures, applies the
Ben-locked thresholds via the shared ``guard1`` core, collapses per session →
per subject (chronic noisy/dead any-trial, spike majority), splits chronic vs
variable, builds the would-be per-subject static-drop dict, and DIFFS it against
the current locked-11 convention (``anatomy.extra_bad_electrodes``). It writes a
report; it does NOT touch the dataset, the cache, or ``anatomy.py`` — wiring the
self-calibrated set into production is the separate, cache-affecting #213 (gated).

Per-session signature JSON schema (one file per session, = ``static_bad_mask``):
    {"subject": int, "trial": int,
     "labels":    [str, ...],     # cleaned contact labels, voltage order
     "clip_frac": [float, ...],   # dV/dt>100σ clip-corruption fraction per contact
     "rmad":      [float, ...]}    # robust spread of filtered V per contact

Laptop-runnable (pure JSON in, JSON out — no BT data, no ckpt):

    .venv/bin/python scripts/neuroprobe/collect_guard1_static.py \
        --detections-dir reports/guard1_static/detections \
        --out reports/guard1_static/collated.json

"no sweep without a collector": this exists so #207 can be dispatched. The RUN
of #207 is Ben-gated (the BT raw h5 was deleted in the /work migration); this
collector is ready the moment those per-session signatures exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from speech_decoding.studies.braintreebank import guard1
from speech_decoding.studies.braintreebank.anatomy import (
    _BT_V14_EXTRA_BAD_ELECTRODES,
    extra_bad_electrodes,
)


def _load_sessions(detections_dir: Path) -> list[dict]:
    sessions: list[dict] = []
    files = sorted(detections_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no per-session *.json under {detections_dir}")
    for f in files:
        rec = json.loads(f.read_text())
        for key in ("subject", "trial", "labels", "clip_frac", "rmad"):
            if key not in rec:
                raise SystemExit(f"{f} missing required key {key!r}")
        sessions.append(rec)
    return sessions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detections-dir", required=True,
                    help="dir of per-session signature JSONs from the #207 precompute")
    ap.add_argument("--out", default="reports/guard1_static/collated.json")
    args = ap.parse_args()

    sessions = _load_sessions(Path(args.detections_dir))
    print(f"[load] {len(sessions)} per-session signatures")

    per_subject = guard1.collate_sessions(sessions)
    would_be = {sid: rec["static"] for sid, rec in per_subject.items()}

    # locked-11 over the union of (precompute-seen subjects, locked-11 subjects),
    # so a subject locked-11 dropped but the self-calibration cleared shows up.
    locked_subjects = set(would_be) | set(_BT_V14_EXTRA_BAD_ELECTRODES)
    locked = {s: extra_bad_electrodes(s) for s in locked_subjects}
    diff = guard1.diff_static_drops(would_be, locked)

    n_added = sum(len(d["added"]) for d in diff.values())
    n_removed = sum(len(d["removed"]) for d in diff.values())
    print(f"[diff] vs locked-11: +{n_added} added / -{n_removed} removed across "
          f"{len(diff)} subjects")
    for sid in sorted(diff):
        d = diff[sid]
        if d["added"] or d["removed"]:
            print(f"[diff]   subj {sid}: +{d['added']} -{d['removed']}")

    report = {
        "n_sessions": len(sessions),
        "thresholds": {
            "spike_clip_frac": guard1.SPIKE_CLIP_FRAC,
            "noisy_z": guard1.NOISY_Z,
            "dead_ratio": guard1.DEAD_RATIO,
        },
        "per_subject": {
            str(sid): {
                "n_trials": rec["n_trials"],
                "spike": sorted(rec["spike"]),
                "noisy": sorted(rec["noisy"]),
                "dead": sorted(rec["dead"]),
                "static": sorted(rec["static"]),
                "chronic": sorted(rec["chronic"]),
                "variable": sorted(rec["variable"]),
            }
            for sid, rec in sorted(per_subject.items())
        },
        "diff_vs_locked11": {str(sid): diff[sid] for sid in sorted(diff)},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
