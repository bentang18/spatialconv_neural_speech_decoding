#!/usr/bin/env python
"""GUARD-1 static-drop collector for the Cogan D-cohort — Cogan sibling of
``collect_guard1_static.py``.

The Cogan precompute (``precompute_guard1_static_cogan.py``, one SLURM-array task
per manifest run) scans each run's pre-CAR notch+HPF voltage with the three locked
detectors and dumps a per-run signature ``cogan{sid}_t{tid}.json``. This collector
reads those signatures, re-derives the STATIC drop **per run** via
``guard1.per_session_static_drops`` (locked thresholds in ONE place + identity-guard
against scan/collect drift), and emits a per-run static-drop MAP as neutral JSON.

Per-run decision (same convention as BT, Ben 2026-06-18): electrode quality drifts
across a subject's sessions, so the drop is a per-``(subject, trial)`` decision —
each run's ``spike ∪ noisy ∪ dead`` IS its drop, no cross-trial aggregation.

Cogan key note: the signature ``subject`` field is the GLOBAL subject id (e.g.
``1005``), ``trial`` the manifest ``trial_id`` — so the map key ``"1005_0"`` reads
as (global_subject_id, trial_id), matching how the scan tagged each run.

Coverage guard: the scan corpus = the manifest runs. With ``--manifest`` the
collector fails loud if any manifest run lacks a signature (a missing run would
silently be treated as clean — every run needs an explicit static decision).

Output is a NEUTRAL JSON map only — this collector deliberately does NOT bake into
the loader (the bake mechanism, code-literal vs runtime data-file, is unresolved).

Laptop-runnable (pure JSON in, JSON out — no Cogan data):

    .venv/bin/python scripts/neuroprobe/collect_guard1_static_cogan.py \
        --detections-dir /work/ht203/cogan_v3/guard1/detections \
        --manifest /work/ht203/cogan_v3/manifest/cogan_manifest.csv \
        --out /work/ht203/cogan_v3/guard1/cogan_guard1_static.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from speech_decoding.studies.braintreebank import guard1

_REQUIRED_KEYS = ("subject", "trial", "labels", "clip_frac", "rmad", "static")


def load_sessions(detections_dir: Path) -> list[dict]:
    files = sorted(detections_dir.glob("cogan*.json"))
    if not files:
        raise SystemExit(f"no cogan*.json signatures under {detections_dir}")
    sessions: list[dict] = []
    for f in files:
        rec = json.loads(f.read_text())
        for key in _REQUIRED_KEYS:
            if key not in rec:
                raise SystemExit(f"{f} missing required key {key!r}")
        sessions.append(rec)
    return sessions


def manifest_run_keys(manifest_path: Path) -> set[tuple[int, int]]:
    """Expected (global_subject_id, trial_id) keys — one per manifest run."""
    with open(manifest_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {(int(r["global_subject_id"]), int(r["trial_id"])) for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detections-dir", required=True,
                    help="dir of cogan{sid}_t{tid}.json signatures from the precompute")
    ap.add_argument("--manifest", default=None,
                    help="cogan_manifest.csv — if given, enforce full run coverage")
    ap.add_argument("--out", required=True,
                    help="output JSON static-map path")
    args = ap.parse_args()

    sessions = load_sessions(Path(args.detections_dir))
    print(f"[load] {len(sessions)} per-run signatures")

    # Per-run assembler + scan/collect identity guard (re-derived == emitted static).
    per_run = guard1.per_session_static_drops(sessions)

    if args.manifest:
        expected = manifest_run_keys(Path(args.manifest))
        present = set(per_run)
        extra = sorted(present - expected)
        missing = sorted(expected - present)
        if extra:
            print(f"[warn] {len(extra)} signature(s) not in manifest: {extra[:10]}"
                  + (" ..." if len(extra) > 10 else ""))
        if missing:
            raise SystemExit(
                f"[FAIL] {len(missing)} manifest run(s) have NO signature — cannot "
                f"emit a per-run map without every run: {missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
            )
        print(f"[coverage] all {len(expected)} manifest runs present")

    n_drop = sum(1 for v in per_run.values() if v)
    n_contacts = sum(len(v) for v in per_run.values())
    print(f"[static] {n_drop}/{len(per_run)} runs drop >=1 contact "
          f"({n_contacts} contacts total)")
    by_key = {(int(s["subject"]), int(s["trial"])): s for s in sessions}
    for key in sorted(per_run):
        v = per_run[key]
        if v:
            rec = by_key[key]
            print(f"[static]   cogan{key[0]}_t{key[1]}: spike={rec.get('spike')} "
                  f"noisy={rec.get('noisy')} dead={rec.get('dead')}")

    report = {
        "n_runs": len(sessions),
        "thresholds": {
            "spike_clip_frac": guard1.SPIKE_CLIP_FRAC,
            "noisy_z": guard1.NOISY_Z,
            "dead_ratio": guard1.DEAD_RATIO,
        },
        "n_runs_with_drop": n_drop,
        "n_contacts_dropped": n_contacts,
        # key "{global_subject_id}_{trial_id}" -> sorted static labels (or [])
        "per_run": {f"{s}_{t}": per_run[(s, t)] for (s, t) in sorted(per_run)},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
