#!/usr/bin/env python
"""GUARD-1 static-drop collector (#208) — pairs the #207 DCC precompute.

The #207 precompute (DCC) scans each BT (subject, trial)'s pre-CAR notch+HPF
voltage with the three locked detectors (``guard1.static_bad_mask``) and dumps a
per-session signature JSON. This collector reads those signatures, re-derives the
STATIC drop **per session** via ``guard1.per_session_static_drops`` (which keeps
the locked thresholds in ONE place and identity-guards against scan/collect
drift), and emits a per-session drop dict ``{(subject, trial): (labels,...)}``
ready to bake into ``anatomy._BT_V14_EXTRA_BAD_ELECTRODES_PER_SESSION`` (#213).

Per-session decision (Ben 2026-06-18, "go per session on both eval lite and
pretrain"): electrode quality drifts across a subject's sessions, so the drop is
a per-(subject, trial) decision applied to BOTH the eval-Lite and pretrain paths.
Each session's classified ``static`` IS its drop — there is **no** cross-trial
aggregation (no any/majority/chronic-vs-variable heuristic). The old per-subject
collapse (``guard1.collate_sessions``/``aggregate_subject``/``diff_static_drops``)
is superseded and off this path.

Per-session signature JSON schema (one file per session, = ``session_signature``):
    {"subject": int, "trial": int,
     "labels":    [str, ...],     # cleaned contact labels, voltage order
     "clip_frac": [float, ...],   # dV/dt>100σ clip-corruption fraction per contact
     "rmad":      [float, ...],    # robust spread of filtered V per contact
     "spike"/"noisy"/"dead"/"static": [str, ...]}   # fired detectors (provenance)

Coverage guard: the v14 scan corpus is BT_FULL minus the excluded subjects (S5)
= 25 sessions. The collector fails loud if any expected session is missing — the
per-session bake needs an explicit entry for EVERY session.

Laptop-runnable (pure JSON in, JSON + paste-ready literal out — no BT data):

    .venv/bin/python scripts/neuroprobe/collect_guard1_static.py \
        --detections-dir reports/guard1_static/detections \
        --out reports/guard1_static/per_session.json \
        --literal-out reports/guard1_static/per_session_literal.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from speech_decoding.studies.braintreebank import guard1
from speech_decoding.studies.braintreebank.manifest import (
    BT_FULL_SESSIONS,
    V14_EXCLUDED_SUBJECT_IDS,
)

_REQUIRED_KEYS = ("subject", "trial", "labels", "clip_frac", "rmad", "static")


def _expected_sessions() -> set[tuple[int, int]]:
    """The v14 scan corpus: BT_FULL minus the excluded subjects (S5)."""
    excluded = set(V14_EXCLUDED_SUBJECT_IDS)
    return {(s, t) for (s, t) in BT_FULL_SESSIONS if s not in excluded}


def _load_sessions(detections_dir: Path) -> list[dict]:
    files = sorted(detections_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no per-session *.json under {detections_dir}")
    sessions: list[dict] = []
    for f in files:
        rec = json.loads(f.read_text())
        for key in _REQUIRED_KEYS:
            if key not in rec:
                raise SystemExit(f"{f} missing required key {key!r}")
        sessions.append(rec)
    return sessions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detections-dir", required=True,
                    help="dir of per-session signature JSONs from the #207 precompute")
    ap.add_argument("--out", default="reports/guard1_static/per_session.json")
    ap.add_argument("--literal-out", default=None,
                    help="optional path to write the paste-ready Python dict literal")
    args = ap.parse_args()

    sessions = _load_sessions(Path(args.detections_dir))
    print(f"[load] {len(sessions)} per-session signatures")

    # Per-session assembler + identity guard (re-derived static == emitted).
    per_session = guard1.per_session_static_drops(sessions)

    expected = _expected_sessions()
    present = set(per_session)
    extra = sorted(present - expected)
    missing = sorted(expected - present)
    if extra:
        print(f"[warn] {len(extra)} session(s) outside the v14 corpus: {extra}")
    if missing:
        raise SystemExit(
            f"[FAIL] {len(missing)} expected session(s) missing — cannot bake a "
            f"per-session set without every session: {missing}"
        )
    print(f"[coverage] all {len(expected)} expected sessions present")

    n_drop = sum(1 for v in per_session.values() if v)
    n_contacts = sum(len(v) for v in per_session.values())
    print(f"[static] {n_drop}/{len(per_session)} sessions drop ≥1 contact "
          f"({n_contacts} contacts total)")
    by_label = {(int(s["subject"]), int(s["trial"])): s for s in sessions}
    for key in sorted(per_session):
        v = per_session[key]
        if v:
            rec = by_label[key]
            print(f"[static]   btbank{key[0]}_t{key[1]}: spike={rec.get('spike')} "
                  f"noisy={rec.get('noisy')} dead={rec.get('dead')}")

    literal = guard1.format_per_session_literal(per_session)
    report = {
        "n_sessions": len(sessions),
        "thresholds": {
            "spike_clip_frac": guard1.SPIKE_CLIP_FRAC,
            "noisy_z": guard1.NOISY_Z,
            "dead_ratio": guard1.DEAD_RATIO,
        },
        "missing_sessions": [list(m) for m in missing],
        "per_session": {
            f"{s}_{t}": per_session[(s, t)] for (s, t) in sorted(per_session)
        },
        "per_session_dict_literal": literal,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[write] {out_path}")

    if args.literal_out:
        lit_path = Path(args.literal_out)
        lit_path.parent.mkdir(parents=True, exist_ok=True)
        lit_path.write_text(literal + "\n")
        print(f"[write] {lit_path}")
    print("\n" + literal)


if __name__ == "__main__":
    main()
