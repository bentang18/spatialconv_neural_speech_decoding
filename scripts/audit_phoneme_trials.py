#!/usr/bin/env python3
"""Per-patient `#34` audit CLI.

Runs the structural + timing + audio-neural alignment predicates for one
patient, writes the JSON verdict + diagnostic plot + exclusion-candidates CSV
to `reports/phoneme_audit_2026_04_16/`.

Usage:
    .venv/bin/python -m scripts.audit_phoneme_trials --patient S14

Exit code is 0 only when verdict == "pass" or "warn".
"""
from __future__ import annotations

import argparse
import sys

from speech_decoding.v14.audit.runner import run_patient_audit, write_result


def _print_summary(result) -> None:
    print()
    print(f"── {result['patient']} verdict: {result['verdict']} ──")
    for c in result["checks"]:
        mark = "✓" if c["passed"] else "✗"
        tag = f"[{c['level']}]"
        print(f"  {mark} {tag:6} {c['name']:40}  {c['detail']}")
    if result["workarounds"]:
        print("  workarounds:")
        for w in result["workarounds"]:
            print(f"    - {w}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--patient", required=True, help="e.g. S14")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    result = run_patient_audit(args.patient, verbose=not args.quiet)
    out = write_result(result)
    _print_summary(result)
    print(f"\nwrote {out}")
    return 0 if result["verdict"] in ("pass", "warn") else 1


if __name__ == "__main__":
    sys.exit(main())
