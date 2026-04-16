#!/usr/bin/env python3
"""Rollup: join every per-patient `#34` audit JSON into a single markdown.

Usage:
    .venv/bin/python -m scripts.rollup_phoneme_audit

Emits `reports/phoneme_audit_2026_04_16/README.md` with:
- per-patient verdict table
- per-check pass/fail matrix across patients
- pointer to per-patient JSONs, exclusion CSVs, alignment plots
- closure-rule footer (core patients S14, S26, S33, S62 must be pass/warn)
"""
from __future__ import annotations

import json
from pathlib import Path

from speech_decoding.v14.audit import paths

CORE_PATIENTS = ["S14", "S26", "S33", "S62"]
EXTENDED_PATIENTS = ["S16", "S23", "S39"]


def _load_results(out_dir: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for p in sorted(out_dir.glob("S*.json")):
        with p.open() as f:
            r = json.load(f)
        results[r["patient"]] = r
    return results


def _verdict_table(results: dict[str, dict]) -> str:
    rows = ["| Patient | Verdict | Tier | Silent-suspect | Audio offset (s) | MAD (ms) |",
            "|---------|---------|------|----------------|------------------|----------|"]
    for patient in CORE_PATIENTS + EXTENDED_PATIENTS:
        if patient not in results:
            tier = "core" if patient in CORE_PATIENTS else "extended"
            rows.append(f"| {patient} | _(not run)_ | {tier} | — | — | — |")
            continue
        r = results[patient]
        tier = "core" if patient in CORE_PATIENTS else "extended"
        audio = r.get("metadata", {}).get("audio", {})
        bulk = audio.get("clock_fit_a")
        res_std = audio.get("clock_fit_residual_std_s")
        n_silent = audio.get("n_silent_suspect")
        n_trials = r.get("metadata", {}).get("n_epochs", "—")
        silent_cell = f"{n_silent}/{n_trials}" if n_silent is not None else "—"
        bulk_cell = f"{bulk:.3f}" if isinstance(bulk, (int, float)) else "—"
        res_cell = f"{res_std * 1000:.1f}" if isinstance(res_std, (int, float)) else "—"
        rows.append(
            f"| {patient} | **{r['verdict']}** | {tier} | {silent_cell} | {bulk_cell} | {res_cell} |"
        )
    return "\n".join(rows)


def _check_matrix(results: dict[str, dict]) -> str:
    if not results:
        return "_no per-patient results yet_"
    all_check_names: list[str] = []
    seen = set()
    for r in results.values():
        for c in r["checks"]:
            if c["name"] not in seen:
                all_check_names.append(c["name"])
                seen.add(c["name"])
    header = "| Check | " + " | ".join(results.keys()) + " |"
    sep = "|" + "---|" * (1 + len(results))
    rows = [header, sep]
    for name in all_check_names:
        cells = []
        for patient, r in results.items():
            found = next((c for c in r["checks"] if c["name"] == name), None)
            if found is None:
                cells.append("—")
            else:
                cells.append("✓" if found["passed"] else "✗")
        rows.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _closure_footer(results: dict[str, dict]) -> str:
    core_status = []
    for p in CORE_PATIENTS:
        if p not in results:
            core_status.append(f"- {p}: NOT RUN")
        else:
            v = results[p]["verdict"]
            core_status.append(f"- {p}: {v}")
    all_ok = all(
        p in results and results[p]["verdict"] in ("pass", "warn")
        for p in CORE_PATIENTS
    )
    banner = "**`#34` CLOSED**" if all_ok else "**`#34` still OPEN**"
    return banner + "\n\n" + "\n".join(core_status)


def main() -> int:
    out_dir = paths.reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = _load_results(out_dir)

    md = [
        "# `#34` phoneme-loading + trial-timing audit — rollup",
        "",
        f"Patients found: {', '.join(results.keys()) or '(none yet)'}.",
        "",
        "## Verdict table",
        "",
        _verdict_table(results),
        "",
        "## Per-check pass/fail matrix",
        "",
        _check_matrix(results),
        "",
        "## Closure rule",
        "",
        _closure_footer(results),
        "",
        "## Artifacts per patient",
        "",
        "- `{patient}.json` — full audit result",
        "- `{patient}_exclusion_candidates.csv` — silent-suspect trials to drop at loader time",
        "- `plots/{patient}_audio_alignment.png` — audio RMS envelope + per-trial offset distribution",
        "",
    ]

    out = out_dir / "README.md"
    out.write_text("\n".join(md))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
