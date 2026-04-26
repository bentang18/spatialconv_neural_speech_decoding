"""A3: Continuous corpus audit across 4 speech-task BIDS roots on Box.

Prefers BIDS JSON sidecars (`*_ieeg.json`) for duration — they carry
`SamplingFrequency`, `RecordingDuration`, and `SEEGChannelCount` already
(~hundreds of bytes vs. the 1-GB EDF). Falls back to an MNE EDF-header
read (`preload=False`) only if the JSON sidecar is missing.

The original EDF-header-only version stalled on Box-mount IO for PS+Lex
(~400 EDFs ≥ 1 GB each). JSON-first is ~1000× faster and avoids the
file-handle opens that cause the mount to hang.

Task roots on Box:
    BIDS-1.4_Phoneme_sequencing       (PS)          — `/BIDS/sub-D*/ieeg`
    BIDS-1.0_LexicalDecRepDelay       (LexDelay)    — `/BIDS/sub-D*/ieeg`
    BIDS-1.0_LexicalDecRepNoDelay     (LexNoDelay)  — `/BIDS/sub-D*/ieeg`
    BIDS-1.4_SentenceRep              (SentenceRep) — `/BIDS/sub-D*/ieeg`

Skips `ieeg/practice/` subdirs — those are pre-task practice recordings,
not the task corpus.

Outputs `reports/seeg_corpus_audit_2026_04_24/{per_run.csv,
per_patient.csv, summary.md}`. The summary lines get copied into
`docs/references/data_reference.md` §corpus so the Stage-3 ceiling sits
next to the 6.79 h uECoG number.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


BOX_ROOT = Path("/Users/bentang/Library/CloudStorage/Box-Box/CoganLab")
TASKS = {
    "PS":          BOX_ROOT / "BIDS-1.4_Phoneme_sequencing" / "BIDS",
    "LexDelay":    BOX_ROOT / "BIDS-1.0_LexicalDecRepDelay" / "BIDS",
    "LexNoDelay":  BOX_ROOT / "BIDS-1.0_LexicalDecRepNoDelay" / "BIDS",
    "SentenceRep": BOX_ROOT / "BIDS-1.4_SentenceRep" / "BIDS",
}
OUT_DIR = Path("reports/seeg_corpus_audit_2026_04_24")


def read_json_sidecar(edf: Path) -> tuple[float, float, int] | None:
    """Return (dur_s, sfreq, n_ch) from the matching `*_ieeg.json`, or None."""
    js = edf.with_suffix(".json")
    if not js.exists():
        return None
    try:
        meta = json.loads(js.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    sfreq = meta.get("SamplingFrequency")
    dur = meta.get("RecordingDuration")
    n_ch = meta.get("SEEGChannelCount") or meta.get("ECOGChannelCount") or 0
    if sfreq is None or dur is None:
        return None
    return float(dur), float(sfreq), int(n_ch)


def read_edf_header_dur(edf: Path) -> tuple[float, float, int]:
    """MNE EDF-header fallback — only called when JSON sidecar is absent."""
    import mne
    raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
    return int(raw.n_times) / float(raw.info["sfreq"]), float(raw.info["sfreq"]), len(raw.ch_names)


def _process_one_edf(label: str, bids_id: str, edf: Path) -> dict:
    """Worker: read the JSON sidecar, fall back to EDF header, return one row."""
    meta = read_json_sidecar(edf)
    if meta is not None:
        dur_s, sfreq, n_ch = meta
        return {
            "task": label, "bids_id": bids_id, "edf": edf.name,
            "dur_s": round(dur_s, 3), "sfreq": round(sfreq, 1),
            "n_ch": n_ch, "source": "json", "error": "",
        }
    try:
        dur_s, sfreq, n_ch = read_edf_header_dur(edf)
        return {
            "task": label, "bids_id": bids_id, "edf": edf.name,
            "dur_s": round(dur_s, 3), "sfreq": round(sfreq, 1),
            "n_ch": n_ch, "source": "edf", "error": "",
        }
    except Exception as err:  # noqa: BLE001
        return {
            "task": label, "bids_id": bids_id, "edf": edf.name,
            "dur_s": 0.0, "sfreq": 0.0, "n_ch": 0,
            "source": "error", "error": str(err)[:80],
        }


def audit_root(label: str, root: Path, n_workers: int = 16) -> list[dict]:
    if not root.exists():
        print(f"[{label}] MISSING root {root}", file=sys.stderr)
        return []
    patients = sorted(p for p in root.glob("sub-D*") if p.is_dir())
    print(f"[{label}] {len(patients)} D-patients under {root}", flush=True)
    t_start = time.time()

    # Enumerate all EDF paths up-front. Box IO is latency-bound (Path.glob()
    # serializes calls); parallelizing here is load-bearing.
    jobs: list[tuple[str, str, Path]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as tx:
        edf_lists = list(tx.map(
            lambda pt: (pt.name, sorted(pt.glob("ieeg/*_ieeg.edf"))),
            patients,
        ))
    for bids_id, edfs in edf_lists:
        for edf in edfs:
            jobs.append((label, bids_id, edf))

    print(f"[{label}] {len(jobs)} EDFs to probe", flush=True)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=n_workers) as tx:
        for row in tx.map(lambda j: _process_one_edf(*j), jobs):
            rows.append(row)

    n_json = sum(1 for r in rows if r["source"] == "json")
    n_edf_fallback = sum(1 for r in rows if r["source"] == "edf")
    n_fail = sum(1 for r in rows if r["source"] == "error")
    dt = time.time() - t_start
    print(
        f"[{label}] {len(rows)} runs  json={n_json}  edf_fallback={n_edf_fallback}  "
        f"failed={n_fail}  in {dt:.1f}s",
        flush=True,
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS.keys()))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for label in args.tasks:
        root = TASKS[label]
        all_rows.extend(audit_root(label, root))

    per_run = args.out_dir / "per_run.csv"
    with per_run.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "bids_id", "edf", "dur_s", "sfreq",
                        "n_ch", "source", "error"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # Per-patient × task rollup
    pt_rollup: dict[tuple[str, str], dict] = {}
    for r in all_rows:
        key = (r["task"], r["bids_id"])
        bucket = pt_rollup.setdefault(
            key, {"task": r["task"], "bids_id": r["bids_id"], "n_runs": 0,
                  "hours": 0.0, "sfreq_max": 0.0, "n_ch_max": 0},
        )
        bucket["n_runs"] += 1
        bucket["hours"] += r["dur_s"] / 3600.0
        bucket["sfreq_max"] = max(bucket["sfreq_max"], r["sfreq"])
        bucket["n_ch_max"] = max(bucket["n_ch_max"], r["n_ch"])

    per_pt = args.out_dir / "per_patient.csv"
    pt_rows = sorted(pt_rollup.values(), key=lambda r: (r["task"], r["bids_id"]))
    for r in pt_rows:
        r["hours"] = round(r["hours"], 3)
    with per_pt.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["task", "bids_id", "n_runs", "hours", "sfreq_max", "n_ch_max"],
        )
        writer.writeheader()
        writer.writerows(pt_rows)

    # Task-level rollup
    task_totals: dict[str, tuple[float, int]] = {}
    for label in args.tasks:
        rows = [r for r in pt_rows if r["task"] == label]
        task_totals[label] = (sum(r["hours"] for r in rows), len(rows))
    grand_hours = sum(h for h, _ in task_totals.values())
    union_patients = {r["bids_id"] for r in pt_rows}

    # Patient-level totals (union across tasks)
    pt_hours: dict[str, tuple[float, list[str]]] = {}
    for r in pt_rows:
        h, ts = pt_hours.get(r["bids_id"], (0.0, []))
        pt_hours[r["bids_id"]] = (h + r["hours"], ts + [r["task"]])

    lines: list[str] = [
        "# A3 — sEEG D-cohort continuous corpus audit (2026-04-24)",
        "",
        "JSON-sidecar-first scan of all 4 speech-task BIDS roots on Box.",
        "Reads `sub-D*/ieeg/*_ieeg.edf` (non-recursive → skips `practice/`).",
        f"Extracted duration from BIDS `*_ieeg.json` for {sum(1 for r in all_rows if r['source'] == 'json')} runs; "
        f"fell back to EDF header for {sum(1 for r in all_rows if r['source'] == 'edf')}; "
        f"{sum(1 for r in all_rows if r['source'] == 'error')} failures.",
        "",
        "## Per-task totals",
        "",
        "| task | n_patients | hours |",
        "|---|---:|---:|",
    ]
    for label, (hours, n_pt) in task_totals.items():
        lines.append(f"| {label} | {n_pt} | {round(hours, 2)} |")
    lines += [
        "",
        f"**Grand total**: {round(grand_hours, 2)} h across "
        f"**{len(union_patients)}** unique D-patients (union over tasks).",
        "",
        "For comparison (uECoG — see `docs/references/data_reference.md`): "
        "2.83 h PS + 3.96 h lexical = 6.79 h / 27 unique S-patients.",
        "",
        "## Top 10 D-patients by total sEEG hours",
        "",
        "| bids_id | total_hours | tasks |",
        "|---|---:|---|",
    ]
    for pid in sorted(pt_hours, key=lambda p: -pt_hours[p][0])[:10]:
        h, ts = pt_hours[pid]
        lines.append(f"| {pid} | {round(h, 2)} | {','.join(sorted(set(ts)))} |")
    lines.append("")
    summary = args.out_dir / "summary.md"
    summary.write_text("\n".join(lines))

    print(f"\nwrote {per_run}  ({len(all_rows)} rows)")
    print(f"wrote {per_pt}  ({len(pt_rows)} rows)")
    print(f"wrote {summary}")
    print("\nPer-task totals:")
    for label, (hours, n_pt) in task_totals.items():
        print(f"  {label:<12} {hours:>7.2f} h  ({n_pt} pts)")
    print(f"  {'GRAND':<12} {grand_hours:>7.2f} h  ({len(union_patients)} unique pts)")


if __name__ == "__main__":
    main()
