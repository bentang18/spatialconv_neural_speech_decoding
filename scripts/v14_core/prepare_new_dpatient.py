"""B3: Cohort-wide D-patient prep orchestrator + manifest writer.

Joins the outputs of A1 (sig-channel), A2 (events + muscle), A3 (corpus
hours — optional, written when available), A4 (z-score distribution),
A6 (sensor geometry), B1 (support cache), and B2 (coord cache) into a
single manifest row per D-patient.

Idempotent: reruns produce a byte-identical `data/dcohort_manifest.csv`
given the same inputs.

Columns (one row per D-patient, keyed by short ID e.g. `D40`):

- patient               — short ID
- bids_id               — full `D0040`-style
- n_elec_support_cache  — rows in `data/atlas/support_cache_v2c_snap_dcohort/<D>_support_tier1.csv`
- n_elec_coord_cache    — rows in `data/dcohort_coords/<D>_RAS.csv`
- argmax_in_tier1       — electrodes whose Tier-1 argmax > 0 (from B1 QC)
- low_tier1_mass        — electrodes where max-Tier-1 support < 20 (QC)
- lh_tier1              — LH Tier-1 count (from A1 coverage row, if PS)
- rh_tier1              — RH Tier-1 count (ditto)
- n_sig_resp_hg         — PS resp-phase sig-channels (A1)
- frac_sig_resp         — PS resp-phase sig fraction (A1)
- usability_flag        — ok / low / likely_bad / unknown (A1)
- n_trials_audio        — PS `Audio` event count (A2)
- n_unique_tokens       — distinct `stim_file` tokens (A2)
- n_oov_tokens          — count not in the 52-token PS manifest
- events_format         — `bids` / `legacy_matlab` / `missing` (A2)
- n_muscle_ch           — muscle artifact channels (A2)
- hours_ps              — continuous-corpus PS hours (A3; blank if audit not complete)
- zscore_ok_baseline    — A4 verdict: `True` / `False` / `unknown`
- zscore_ok_prod        — A4 verdict
- median_spacing_mm     — median inter-contact mm (A6)
- p95_spacing_mm        — p95 inter-contact mm (A6)
- status                — ready / partial / missing_support / missing_coord

A patient is `ready` iff both B1 and B2 caches exist. `partial` means one
is present but the other is missing. Phase-A audit fields can be empty
without blocking `ready` — the caches are the hard prerequisites.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


SUPPORT_DIR = Path("data/atlas/support_cache_v2c_snap_dcohort")
COORD_DIR = Path("data/dcohort_coords")
MANIFEST_PATH = Path("data/dcohort_manifest.csv")

A1_CSV = Path("reports/seeg_sig_channels_2026_04_24/per_patient.csv")
A2_CSV = Path("reports/seeg_events_audit_2026_04_24/per_patient.csv")
A3_CSV = Path("reports/seeg_corpus_audit_2026_04_24/per_patient.csv")
A4_JSON = Path("reports/seeg_zscore_recipe_2026_04_24/per_patient.json")
A6_CSV = Path("reports/seeg_sensor_geometry_2026_04_24/per_patient.csv")


MANIFEST_COLUMNS = [
    "patient", "bids_id",
    "n_elec_support_cache", "n_elec_coord_cache",
    "argmax_in_tier1", "low_tier1_mass",
    "lh_tier1", "rh_tier1",
    "n_sig_resp_hg", "frac_sig_resp", "usability_flag",
    "n_trials_audio", "n_unique_tokens", "n_oov_tokens",
    "events_format", "n_muscle_ch",
    "hours_ps",
    "zscore_ok_baseline", "zscore_ok_prod",
    "median_spacing_mm", "p95_spacing_mm",
    "status",
]


def short_to_bids(pid: str) -> str:
    """`D40` → `D0040`, `D7` → `D0007`, `D126` → `D0126`."""
    m = re.match(r"^D(\d+)$", pid)
    if not m:
        return pid
    return f"D{int(m.group(1)):04d}"


def load_csv_by(bids_key: str, path: Path) -> dict[str, dict[str, str]]:
    """Read a CSV and key by the value of `bids_key` column."""
    if not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get(bids_key, "")
            # tolerate `sub-D0040` vs `D0040` by stripping `sub-`
            if key.startswith("sub-"):
                key = key[4:]
            out[key] = row
    return out


def load_a3_ps(path: Path) -> dict[str, dict[str, str]]:
    """A3 per_patient.csv has one row per (task, bids_id). Keep only PS rows."""
    if not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("task") != "PS":
                continue
            key = row.get("bids_id", "")
            if key.startswith("sub-"):
                key = key[4:]
            out[key] = row
    return out


def load_a4(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {r["bids_id"]: r for r in data}


def discover_support_patients(support_dir: Path) -> list[str]:
    pids: list[str] = []
    for p in support_dir.glob("D*_support_tier1.csv"):
        pid = p.name.replace("_support_tier1.csv", "")
        pids.append(pid)
    return sorted(pids, key=lambda p: int(p[1:]) if p[1:].isdigit() else 0)


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f) - 1  # minus header


def qc_tier1_counts(support_csv: Path) -> tuple[int, int]:
    """Count rows where argmax > 0 (`argmax_in_tier1`) and where
    max-support < 20 among argmax_in_tier1 rows (`low_tier1_mass`)."""
    if not support_csv.exists():
        return 0, 0
    argmax_any = 0
    low_mass = 0
    with support_csv.open() as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            vals = [float(v) for v in row[1:] if v]
            if not vals:
                continue
            m = max(vals)
            if m > 0:
                argmax_any += 1
                if m < 20.0:
                    low_mass += 1
    return argmax_any, low_mass


def manifest_row(
    pid: str,
    a1: dict[str, dict[str, str]],
    a2: dict[str, dict[str, str]],
    a3: dict[str, dict[str, str]],
    a4: dict[str, dict],
    a6: dict[str, dict[str, str]],
) -> dict[str, str]:
    bids_id = short_to_bids(pid)
    support_csv = SUPPORT_DIR / f"{pid}_support_tier1.csv"
    coord_csv = COORD_DIR / f"{pid}_RAS.csv"

    n_support = count_csv_rows(support_csv)
    n_coord = count_csv_rows(coord_csv)
    argmax, low_mass = qc_tier1_counts(support_csv)

    a1_row = a1.get(bids_id, {})
    a2_row = a2.get(bids_id, {})
    a3_row = a3.get(bids_id, {})
    a4_row = a4.get(bids_id, {})
    a6_row = a6.get(pid, {})  # A6 keyed by short ID

    if n_support > 0 and n_coord > 0:
        status = "ready"
    elif n_support > 0 or n_coord > 0:
        status = "partial"
    elif n_coord == 0:
        status = "missing_coord"
    else:
        status = "missing_support"

    # A4 verdicts
    zb = a4_row.get("looks_raw_baseline")
    zp = a4_row.get("looks_zscored_production")
    zb_str = "True" if zb else ("False" if zb is False else "unknown")
    zp_str = "True" if zp else ("False" if zp is False else "unknown")

    return {
        "patient": pid,
        "bids_id": bids_id,
        "n_elec_support_cache": str(n_support),
        "n_elec_coord_cache": str(n_coord),
        "argmax_in_tier1": str(argmax),
        "low_tier1_mass": str(low_mass),
        "lh_tier1": a1_row.get("lh_tier1", ""),
        "rh_tier1": a1_row.get("rh_tier1", ""),
        "n_sig_resp_hg": a1_row.get("n_sig_resp", ""),
        "frac_sig_resp": a1_row.get("frac_sig_resp", ""),
        "usability_flag": a1_row.get("usability_flag", "unknown"),
        "n_trials_audio": a2_row.get("n_audio", ""),
        "n_unique_tokens": a2_row.get("n_unique_stim", ""),
        "n_oov_tokens": a2_row.get("n_stim_out_of_vocab", ""),
        "events_format": a2_row.get("format", "missing"),
        "n_muscle_ch": a2_row.get("n_muscle_ch", ""),
        "hours_ps": a3_row.get("hours", ""),
        "zscore_ok_baseline": zb_str,
        "zscore_ok_prod": zp_str,
        "median_spacing_mm": a6_row.get("median_mm", ""),
        "p95_spacing_mm": a6_row.get("p95_mm", ""),
        "status": status,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--support-dir", type=Path, default=SUPPORT_DIR)
    ap.add_argument("--coord-dir", type=Path, default=COORD_DIR)
    ap.add_argument("--out-csv", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--a1", type=Path, default=A1_CSV)
    ap.add_argument("--a2", type=Path, default=A2_CSV)
    ap.add_argument("--a3", type=Path, default=A3_CSV)
    ap.add_argument("--a4", type=Path, default=A4_JSON)
    ap.add_argument("--a6", type=Path, default=A6_CSV)
    args = ap.parse_args()

    a1 = load_csv_by("bids_id", args.a1)
    a2 = load_csv_by("bids_id", args.a2)
    a3 = load_a3_ps(args.a3)  # filter to PS task only
    a4 = load_a4(args.a4)
    a6 = load_csv_by("patient", args.a6)

    # Union patients: any D in support cache, coord cache, or A-audits
    patients = set(discover_support_patients(args.support_dir))
    for p in args.coord_dir.glob("D*_RAS.csv"):
        patients.add(p.name.replace("_RAS.csv", ""))
    # Also pull from A1 so we notice audit-only patients with no cache (rare)
    for bids_id in a1:
        m = re.match(r"^D(\d+)$", bids_id)
        if m:
            patients.add(f"D{int(m.group(1))}")

    patients_sorted = sorted(patients, key=lambda p: int(p[1:]) if p[1:].isdigit() else 0)

    rows = [manifest_row(p, a1, a2, a3, a4, a6) for p in patients_sorted]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    ready = sum(1 for r in rows if r["status"] == "ready")
    partial = sum(1 for r in rows if r["status"] == "partial")
    print(f"wrote {args.out_csv} — {len(rows)} D-patients total, "
          f"{ready} ready, {partial} partial")
    if not args.a3.exists():
        print(f"  note: A3 corpus audit not yet complete — hours_ps blank "
              f"({args.a3}). Rerun this script after A3 finishes to populate.")


if __name__ == "__main__":
    main()
