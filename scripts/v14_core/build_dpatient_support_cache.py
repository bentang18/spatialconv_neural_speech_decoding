"""B1: Build per-D-patient Tier-1 support cache from pre-baked 3mm BNA CSVs.

Matches the existing uECoG support-cache contract
(`src/speech_decoding/v14/support_cache.py:TIER1_COLUMNS`) exactly:
    header = ("name",) + TIER1_COLUMNS  # 1 + 15 cols
    row order = `.electrodeNames` sequence
    values = raw BNA probability in [0, 100], float32, 6-dec precision

Source per patient:
    ECoG_Recon/<D>/elec_recon/<D>_elec_location_radius_3mm_aparc.BN_atlas+aseg.mgz.csv

Outputs:
    data/atlas/support_cache_v2c_snap_dcohort/<D>_support_tier1.csv
    docs/qc/dcohort_support_cache_qc_2026_04_24.md (cohort-wide QC report)

The 3mm voxel-count recipe is parity-verified (reports/bna_parity_dpatients_
2026_04_21/): 74% Tier-1 argmax agreement with our fsaverage snap; pre-baked
CSVs are the Stage-3 source-of-truth for D-patient support.

For contacts with no Tier-1 hit (argmax Unknown or non-Tier-1), the row is
all-zero. This matches the Phase-1 contract — electrodes with
`max over Tier-1 == 0` are still emitted, their embedding contribution is zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from speech_decoding.v14.support_cache import (
    TIER1_COLUMNS,
    PatientQC,
    qc_row_for_patient,
    write_qc_report,
    write_support_cache,
)
from speech_decoding.v14.token_spec import DEFAULT_BASE_PARCELS


BOX_RECON = Path("/Users/bentang/Library/CloudStorage/Box-Box/ECoG_Recon")
OUT_CACHE_DIR = Path("data/atlas/support_cache_v2c_snap_dcohort")
QC_PATH = Path("docs/qc/dcohort_support_cache_qc_2026_04_24.md")

# Tier-1 lookup: column-order names with LH-only suffix (Phase-1 15 parcels)
TIER1_LOOKUP_KEYS: tuple[str, ...] = tuple(f"{name}_L" for name, _ in DEFAULT_BASE_PARCELS)


def parse_electrode_names(enames_file: Path) -> list[str]:
    """Skip 2 header lines (date + column header)."""
    return [
        line.strip().split()[0]
        for line in enames_file.read_text().splitlines()[2:]
        if line.strip()
    ]


def parse_bna_csv(csv_path: Path) -> dict[str, dict[str, float]]:
    """Parse 3mm BNA CSV → {electrode_name: {label: proportion}}.

    CSV row: `argmax_label, electrode_name, label_1, prop_1, label_2, prop_2, ...`
    Proportions in [0, 1].
    """
    out: dict[str, dict[str, float]] = {}
    for raw_line in csv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[1]:
            continue
        name = parts[1]
        tail = parts[2:]
        props: dict[str, float] = {}
        # Pairs: label_i, prop_i
        for i in range(0, len(tail) - 1, 2):
            label = tail[i]
            prop_str = tail[i + 1]
            if not label or not prop_str:
                continue
            try:
                props[label] = float(prop_str)
            except ValueError:
                continue
        out[name] = props
    return out


def build_patient_support_from_csv(
    enames_file: Path,
    bna_csv: Path,
) -> tuple[tuple[str, ...], np.ndarray, int]:
    """Return (names, support[N_e, 15] float32 in [0, 100], n_missing_in_csv)."""
    names = parse_electrode_names(enames_file)
    by_name = parse_bna_csv(bna_csv)
    n_e = len(names)
    support = np.zeros((n_e, len(TIER1_COLUMNS)), dtype=np.float32)
    n_missing = 0
    for row_i, nm in enumerate(names):
        props = by_name.get(nm)
        if props is None:
            n_missing += 1
            continue
        for col_i, key in enumerate(TIER1_LOOKUP_KEYS):
            if key in props:
                # Scale [0, 1] -> [0, 100] to match uECoG support_cache convention
                support[row_i, col_i] = float(props[key]) * 100.0
    return tuple(names), support, n_missing


def discover_patients(recon_root: Path) -> list[str]:
    return sorted(
        p.name for p in recon_root.glob("D*")
        if p.is_dir() and "_" not in p.name and p.name[1:].isdigit()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-root", type=Path, default=BOX_RECON)
    ap.add_argument("--out-dir", type=Path, default=OUT_CACHE_DIR)
    ap.add_argument("--qc-path", type=Path, default=QC_PATH)
    ap.add_argument("--patients", nargs="+", default=None,
                    help="Override D-patient list (default: all D* under recon-root)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.qc_path.parent.mkdir(parents=True, exist_ok=True)

    patients = args.patients or discover_patients(args.recon_root)
    print(f"building support cache for {len(patients)} D-patients under {args.recon_root}")

    qc_rows: list[PatientQC] = []
    skipped: list[str] = []
    written: list[str] = []

    for pid in patients:
        pt_dir = args.recon_root / pid
        enames_file = pt_dir / "elec_recon" / f"{pid}.electrodeNames"
        bna_csv = (
            pt_dir / "elec_recon"
            / f"{pid}_elec_location_radius_3mm_aparc.BN_atlas+aseg.mgz.csv"
        )
        if not enames_file.exists():
            print(f"  {pid}: SKIP (no .electrodeNames)")
            skipped.append(pid)
            continue
        if not bna_csv.exists():
            print(f"  {pid}: SKIP (no 3mm BNA csv)")
            skipped.append(pid)
            continue

        names, support, n_missing = build_patient_support_from_csv(enames_file, bna_csv)
        out_path = args.out_dir / f"{pid}_support_tier1.csv"
        write_support_cache(out_path, names, support)

        qc = qc_row_for_patient(pid, support)
        qc_rows.append(qc)
        written.append(pid)
        print(
            f"  {pid}: n_elec={qc.n_electrodes}  tier1_ch={qc.argmax_in_tier1}  "
            f"low_mass={qc.low_tier1_mass}  missing_in_csv={n_missing}"
        )

    write_qc_report(args.qc_path, qc_rows)
    print(f"\nwrote {len(written)} caches to {args.out_dir}")
    print(f"wrote QC report to {args.qc_path}")
    if skipped:
        print(f"skipped {len(skipped)} patients: {', '.join(skipped[:20])}"
              f"{'...' if len(skipped) > 20 else ''}")


if __name__ == "__main__":
    main()
