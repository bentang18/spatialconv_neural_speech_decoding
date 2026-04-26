"""B2: Per-D-patient RAS coordinate cache.

Parses `<D>_elec_locations_RAS_brainshifted.txt` and writes
`data/dcohort_coords/<D>_RAS.csv` with columns `name, x, y, z, hemisphere`
in `.electrodeNames` order.

RAS source format (whitespace-separated):
    shaft_prefix contact_idx x y z hemi D

Output CSV row order = the patient's `.electrodeNames` (skipping the 2-line
header) so a downstream consumer can key by row index matching the `.fif`
channel order.

Contacts in `.electrodeNames` that have no matching RAS entry are dropped
with a warning — matches the tolerance in `scripts/seeg_sensor_geometry.py`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BOX_RECON = Path("/Users/bentang/Library/CloudStorage/Box-Box/ECoG_Recon")
OUT_DIR = Path("data/dcohort_coords")


def parse_electrode_names(enames_file: Path) -> list[str]:
    return [
        line.strip().split()[0]
        for line in enames_file.read_text().splitlines()[2:]
        if line.strip()
    ]


def parse_ras(ras_file: Path) -> dict[str, tuple[float, float, float, str]]:
    out: dict[str, tuple[float, float, float, str]] = {}
    for line in ras_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        prefix = parts[0]
        num = parts[1]
        try:
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        hemi = parts[5]
        out[f"{prefix}{num}"] = (x, y, z, hemi)
    return out


def write_coord_csv(out_path: Path, names: list[str],
                    coords: list[tuple[float, float, float, str]]) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "x", "y", "z", "hemisphere"])
        for name, (x, y, z, h) in zip(names, coords):
            writer.writerow([name, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", h])


def build_one(pt_dir: Path, out_dir: Path) -> tuple[int, int] | None:
    """Returns (n_written, n_missing) or None if the patient is un-parseable."""
    pid = pt_dir.name
    enames_file = pt_dir / "elec_recon" / f"{pid}.electrodeNames"
    ras_file = pt_dir / "elec_recon" / f"{pid}_elec_locations_RAS_brainshifted.txt"
    if not enames_file.exists() or not ras_file.exists():
        return None
    names = parse_electrode_names(enames_file)
    ras = parse_ras(ras_file)
    aligned_names: list[str] = []
    aligned_coords: list[tuple[float, float, float, str]] = []
    n_missing = 0
    for n in names:
        entry = ras.get(n)
        if entry is None:
            n_missing += 1
            continue
        aligned_names.append(n)
        aligned_coords.append(entry)
    out_path = out_dir / f"{pid}_RAS.csv"
    write_coord_csv(out_path, aligned_names, aligned_coords)
    return len(aligned_names), n_missing


def discover_patients(recon_root: Path) -> list[str]:
    return sorted(
        p.name for p in recon_root.glob("D*")
        if p.is_dir() and "_" not in p.name and p.name[1:].isdigit()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-root", type=Path, default=BOX_RECON)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--patients", nargs="+", default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    patients = args.patients or discover_patients(args.recon_root)
    print(f"building RAS coord cache for {len(patients)} D-patients under {args.recon_root}")

    written: list[str] = []
    skipped: list[str] = []
    for pid in patients:
        result = build_one(args.recon_root / pid, args.out_dir)
        if result is None:
            skipped.append(pid)
            print(f"  {pid}: SKIP (missing RAS or electrodeNames)")
            continue
        n_written, n_missing = result
        written.append(pid)
        print(f"  {pid}: wrote {n_written} rows  (dropped {n_missing} contacts without RAS)")

    print(f"\nwrote {len(written)} coord CSVs to {args.out_dir}")
    if skipped:
        print(f"skipped {len(skipped)}: {', '.join(skipped[:20])}"
              f"{'...' if len(skipped) > 20 else ''}")


if __name__ == "__main__":
    main()
