"""A6: sEEG sensor geometry probe — inter-contact spacing per shaft.

Each D-patient's `ECoG_Recon/<D>/elec_recon/<D>.electrodeNames` and
`<D>_elec_locations_RAS_brainshifted.txt` give name + 3D coord per contact.
Shaft prefix is inferred by stripping the trailing digit run (e.g. `L1IF1` →
shaft `L1IF`, `RAIN12` → shaft `RAIN`).

For each shaft we report consecutive inter-contact Euclidean distance (mm) —
this is the relevant scale for per-electrode attention kernels (distance-bias
or Fourier MNI encoding).

Outputs:
    reports/seeg_sensor_geometry_2026_04_24/per_patient.csv
    reports/seeg_sensor_geometry_2026_04_24/per_shaft.csv
    reports/seeg_sensor_geometry_2026_04_24/summary.md
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


BOX_RECON = Path("/Users/bentang/Library/CloudStorage/Box-Box/ECoG_Recon")
OUT_DIR = Path("reports/seeg_sensor_geometry_2026_04_24")


SHAFT_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*?)(\d+)$")


def parse_shaft(name: str) -> tuple[str, int] | None:
    """Split electrode name into (shaft_prefix, contact_idx)."""
    m = SHAFT_RE.match(name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def load_patient(pt_dir: Path) -> tuple[list[str], np.ndarray] | None:
    pid = pt_dir.name
    ename_file = pt_dir / "elec_recon" / f"{pid}.electrodeNames"
    ras_file = pt_dir / "elec_recon" / f"{pid}_elec_locations_RAS_brainshifted.txt"
    if not ename_file.exists() or not ras_file.exists():
        return None
    enames = [
        line.strip().split()[0]
        for line in ename_file.read_text().splitlines()[2:]
        if line.strip()
    ]
    ras_lines = [line for line in ras_file.read_text().splitlines() if line.strip()]
    coords: list[tuple[float, float, float]] = []
    ras_names: list[str] = []
    for line in ras_lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        prefix = parts[0]
        num = parts[1]
        try:
            x, y, z = (float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        ras_names.append(f"{prefix}{num}")
        coords.append((x, y, z))
    # Build name -> coord using RAS file; then reorder to .electrodeNames
    name_to_xyz = {n: c for n, c in zip(ras_names, coords)}
    aligned: list[tuple[float, float, float] | None] = [name_to_xyz.get(n) for n in enames]
    if any(c is None for c in aligned):
        missing = [n for n, c in zip(enames, aligned) if c is None]
        # Tolerate: keep only contacts that resolved
        enames = [n for n, c in zip(enames, aligned) if c is not None]
        aligned = [c for c in aligned if c is not None]
        if missing[:3]:
            print(f"  {pid}: {len(missing)} contacts in .electrodeNames missing from RAS; "
                  f"sample {missing[:3]}")
    xyz = np.asarray(aligned, dtype=np.float64)
    return enames, xyz


def shaft_spacings(names: list[str], xyz: np.ndarray) -> dict[str, list[float]]:
    """Return {shaft: [consecutive mm distances]}."""
    buckets: dict[str, list[tuple[int, np.ndarray]]] = {}
    for i, n in enumerate(names):
        parsed = parse_shaft(n)
        if parsed is None:
            continue
        shaft, idx = parsed
        buckets.setdefault(shaft, []).append((idx, xyz[i]))
    out: dict[str, list[float]] = {}
    for shaft, entries in buckets.items():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda e: e[0])
        dists: list[float] = []
        for (i_a, x_a), (i_b, x_b) in zip(entries[:-1], entries[1:]):
            if i_b - i_a != 1:
                # non-consecutive: skip (gaps happen with rejected contacts)
                continue
            dists.append(float(np.linalg.norm(x_b - x_a)))
        if dists:
            out[shaft] = dists
    return out


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-root", type=Path, default=BOX_RECON)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--patients", nargs="+", default=None,
                    help="Override D-patient list (default: all D* under recon-root)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.patients:
        pt_dirs = [args.recon_root / p for p in args.patients]
    else:
        pt_dirs = sorted(
            p for p in args.recon_root.glob("D*")
            if p.is_dir() and "_" not in p.name and p.name[1:].isdigit()
        )
    print(f"auditing {len(pt_dirs)} D-patient recons under {args.recon_root}")

    per_shaft_rows: list[dict] = []
    per_patient_rows: list[dict] = []
    all_dists: list[float] = []

    for pt_dir in pt_dirs:
        pid = pt_dir.name
        loaded = load_patient(pt_dir)
        if loaded is None:
            print(f"  {pid}: skip (missing files)")
            continue
        names, xyz = loaded
        if xyz.size == 0:
            continue
        spacings = shaft_spacings(names, xyz)
        pt_dists: list[float] = []
        for shaft, dists in spacings.items():
            pt_dists.extend(dists)
            per_shaft_rows.append({
                "patient": pid, "shaft": shaft, "n_gaps": len(dists),
                "median_mm": round(float(np.median(dists)), 3),
                "min_mm": round(min(dists), 3),
                "max_mm": round(max(dists), 3),
                "p90_mm": round(pct(dists, 90), 3),
            })
        all_dists.extend(pt_dists)
        if pt_dists:
            per_patient_rows.append({
                "patient": pid,
                "n_contacts": len(names),
                "n_shafts": len(spacings),
                "n_gaps": len(pt_dists),
                "median_mm": round(float(np.median(pt_dists)), 3),
                "p90_mm": round(pct(pt_dists, 90), 3),
                "p95_mm": round(pct(pt_dists, 95), 3),
                "max_mm": round(max(pt_dists), 3),
            })
            print(f"  {pid}: shafts={len(spacings)} gaps={len(pt_dists)} "
                  f"median={per_patient_rows[-1]['median_mm']}mm "
                  f"p90={per_patient_rows[-1]['p90_mm']}mm")

    with (args.out_dir / "per_shaft.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient", "shaft", "n_gaps",
                                               "median_mm", "min_mm", "max_mm", "p90_mm"])
        writer.writeheader()
        writer.writerows(per_shaft_rows)

    with (args.out_dir / "per_patient.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient", "n_contacts", "n_shafts",
                                               "n_gaps", "median_mm", "p90_mm",
                                               "p95_mm", "max_mm"])
        writer.writeheader()
        writer.writerows(per_patient_rows)

    lines: list[str] = [
        "# A6 — sEEG inter-contact spacing (2026-04-24)",
        "",
        f"{len(per_patient_rows)} D-patients × all shafts, consecutive-contact distances on "
        "`<D>_elec_locations_RAS_brainshifted.txt`.",
        "",
        "## Pooled distribution (mm, consecutive contacts within a shaft)",
        "",
    ]
    if all_dists:
        lines += [
            f"- n = {len(all_dists)} consecutive-contact gaps across "
            f"{len(per_patient_rows)} patients",
            f"- median: **{round(float(np.median(all_dists)), 2)} mm**",
            f"- p90: {round(pct(all_dists, 90), 2)} mm",
            f"- p95: {round(pct(all_dists, 95), 2)} mm",
            f"- min: {round(min(all_dists), 2)} mm, max: {round(max(all_dists), 2)} mm",
            "",
            "## Implication for Stage-3 per-electrode attention",
            "",
            "Typical sEEG shaft inter-contact spacing (~3–5 mm) is order-of-magnitude",
            "similar to uECoG 128-strip spacing (~2 mm) but with much larger cross-shaft",
            "gaps. Any Gaussian distance-bias on per-electrode attention should use",
            "bandwidth ≥ p95 of the pooled spacing distribution so neighboring-contact",
            "attention is not suppressed by the kernel. Compare to the MV-BrainFM",
            "14→152 mm head specialization (`memory/reference_mv_brainfm_xu_2026_04.md`)",
            "— bandwidths in the 10–30 mm range cover the typical sEEG shaft; a",
            "long-range head (100+ mm) is still needed for cross-shaft structure.",
            "",
        ]
    (args.out_dir / "summary.md").write_text("\n".join(lines))

    print(f"\nwrote {args.out_dir}/per_shaft.csv ({len(per_shaft_rows)} rows)")
    print(f"wrote {args.out_dir}/per_patient.csv ({len(per_patient_rows)} rows)")
    print(f"wrote {args.out_dir}/summary.md")


if __name__ == "__main__":
    main()
