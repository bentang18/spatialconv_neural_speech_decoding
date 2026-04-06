#!/usr/bin/env python3
"""Diagnostic: data quality check across all patients.

Checks:
1. Sig channels — are .fif files pre-filtered or do they contain all channels?
2. Channel name matching — do .fif ch_names match electrode TSV names?
3. Coordinate system — are TSV coordinates MNI (mm) or normalized (0-1)?
4. Per-channel task modulation — empirical sig channel detection via variance ratio
5. Continuous recordings — do .edf files exist?
6. MNI coordinate files — any RAS/MNI files in BIDS structure?

Run on DCC:
    /work/ht203/miniconda3/envs/speech/bin/python scripts/diagnostic_data_quality.py \
        --bids-root /work/ht203/data/BIDS
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mne
import numpy as np


PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S39", "S57", "S58", "S62"]

# Expected sig channel counts from Spalding 2025 supplementary
# Spalding paper IDs → code IDs (approximate, some mappings uncertain)
EXPECTED_SIG = {
    "S14": 111,  # Spalding S1, 128-ch
    "S16": 111,  # Spalding S2, 128-ch (same count coincidence?)
    "S22": 149,  # Spalding S3, 256-ch — may be wrong mapping
    "S23": 74,   # Spalding S4, 128-ch
    "S26": 63,   # Spalding S5, 128-ch
    "S32": 144,  # Spalding S6, 256-ch
    "S33": 171,  # Spalding S7, 256-ch
    "S39": 201,  # Spalding S8, 256-ch
    # S57, S58, S62 — no Spalding data
}


def find_fif(bids_root: Path, subject: str) -> Path | None:
    deriv = bids_root / "derivatives" / "epoch(phonemeLevel)(CAR)"
    pattern = f"sub-{subject}_task-PhonemeSequence_desc-productionZscore_highgamma.fif"
    fif = deriv / f"sub-{subject}" / "epoch(band)(power)" / pattern
    return fif if fif.exists() else None


def find_electrodes_tsv(bids_root: Path, subject: str) -> Path | None:
    tsv = (
        bids_root / f"sub-{subject}" / "ieeg"
        / f"sub-{subject}_acq-01_space-ACPC_electrodes.tsv"
    )
    return tsv if tsv.exists() else None


def load_tsv_channels(tsv_path: Path) -> list[dict]:
    """Load electrode TSV, return list of {name, x, y, z?, ...}."""
    with open(tsv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def check_edf_files(bids_root: Path, subject: str) -> list[Path]:
    """Find all .edf files for a subject."""
    edfs = []
    # Check main BIDS folder
    main_ieeg = bids_root / f"sub-{subject}" / "ieeg"
    if main_ieeg.exists():
        edfs.extend(main_ieeg.glob("*.edf"))
    # Check derivatives
    deriv = bids_root / "derivatives"
    if deriv.exists():
        for d in deriv.iterdir():
            sub_ieeg = d / f"sub-{subject}" / "ieeg"
            if sub_ieeg.exists():
                edfs.extend(sub_ieeg.glob("*.edf"))
    return sorted(set(edfs))


def check_mni_files(bids_root: Path, subject: str) -> list[Path]:
    """Look for any MNI/RAS coordinate files."""
    mni_files = []
    sub_dir = bids_root / f"sub-{subject}"
    if sub_dir.exists():
        # Check for coordsystem JSON
        for f in sub_dir.rglob("*coordsystem*"):
            mni_files.append(f)
        # Check for any elec_locations files
        for f in sub_dir.rglob("*location*"):
            mni_files.append(f)
        for f in sub_dir.rglob("*RAS*"):
            mni_files.append(f)
        for f in sub_dir.rglob("*MNI*"):
            mni_files.append(f)
    # Check derivatives too
    deriv = bids_root / "derivatives"
    if deriv.exists():
        for d in deriv.iterdir():
            sub_d = d / f"sub-{subject}"
            if sub_d.exists():
                for f in sub_d.rglob("*coordsystem*"):
                    mni_files.append(f)
    return sorted(set(mni_files))


def compute_channel_modulation(epochs_data: np.ndarray, sfreq: float) -> np.ndarray:
    """Compute task modulation ratio per channel.

    Simple metric: variance in production window (0.0-0.5s) vs
    baseline window (-0.5 to -0.1s relative to epoch start).

    For productionZscore epochs (tmin~=-0.5, tmax~=0.5 at 200Hz):
    - Baseline: first ~80 samples (t < -0.1s)
    - Production: last ~100 samples (t > 0.0s)

    Returns ratio of production variance / baseline variance per channel.
    Higher = more task-modulated.
    """
    n_trials, n_ch, n_times = epochs_data.shape

    # Assume epoch spans roughly -0.5 to 0.5s (or similar)
    # Use first 1/3 as baseline, last 1/2 as production
    t_base_end = n_times // 3
    t_prod_start = n_times // 2

    baseline = epochs_data[:, :, :t_base_end]   # (trials, ch, T_base)
    production = epochs_data[:, :, t_prod_start:]  # (trials, ch, T_prod)

    # Mean variance across trials
    base_var = np.var(baseline, axis=(0, 2)) + 1e-10  # (n_ch,)
    prod_var = np.var(production, axis=(0, 2)) + 1e-10  # (n_ch,)

    return prod_var / base_var


def diagnose_patient(bids_root: Path, subject: str) -> dict:
    """Run all diagnostics for one patient."""
    result = {"subject": subject}

    # --- 1. Load .fif ---
    fif_path = find_fif(bids_root, subject)
    if fif_path is None:
        result["fif_found"] = False
        print(f"  {subject}: .fif NOT FOUND")
        return result
    result["fif_found"] = True

    epochs = mne.read_epochs(str(fif_path), preload=True, verbose=False)
    fif_ch_names = set(epochs.ch_names)
    n_fif_ch = len(fif_ch_names)
    data = epochs.get_data()  # (n_epochs, n_ch, T)

    result["n_fif_channels"] = n_fif_ch
    result["n_epochs"] = data.shape[0]
    result["n_times"] = data.shape[2]
    result["sfreq"] = epochs.info["sfreq"]
    result["tmin"] = epochs.tmin
    result["tmax"] = epochs.tmax

    print(f"\n{'='*60}")
    print(f"  {subject}")
    print(f"{'='*60}")
    print(f"  .fif: {n_fif_ch} channels, {data.shape[0]} epochs, "
          f"T={data.shape[2]} ({epochs.tmin:.3f}s to {epochs.tmax:.3f}s), "
          f"sfreq={epochs.info['sfreq']:.0f}Hz")

    # --- 2. Load electrode TSV ---
    tsv_path = find_electrodes_tsv(bids_root, subject)
    if tsv_path is None:
        print(f"  TSV: NOT FOUND")
        result["tsv_found"] = False
    else:
        result["tsv_found"] = True
        tsv_rows = load_tsv_channels(tsv_path)
        tsv_names = set(r["name"] for r in tsv_rows)
        n_tsv_ch = len(tsv_names)

        # Check coordinate ranges
        coords_x = [float(r["x"]) for r in tsv_rows if r.get("x", "n/a") != "n/a"]
        coords_y = [float(r["y"]) for r in tsv_rows if r.get("y", "n/a") != "n/a"]
        has_z = any(r.get("z", "n/a") != "n/a" for r in tsv_rows)
        coords_z = [float(r["z"]) for r in tsv_rows if r.get("z", "n/a") != "n/a"] if has_z else []

        # Check if coordinates look like MNI (mm, range ~-80 to +80) or normalized (0-1)
        x_range = (min(coords_x), max(coords_x)) if coords_x else (0, 0)
        y_range = (min(coords_y), max(coords_y)) if coords_y else (0, 0)
        z_range = (min(coords_z), max(coords_z)) if coords_z else (0, 0)

        is_mni = abs(x_range[1]) > 10 or abs(y_range[1]) > 10  # MNI coords > 10mm
        coord_type = "MNI (mm)" if is_mni else "Normalized (0-1)"

        result["n_tsv_channels"] = n_tsv_ch
        result["coord_type"] = coord_type
        result["x_range"] = x_range
        result["y_range"] = y_range
        result["z_range"] = z_range
        result["has_z"] = has_z

        # TSV columns
        all_cols = list(tsv_rows[0].keys()) if tsv_rows else []
        print(f"  TSV: {n_tsv_ch} channels, columns={all_cols}")
        print(f"  Coords: {coord_type}")
        print(f"    x: [{x_range[0]:.4f}, {x_range[1]:.4f}]")
        print(f"    y: [{y_range[0]:.4f}, {y_range[1]:.4f}]")
        if has_z:
            print(f"    z: [{z_range[0]:.4f}, {z_range[1]:.4f}]")

        # Print first 3 rows for inspection
        print(f"  First 3 TSV rows:")
        for row in tsv_rows[:3]:
            print(f"    {row}")

        # Channel name matching
        in_both = fif_ch_names & tsv_names
        fif_only = fif_ch_names - tsv_names
        tsv_only = tsv_names - fif_ch_names

        result["ch_in_both"] = len(in_both)
        result["ch_fif_only"] = len(fif_only)
        result["ch_tsv_only"] = len(tsv_only)

        print(f"  Channel matching:")
        print(f"    In both .fif and TSV: {len(in_both)}")
        print(f"    In .fif only: {len(fif_only)}")
        if fif_only:
            print(f"      Names: {sorted(fif_only)[:10]}{'...' if len(fif_only) > 10 else ''}")
        print(f"    In TSV only: {len(tsv_only)}")
        if tsv_only:
            print(f"      Names: {sorted(tsv_only)[:10]}{'...' if len(tsv_only) > 10 else ''}")

    # --- 3. Sig channel analysis ---
    expected = EXPECTED_SIG.get(subject)
    if expected:
        print(f"  Expected sig channels (Spalding): {expected}")
        print(f"  .fif has: {n_fif_ch} → {'MATCHES' if n_fif_ch == expected else 'MISMATCH'} "
              f"({'pre-filtered' if n_fif_ch == expected else 'likely ALL channels'})")

    # Compute empirical modulation ratio
    modulation = compute_channel_modulation(data, epochs.info["sfreq"])
    high_mod = np.sum(modulation > 1.5)  # channels with 50%+ more production variance
    low_mod = np.sum(modulation < 1.1)   # channels with <10% more production variance

    result["modulation_high"] = int(high_mod)
    result["modulation_low"] = int(low_mod)
    result["modulation_median"] = float(np.median(modulation))
    result["modulation_mean"] = float(np.mean(modulation))

    print(f"  Task modulation (prod_var / base_var):")
    print(f"    Median: {np.median(modulation):.3f}, Mean: {np.mean(modulation):.3f}")
    print(f"    High (>1.5): {high_mod}/{n_fif_ch}")
    print(f"    Low (<1.1):  {low_mod}/{n_fif_ch}")

    # Sort channels by modulation and show top/bottom 5
    ch_mod = list(zip(epochs.ch_names, modulation))
    ch_mod.sort(key=lambda x: x[1], reverse=True)
    print(f"    Top 5:    {[(n, f'{m:.2f}') for n, m in ch_mod[:5]]}")
    print(f"    Bottom 5: {[(n, f'{m:.2f}') for n, m in ch_mod[-5:]]}")

    # --- 4. Check for .edf files ---
    edfs = check_edf_files(bids_root, subject)
    result["n_edf_files"] = len(edfs)
    if edfs:
        # Get file sizes
        edf_info = [(p.name, p.stat().st_size / 1e6) for p in edfs]
        print(f"  EDF files: {len(edfs)}")
        for name, size_mb in edf_info:
            print(f"    {name}: {size_mb:.1f} MB")
    else:
        print(f"  EDF files: NONE FOUND")

    # --- 5. Check for MNI/RAS files ---
    mni_files = check_mni_files(bids_root, subject)
    result["mni_files"] = [str(f) for f in mni_files]
    if mni_files:
        print(f"  MNI/coordsystem files:")
        for f in mni_files:
            print(f"    {f}")
            # If it's a small text/json file, print contents
            if f.suffix in (".json", ".txt", ".tsv") and f.stat().st_size < 5000:
                try:
                    content = f.read_text(encoding="utf-8-sig")
                    for line in content.strip().split("\n")[:10]:
                        print(f"      {line}")
                except Exception:
                    pass
    else:
        print(f"  MNI/coordsystem files: NONE FOUND")

    return result


def main():
    parser = argparse.ArgumentParser(description="Data quality diagnostic")
    parser.add_argument("--bids-root", required=True, help="Path to BIDS root")
    parser.add_argument("--patients", nargs="*", default=PATIENTS, help="Patient IDs")
    args = parser.parse_args()

    bids_root = Path(args.bids_root)
    if not bids_root.exists():
        print(f"ERROR: BIDS root not found: {bids_root}")
        sys.exit(1)

    # Also list top-level BIDS structure
    print("=" * 60)
    print("BIDS ROOT STRUCTURE")
    print("=" * 60)
    for item in sorted(bids_root.iterdir()):
        if item.is_dir():
            print(f"  {item.name}/")
            # Show 1 level deeper
            for sub in sorted(item.iterdir())[:5]:
                print(f"    {sub.name}")
            remaining = len(list(item.iterdir())) - 5
            if remaining > 0:
                print(f"    ... +{remaining} more")

    print("\n")

    results = []
    for patient in args.patients:
        try:
            r = diagnose_patient(bids_root, patient)
            results.append(r)
        except Exception as e:
            print(f"\n  {patient}: ERROR — {e}")
            results.append({"subject": patient, "error": str(e)})

    # Summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Patient':>8} {'fif_ch':>7} {'tsv_ch':>7} {'match':>6} {'expected':>9} "
          f"{'pre-filt?':>10} {'mod_med':>8} {'hi_mod':>7} {'lo_mod':>7} {'edfs':>5} {'coord':>12}")
    print("-" * 80)

    for r in results:
        if r.get("error"):
            print(f"{r['subject']:>8} ERROR: {r['error'][:50]}")
            continue
        if not r.get("fif_found"):
            print(f"{r['subject']:>8} .fif NOT FOUND")
            continue

        exp = EXPECTED_SIG.get(r["subject"], "?")
        pre_filt = "YES" if r["n_fif_channels"] == exp else "NO" if isinstance(exp, int) else "?"

        print(f"{r['subject']:>8} {r['n_fif_channels']:>7} {r.get('n_tsv_channels', '?'):>7} "
              f"{r.get('ch_in_both', '?'):>6} {str(exp):>9} {pre_filt:>10} "
              f"{r.get('modulation_median', 0):>8.3f} {r.get('modulation_high', '?'):>7} "
              f"{r.get('modulation_low', '?'):>7} {r.get('n_edf_files', 0):>5} "
              f"{r.get('coord_type', '?'):>12}")


if __name__ == "__main__":
    main()
