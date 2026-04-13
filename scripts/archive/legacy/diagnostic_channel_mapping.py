#!/usr/bin/env python3
"""Verify .fif channel names vs RAS electrode numbers for all patients.

The critical question: do .fif channel names match RAS physical electrode numbers?
If yes: direct RAS lookup works. If no: chanMap bridge needed.

Run on DCC:
    /work/ht203/miniconda3/envs/speech/bin/python scripts/diagnostic_channel_mapping.py \
        --bids-root /work/ht203/data/BIDS --ras-dir /work/ht203/data/mni_coords
"""
from __future__ import annotations
import argparse
from pathlib import Path
import mne

PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S39", "S57", "S58", "S62"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--ras-dir", required=True)
    args = parser.parse_args()

    bids_root = Path(args.bids_root)
    ras_dir = Path(args.ras_dir)

    print(f"{'Patient':>8} {'fif_ch':>7} {'RAS_el':>7} {'fif_range':>15} {'RAS_range':>15} "
          f"{'overlap':>8} {'fif_only':>9} {'RAS_only':>9} {'match?':>7}")
    print("-" * 95)

    for subj in PATIENTS:
        # Load .fif channel names
        fif_path = (bids_root / "derivatives" / "epoch(phonemeLevel)(CAR)" /
                    f"sub-{subj}" / "epoch(band)(power)" /
                    f"sub-{subj}_task-PhonemeSequence_desc-productionZscore_highgamma.fif")
        if not fif_path.exists():
            print(f"{subj:>8} FIF NOT FOUND")
            continue

        epochs = mne.read_epochs(str(fif_path), preload=False, verbose=False)
        fif_names = set(epochs.ch_names)
        fif_sorted = sorted(epochs.ch_names, key=lambda x: int(x))

        # Load RAS electrode numbers
        ras_path = ras_dir / f"{subj}_RAS.txt"
        if not ras_path.exists():
            print(f"{subj:>8} RAS NOT FOUND")
            continue

        ras_nums = set()
        with open(ras_path) as f:
            for line in f:
                parts = line.strip().split()
                ras_nums.add(parts[1])  # Keep as string for comparison

        overlap = fif_names & ras_nums
        fif_only = fif_names - ras_nums
        ras_only = ras_nums - fif_names

        fif_range = f"[{fif_sorted[0]}-{fif_sorted[-1]}]"
        ras_sorted = sorted(ras_nums, key=lambda x: int(x))
        ras_range = f"[{ras_sorted[0]}-{ras_sorted[-1]}]"

        match = "YES" if len(overlap) == len(fif_names) == len(ras_nums) else "PARTIAL" if overlap else "NO"

        print(f"{subj:>8} {len(fif_names):>7} {len(ras_nums):>7} {fif_range:>15} {ras_range:>15} "
              f"{len(overlap):>8} {len(fif_only):>9} {len(ras_only):>9} {match:>7}")

        # Show first few mismatches
        if fif_only and len(fif_only) <= 10:
            print(f"         fif_only: {sorted(fif_only, key=lambda x: int(x))}")
        if ras_only and len(ras_only) <= 10:
            print(f"         ras_only: {sorted(ras_only, key=lambda x: int(x))}")
        if fif_only and len(fif_only) > 10:
            fif_only_sorted = sorted(fif_only, key=lambda x: int(x))
            print(f"         fif_only (first 5): {fif_only_sorted[:5]}... ({len(fif_only)} total)")
        if ras_only and len(ras_only) > 10:
            ras_only_sorted = sorted(ras_only, key=lambda x: int(x))
            print(f"         ras_only (first 5): {ras_only_sorted[:5]}... ({len(ras_only)} total)")


if __name__ == "__main__":
    main()
