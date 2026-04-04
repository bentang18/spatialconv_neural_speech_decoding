#!/usr/bin/env python3
"""Diagnose per-phoneme epoch alignment quality across patients.

For each patient, checks:
1. Whether the fif epoch interleaving matches our assumption [p1_t1, p2_t1, p3_t1, ...]
2. Whether phoneme timing CSV exists and has matching trial count
3. Whether phoneme onset times from CSV fall within the epoch window
4. Summary statistics of timing alignment quality

Usage:
  python scripts/diagnose_epoch_alignment.py --paths configs/paths.yaml
  python scripts/diagnose_epoch_alignment.py --paths configs/paths.yaml --patient S14
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import mne
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PS_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S36", "S39", "S57", "S58", "S62"]


def load_fif_epochs(bids_root, subject):
    """Load raw MNE epochs and return event structure."""
    deriv = Path(bids_root) / "derivatives" / "epoch(phonemeLevel)(CAR)"
    pattern = f"sub-{subject}_task-PhonemeSequence_desc-productionZscore_highgamma.fif"
    fif_path = deriv / f"sub-{subject}" / "epoch(band)(power)" / pattern
    if not fif_path.exists():
        return None, None
    epochs = mne.read_epochs(str(fif_path), preload=False, verbose=False)
    return epochs, fif_path


def load_phoneme_csv(bids_root, subject):
    """Load phoneme timing CSV if it exists."""
    path = (
        Path(bids_root) / "derivatives" / "phoneme" / f"sub-{subject}"
        / "event" / f"sub-{subject}_task-phoneme_acq-01_run-01_desc-production_events.csv"
    )
    if not path.exists():
        return None, path

    rows_by_trial = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            trial = int(row["trial"])
            rows_by_trial.setdefault(trial, []).append(row)
    return rows_by_trial, path


def diagnose_patient(bids_root, subject):
    """Run full alignment diagnostic for one patient."""
    print(f"\n{'='*70}")
    print(f"  {subject}")
    print(f"{'='*70}")

    # --- 1. Load fif epochs ---
    epochs, fif_path = load_fif_epochs(bids_root, subject)
    if epochs is None:
        print(f"  [SKIP] No .fif file found")
        return

    all_event_ids = epochs.events[:, 2]
    event_samples = epochs.events[:, 0]  # sample numbers on 2kHz clock
    inv_event_id = {v: k for k, v in epochs.event_id.items()}
    n_total = len(all_event_ids)
    n_phons = 3

    print(f"  .fif: {n_total} epochs, event_id map: {epochs.event_id}")
    print(f"  Epoch tmin={epochs.tmin:.3f}s, tmax={epochs.tmax:.3f}s")

    if n_total % n_phons != 0:
        print(f"  [ERROR] {n_total} epochs not divisible by {n_phons}!")
        return

    n_trials = n_total // n_phons

    # --- 2. Check interleaving pattern ---
    # Our assumption: [p1_t1, p2_t1, p3_t1, p1_t2, p2_t2, p3_t2, ...]
    # Check: are event IDs within each triplet different (3 distinct phonemes)?
    # And are event samples within each triplet close together (same trial)?
    triplet_issues = 0
    timing_gaps = []

    for t in range(min(n_trials, 10)):  # check first 10 trials
        idx_start = t * n_phons
        triplet_ids = [inv_event_id[all_event_ids[idx_start + j]] for j in range(n_phons)]
        triplet_samples = [event_samples[idx_start + j] for j in range(n_phons)]
        # Time gaps between consecutive phonemes in this triplet (in seconds at 2kHz)
        gaps = [(triplet_samples[j+1] - triplet_samples[j]) / 2000.0 for j in range(n_phons - 1)]
        timing_gaps.extend(gaps)

        if t < 5:
            gap_str = ", ".join(f"{g*1000:.0f}ms" for g in gaps)
            print(f"  Trial {t:3d}: {triplet_ids}  gaps: [{gap_str}]")

    if timing_gaps:
        gaps_arr = np.array(timing_gaps)
        print(f"\n  Inter-phoneme gaps (within triplet): "
              f"mean={gaps_arr.mean()*1000:.0f}ms, "
              f"std={gaps_arr.std()*1000:.0f}ms, "
              f"min={gaps_arr.min()*1000:.0f}ms, "
              f"max={gaps_arr.max()*1000:.0f}ms")

        # Check for suspicious gaps (negative = wrong order, very large = wrong triplet)
        neg_gaps = (gaps_arr < 0).sum()
        huge_gaps = (gaps_arr > 2.0).sum()  # >2s between phonemes in same triplet
        if neg_gaps > 0:
            print(f"  [WARNING] {neg_gaps} NEGATIVE gaps — epochs may not be in temporal order!")
        if huge_gaps > 0:
            print(f"  [WARNING] {huge_gaps} gaps >2s — triplets may cross trial boundaries!")

    # --- 3. Check inter-trial gaps (between last phoneme of trial N and first of N+1) ---
    inter_trial_gaps = []
    for t in range(min(n_trials - 1, 50)):
        last_sample = event_samples[(t + 1) * n_phons - 1]
        next_sample = event_samples[(t + 1) * n_phons]
        gap = (next_sample - last_sample) / 2000.0
        inter_trial_gaps.append(gap)

    if inter_trial_gaps:
        itg = np.array(inter_trial_gaps)
        print(f"\n  Inter-trial gaps (p3→p1): "
              f"mean={itg.mean():.2f}s, "
              f"std={itg.std():.2f}s, "
              f"min={itg.min():.2f}s, "
              f"max={itg.max():.2f}s")

    # --- 4. Check phoneme CSV alignment ---
    csv_data, csv_path = load_phoneme_csv(bids_root, subject)
    if csv_data is None:
        print(f"\n  [INFO] No phoneme timing CSV: {csv_path}")
        return

    csv_n_trials = len(csv_data)
    print(f"\n  Phoneme CSV: {csv_n_trials} trials (fif has {n_trials} trials)")
    if csv_n_trials != n_trials:
        print(f"  [WARNING] Trial count mismatch! CSV={csv_n_trials}, fif={n_trials}")

    # --- 5. Compare fif labels vs CSV labels ---
    mismatches = 0
    for t in range(min(n_trials, csv_n_trials)):
        # Labels from fif
        fif_labels = [inv_event_id[all_event_ids[t * n_phons + j]] for j in range(n_phons)]

        # Labels from CSV (trial numbers in CSV may be 0-indexed or 1-indexed)
        csv_trial_key = t if t in csv_data else t + 1
        if csv_trial_key not in csv_data:
            continue

        csv_rows = sorted(csv_data[csv_trial_key], key=lambda r: int(r["phoneme_idx"]))
        csv_labels = [r["phoneme"] for r in csv_rows]

        if fif_labels != csv_labels:
            mismatches += 1
            if mismatches <= 5:
                print(f"  Trial {t}: fif={fif_labels} vs csv={csv_labels} [MISMATCH]")

    if mismatches > 0:
        print(f"  [WARNING] {mismatches}/{min(n_trials, csv_n_trials)} label mismatches!")
    else:
        print(f"  Labels match across all {min(n_trials, csv_n_trials)} trials ✓")

    # --- 6. Check phoneme onset timing relative to epoch t=0 ---
    onset_offsets = []
    for t in range(min(n_trials, csv_n_trials)):
        csv_trial_key = t if t in csv_data else t + 1
        if csv_trial_key not in csv_data:
            continue
        csv_rows = sorted(csv_data[csv_trial_key], key=lambda r: int(r["phoneme_idx"]))
        response_onset = float(csv_rows[0]["response_onset"])
        for row in csv_rows:
            phon_onset = float(row["onset"]) - response_onset  # relative to response
            onset_offsets.append(phon_onset)

    if onset_offsets:
        oo = np.array(onset_offsets)
        print(f"\n  Phoneme onsets (relative to response_onset): "
              f"mean={oo.mean()*1000:.0f}ms, "
              f"std={oo.std()*1000:.0f}ms, "
              f"min={oo.min()*1000:.0f}ms, "
              f"max={oo.max()*1000:.0f}ms")

        # First phoneme onsets (should be near t=0 if epoch is locked to response onset)
        first_phon = oo[::3]  # every 3rd is position-1
        print(f"  First phoneme onset: "
              f"mean={first_phon.mean()*1000:.0f}ms, "
              f"std={first_phon.std()*1000:.0f}ms")
        if abs(first_phon.mean()) > 0.3:
            print(f"  [WARNING] First phoneme onset ~{first_phon.mean()*1000:.0f}ms from epoch t=0 "
                  f"— systematic offset!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--patient", default=None, help="Single patient to check (default: all)")
    args = parser.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)
    bids_root = paths.get("ps_bids_root") or paths["bids_root"]

    patients = [args.patient] if args.patient else PS_PATIENTS

    for subject in patients:
        try:
            diagnose_patient(bids_root, subject)
        except Exception as e:
            print(f"\n  {subject}: [ERROR] {e}")

    print(f"\n{'='*70}")
    print("  Done. Check [WARNING] and [ERROR] flags above.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
