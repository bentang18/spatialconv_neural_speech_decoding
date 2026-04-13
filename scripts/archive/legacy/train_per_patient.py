#!/usr/bin/env python3
"""Per-patient training script.

Trains and evaluates on each PS patient individually with stratified CV.
Compares against Zac's per-patient SVM baseline.

Usage:
    python scripts/train_per_patient.py [--config configs/default.yaml]
                                        [--patients S14 S33 ...]
                                        [--device cpu]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

from speech_decoding.data.bids_dataset import load_patient_data, load_per_position_data
from speech_decoding.training.trainer import train_per_patient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# All PS patients with preprocessed data
PS_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S36", "S39", "S57", "S58", "S62"]


def main():
    parser = argparse.ArgumentParser(description="Per-patient CTC training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--patients", nargs="+", default=None,
                        help="Patient IDs to train. Default: all PS patients.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tmin", type=float, default=-0.5)
    parser.add_argument("--tmax", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    bids_root = Path(paths["ps_bids_root"])
    patients = args.patients or PS_PATIENTS

    all_results = []
    for pid in patients:
        logger.info("=" * 60)
        logger.info("Training patient %s", pid)

        try:
            loss_type = config["training"].get("loss_type", "ctc")
            if loss_type == "ce_perpos":
                ds = load_per_position_data(
                    pid, bids_root, task="PhonemeSequence", n_phons=3,
                    tmin=args.tmin, tmax=args.tmax,
                )
            else:
                ds = load_patient_data(
                    pid, bids_root, task="PhonemeSequence", n_phons=3,
                    tmin=args.tmin, tmax=args.tmax,
                )
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", pid, e)
            continue

        logger.info("%s: %d trials, grid %s", pid, len(ds), ds.grid_shape)

        for seed in config["evaluation"]["seeds"]:
            result = train_per_patient(ds, config, seed=seed, device=args.device)
            all_results.append(result)
            logger.info(
                "%s seed=%d: PER=%.3f±%.3f, bal_acc=%.3f±%.3f",
                pid, seed,
                result["per_mean"], result["per_std"],
                result["bal_acc_mean_mean"], result["bal_acc_mean_std"],
            )

    # Summary table
    if all_results:
        print("\n" + "=" * 70)
        print(f"{'Patient':<10} {'Seed':<6} {'PER':<12} {'Bal Acc':<12} "
              f"{'Len Acc':<12} {'Blank%':<10}")
        print("-" * 70)
        for r in all_results:
            print(
                f"{r['patient_id']:<10} {r['seed']:<6} "
                f"{r['per_mean']:.3f}±{r['per_std']:.3f}  "
                f"{r['bal_acc_mean_mean']:.3f}±{r['bal_acc_mean_std']:.3f}  "
                f"{r['length_accuracy_mean']:.3f}±{r['length_accuracy_std']:.3f}  "
                f"{r['blank_ratio_mean']:.2f}±{r['blank_ratio_std']:.2f}"
            )


if __name__ == "__main__":
    main()
