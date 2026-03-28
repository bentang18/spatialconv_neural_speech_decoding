#!/usr/bin/env python3
"""Run the per-patient CE baseline with grouped-by-token CV.

Reproduces the PER=0.700 config (per_patient_ce_s10_pool48.yaml) but
with proper grouped-by-token CV to quantify leakage from stratified splits.

Usage:
  python scripts/run_grouped_baseline.py --patient S14
  python scripts/run_grouped_baseline.py --patient S14 --device mps
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.evaluation.grouped_cv import (
    build_token_groups,
    load_or_create_splits,
)
from speech_decoding.training.trainer import _train_fold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Grouped-CV baseline")
    parser.add_argument("--config", default="configs/per_patient_ce_s10_pool48.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--patient", default="S14")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="results/grouped_baseline")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    bids_root = paths.get("ps_bids_root") or paths["bids_root"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = load_patient_data(
        args.patient, bids_root, task="PhonemeSequence",
        n_phons=3, tmin=-0.5, tmax=1.0,
    )
    logger.info("%s: %d trials, grid %s", args.patient, len(ds), ds.grid_shape)

    # Create grouped-by-token splits
    labels = ds.ctc_labels
    splits_path = output_dir / f"splits_{args.patient}.json"
    splits = load_or_create_splits(
        labels, patient_id=args.patient, n_folds=args.n_folds,
        save_path=splits_path,
    )

    # Run each fold through the full production trainer
    fold_results = []
    for fold_idx, fold in enumerate(splits):
        train_idx = np.array(fold["train_indices"])
        val_idx = np.array(fold["val_indices"])
        logger.info("Fold %d/%d: %d train, %d val",
                     fold_idx + 1, args.n_folds, len(train_idx), len(val_idx))

        result = _train_fold(
            ds, config, train_idx, val_idx, args.seed, args.device,
        )
        fold_results.append(result)
        logger.info(
            "Fold %d: PER=%.3f, bal_acc=%.3f",
            fold_idx + 1, result["per"], result["bal_acc_mean"],
        )

    # Aggregate
    pers = [r["per"] for r in fold_results]
    bal_accs = [r["bal_acc_mean"] for r in fold_results]
    summary = {
        "patient": args.patient,
        "cv_type": "grouped_by_token",
        "config": args.config,
        "seed": args.seed,
        "n_folds": args.n_folds,
        "mean_per": float(np.mean(pers)),
        "std_per": float(np.std(pers)),
        "mean_bal_acc": float(np.mean(bal_accs)),
        "fold_pers": pers,
        "fold_bal_accs": bal_accs,
    }

    logger.info("=" * 60)
    logger.info("GROUPED-BY-TOKEN CV RESULTS for %s", args.patient)
    logger.info("  PER: %.3f ± %.3f", summary["mean_per"], summary["std_per"])
    logger.info("  Bal Acc: %.3f", summary["mean_bal_acc"])
    logger.info("  Fold PERs: %s", [f"{p:.3f}" for p in pers])
    logger.info("  (Compare to stratified CV PER = 0.700)")

    with open(output_dir / f"grouped_baseline_{args.patient}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
