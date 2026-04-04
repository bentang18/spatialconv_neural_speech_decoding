#!/usr/bin/env python3
"""Run per-patient CE + phonological auxiliary sweeps."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.training.phonological_aux_trainer import train_per_patient_phonological_aux

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Diagnose phonological auxiliary loss on one patient.")
    parser.add_argument("--config", default="configs/per_patient_phonological_aux.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--patient", default="S14")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambda-sweep", nargs="+", type=float, default=[0.0, 0.1, 0.3, 1.0])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    bids_root = Path(paths["ps_bids_root"])
    ds = load_patient_data(
        args.patient,
        bids_root,
        task="PhonemeSequence",
        n_phons=3,
        tmin=-0.5,
        tmax=1.0,
    )
    logger.info("%s loaded: trials=%d grid=%s", args.patient, len(ds), ds.grid_shape)

    rows = []
    for lam in args.lambda_sweep:
        run_config = yaml.safe_load(yaml.safe_dump(config))
        run_config["training"]["phonological_aux_lambda"] = lam
        result = train_per_patient_phonological_aux(ds, run_config, seed=42, device=args.device)
        rows.append((lam, result))
        logger.info(
            "lambda=%.2f: PER=%.3f bal_acc=%.3f feature_acc=%s feature_exact=%s",
            lam,
            result["per_mean"],
            result["bal_acc_mean_mean"],
            f"{result['feature_acc_mean']:.3f}" if "feature_acc_mean" in result else "n/a",
            f"{result['feature_exact_mean']:.3f}" if "feature_exact_mean" in result else "n/a",
        )

    print("\nLambda  PER        BalAcc     FeatureAcc  FeatureExact")
    print("-------------------------------------------------------")
    for lam, result in rows:
        feat_acc = result.get("feature_acc_mean", float("nan"))
        feat_exact = result.get("feature_exact_mean", float("nan"))
        feat_acc_str = f"{feat_acc:.3f}" if feat_acc == feat_acc else "n/a"
        feat_exact_str = f"{feat_exact:.3f}" if feat_exact == feat_exact else "n/a"
        print(
            f"{lam:<6.2f} "
            f"{result['per_mean']:.3f}±{result['per_std']:.3f}  "
            f"{result['bal_acc_mean_mean']:.3f}±{result['bal_acc_mean_std']:.3f}  "
            f"{feat_acc_str:<10} {feat_exact_str}"
        )


if __name__ == "__main__":
    main()
