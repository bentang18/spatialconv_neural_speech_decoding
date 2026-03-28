#!/usr/bin/env python3
"""Per-patient regression training: CE + masked MSE on speech targets."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.training.regression_trainer import train_per_patient_regression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PS_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S36", "S39", "S57", "S58", "S62"]


def _feature_path(results_dir: Path, feature_type: str, patient_id: str) -> Path:
    return results_dir / "audio_features" / feature_type / f"sub-{patient_id}_{feature_type}.npz"


def main():
    parser = argparse.ArgumentParser(description="Train CE + regression per patient.")
    parser.add_argument("--config", default="configs/per_patient_regression.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--patients", nargs="+", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambda-sweep", nargs="+", type=float, default=None)
    parser.add_argument("--feature-type", choices=["hubert", "mel"], default=None)
    parser.add_argument("--target-mode", choices=["frame", "segment"], default=None)
    parser.add_argument("--pca-scope", choices=["all_frames", "speech_only"], default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    bids_root = Path(paths["ps_bids_root"])
    results_dir = Path(paths["results_dir"])
    patients = args.patients or PS_PATIENTS
    lambdas = args.lambda_sweep or [config["training"]["regression_lambda"]]
    if 0.0 not in lambdas:
        lambdas = [0.0] + list(lambdas)
    if args.feature_type is not None:
        config.setdefault("data", {})["feature_type"] = args.feature_type
    if args.target_mode is not None:
        config["training"]["regression_target_mode"] = args.target_mode
    if args.pca_scope is not None:
        config["training"]["regression_pca_scope"] = args.pca_scope
    feature_type = config.get("data", {}).get("feature_type", "hubert")

    rows = []
    for patient_id in patients:
        feat_path = _feature_path(results_dir, feature_type, patient_id)
        if not feat_path.exists():
            logger.error("Missing features for %s: %s", patient_id, feat_path)
            continue
        feat = np.load(feat_path)
        embeddings = feat["embeddings"]
        speech_mask = feat["speech_mask"]
        segment_mask = feat["segment_mask"] if "segment_mask" in feat else None

        ds = load_patient_data(
            patient_id,
            bids_root,
            task=config["data"]["task"],
            n_phons=config["data"]["n_phons"],
            tmin=config["data"]["tmin"],
            tmax=config["data"]["tmax"],
        )
        if len(ds) != len(embeddings):
            raise ValueError(
                f"{patient_id}: dataset trials ({len(ds)}) != embeddings ({len(embeddings)})"
            )

        for lam in lambdas:
            run_config = yaml.safe_load(yaml.safe_dump(config))
            run_config["training"]["regression_lambda"] = lam
            for seed in run_config["evaluation"]["seeds"]:
                result = train_per_patient_regression(
                    ds,
                    embeddings,
                    speech_mask,
                    run_config,
                    seed=seed,
                    device=args.device,
                    segment_mask=segment_mask,
                )
                rows.append((patient_id, seed, lam, result))
                logger.info(
                    "%s seed=%d lambda=%.2f: PER=%.3f bal_acc=%.3f r2_speech=%s",
                    patient_id,
                    seed,
                    lam,
                    result["per_mean"],
                    result["bal_acc_mean_mean"],
                    f"{result['r2_speech_mean']:.3f}" if "r2_speech_mean" in result else "n/a",
                )

    if rows:
        print("\nPatient    Seed  Lambda  PER        BalAcc     R2Speech")
        print("----------------------------------------------------------")
        for patient_id, seed, lam, result in rows:
            r2 = result.get("r2_speech_mean", float("nan"))
            r2_str = f"{r2:.3f}" if np.isfinite(r2) else "n/a"
            print(
                f"{patient_id:<10} {seed:<5} {lam:<6.2f} "
                f"{result['per_mean']:.3f}±{result['per_std']:.3f}  "
                f"{result['bal_acc_mean_mean']:.3f}±{result['bal_acc_mean_std']:.3f}  "
                f"{r2_str}"
            )


if __name__ == "__main__":
    main()
