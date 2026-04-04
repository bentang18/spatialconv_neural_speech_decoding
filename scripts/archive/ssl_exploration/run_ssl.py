#!/usr/bin/env python3
"""Unified SSL training + evaluation script.

Supports all SSL methods: jepa, byol, dino, vicreg, lewm, masked (PretrainModel).
Uses Stage2Trainer for SSL training, Stage3Evaluator for evaluation.

Usage:
  # Pure BYOL (verify the 0.040 anomaly):
  python scripts/run_ssl.py --mode byol --target S14 --steps 5000

  # DINO:
  python scripts/run_ssl.py --mode dino --target S14 --steps 5000

  # JEPA (re-run with EMA fix):
  python scripts/run_ssl.py --mode jepa --target S14 --steps 5000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.pretraining.pretrain_model import PretrainModel
from speech_decoding.pretraining.stage2_trainer import Stage2Trainer, Stage2Config
from speech_decoding.pretraining.stage3_evaluator import Stage3Evaluator, Stage3Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PS_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S33", "S39", "S57", "S58", "S62"]
# S32 removed (duplicate of S36), S36 removed (duplicate of S32)
DEV_PATIENT = "S26"


def build_ssl_model(mode: str, config: dict, grid_shape: tuple[int, int]):
    """Instantiate SSL model by mode name."""
    config["ema_total_steps"] = config.get("ema_total_steps", 5000)

    if mode == "jepa":
        from speech_decoding.pretraining.jepa_model import JEPAModel
        return JEPAModel(config, grid_shape)
    elif mode == "byol":
        from speech_decoding.pretraining.byol_model import BYOLModel
        return BYOLModel(config, grid_shape)
    elif mode == "dino":
        from speech_decoding.pretraining.dino_model import DINOModel
        return DINOModel(config, grid_shape)
    elif mode == "vicreg":
        from speech_decoding.pretraining.vicreg_model import VICRegModel
        return VICRegModel(config, grid_shape)
    elif mode == "lewm":
        from speech_decoding.pretraining.lewm_model import LeWMModel
        return LeWMModel(config, grid_shape)
    elif mode == "masked":
        return PretrainModel(config, grid_shape)
    else:
        raise ValueError(f"Unknown SSL mode: {mode}")


def load_source_data(patients, bids_root):
    """Load grid data for source patients."""
    patient_data = {}
    for pid in patients:
        ds = load_patient_data(pid, bids_root, task="PhonemeSequence",
                               n_phons=3, tmin=0.0, tmax=1.0)
        grids = []
        for i in range(len(ds)):
            g, _, _ = ds[i]
            grids.append(g)
        patient_data[pid] = torch.tensor(np.stack(grids), dtype=torch.float32)
        logger.info("  %s: %d trials", pid, len(grids))
    return patient_data


def load_target_data(target, bids_root):
    """Load grid data and labels for target patient."""
    ds = load_patient_data(target, bids_root, task="PhonemeSequence",
                           n_phons=3, tmin=0.0, tmax=1.0)
    grids, labels = [], []
    for i in range(len(ds)):
        g, l, _ = ds[i]
        grids.append(g)
        labels.append(l)
    return torch.tensor(np.stack(grids), dtype=torch.float32), labels


def main():
    p = argparse.ArgumentParser(description="Unified SSL training + evaluation")
    p.add_argument("--mode", required=True,
                   choices=["jepa", "byol", "dino", "vicreg", "lewm", "masked"],
                   help="SSL method")
    p.add_argument("--target", default="S14", help="Target patient for Stage 3")
    p.add_argument("--paths", default="configs/paths.yaml")
    p.add_argument("--config", default="configs/pretrain_base.yaml")
    p.add_argument("--steps", type=int, default=5000, help="Stage 2 training steps")
    p.add_argument("--device", default="mps")
    p.add_argument("--output-dir", default="results/ssl")
    p.add_argument("--include-target-in-ssl", action="store_true",
                   help="Include target patient in SSL training (no labels used)")
    args = p.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)
    with open(args.config) as f:
        model_config = yaml.safe_load(f)

    bids_root = paths.get("ps_bids_root") or paths["bids_root"]
    output_dir = Path(args.output_dir) / f"{args.mode}" / args.target
    output_dir.mkdir(parents=True, exist_ok=True)

    # Source patients: exclude dev, optionally include target for SSL
    source_patients = [pid for pid in PS_PATIENTS if pid != DEV_PATIENT]
    if not args.include_target_in_ssl:
        source_patients = [pid for pid in source_patients if pid != args.target]
    logger.info("Mode: %s, Target: %s, Sources: %s", args.mode, args.target, source_patients)

    # Load data
    logger.info("Loading source data...")
    patient_data = load_source_data(source_patients, bids_root)

    # Build model
    model_config["ema_total_steps"] = args.steps
    grid_shape = (8, 16)  # S14 default; model handles different shapes via SpatialConv
    model = build_ssl_model(args.mode, model_config, grid_shape)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s, params: %d", args.mode, n_params)

    # Stage 2: SSL training
    s2_cfg = Stage2Config(steps=args.steps, batch_size=8)
    trainer = Stage2Trainer(model, s2_cfg, device=args.device)
    logger.info("Stage 2: %d steps on %d patients", args.steps, len(patient_data))
    metrics = trainer.train(patient_data)
    logger.info("Stage 2 final loss: %.4f", metrics[-1]["loss"])

    # Save checkpoint
    torch.save(model.state_dict(), output_dir / "stage2_checkpoint.pt")

    # Transfer to PretrainModel for Stage 3
    pretrain_model = PretrainModel(model_config, grid_shape)
    if hasattr(model, "transfer_encoder_weights"):
        model.transfer_encoder_weights(pretrain_model)
        logger.info("Transferred %s encoder weights to PretrainModel", args.mode)
    else:
        # masked mode: model IS a PretrainModel
        pretrain_model.load_state_dict(model.state_dict())

    # Stage 3: Evaluate
    logger.info("Loading target data for %s...", args.target)
    target_grids, target_labels = load_target_data(args.target, bids_root)

    s3_cfg = Stage3Config(epochs=100, patience=10, n_folds=5)
    evaluator = Stage3Evaluator(pretrain_model, s3_cfg, device=args.device)
    results = evaluator.evaluate(target_grids, target_labels, patient_id=args.target)

    logger.info("%s → %s: PER %.3f ± %.3f",
                args.mode, args.target, results["mean_per"], results["std_per"])
    logger.info("Fold PERs: %s", results["fold_pers"])
    logger.info("Content collapse: %s", results["content_collapse"])

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_dir / "results.json")


if __name__ == "__main__":
    main()
