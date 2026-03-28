#!/usr/bin/env python3
"""NCA-JEPA pretraining pipeline CLI.

Usage:
  # Method B (neural-only SSL):
  python scripts/train_pretrain.py --method B --paths configs/paths.yaml --target S14

  # Method C (smooth AR -> neural):
  python scripts/train_pretrain.py --method C --paths configs/paths.yaml --target S14

  # Method A-minimal (switching LDS -> neural):
  python scripts/train_pretrain.py --method A --generator switching_lds --paths configs/paths.yaml --target S14
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
from speech_decoding.pretraining.stage1_trainer import Stage1Trainer, Stage1Config
from speech_decoding.pretraining.stage2_trainer import Stage2Trainer, Stage2Config
from speech_decoding.pretraining.stage3_evaluator import Stage3Evaluator, Stage3Config
from speech_decoding.pretraining.synthetic_pipeline import SyntheticDataPipeline, SyntheticConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PS_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S36", "S39", "S57", "S58", "S62"]
DEV_PATIENT = "S26"


def parse_args():
    p = argparse.ArgumentParser(description="NCA-JEPA pretraining pipeline")
    p.add_argument("--method", required=True, choices=["B", "C", "A"],
                   help="B=neural-only, C=smooth-AR, A=structured")
    p.add_argument("--paths", required=True, help="paths.yaml")
    p.add_argument("--target", required=True, help="Target patient for evaluation")
    p.add_argument("--config", default="configs/pretrain_base.yaml")
    p.add_argument("--generator", default="smooth_ar", help="Generator for Method A")
    p.add_argument("--device", default="mps")
    p.add_argument("--output-dir", default="results/pretrain")
    p.add_argument("--s-total", type=int, default=5000, help="Total optimizer steps")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)
    with open(args.config) as f:
        model_config = yaml.safe_load(f)

    bids_root = paths.get("ps_bids_root") or paths["bids_root"]
    output_dir = Path(args.output_dir) / f"method_{args.method}" / args.target
    output_dir.mkdir(parents=True, exist_ok=True)

    source_patients = [p for p in PS_PATIENTS if p != args.target and p != DEV_PATIENT]
    logger.info("Target: %s, Dev: %s, Sources: %s", args.target, DEV_PATIENT, source_patients)

    # Initialize model
    model = PretrainModel(model_config, grid_shape=(8, 16))
    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    # Stage 1: Synthetic (Methods C and A only)
    if args.method in ("C", "A"):
        gen_name = "smooth_ar" if args.method == "C" else args.generator
        synth_cfg = SyntheticConfig(
            generator=gen_name,
            grid_shapes=[(8, 16), (12, 22)],
        )
        pipeline = SyntheticDataPipeline(synth_cfg)
        s1_cfg = Stage1Config(steps=args.s_total // 2, batch_size=8, T=300)
        trainer = Stage1Trainer(model, pipeline, s1_cfg, device=args.device)
        logger.info("Stage 1: %d steps with %s generator", s1_cfg.steps, gen_name)
        s1_metrics = trainer.train()
        torch.save(model.state_dict(), output_dir / "stage1_checkpoint.pt")
        logger.info("Stage 1 final loss: %.4f", s1_metrics[-1]["loss"])

    # Stage 2: Neural adaptation
    patient_data = {}
    for pid in source_patients:
        ds = load_patient_data(pid, bids_root, task="PhonemeSequence",
                                n_phons=3, tmin=-0.5, tmax=1.0)
        grids = []
        for i in range(len(ds)):
            g, _, _ = ds[i]
            grids.append(g)
        patient_data[pid] = torch.tensor(np.stack(grids), dtype=torch.float32)

    s2_steps = args.s_total // 2 if args.method in ("C", "A") else args.s_total
    s2_cfg = Stage2Config(steps=s2_steps, batch_size=8)
    s2_trainer = Stage2Trainer(model, s2_cfg, device=args.device)
    logger.info("Stage 2: %d steps on %d source patients", s2_steps, len(patient_data))
    s2_metrics = s2_trainer.train(patient_data)
    torch.save(model.state_dict(), output_dir / "stage2_checkpoint.pt")
    logger.info("Stage 2 final loss: %.4f", s2_metrics[-1]["loss"])

    # Stage 3: Evaluate on target
    ds_target = load_patient_data(args.target, bids_root, task="PhonemeSequence",
                                   n_phons=3, tmin=-0.5, tmax=1.0)
    target_grids, target_labels = [], []
    for i in range(len(ds_target)):
        g, l, _ = ds_target[i]
        target_grids.append(g)
        target_labels.append(l)
    target_grids = torch.tensor(np.stack(target_grids), dtype=torch.float32)

    s3_cfg = Stage3Config(epochs=100, patience=10, n_folds=5)
    evaluator = Stage3Evaluator(model, s3_cfg, device=args.device)
    results = evaluator.evaluate(target_grids, target_labels, patient_id=args.target)

    logger.info("Method %s -> Target %s: PER %.3f +/- %.3f",
                args.method, args.target, results["mean_per"], results["std_per"])
    logger.info("Content collapse: %s", results["content_collapse"])

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
