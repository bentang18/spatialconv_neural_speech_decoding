#!/usr/bin/env python3
"""Stage 3 LR sweep on a JEPA checkpoint.

Evaluates a 3x3 grid of (lr, readin_lr_mult) configurations
on frozen JEPA features with grouped-by-token CV.

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/lr_sweep_stage3.py \
    --checkpoint results/pretrain/method_B_jepa/S14/stage2_checkpoint.pt \
    --paths configs/paths.yaml --target S14 --device mps
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.pretraining.pretrain_model import PretrainModel
from speech_decoding.pretraining.jepa_model import JEPAModel
from speech_decoding.pretraining.stage3_evaluator import Stage3Evaluator, Stage3Config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 LR sweep on JEPA checkpoint")
    p.add_argument("--checkpoint", required=True, help="Stage 2 checkpoint path")
    p.add_argument("--paths", required=True, help="paths.yaml")
    p.add_argument("--config", default="configs/pretrain_base.yaml")
    p.add_argument("--target", required=True, help="Target patient for evaluation")
    p.add_argument("--device", default="mps")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    bids_root = paths.get("ps_bids_root") or paths["bids_root"]

    # Load JEPA checkpoint and transfer target encoder to PretrainModel
    config["ema_total_steps"] = 5000
    ssl_model = JEPAModel(config, grid_shape=(8, 16))
    ssl_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    base_model = PretrainModel(config, grid_shape=(8, 16))
    ssl_model.transfer_encoder_weights(base_model)
    logger.info("Loaded JEPA checkpoint, transferred target encoder to PretrainModel")

    # Save the pretrained state dict for reloading before each sweep point
    pretrained_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    # Load target patient data
    ds = load_patient_data(args.target, bids_root, task="PhonemeSequence",
                           n_phons=3, tmin=-0.5, tmax=1.0)
    target_grids, target_labels = [], []
    for i in range(len(ds)):
        g, l, _ = ds[i]
        target_grids.append(g)
        target_labels.append(l)
    target_grids = torch.tensor(np.stack(target_grids), dtype=torch.float32)
    logger.info("%s: %d trials, grid %s", args.target, len(target_grids),
                tuple(target_grids.shape[1:3]))

    # Sweep grid
    lrs = [3e-4, 1e-3, 3e-3]
    readin_lr_mults = [1.0, 3.0, 10.0]

    all_results = {}
    total = len(lrs) * len(readin_lr_mults)
    idx = 0

    for lr in lrs:
        for readin_lr_mult in readin_lr_mults:
            idx += 1
            tag = f"lr={lr:.0e}_mult={readin_lr_mult:.1f}"
            logger.info("\n=== [%d/%d] %s ===", idx, total, tag)

            # Reload pretrained weights before each sweep point
            base_model.load_state_dict(pretrained_state)

            s3_cfg = Stage3Config(
                lr=lr,
                readin_lr_mult=readin_lr_mult,
                epochs=300,
                patience=7,
                eval_every=10,
                warmup_epochs=20,
                n_folds=5,
                batch_size=16,
                grad_clip=5.0,
                loss_mode="ce",
                head_type="linear",
            )

            evaluator = Stage3Evaluator(base_model, s3_cfg, device=args.device)
            t0 = time.time()
            results = evaluator.evaluate(target_grids, target_labels,
                                         patient_id=args.target)
            elapsed = time.time() - t0

            all_results[tag] = {
                "lr": lr,
                "readin_lr_mult": readin_lr_mult,
                "mean_per": results["mean_per"],
                "std_per": results["std_per"],
                "fold_pers": results["fold_pers"],
                "content_collapse": results["content_collapse"],
                "elapsed_sec": round(elapsed, 1),
            }
            logger.info("%s -> PER %.3f +/- %.3f (%.1fs)",
                        tag, results["mean_per"], results["std_per"], elapsed)

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("Stage 3 LR Sweep Results — %s (JEPA target encoder)", args.target)
    logger.info("=" * 70)
    logger.info("%-30s  %-10s  %-10s  %-8s", "Config", "PER", "Std", "Time")
    logger.info("-" * 70)

    best_tag = None
    best_per = float("inf")
    for tag, r in all_results.items():
        logger.info("%-30s  %-10.3f  %-10.3f  %-8.1fs",
                     tag, r["mean_per"], r["std_per"], r["elapsed_sec"])
        if r["mean_per"] < best_per:
            best_per = r["mean_per"]
            best_tag = tag

    logger.info("-" * 70)
    logger.info("Best: %s -> PER %.3f", best_tag, best_per)

    # Also print as a grid
    logger.info("\n--- PER Grid (rows=lr, cols=readin_lr_mult) ---")
    header = "lr \\ mult   " + "  ".join(f"{m:>8.1f}" for m in readin_lr_mults)
    logger.info(header)
    for lr in lrs:
        row = f"{lr:.0e}      "
        for mult in readin_lr_mults:
            tag = f"lr={lr:.0e}_mult={mult:.1f}"
            row += f"  {all_results[tag]['mean_per']:>8.3f}"
        logger.info(row)

    # Save results
    out_path = Path(args.checkpoint).parent / "lr_sweep_stage3.json"
    with open(out_path, "w") as f:
        json.dump({
            "target": args.target,
            "checkpoint": args.checkpoint,
            "sweep_grid": {"lrs": lrs, "readin_lr_mults": readin_lr_mults},
            "results": all_results,
            "best": {"tag": best_tag, "mean_per": best_per},
        }, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
