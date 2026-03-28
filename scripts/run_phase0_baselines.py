"""Phase 0: Baseline evaluation — Methods E, D, spatial-only.

Gate 0 decision:
  D ≈ E → architecture has no value. Debug.
  spatial-only ≈ D → temporal model adds nothing.
  D > spatial-only > E → proceed to SSL.

Usage:
  python scripts/run_phase0_baselines.py --paths configs/paths.yaml --patients S14
  python scripts/run_phase0_baselines.py --paths configs/paths.yaml --all
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.models.spatial_conv import SpatialConvReadIn
from speech_decoding.models.backbone import SharedBackbone
from speech_decoding.evaluation.grouped_cv import (
    build_token_groups,
    load_or_create_splits,
    validate_fold_coverage,
)
from speech_decoding.evaluation.content_collapse import content_collapse_report
from speech_decoding.evaluation.metrics import compute_per

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PS_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S36", "S39", "S57", "S58", "S62"]


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 baselines")
    p.add_argument("--paths", type=str, required=True, help="Path to paths.yaml")
    p.add_argument("--patients", nargs="+", default=None, help="Patient IDs (default: all PS)")
    p.add_argument("--all", action="store_true", help="Run all PS patients")
    p.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds")
    p.add_argument("--device", default="mps", help="Device (mps/cpu/cuda)")
    p.add_argument("--output-dir", default="results/phase0", help="Output directory")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100, help="Max epochs for Method D")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temporal-stride", type=int, default=10, help="200Hz → 20Hz")
    p.add_argument("--pool-h", type=int, default=4)
    p.add_argument("--pool-w", type=int, default=8)
    return p.parse_args()


def train_ce_fold(
    readin, backbone, head, train_data, val_data,
    epochs, lr, device, freeze_backbone=False, freeze_readin=False,
):
    """Train one fold with CE mean-pool loss. Returns val PER + predictions."""
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
    if freeze_readin:
        for p in readin.parameters():
            p.requires_grad = False

    params = [p for p in list(readin.parameters()) + list(backbone.parameters()) + list(head.parameters())
              if p.requires_grad]
    if not params:
        params = list(head.parameters())

    optimizer = AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        if not freeze_readin:
            readin.train()
        if backbone is not None and not freeze_backbone:
            backbone.train()
        head.train()

        grids, labels = train_data
        grids = grids.to(device)
        targets = torch.tensor(labels, dtype=torch.long, device=device)

        shared = readin(grids)
        if backbone is not None:
            h = backbone(shared)
        else:
            h = shared.permute(0, 2, 1)
        pooled = h.mean(dim=1)
        logits = head(pooled)

        per_pos = logits.view(-1, 3, 9)
        loss = sum(
            F.cross_entropy(per_pos[:, pos, :], targets[:, pos] - 1)
            for pos in range(3)
        ) / 3

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        readin.eval()
        if backbone is not None:
            backbone.eval()
        head.eval()
        with torch.no_grad():
            v_grids, v_labels = val_data
            v_grids = v_grids.to(device)
            v_targets = torch.tensor(v_labels, dtype=torch.long, device=device)

            v_shared = readin(v_grids)
            if backbone is not None:
                v_h = backbone(v_shared)
            else:
                v_h = v_shared.permute(0, 2, 1)
            v_pooled = v_h.mean(dim=1)
            v_logits = head(v_pooled)

            v_per_pos = v_logits.view(-1, 3, 9)
            val_loss = sum(
                F.cross_entropy(v_per_pos[:, pos, :], v_targets[:, pos] - 1)
                for pos in range(3)
            ) / 3

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "readin": deepcopy(readin.state_dict()),
                "head": deepcopy(head.state_dict()),
            }
            if backbone is not None:
                best_state["backbone"] = deepcopy(backbone.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    readin.load_state_dict(best_state["readin"])
    head.load_state_dict(best_state["head"])
    if backbone is not None and "backbone" in best_state:
        backbone.load_state_dict(best_state["backbone"])

    readin.eval()
    if backbone is not None:
        backbone.eval()
    head.eval()

    with torch.no_grad():
        v_grids, v_labels = val_data
        v_grids = v_grids.to(device)
        v_shared = readin(v_grids)
        if backbone is not None:
            v_h = backbone(v_shared)
        else:
            v_h = v_shared.permute(0, 2, 1)
        v_pooled = v_h.mean(dim=1)
        v_logits = head(v_pooled)
        v_per_pos = v_logits.view(-1, 3, 9)
        preds = v_per_pos.argmax(dim=-1).cpu().numpy() + 1
        v_targets_np = np.array(v_labels)

    total, errors = 0, 0
    for pred_seq, true_seq in zip(preds, v_targets_np):
        for p, t in zip(pred_seq, true_seq):
            total += 1
            if p != t:
                errors += 1
    per = errors / total if total > 0 else 1.0
    return per, preds, v_targets_np


def main():
    args = parse_args()
    import yaml
    with open(args.paths) as f:
        paths = yaml.safe_load(f)
    bids_root = paths["bids_root"]

    patients = args.patients or (PS_PATIENTS if args.all else ["S14"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for patient_id in patients:
        logger.info("=" * 60)
        logger.info("Patient %s", patient_id)

        ds = load_patient_data(patient_id, bids_root, task="PhonemeSequence",
                                n_phons=3, tmin=-0.5, tmax=1.0)
        grid_h, grid_w = ds.grid_shape

        all_grids = []
        all_labels = []
        for i in range(len(ds)):
            grid, label, _ = ds[i]
            all_grids.append(grid)
            all_labels.append(label)
        all_grids = torch.tensor(np.stack(all_grids), dtype=torch.float32)

        splits_path = output_dir / f"splits_{patient_id}.json"
        splits = load_or_create_splits(all_labels, patient_id, n_folds=args.n_folds,
                                        save_path=splits_path)

        patient_results = {"E": [], "D": [], "spatial_only": []}

        for fold_idx, fold in enumerate(splits):
            train_idx = fold["train_indices"]
            val_idx = fold["val_indices"]
            train_data = (all_grids[train_idx], [all_labels[i] for i in train_idx])
            val_data = (all_grids[val_idx], [all_labels[i] for i in val_idx])

            device = args.device

            # Method E: frozen random init backbone + readin, train only CE head
            readin_e = SpatialConvReadIn(grid_h, grid_w, pool_h=args.pool_h, pool_w=args.pool_w).to(device)
            backbone_e = SharedBackbone(D=readin_e.out_dim, H=64, temporal_stride=args.temporal_stride).to(device)
            head_e = nn.Linear(128, 27).to(device)
            per_e, _, _ = train_ce_fold(
                readin_e, backbone_e, head_e, train_data, val_data,
                epochs=args.epochs, lr=args.lr, device=device,
                freeze_backbone=True, freeze_readin=True,
            )
            patient_results["E"].append(per_e)
            logger.info("  Fold %d Method E PER: %.3f", fold_idx, per_e)

            # Method D: supervised from scratch, all params trainable
            readin_d = SpatialConvReadIn(grid_h, grid_w, pool_h=args.pool_h, pool_w=args.pool_w).to(device)
            backbone_d = SharedBackbone(D=readin_d.out_dim, H=64, temporal_stride=args.temporal_stride).to(device)
            head_d = nn.Linear(128, 27).to(device)
            per_d, _, _ = train_ce_fold(
                readin_d, backbone_d, head_d, train_data, val_data,
                epochs=args.epochs, lr=args.lr, device=device,
            )
            patient_results["D"].append(per_d)
            logger.info("  Fold %d Method D PER: %.3f", fold_idx, per_d)

            # Spatial-only: Conv2d → temporal mean pool → CE head, no BiGRU
            readin_s = SpatialConvReadIn(grid_h, grid_w, pool_h=args.pool_h, pool_w=args.pool_w).to(device)
            head_s = nn.Linear(readin_s.out_dim, 27).to(device)
            per_s, preds_s, targets_s = train_ce_fold(
                readin_s, None, head_s, train_data, val_data,
                epochs=args.epochs, lr=args.lr, device=device,
            )
            patient_results["spatial_only"].append(per_s)
            logger.info("  Fold %d Spatial-only PER: %.3f", fold_idx, per_s)

        results[patient_id] = {
            method: {
                "mean_per": float(np.mean(pers)),
                "std_per": float(np.std(pers)),
                "fold_pers": pers,
            }
            for method, pers in patient_results.items()
        }
        for method in ["E", "D", "spatial_only"]:
            r = results[patient_id][method]
            logger.info("  %s: PER %.3f ± %.3f", method, r["mean_per"], r["std_per"])

    with open(output_dir / "phase0_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_dir / "phase0_results.json")

    print("\n" + "=" * 70)
    print(f"{'Patient':<10} {'Method E':<15} {'Method D':<15} {'Spatial-only':<15}")
    print("-" * 70)
    for patient_id, r in results.items():
        print(f"{patient_id:<10} "
              f"{r['E']['mean_per']:.3f} ± {r['E']['std_per']:.3f}  "
              f"{r['D']['mean_per']:.3f} ± {r['D']['std_per']:.3f}  "
              f"{r['spatial_only']['mean_per']:.3f} ± {r['spatial_only']['std_per']:.3f}")

    print("\n--- Gate 0 Check ---")
    for patient_id, r in results.items():
        e, d, s = r["E"]["mean_per"], r["D"]["mean_per"], r["spatial_only"]["mean_per"]
        if abs(d - e) < 0.02:
            print(f"  {patient_id}: WARNING — D ≈ E ({d:.3f} vs {e:.3f}). Architecture has no value.")
        elif abs(s - d) < 0.02:
            print(f"  {patient_id}: WARNING — spatial-only ≈ D ({s:.3f} vs {d:.3f}). Temporal model adds nothing.")
        elif d < s < e:
            print(f"  {patient_id}: PASS — D ({d:.3f}) > spatial-only ({s:.3f}) > E ({e:.3f}). Proceed to SSL.")
        else:
            print(f"  {patient_id}: D={d:.3f}, spatial-only={s:.3f}, E={e:.3f} — check ordering.")


if __name__ == "__main__":
    main()
