#!/usr/bin/env python3
"""Exp 79: Domain Adversarial Neural Network (DANN) with gradient reversal.

Forces backbone to learn patient-INVARIANT features by adversarially
training a patient classifier. Gradient reversal flips sign of patient
classifier gradients, penalizing patient-discriminable representations.

Key difference from baseline: S1 loss = CE_phoneme + lambda * CE_patient
where lambda anneals 0→1 and patient classifier gradients are reversed.
"""
from __future__ import annotations
import math, sys, time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare
from arch_ablation_base import (
    DEVICE, SEED, SpatialReadIn, Backbone, ArticulatoryBottleneckHead,
    augment, compute_loss, train_eval_fold,
    S1_EPOCHS, S1_BATCH_SIZE, S1_LR, S1_READIN_LR_MULT,
    S1_WEIGHT_DECAY, S1_GRAD_CLIP, S1_WARMUP_EPOCHS, S1_PATIENCE,
    S1_EVAL_EVERY, S1_VAL_FRACTION, MIXUP_ALPHA, SOURCE_KNN_WEIGHT,
)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class PatientDiscriminator(nn.Module):
    def __init__(self, d_in, n_patients):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, n_patients),
        )

    def forward(self, x, alpha=1.0):
        x = GradientReversal.apply(x, alpha)
        return self.net(x)


def train_stage1_dann(all_data, ReadInCls, BackboneCls, HeadCls):
    torch.manual_seed(SEED); np.random.seed(SEED)
    pids = prepare.SOURCE_PATIENTS
    pid_to_idx = {p: i for i, p in enumerate(pids)}

    read_ins = {}
    for pid in pids:
        H, W = all_data[pid]["grid_shape"]
        read_ins[pid] = ReadInCls(H, W).to(DEVICE)
    d_flat = list(read_ins.values())[0].d_flat

    backbone = BackboneCls(d_in=d_flat).to(DEVICE)
    head = HeadCls(d_in=backbone.out_dim).to(DEVICE)
    discriminator = PatientDiscriminator(backbone.out_dim, len(pids)).to(DEVICE)

    source_train, source_val = {}, {}
    for pid in pids:
        n = len(all_data[pid]["labels"])
        perm = np.random.permutation(n)
        n_val = max(1, int(round(S1_VAL_FRACTION * n)))
        source_train[pid] = sorted(perm[n_val:].tolist())
        source_val[pid] = sorted(perm[:n_val].tolist())

    readin_params = []
    for ri in read_ins.values():
        readin_params.extend(ri.parameters())
    optimizer = AdamW([
        {"params": readin_params, "lr": S1_LR * S1_READIN_LR_MULT},
        {"params": backbone.parameters(), "lr": S1_LR},
        {"params": head.parameters(), "lr": S1_LR},
        {"params": discriminator.parameters(), "lr": S1_LR},
    ], weight_decay=S1_WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < S1_WARMUP_EPOCHS:
            return (epoch + 1) / S1_WARMUP_EPOCHS
        progress = (epoch - S1_WARMUP_EPOCHS) / max(S1_EPOCHS - S1_WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    best_val_loss, best_state, patience_ctr = float("inf"), None, 0
    dann_lambda_max = 0.5  # max adversarial weight

    for epoch in range(S1_EPOCHS):
        backbone.train(); head.train(); discriminator.train()
        for ri in read_ins.values(): ri.train()
        np.random.shuffle(pids)
        epoch_loss, epoch_dann, n_batches = 0.0, 0.0, 0

        # DANN lambda: anneal from 0 to max using sigmoid schedule
        p = epoch / S1_EPOCHS
        dann_alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0  # 0→1
        dann_weight = dann_lambda_max * dann_alpha

        for pid in pids:
            grids = all_data[pid]["grids"]
            labels = all_data[pid]["labels"]
            tr_idx = source_train[pid]
            perm = np.random.permutation(len(tr_idx))
            for start in range(0, len(tr_idx), S1_BATCH_SIZE):
                batch_idx = [tr_idx[perm[i]] for i in range(start, min(start + S1_BATCH_SIZE, len(tr_idx)))]
                x = augment(grids[batch_idx]).to(DEVICE)
                y = [labels[i] for i in batch_idx]
                B = x.shape[0]

                mixup_y, mixup_lam = None, 1.0
                if MIXUP_ALPHA > 0 and B > 1:
                    mixup_lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                    perm_mix = torch.randperm(B)
                    x = mixup_lam * x + (1 - mixup_lam) * x[perm_mix]
                    mixup_y = [y[i] for i in perm_mix.tolist()]

                optimizer.zero_grad()
                feat = read_ins[pid](x)
                h = backbone(feat)  # (B, T', 2H)
                logits = head(h)

                # Phoneme CE loss
                ce_loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)

                # DANN: patient classifier on mean-pooled features
                h_pooled = h.mean(dim=1).detach() if epoch < 5 else h.mean(dim=1)
                patient_logits = discriminator(h_pooled, alpha=dann_alpha)
                patient_target = torch.full((B,), pid_to_idx[pid], dtype=torch.long, device=DEVICE)
                dann_loss = F.cross_entropy(patient_logits, patient_target)

                loss = ce_loss + dann_weight * dann_loss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()) +
                    readin_params + list(discriminator.parameters()), S1_GRAD_CLIP)
                optimizer.step()
                epoch_loss += ce_loss.item()
                epoch_dann += dann_loss.item()
                n_batches += 1
        scheduler.step()

        if (epoch + 1) % S1_EVAL_EVERY == 0:
            backbone.eval(); head.eval()
            for ri in read_ins.values(): ri.eval()
            val_loss, val_batches = 0.0, 0
            with torch.no_grad():
                for pid in pids:
                    vi = source_val[pid]
                    if not vi: continue
                    x = all_data[pid]["grids"][vi].to(DEVICE)
                    y = [all_data[pid]["labels"][i] for i in vi]
                    feat = read_ins[pid](x)
                    h = backbone(feat)
                    logits = head(h)
                    val_loss += compute_loss(logits, y).item()
                    val_batches += 1
            val_loss /= max(val_batches, 1)
            avg_ce = epoch_loss / max(n_batches, 1)
            avg_dann = epoch_dann / max(n_batches, 1)
            print(f"  S1 epoch {epoch+1}: ce={avg_ce:.4f} dann={avg_dann:.4f} "
                  f"val={val_loss:.4f} alpha={dann_alpha:.3f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "backbone": deepcopy(backbone.state_dict()),
                    "head": deepcopy(head.state_dict()),
                    "read_ins": {pid: deepcopy(ri.state_dict()) for pid, ri in read_ins.items()},
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= S1_PATIENCE:
                    print(f"  S1 early stop at epoch {epoch+1}")
                    break

    if best_state:
        backbone.load_state_dict(best_state["backbone"])
        head.load_state_dict(best_state["head"])
        for pid, ri in read_ins.items():
            if pid in best_state["read_ins"]:
                ri.load_state_dict(best_state["read_ins"][pid])
    return backbone, head, read_ins


def run():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    all_data = prepare.load_all_patients()
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)
    N, H, W, T = grids.shape

    print("=== exp79_dann ===")
    print(f"Target: {prepare.TARGET_PATIENT} | Trials: {N} | DANN gradient reversal")

    backbone, head, read_ins = train_stage1_dann(
        all_data, SpatialReadIn, Backbone, ArticulatoryBottleneckHead)

    fold_pers = []
    all_preds, all_refs = [], []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        per, preds, methods = train_eval_fold(
            backbone, head, grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx],
            read_ins=read_ins, all_data=all_data, ReadInCls=SpatialReadIn)
        fold_pers.append(per)
        all_preds.extend(preds)
        all_refs.extend([labels[i] for i in va_idx])
        best_method = min(methods, key=lambda k: methods[k][0])
        print(f"  Fold {fi+1}: PER={per:.4f} (lin={methods['linear'][0]:.4f} "
              f"knn={methods['knn'][0]:.4f} best={best_method}) ({time.time()-t0:.1f}s)")
        if time.time() - t0 > prepare.TIME_BUDGET: break

    mean_per = float(np.mean(fold_pers))
    collapse = prepare.compute_content_collapse(all_preds)
    print(f"\n---\nval_per:            {mean_per:.6f}")
    print(f"val_per_std:        {float(np.std(fold_pers)):.6f}")
    print(f"fold_pers:          {fold_pers}")
    print(f"collapsed:          {collapse['collapsed']}")
    print(f"mean_entropy:       {collapse['mean_entropy']:.3f}")
    print(f"training_seconds:   {time.time()-t0:.1f}")


if __name__ == "__main__":
    run()
