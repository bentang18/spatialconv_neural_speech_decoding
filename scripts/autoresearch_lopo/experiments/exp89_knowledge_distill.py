#!/usr/bin/env python3
"""Exp 89: Knowledge distillation from per-patient teacher models.

Per-patient CE gets ~0.70 PER on S14 (stratified). These per-patient
models learn strong patient-specific features. Can we distill this
knowledge into the cross-patient backbone?

Approach:
1. Train per-patient teachers on each source patient
2. Extract teacher features (backbone embeddings)
3. Add distillation loss: MSE between student backbone and teacher backbone
4. S1 loss = CE_phoneme + distill_weight * MSE_features

This is DIFFERENT from pseudo-labeling: we're aligning feature
representations, not just labels. The teacher provides a richer
signal about what the "right" features should look like.
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
    S1_EVAL_EVERY, S1_VAL_FRACTION, MIXUP_ALPHA,
)


def train_teacher(grids, labels, grid_shape, n_epochs=100):
    """Train a single per-patient teacher model."""
    H, W = grid_shape
    ri = SpatialReadIn(H, W).to(DEVICE)
    bb = Backbone(d_in=ri.d_flat).to(DEVICE)
    head = ArticulatoryBottleneckHead(d_in=bb.out_dim).to(DEVICE)

    n = len(labels)
    perm = np.random.permutation(n)
    n_val = max(1, int(round(0.2 * n)))
    tr_idx, va_idx = perm[n_val:].tolist(), perm[:n_val].tolist()

    optimizer = AdamW(list(ri.parameters()) + list(bb.parameters()) + list(head.parameters()),
                      lr=1e-3, weight_decay=1e-4)

    best_loss, best_state = float("inf"), None
    for epoch in range(n_epochs):
        bb.train(); ri.train(); head.train()
        ep_perm = np.random.permutation(len(tr_idx))
        for start in range(0, len(tr_idx), 16):
            idx = [tr_idx[ep_perm[i]] for i in range(start, min(start + 16, len(tr_idx)))]
            x = augment(grids[idx]).to(DEVICE)
            y = [labels[i] for i in idx]
            optimizer.zero_grad()
            loss = compute_loss(head(bb(ri(x))), y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0 and va_idx:
            bb.eval(); ri.eval(); head.eval()
            with torch.no_grad():
                vl = compute_loss(head(bb(ri(grids[va_idx].to(DEVICE)))),
                                  [labels[i] for i in va_idx]).item()
            if vl < best_loss:
                best_loss = vl
                best_state = {"ri": deepcopy(ri.state_dict()), "bb": deepcopy(bb.state_dict())}

    if best_state:
        ri.load_state_dict(best_state["ri"])
        bb.load_state_dict(best_state["bb"])
    ri.eval(); bb.eval()
    return ri, bb


def train_stage1_distill(all_data, teachers):
    """S1 with knowledge distillation from per-patient teachers."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    pids = prepare.SOURCE_PATIENTS

    read_ins = {}
    for pid in pids:
        H, W = all_data[pid]["grid_shape"]
        read_ins[pid] = SpatialReadIn(H, W).to(DEVICE)
    d_flat = list(read_ins.values())[0].d_flat

    backbone = Backbone(d_in=d_flat).to(DEVICE)
    head = ArticulatoryBottleneckHead(d_in=backbone.out_dim).to(DEVICE)

    source_train, source_val = {}, {}
    for pid in pids:
        n = len(all_data[pid]["labels"])
        perm = np.random.permutation(n)
        n_val = max(1, int(round(S1_VAL_FRACTION * n)))
        source_train[pid] = sorted(perm[n_val:].tolist())
        source_val[pid] = sorted(perm[:n_val].tolist())

    readin_params = [p for ri in read_ins.values() for p in ri.parameters()]
    optimizer = AdamW([
        {"params": readin_params, "lr": S1_LR * S1_READIN_LR_MULT},
        {"params": backbone.parameters(), "lr": S1_LR},
        {"params": head.parameters(), "lr": S1_LR},
    ], weight_decay=S1_WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < S1_WARMUP_EPOCHS:
            return (epoch + 1) / S1_WARMUP_EPOCHS
        progress = (epoch - S1_WARMUP_EPOCHS) / max(S1_EPOCHS - S1_WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    best_val_loss, best_state, patience_ctr = float("inf"), None, 0
    distill_weight = 0.5

    for epoch in range(S1_EPOCHS):
        backbone.train(); head.train()
        for ri in read_ins.values(): ri.train()
        np.random.shuffle(pids)
        epoch_ce, epoch_distill, n_batches = 0.0, 0.0, 0

        for pid in pids:
            grids = all_data[pid]["grids"]
            labels = all_data[pid]["labels"]
            tr_idx = source_train[pid]
            teacher_ri, teacher_bb = teachers[pid]
            perm = np.random.permutation(len(tr_idx))
            for start in range(0, len(tr_idx), S1_BATCH_SIZE):
                batch_idx = [tr_idx[perm[i]] for i in range(start, min(start + S1_BATCH_SIZE, len(tr_idx)))]
                x = augment(grids[batch_idx]).to(DEVICE)
                y = [labels[i] for i in batch_idx]

                optimizer.zero_grad()

                # Student forward
                feat = read_ins[pid](x)
                h_student = backbone(feat)
                logits = head(h_student)
                ce_loss = compute_loss(logits, y)

                # Teacher forward (no mixup for distillation)
                with torch.no_grad():
                    t_feat = teacher_ri(x)
                    h_teacher = teacher_bb(t_feat)

                # Distillation: align mean-pooled features
                s_pooled = h_student.mean(dim=1)
                t_pooled = h_teacher.mean(dim=1)
                distill_loss = F.mse_loss(s_pooled, t_pooled)

                loss = ce_loss + distill_weight * distill_loss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()) + readin_params,
                    S1_GRAD_CLIP)
                optimizer.step()
                epoch_ce += ce_loss.item()
                epoch_distill += distill_loss.item()
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
                    logits = head(backbone(read_ins[pid](x)))
                    val_loss += compute_loss(logits, y).item()
                    val_batches += 1
            val_loss /= max(val_batches, 1)
            print(f"  S1 epoch {epoch+1}: ce={epoch_ce/max(n_batches,1):.4f} "
                  f"distill={epoch_distill/max(n_batches,1):.4f} val={val_loss:.4f}")
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

    print("=== exp89_knowledge_distill ===")
    print(f"Target: {prepare.TARGET_PATIENT} | Knowledge distillation from per-patient teachers")

    # Train per-patient teachers
    print("\n--- Training per-patient teachers ---")
    teachers = {}
    for pid in prepare.SOURCE_PATIENTS:
        t_start = time.time()
        ri, bb = train_teacher(
            all_data[pid]["grids"], all_data[pid]["labels"],
            all_data[pid]["grid_shape"])
        teachers[pid] = (ri, bb)
        print(f"  Teacher {pid}: {time.time()-t_start:.1f}s")

    # S1 with distillation
    print("\n--- S1 with distillation ---")
    backbone, head, read_ins = train_stage1_distill(all_data, teachers)

    # S2 + eval
    print("\n--- S2 + eval ---")
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
        print(f"  Fold {fi+1}: PER={per:.4f} (best={best_method}) ({time.time()-t0:.1f}s)")
        if time.time() - t0 > 1200: break  # 20 min budget

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
