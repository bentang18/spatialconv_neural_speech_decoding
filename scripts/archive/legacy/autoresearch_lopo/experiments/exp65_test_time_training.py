#!/usr/bin/env python3
"""Exp 65: Test-time training with masked prediction on target val set.

After Stage 2 adaptation, before evaluation: run a few gradient steps
of reconstruction on the validation grids (no labels). Adapts backbone
to target patient's temporal dynamics at test time.

Baseline (exp33): No TTT, PER=0.762
Change: 10 gradient steps of MSE reconstruction on val grids before eval
"""
from __future__ import annotations

import math
import os
import sys
import time
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
    augment, SpatialReadIn, Backbone, ArticulatoryBottleneckHead,
    focal_ce, compute_loss, decode, extract_embeddings,
    knn_predict, simpleshot_predict, prototype_predict,
    DEVICE, SEED, S1_EPOCHS, S1_BATCH_SIZE, S1_LR, S1_READIN_LR_MULT,
    S1_WEIGHT_DECAY, S1_GRAD_CLIP, S1_WARMUP_EPOCHS, S1_PATIENCE,
    S1_EVAL_EVERY, S1_VAL_FRACTION, S2_EPOCHS, S2_BATCH_SIZE, S2_LR,
    S2_BACKBONE_LR_MULT, S2_READIN_LR_MULT, S2_WEIGHT_DECAY, S2_GRAD_CLIP,
    S2_WARMUP_EPOCHS, S2_PATIENCE, S2_EVAL_EVERY, MIXUP_ALPHA,
    TTA_COPIES, KNN_K, SOURCE_KNN_WEIGHT,
    train_stage1,
)

TTT_STEPS = 10
TTT_LR = 1e-4
TTT_MASK_RATIO = 0.4


def ttt_adapt(backbone_copy, target_ri, val_grids):
    """TTT: masked reconstruction on val grids (no labels)."""
    backbone_copy.train()
    target_ri.eval()  # don't adapt read-in
    opt = AdamW(backbone_copy.parameters(), lr=TTT_LR, weight_decay=0.0)

    for _ in range(TTT_STEPS):
        x = augment(val_grids).to(DEVICE)
        with torch.no_grad():
            feat = target_ri(x)

        # Get conv features as targets
        feat_normed = backbone_copy.ln(feat.permute(0, 2, 1)).permute(0, 2, 1)
        conv_out = backbone_copy.temporal_conv(feat_normed)  # (B, d, T')
        conv_feat = conv_out.permute(0, 2, 1)  # (B, T', d)
        T = conv_feat.shape[1]
        d = conv_feat.shape[2]
        targets = conv_feat.detach()

        # Mask and reconstruct
        n_mask = max(1, int(T * TTT_MASK_RATIO))
        mask_idx = torch.randperm(T, device=conv_feat.device)[:n_mask]
        masked = conv_feat.clone()
        masked[:, mask_idx, :] = 0.0

        h, _ = backbone_copy.gru(masked)
        preds = h[:, mask_idx, :d]  # use first d dims of GRU output

        loss = F.mse_loss(preds, targets[:, mask_idx, :])
        opt.zero_grad()
        loss.backward()
        opt.step()

    backbone_copy.eval()


def train_eval_fold(backbone, head_init, train_grids, train_labels,
                    val_grids, val_labels, read_ins, all_data, ReadInCls):
    """Standard S2 + TTT before eval."""
    backbone_copy = deepcopy(backbone)
    target_ri = ReadInCls(
        prepare.PATIENT_GRIDS[prepare.TARGET_PATIENT][0],
        prepare.PATIENT_GRIDS[prepare.TARGET_PATIENT][1]).to(DEVICE)
    head = deepcopy(head_init).to(DEVICE)

    n_train = len(train_grids)
    optimizer = AdamW([
        {"params": target_ri.parameters(), "lr": S2_LR * S2_READIN_LR_MULT},
        {"params": backbone_copy.parameters(), "lr": S2_LR * S2_BACKBONE_LR_MULT},
        {"params": head.parameters(), "lr": S2_LR},
    ], weight_decay=S2_WEIGHT_DECAY)

    def lr_lambda(epoch):
        if S2_WARMUP_EPOCHS > 0 and epoch < S2_WARMUP_EPOCHS:
            return (epoch + 1) / S2_WARMUP_EPOCHS
        progress = (epoch - S2_WARMUP_EPOCHS) / max(S2_EPOCHS - S2_WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    best_val_loss = float("inf"); best_state = None; patience_ctr = 0

    for epoch in range(S2_EPOCHS):
        backbone_copy.train(); target_ri.train(); head.train()
        perm = torch.randperm(n_train)
        for start in range(0, n_train, S2_BATCH_SIZE):
            idx = perm[start:start + S2_BATCH_SIZE]
            x = augment(train_grids[idx]).to(DEVICE)
            y = [train_labels[i] for i in idx.tolist()]
            mixup_y, mixup_lam = None, 1.0
            if MIXUP_ALPHA > 0 and len(idx) > 1:
                mixup_lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm_mix = torch.randperm(x.shape[0])
                x = mixup_lam * x + (1 - mixup_lam) * x[perm_mix]
                mixup_y = [y[i] for i in perm_mix.tolist()]
            optimizer.zero_grad()
            feat = target_ri(x); h = backbone_copy(feat); logits = head(h)
            loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)
            if math.isnan(loss.item()): sys.exit(1)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(backbone_copy.parameters()) + list(target_ri.parameters()) + list(head.parameters()),
                S2_GRAD_CLIP)
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % S2_EVAL_EVERY == 0:
            backbone_copy.eval(); target_ri.eval(); head.eval()
            with torch.no_grad():
                feat = target_ri(val_grids.to(DEVICE)); h = backbone_copy(feat)
                logits = head(h); val_loss = compute_loss(logits, val_labels).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "backbone": deepcopy(backbone_copy.state_dict()),
                    "readin": deepcopy(target_ri.state_dict()),
                    "head": deepcopy(head.state_dict()),
                }; patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= S2_PATIENCE: break

    if best_state:
        backbone_copy.load_state_dict(best_state["backbone"])
        target_ri.load_state_dict(best_state["readin"])
        head.load_state_dict(best_state["head"])

    # === TTT: adapt backbone to val distribution ===
    ttt_adapt(backbone_copy, target_ri, val_grids)

    backbone_copy.eval(); target_ri.eval(); head.eval()

    with torch.no_grad():
        if TTA_COPIES > 1:
            logits_sum = torch.zeros(len(val_grids), prepare.N_POSITIONS, prepare.N_CLASSES, device=DEVICE)
            logits_sum += head(backbone_copy(target_ri(val_grids.to(DEVICE))))
            for _ in range(TTA_COPIES - 1):
                logits_sum += head(backbone_copy(target_ri(augment(val_grids).to(DEVICE))))
            logits = logits_sum / TTA_COPIES
        else:
            logits = head(backbone_copy(target_ri(val_grids.to(DEVICE))))
    linear_preds = decode(logits)
    linear_per = prepare.compute_per(linear_preds, val_labels)

    train_emb = extract_embeddings(backbone_copy, target_ri, train_grids)
    train_emb_all = train_emb; train_labels_all = list(train_labels)
    if read_ins and all_data:
        se, sl = [], []
        for pid in prepare.SOURCE_PATIENTS:
            ri = read_ins[pid]; ri.eval()
            se.append(extract_embeddings(backbone_copy, ri, all_data[pid]["grids"]))
            sl.extend(all_data[pid]["labels"])
        train_emb_all = torch.cat([train_emb, torch.cat(se) * SOURCE_KNN_WEIGHT])
        train_labels_all = list(train_labels) + sl

    if TTA_COPIES > 1:
        ve = extract_embeddings(backbone_copy, target_ri, val_grids)
        for _ in range(TTA_COPIES - 1):
            ve = ve + extract_embeddings(backbone_copy, target_ri, augment(val_grids))
        val_emb = ve / TTA_COPIES
    else:
        val_emb = extract_embeddings(backbone_copy, target_ri, val_grids)

    kp = knn_predict(train_emb_all, train_labels_all, val_emb)
    knn_per = prepare.compute_per(kp, val_labels)
    sp = simpleshot_predict(train_emb, list(train_labels), val_emb)
    ss_per = prepare.compute_per(sp, val_labels)
    pp = prototype_predict(train_emb, list(train_labels), val_emb)
    proto_per = prepare.compute_per(pp, val_labels)

    methods = {"linear": (linear_per, linear_preds), "knn": (knn_per, kp),
               "simpleshot": (ss_per, sp), "prototype": (proto_per, pp)}
    best_name = min(methods, key=lambda k: methods[k][0])
    return methods[best_name][0], methods[best_name][1], methods


if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    all_data = prepare.load_all_patients()
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)

    N, H, W, T = grids.shape
    print(f"=== exp65_test_time_training ===")
    print(f"Target: {prepare.TARGET_PATIENT} | TTT: {TTT_STEPS} steps @ lr={TTT_LR}")
    print(f"Device: {DEVICE}\n")

    print("=== Stage 1 ===")
    s1_start = time.time()
    backbone, head, read_ins = train_stage1(all_data, SpatialReadIn, Backbone, ArticulatoryBottleneckHead)
    s1_time = time.time() - s1_start
    print(f"Stage 1 done in {s1_time:.1f}s\n")

    print("=== Stage 2 + TTT ===")
    fold_pers = []; all_preds = []
    method_pers = {m: [] for m in ["linear", "knn", "simpleshot", "prototype"]}

    for fi, (tr_idx, va_idx) in enumerate(splits):
        ft0 = time.time()
        per, preds, methods = train_eval_fold(
            backbone, head, grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx],
            read_ins=read_ins, all_data=all_data, ReadInCls=SpatialReadIn)
        fold_pers.append(per); all_preds.extend(preds)
        for m, (p, _) in methods.items(): method_pers[m].append(p)
        print(f"  Fold {fi+1}: PER={per:.4f} ({time.time()-ft0:.1f}s)")
        if time.time() - t0 > prepare.TIME_BUDGET: break

    mean_per = float(np.mean(fold_pers))
    collapse = prepare.compute_content_collapse(all_preds)
    print(f"\n---")
    print(f"val_per:            {mean_per:.6f}")
    print(f"val_per_std:        {float(np.std(fold_pers)):.6f}")
    print(f"fold_pers:          {fold_pers}")
    print(f"training_seconds:   {time.time()-t0:.1f}")
    print(f"collapsed:          {collapse['collapsed']}")
    print(f"mean_entropy:       {collapse['mean_entropy']:.3f}")
    print(f"stereotypy:         {collapse['stereotypy']:.3f}")
    print(f"unique_ratio:       {collapse['unique_ratio']:.3f}")
