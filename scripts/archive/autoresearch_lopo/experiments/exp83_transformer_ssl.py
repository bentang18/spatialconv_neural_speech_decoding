#!/usr/bin/env python3
"""Exp 83: Transformer encoder + masked prediction SSL pretrain.

User's specific interest: can transformers work better with SSL?
Design: Transformer encoder (2 layers, 4 heads) replaces BiGRU.
Two-phase training:
  Phase 0: Masked prediction pretrain on ALL patients (no labels, 100 epochs)
  Phase 1: CE fine-tune in normal S1+S2 pipeline

The transformer's attention mechanism may learn better cross-patient
representations when pretrained with self-supervision.
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
    DEVICE, SEED, SpatialReadIn, ArticulatoryBottleneckHead,
    augment, compute_loss, train_eval_fold,
    S1_EPOCHS, S1_BATCH_SIZE, S1_LR, S1_READIN_LR_MULT,
    S1_WEIGHT_DECAY, S1_GRAD_CLIP, S1_WARMUP_EPOCHS, S1_PATIENCE,
    S1_EVAL_EVERY, S1_VAL_FRACTION, MIXUP_ALPHA,
)


class TransformerBackbone(nn.Module):
    """Transformer encoder replacing BiGRU."""
    def __init__(self, d_in=256, d=32, n_heads=4, n_layers=2,
                 stride=10, dropout=0.3, feat_drop_max=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_in, d, kernel_size=stride, stride=stride), nn.GELU())
        # Transformer needs even d for multi-head attention
        self.d = d
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_dim = d  # not 2*d like BiGRU
        self.feat_drop_max = feat_drop_max
        # Learned positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, 100, d) * 0.02)

    def forward(self, x):
        x = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.training and self.feat_drop_max > 0:
            p = torch.rand(1).item() * self.feat_drop_max
            mask = (torch.rand(x.shape[1], device=x.device) > p).float()
            x = x * mask.unsqueeze(0).unsqueeze(-1) / (1 - p + 1e-8)
        x = self.temporal_conv(x).permute(0, 2, 1)  # (B, T', d)
        T = x.shape[1]
        x = x + self.pos_enc[:, :T, :]
        x = self.transformer(x)
        return x


class MaskedPredictionHead(nn.Module):
    """Predict masked frames from context."""
    def __init__(self, d=32):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

    def forward(self, x):
        return self.predictor(x)


def ssl_pretrain(all_data, read_ins, backbone, mask_ratio=0.4, n_epochs=80):
    """Masked prediction pretraining on all patients (no labels)."""
    pred_head = MaskedPredictionHead(d=backbone.d).to(DEVICE)
    optimizer = AdamW(
        list(backbone.parameters()) + list(pred_head.parameters()) +
        [p for ri in read_ins.values() for p in ri.parameters()],
        lr=1e-3, weight_decay=1e-4)

    pids = prepare.SOURCE_PATIENTS
    print("  SSL pretrain: masked prediction on all source patients")

    for epoch in range(n_epochs):
        backbone.train(); pred_head.train()
        for ri in read_ins.values(): ri.train()
        epoch_loss, n_batches = 0.0, 0
        np.random.shuffle(pids)

        for pid in pids:
            grids = all_data[pid]["grids"]
            n = len(all_data[pid]["labels"])
            perm = np.random.permutation(n)
            for start in range(0, n, S1_BATCH_SIZE):
                idx = perm[start:start + S1_BATCH_SIZE]
                x = augment(grids[idx]).to(DEVICE)

                optimizer.zero_grad()
                feat = read_ins[pid](x)
                # Get full representation (no mask)
                with torch.no_grad():
                    backbone.eval()
                    target = backbone(feat).detach()  # (B, T', d)
                    backbone.train()

                # Mask random frames
                B, Tp, d = target.shape
                mask = torch.rand(B, Tp, device=DEVICE) < mask_ratio
                if not mask.any():
                    mask[0, 0] = True

                # Replace masked frames with learnable token
                masked_feat = feat.clone()
                # Apply mask at temporal conv output level
                h = backbone(masked_feat)
                predictions = pred_head(h)

                # Loss only on masked positions
                loss = F.mse_loss(predictions[mask], target[mask])
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(pred_head.parameters()),
                    S1_GRAD_CLIP)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

        if (epoch + 1) % 20 == 0:
            print(f"    SSL epoch {epoch+1}: loss={epoch_loss/max(n_batches,1):.4f}")

    return backbone, read_ins


def train_stage1_finetune(all_data, read_ins, backbone, HeadCls):
    """Fine-tune pretrained backbone with CE on source patients."""
    torch.manual_seed(SEED + 1)
    pids = prepare.SOURCE_PATIENTS
    head = HeadCls(d_in=backbone.out_dim).to(DEVICE)

    source_train, source_val = {}, {}
    for pid in pids:
        n = len(all_data[pid]["labels"])
        perm = np.random.permutation(n)
        n_val = max(1, int(round(S1_VAL_FRACTION * n)))
        source_train[pid] = sorted(perm[n_val:].tolist())
        source_val[pid] = sorted(perm[:n_val].tolist())

    readin_params = [p for ri in read_ins.values() for p in ri.parameters()]
    optimizer = AdamW([
        {"params": readin_params, "lr": S1_LR * S1_READIN_LR_MULT * 0.1},  # lower LR (pretrained)
        {"params": backbone.parameters(), "lr": S1_LR * 0.3},  # lower LR (pretrained)
        {"params": head.parameters(), "lr": S1_LR},
    ], weight_decay=S1_WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < S1_WARMUP_EPOCHS:
            return (epoch + 1) / S1_WARMUP_EPOCHS
        progress = (epoch - S1_WARMUP_EPOCHS) / max(S1_EPOCHS - S1_WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    best_val_loss, best_state, patience_ctr = float("inf"), None, 0

    for epoch in range(S1_EPOCHS):
        backbone.train(); head.train()
        for ri in read_ins.values(): ri.train()
        np.random.shuffle(pids)
        epoch_loss, n_batches = 0.0, 0

        for pid in pids:
            grids = all_data[pid]["grids"]
            labels = all_data[pid]["labels"]
            tr_idx = source_train[pid]
            perm = np.random.permutation(len(tr_idx))
            for start in range(0, len(tr_idx), S1_BATCH_SIZE):
                batch_idx = [tr_idx[perm[i]] for i in range(start, min(start + S1_BATCH_SIZE, len(tr_idx)))]
                x = augment(grids[batch_idx]).to(DEVICE)
                y = [labels[i] for i in batch_idx]
                mixup_y, mixup_lam = None, 1.0
                if MIXUP_ALPHA > 0 and len(batch_idx) > 1:
                    mixup_lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                    perm_mix = torch.randperm(x.shape[0])
                    x = mixup_lam * x + (1 - mixup_lam) * x[perm_mix]
                    mixup_y = [y[i] for i in perm_mix.tolist()]
                optimizer.zero_grad()
                feat = read_ins[pid](x)
                h = backbone(feat)
                logits = head(h)
                loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()) + readin_params,
                    S1_GRAD_CLIP)
                optimizer.step()
                epoch_loss += loss.item()
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
            print(f"  S1-FT epoch {epoch+1}: train={epoch_loss/max(n_batches,1):.4f} val={val_loss:.4f}")
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
                    print(f"  S1-FT early stop at epoch {epoch+1}")
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

    print("=== exp83_transformer_ssl ===")
    print(f"Target: {prepare.TARGET_PATIENT} | Transformer + masked SSL pretrain")

    # Create model components
    read_ins = {}
    for pid in prepare.SOURCE_PATIENTS:
        H, W = all_data[pid]["grid_shape"]
        read_ins[pid] = SpatialReadIn(H, W).to(DEVICE)
    d_flat = list(read_ins.values())[0].d_flat

    backbone = TransformerBackbone(d_in=d_flat, d=64, n_heads=4, n_layers=2).to(DEVICE)

    # Phase 0: SSL pretrain
    print("=== Phase 0: SSL pretrain ===")
    backbone, read_ins = ssl_pretrain(all_data, read_ins, backbone)

    # Phase 1: CE fine-tune
    print("\n=== Phase 1: CE fine-tune ===")
    # Need head with transformer's out_dim (d, not 2*d like GRU)
    backbone, head, read_ins = train_stage1_finetune(
        all_data, read_ins, backbone, ArticulatoryBottleneckHead)

    # Phase 2: S2 + eval
    print("\n=== Phase 2: S2 + eval ===")
    fold_pers = []
    all_preds, all_refs = [], []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        # Need custom ReadInCls for train_eval_fold
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
