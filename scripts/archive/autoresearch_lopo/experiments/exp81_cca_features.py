#!/usr/bin/env python3
"""Exp 81: CCA alignment on backbone features (Spalding hybrid).

Spalding's key: explicit CCA alignment using condition averages.
We combine our learned backbone features with Spalding's alignment:
  1. Train S1 normally (backbone learns features)
  2. Extract embeddings for all patients
  3. CCA-align source embeddings → target embedding space
  4. k-NN on aligned features

This tests whether explicit alignment on top of learned features
outperforms the implicit alignment from shared backbone training.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare
from arch_ablation_base import (
    DEVICE, SEED, SpatialReadIn, Backbone, ArticulatoryBottleneckHead,
    train_stage1, extract_embeddings, knn_predict, simpleshot_predict,
)


def cca_align(X_source, y_source, X_target, y_target, n_components=None):
    """CCA-align source embeddings to target space using condition averages.

    Following Spalding: average by phoneme label, then CCA on averages,
    apply transform to all samples.
    """
    # Compute condition averages (per phoneme per position → per unique label tuple)
    def condition_average(X, y):
        label_tuples = [tuple(l) for l in y]
        unique = sorted(set(label_tuples))
        avgs = []
        for u in unique:
            mask = [i for i, l in enumerate(label_tuples) if l == u]
            avgs.append(X[mask].mean(dim=0))
        return torch.stack(avgs), unique

    src_avg, src_labels = condition_average(X_source, y_source)
    tgt_avg, tgt_labels = condition_average(X_target, y_target)

    # Keep only shared conditions
    shared = sorted(set(src_labels) & set(tgt_labels))
    if len(shared) < 5:
        return X_source  # too few shared conditions, return unaligned

    src_idx = [src_labels.index(s) for s in shared]
    tgt_idx = [tgt_labels.index(s) for s in shared]
    S = src_avg[src_idx].numpy()  # (n_shared, d)
    T = tgt_avg[tgt_idx].numpy()  # (n_shared, d)

    # Center
    S_mean, T_mean = S.mean(0), T.mean(0)
    Sc, Tc = S - S_mean, T - T_mean

    # CCA via SVD of cross-covariance
    # Following Spalding: QR decomposition then SVD
    Q_s, R_s = np.linalg.qr(Sc, mode='reduced')
    Q_t, R_t = np.linalg.qr(Tc, mode='reduced')

    U, s, Vt = np.linalg.svd(Q_s.T @ Q_t, full_matrices=False)

    if n_components is None:
        n_components = min(S.shape[1], T.shape[1], len(shared))

    # Projection: source → target space
    # M_s = R_s^-1 @ U[:, :k], M_t = R_t^-1 @ Vt[:k, :].T
    try:
        R_s_inv = np.linalg.pinv(R_s)
        R_t_inv = np.linalg.pinv(R_t)
    except np.linalg.LinAlgError:
        return X_source

    M_s = R_s_inv @ U[:, :n_components]
    M_t = R_t_inv @ Vt[:n_components, :].T

    # Transform source: center, project to CCA space, then to target space
    # X_aligned = (X - S_mean) @ M_s @ pinv(M_t) + T_mean
    try:
        M_t_inv = np.linalg.pinv(M_t)
        transform = M_s @ M_t_inv
    except np.linalg.LinAlgError:
        return X_source

    X_np = X_source.numpy()
    X_aligned = (X_np - S_mean) @ transform + T_mean
    return torch.from_numpy(X_aligned.astype(np.float32))


def run():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    all_data = prepare.load_all_patients()
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)
    N, H, W, T = grids.shape

    print("=== exp81_cca_features ===")
    print(f"Target: {prepare.TARGET_PATIENT} | CCA on backbone features")

    backbone, head, read_ins = train_stage1(
        all_data, SpatialReadIn, Backbone, ArticulatoryBottleneckHead)

    # Extract all source embeddings
    source_data = {}  # pid -> (embeddings, labels)
    for pid in prepare.SOURCE_PATIENTS:
        emb = extract_embeddings(backbone, read_ins[pid], all_data[pid]["grids"])
        source_data[pid] = (emb, all_data[pid]["labels"])

    target_ri = SpatialReadIn(H, W).to(DEVICE)

    fold_pers = []
    all_preds, all_refs = [], []

    for fi, (tr_idx, va_idx) in enumerate(splits):
        train_emb = extract_embeddings(backbone, target_ri, grids[tr_idx])
        train_labels = [labels[i] for i in tr_idx]
        val_emb = extract_embeddings(backbone, target_ri, grids[va_idx])
        val_labels = [labels[i] for i in va_idx]

        # Method 1: unaligned k-NN (baseline comparison)
        knn_preds = knn_predict(train_emb, train_labels, val_emb)
        knn_per = prepare.compute_per(knn_preds, val_labels)

        # Method 2: CCA-aligned source + target k-NN
        aligned_embs, aligned_labs = [train_emb], list(train_labels)
        for pid in prepare.SOURCE_PATIENTS:
            src_emb, src_labels = source_data[pid]
            aligned = cca_align(src_emb, src_labels, train_emb, train_labels)
            aligned_embs.append(aligned * 0.5)  # weight as in baseline
            aligned_labs.extend(src_labels)
        combined_emb = torch.cat(aligned_embs, dim=0)
        knn_cca_preds = knn_predict(combined_emb, aligned_labs, val_emb)
        knn_cca_per = prepare.compute_per(knn_cca_preds, val_labels)

        # Method 3: unaligned source + target k-NN (for comparison)
        unaligned_embs, unaligned_labs = [train_emb], list(train_labels)
        for pid in prepare.SOURCE_PATIENTS:
            src_emb, src_labels = source_data[pid]
            unaligned_embs.append(src_emb * 0.5)
            unaligned_labs.extend(src_labels)
        unaligned_combined = torch.cat(unaligned_embs, dim=0)
        knn_unaligned = knn_predict(unaligned_combined, unaligned_labs, val_emb)
        knn_unaligned_per = prepare.compute_per(knn_unaligned, val_labels)

        best_per = min(knn_per, knn_cca_per, knn_unaligned_per)
        best_preds = [knn_preds, knn_cca_preds, knn_unaligned][
            [knn_per, knn_cca_per, knn_unaligned_per].index(best_per)]
        fold_pers.append(best_per)
        all_preds.extend(best_preds)
        all_refs.extend(val_labels)

        print(f"  Fold {fi+1}: tgt_knn={knn_per:.4f} cca_knn={knn_cca_per:.4f} "
              f"unaligned_knn={knn_unaligned_per:.4f} best={best_per:.4f}")

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
