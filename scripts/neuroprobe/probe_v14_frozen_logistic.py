#!/usr/bin/env python
"""Frozen-encoder L2-logistic probe for ONE Neuroprobe-Lite cell.

Loads the v14 2STFT joint-SSL encoder (``student.encoder.*`` from a Lightning
ckpt), freezes it, extracts a frozen feature per clip, and fits an sklearn
``LogisticRegression`` (leaderboard-parity: StandardScaler + L2, C=1.0) on the
parity-locked train split, scoring AUROC on the held-out test split.

Feature reduction (Ben 2026-06-15 — "mean time, flatten freq + parcels"):
  M4 latent (K parcels, S=104 dual-band freq×time tokens, d) →
    group the S tokens by the model's own ``freq_patch_idx`` (low 0..6, high
    7..9) and MEAN within each group  →  (K, F_p=10, d)   [= mean over time per
    freq band; low groups avg 8 time-patches, high groups avg 16; the low→high
    band jump is a distinct freq index, never averaged across]
    →  keep VALID parcels (latent_valid)  →  (K_valid, 10, d)
    →  flatten  →  K_valid·10·d feature vector.
  CrossSubject restricts to the common-valid parcels of (train S2 ∩ test subj),
  mirroring upstream ``combine_regions`` region-intersection.

Run on DCC (cwd = /work/ht203/repo/speech). One cell:

    ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank \
    EXCA_CACHE_FOLDER=/work/ht203/cache_neuroai \
    EXCA_EXTRACTOR_CACHE_FOLDER=/work/ht203/cache/v14_extractors \
    .venv/bin/python scripts/neuroprobe/probe_v14_frozen_logistic.py \
        --ckpt <ladder-25000.ckpt> --task onset --eval-mode within \
        --subject 1 --trial 1 --fold 0 --out /work/ht203/probe_out/cell.json

The per-cell JSON sidecar is aggregated by the paired collector
``collect_frozen_logistic.py`` (sweep discipline — no sweep without a collector).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# --- run architecture (launch_2stft_fd8ld4_pd4w128_lr3e3.sh = source of truth) -
D_MODEL = 256
N_HEADS = 4
DEPTH_SELF_ATTN = 8
LATENT_PARCEL_DEPTH = 4
ATLAS = "dkt"
SPEC_CACHE_DIR = "/hpc/group/coganlab/ht203/cache_neuroai/v14_2stft_spec_cache"

_MODE_ALIASES = {
    "within": "WithinSession",
    "csession": "CrossSession",
    "csubject": "CrossSubject",
}


def build_frozen_encoder(experiment, ckpt_path: str):
    """Build the encoder from the experiment's brain_model_config and load
    ``student.encoder.*`` from the Lightning ckpt (strict-equivalent)."""
    cfg = experiment.brain_model_config
    head_model = cfg.build(n_outputs=2)
    encoder = head_model.encoder

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    prefix = "student.encoder."
    stripped = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not stripped:
        sys.exit(f"no {prefix!r} keys in ckpt; first keys: {list(sd)[:10]}")
    missing, unexpected = encoder.load_state_dict(stripped, strict=False)
    nonpersistent_ok = {"freq_embed", "key_rope", "rope_joint_token"}
    hard_missing = [m for m in missing if m.split(".")[0] not in nonpersistent_ok]
    if hard_missing or unexpected:
        sys.exit(f"load FAILED: hard_missing={hard_missing} unexpected={list(unexpected)}")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def _freq_patch_index(encoder, et_high: torch.Tensor) -> torch.Tensor:
    """The model's own (S,) freq-patch id for every dual-band token (low 0..F_p_low-1,
    high F_p_low..F_p-1). Grouping the S axis by this id == 'mean time per freq band'."""
    F_p_low, T_low_p, F_p_high, T_high_p = encoder.dual_band_grid_shape(et_high)
    layout = encoder.dual_band_token_layout(T_low_p, T_high_p, device=et_high.device)
    return layout["freq_patch_idx"], (F_p_low + F_p_high)


def reduce_meantime_keepfreq(m4: torch.Tensor, fpi: torch.Tensor, f_p: int) -> torch.Tensor:
    """(B, K, S, d) → (B, K, F_p, d): mean the S tokens that share each freq patch
    (= mean over time within a freq band; per-band time counts differ, handled by
    the group sizes; the low→high jump is a distinct freq id so never crossed)."""
    B, K, S, d = m4.shape
    assert fpi.numel() == S, f"freq_patch_idx ({fpi.numel()}) != S ({S})"
    out = m4.new_zeros(B, K, f_p, d)
    for f in range(f_p):
        sel = fpi == f
        out[:, :, f, :] = m4[:, :, sel, :].mean(dim=2)
    return out


@torch.no_grad()
def extract_split(encoder, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward every batch → (feats (N,K,F_p,d), labels (N,), latent_valid (N,K) bool)."""
    feats, labels, lvs = [], [], []
    fpi = f_p = None
    for batch in loader:
        data = batch.data
        et_low = data["electrode_tokens"].float()
        et_high = data["electrode_tokens_high"].float()
        support = data["support"].float()
        valid_mask = data["valid_mask"].bool() if "valid_mask" in data else None
        taps = encoder(et_low, support, valid_mask,
                       electrode_tokens_high=et_high, return_taps=True)
        m4 = taps["M4"]                       # (B, K, S, d)
        lv = taps["latent_valid"].bool()      # (B, K)
        if fpi is None:
            fpi, f_p = _freq_patch_index(encoder, et_high)
        red = reduce_meantime_keepfreq(m4, fpi, f_p)   # (B, K, F_p, d)
        feats.append(red.cpu().numpy())
        lvs.append(lv.cpu().numpy())
        y = data["target"]
        y = y.reshape(y.shape[0], -1)
        y = y.argmax(dim=1) if y.shape[1] > 1 else y[:, 0]
        labels.append(y.cpu().numpy())
    return (np.concatenate(feats), np.concatenate(labels).astype(int),
            np.concatenate(lvs))


def fit_score(X_tr, y_tr, X_te, y_te, c: float, seed: int) -> dict:
    """StandardScaler + L2 LogisticRegression (upstream-parity), AUROC + bal-acc."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr)
    Xtr, Xte = scaler.transform(X_tr), scaler.transform(X_te)
    clf = LogisticRegression(C=c, random_state=seed, max_iter=10000, tol=1e-3)
    clf.fit(Xtr, y_tr)
    n_classes = int(len(np.unique(y_tr)))
    proba = clf.predict_proba(Xte)
    if n_classes == 2:
        auroc = float(roc_auc_score(y_te, proba[:, 1]))
    else:
        # restrict to classes present in y_te for a defined OVR macro-AUROC
        labels = clf.classes_
        auroc = float(roc_auc_score(y_te, proba, multi_class="ovr",
                                    average="macro", labels=labels))
    bal_acc = float(balanced_accuracy_score(y_te, clf.predict(Xte)))
    return {"auroc": auroc, "bal_acc": bal_acc, "n_classes": n_classes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="onset")
    ap.add_argument("--eval-mode", default="within", choices=list(_MODE_ALIASES))
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--C", type=float, default=1.0, help="L2 inverse-reg (1.0=parity).")
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--out", default=None, help="JSON sidecar path.")
    ap.add_argument("--spec-cache-dir", default=SPEC_CACHE_DIR)
    args = ap.parse_args()

    eval_mode = _MODE_ALIASES[args.eval_mode]
    t0 = time.time()
    if "ROOT_DIR_BRAINTREEBANK" not in os.environ:
        sys.exit("export ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank")

    import speech_decoding.models  # noqa: F401  registers V14ParcelPerceiver
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    experiment = build_v14_experiment(
        mode="lite", task=args.task, eval_mode=eval_mode,
        test_subject_id=args.subject, test_trial_id=args.trial,
        fold_index=args.fold, binary_tasks=True,
        frontend="2stft", pool="mean", mean_pool_std=True,
        atlas=ATLAS, exclude_single_electrode_parcels=True,
        latent_mode="parcel", latent_depth=LATENT_PARCEL_DEPTH,
        d_model=D_MODEL, n_heads=N_HEADS, depth=DEPTH_SELF_ATTN,
        phase4_frozen_probe=True, electrode_set="lite",
        clip_len=1.0, neural_lag_s=0.0,
        spec_cache_dir=args.spec_cache_dir,
        batch_size=64, num_workers=0, precision=None,
    )
    encoder = build_frozen_encoder(experiment, args.ckpt)
    loaders = experiment.data.build(worker_seed=experiment.seed)

    Xtr, ytr, lv_tr = extract_split(encoder, loaders["train"])
    Xte, yte, lv_te = extract_split(encoder, loaders["test"])

    # Valid (covered) parcels are session-constant; require valid across the split.
    valid_tr = lv_tr.all(axis=0)            # (K,)
    valid_te = lv_te.all(axis=0)
    if eval_mode == "CrossSubject":
        sel = valid_tr & valid_te           # common-valid (train S2 ∩ test subj)
    else:
        sel = valid_tr                       # same subject → same valid set
    k_sel = int(sel.sum())
    Xtr2 = Xtr[:, sel, :, :].reshape(Xtr.shape[0], -1)
    Xte2 = Xte[:, sel, :, :].reshape(Xte.shape[0], -1)

    scores = fit_score(Xtr2, ytr, Xte2, yte, args.C, args.seed)
    rec = {
        "task": args.task, "eval_mode": eval_mode,
        "subject": args.subject, "trial": args.trial, "fold": args.fold,
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
        "k_valid_train": int(valid_tr.sum()), "k_valid_test": int(valid_te.sum()),
        "k_sel": k_sel, "f_p": int(Xtr.shape[2]), "d": int(Xtr.shape[3]),
        "feature_dim": int(Xtr2.shape[1]), "C": args.C,
        "reduction": "meantime_keepfreq", "ckpt": args.ckpt,
        "elapsed_s": round(time.time() - t0, 1), **scores,
    }
    print(json.dumps(rec, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rec))
        print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
