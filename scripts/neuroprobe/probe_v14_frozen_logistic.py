#!/usr/bin/env python
"""Frozen-encoder downstream probe for ONE Neuroprobe-Lite cell.

Loads the v14 2STFT joint-SSL encoder (``student.encoder.*`` from a Lightning
ckpt), freezes it, taps the M4 parcel latent, and scores a downstream readout on
the parity-locked train/test split. Three readouts are compared (``--reduction``):

  meantime_keepfreq  (DEFAULT, Ben 2026-06-15 "mean time, flatten freq+parcels")
      group the S=104 dual-band tokens by the model's own ``freq_patch_idx``
      (low 0..6, high 7..9) and MEAN within each group → (K, F_p=10, d)
      [mean over time per freq band; low groups avg 8 time-patches, high 16; the
       low→high band jump is a distinct freq id, never averaged across].
      keep VALID parcels → flatten → K_valid·10·d. sklearn L2-logistic (parity).

  meanall  (Ben 2026-06-15 "mean pool freq as well")
      mean over ALL S tokens → (K, d). keep valid → K_valid·d. sklearn L2-logistic.

  flatten  (Ben 2026-06-15 "simple flatten all linear reg")
      no pooling — keep every token. (K, S, d) → keep valid → K_valid·S·d
      (425,984 for S1). sklearn L2-logistic (heavy L2 shrinkage; p/n ≈ 270).

  attentive  (Ben 2026-06-15 "attentive pool over all ~16×104 tokens")
      single-seed multihead PMA over the K_valid·S valid tokens → d → linear.
      TRAINABLE (gradient-trained head, NOT sklearn); ~263k pool params + linear.
      Early-stopped on the val split. (B35 found this hard to train at ≤3500
      samples/task with a frozen encoder — measured here directly.)

CrossSubject restricts to common-valid parcels (train S2 ∩ test subj), mirroring
upstream region-intersection.

Run on DCC (cwd = /work/ht203/repo/speech). One cell:

    ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank \
    EXCA_CACHE_FOLDER=/work/ht203/cache_neuroai \
    EXCA_EXTRACTOR_CACHE_FOLDER=/work/ht203/cache/v14_extractors \
    .venv/bin/python scripts/neuroprobe/probe_v14_frozen_logistic.py \
        --ckpt <ladder-25000.ckpt> --task onset --eval-mode within \
        --subject 1 --trial 1 --fold 0 --reduction meantime_keepfreq \
        --out /work/ht203/probe_out/cell.json

Per-cell JSON sidecars are aggregated by ``collect_frozen_logistic.py``.
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
_REDUCTIONS = ("meantime_keepfreq", "meanall", "attentive", "flatten")


def build_frozen_encoder(experiment, ckpt_path: str):
    """Build the encoder from brain_model_config and load ``student.encoder.*``."""
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


def _freq_patch_index(encoder, et_high: torch.Tensor):
    """(S,) freq-patch id per dual-band token (low 0..F_p_low-1, high ..F_p-1)."""
    F_p_low, T_low_p, F_p_high, T_high_p = encoder.dual_band_grid_shape(et_high)
    layout = encoder.dual_band_token_layout(T_low_p, T_high_p, device=et_high.device)
    return layout["freq_patch_idx"], (F_p_low + F_p_high)


def reduce_meantime_keepfreq(m4, fpi, f_p):
    """(B,K,S,d) → (B,K,F_p,d): mean S tokens sharing each freq patch (mean-time/band)."""
    B, K, S, d = m4.shape
    assert fpi.numel() == S, f"freq_patch_idx ({fpi.numel()}) != S ({S})"
    out = m4.new_zeros(B, K, f_p, d)
    for f in range(f_p):
        out[:, :, f, :] = m4[:, :, fpi == f, :].mean(dim=2)
    return out


@torch.no_grad()
def extract_split(encoder, loader, reduction):
    """Forward every batch. mean* → pooled (N,K,*,d); attentive → raw (N,K,S,d) fp16.
    Returns (feats, labels (N,), latent_valid (N,K) bool)."""
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
        if reduction == "meantime_keepfreq":
            if fpi is None:
                fpi, f_p = _freq_patch_index(encoder, et_high)
            red = reduce_meantime_keepfreq(m4, fpi, f_p).cpu().numpy()
        elif reduction == "meanall":
            red = m4.mean(dim=2, keepdim=True).cpu().numpy()   # (B,K,1,d)
        else:  # attentive / flatten: keep raw tokens (fp16 to bound RAM)
            red = m4.half().cpu().numpy()                       # (B,K,S,d)
        feats.append(red)
        lvs.append(lv.cpu().numpy())
        y = data["target"]
        y = y.reshape(y.shape[0], -1)
        y = y.argmax(dim=1) if y.shape[1] > 1 else y[:, 0]
        labels.append(y.cpu().numpy())
    return (np.concatenate(feats), np.concatenate(labels).astype(int),
            np.concatenate(lvs))


def _auroc(y_true, proba, classes):
    from sklearn.metrics import roc_auc_score
    if len(classes) == 2:
        return float(roc_auc_score(y_true, proba[:, 1]))
    return float(roc_auc_score(y_true, proba, multi_class="ovr",
                               average="macro", labels=classes))


def fit_sklearn(X_tr, y_tr, X_te, y_te, c, seed):
    """StandardScaler + L2 LogisticRegression (upstream-parity); AUROC + bal-acc."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr)
    Xtr, Xte = scaler.transform(X_tr), scaler.transform(X_te)
    clf = LogisticRegression(C=c, random_state=seed, max_iter=10000, tol=1e-3)
    clf.fit(Xtr, y_tr)
    proba = clf.predict_proba(Xte)
    return {
        "auroc": _auroc(y_te, proba, clf.classes_),
        "bal_acc": float(balanced_accuracy_score(y_te, clf.predict(Xte))),
        "n_classes": int(len(clf.classes_)),
    }


class AttnPool(torch.nn.Module):
    """Single-seed (or n_seed) multihead PMA over M tokens → linear classifier."""

    def __init__(self, d, n_classes, heads=4, n_seed=1):
        super().__init__()
        self.seed = torch.nn.Parameter(torch.randn(n_seed, d) * d ** -0.5)
        self.attn = torch.nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln = torch.nn.LayerNorm(d)
        self.head = torch.nn.Linear(d * n_seed, n_classes)

    def forward(self, x):                       # x: (B, M, d)
        q = self.seed.unsqueeze(0).expand(x.shape[0], -1, -1)
        pooled, _ = self.attn(q, x, x)          # (B, n_seed, d)
        return self.head(self.ln(pooled).reshape(x.shape[0], -1))


def fit_attentive(Xtr, ytr, Xva, yva, Xte, yte, d, n_seed, heads, seed,
                  epochs=150, batch=256, patience=25):
    """Train PMA+linear on frozen tokens, early-stop on val AUROC. Xs: (N,M,d) fp16."""
    import numpy as _np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    classes = _np.unique(ytr)
    n_classes = int(len(classes))
    cls_index = {c: i for i, c in enumerate(classes)}
    ytr_i = _np.array([cls_index[v] for v in ytr])
    # per-channel standardization from train tokens (streamed to bound RAM)
    s, ss, n = _np.zeros(d), _np.zeros(d), 0
    for i in range(0, len(Xtr), 64):
        chunk = Xtr[i:i + 64].astype("float32").reshape(-1, d)
        s += chunk.sum(0); ss += (chunk ** 2).sum(0); n += chunk.shape[0]
    mu = (s / n).astype("float32")
    sd = _np.sqrt(_np.maximum(ss / n - (s / n) ** 2, 1e-8)).astype("float32")
    mu_t = torch.from_numpy(mu).to(device)
    sd_t = torch.from_numpy(sd).to(device)

    model = AttnPool(d, n_classes, heads, n_seed).to(device)
    n_pool = sum(p.numel() for nm, p in model.named_parameters() if not nm.startswith("head"))
    n_total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = torch.nn.CrossEntropyLoss()

    def to_batch(X, idx):
        xb = torch.from_numpy(X[idx].astype("float32")).to(device)
        return (xb - mu_t) / sd_t

    @torch.no_grad()
    def proba(X):
        model.eval()
        out = []
        for i in range(0, len(X), batch):
            xb = to_batch(X, slice(i, i + batch))
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        return _np.concatenate(out)

    rng = _np.random.default_rng(seed)
    best_auroc, best_state, bad, ep = -1.0, None, 0, 0
    has_val = Xva is not None and len(Xva) > 0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            xb = to_batch(Xtr, idx)
            yb = torch.from_numpy(ytr_i[idx]).long().to(device)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
        if has_val:
            va = _auroc(yva, proba(Xva), classes)
        else:
            va = _auroc(ytr, proba(Xtr), classes)   # no val → track train (no early stop)
        if va > best_auroc + 1e-4:
            best_auroc, bad = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if has_val and bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    from sklearn.metrics import balanced_accuracy_score
    p_te = proba(Xte)
    pred = classes[p_te.argmax(1)]
    return {
        "auroc": _auroc(yte, p_te, classes),
        "bal_acc": float(balanced_accuracy_score(yte, pred)),
        "n_classes": n_classes, "val_auroc": float(best_auroc),
        "n_pool_params": int(n_pool), "n_head_params": int(n_total - n_pool),
        "epochs_run": ep + 1, "device": device,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="onset")
    ap.add_argument("--eval-mode", default="within", choices=list(_MODE_ALIASES))
    ap.add_argument("--reduction", default="meantime_keepfreq", choices=_REDUCTIONS)
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--C", type=float, default=1.0, help="L2 inverse-reg (sklearn).")
    ap.add_argument("--n-seed", type=int, default=1, help="attentive: #PMA seed queries.")
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

    Xtr, ytr, lv_tr = extract_split(encoder, loaders["train"], args.reduction)
    Xte, yte, lv_te = extract_split(encoder, loaders["test"], args.reduction)
    has_val = "val" in loaders
    Xva = yva = None
    if has_val and args.reduction == "attentive":
        Xva, yva, _ = extract_split(encoder, loaders["val"], args.reduction)

    # Valid (covered) parcels are session-constant; require valid across the split.
    valid_tr = lv_tr.all(axis=0)
    valid_te = lv_te.all(axis=0)
    sel = (valid_tr & valid_te) if eval_mode == "CrossSubject" else valid_tr
    k_sel = int(sel.sum())

    if args.reduction == "attentive":
        # (N,K,S,d) → keep valid parcels → flatten parcels×time-freq into M tokens.
        def flat(X):
            X = X[:, sel, :, :]
            return X.reshape(X.shape[0], -1, X.shape[-1])     # (N, K_sel*S, d)
        Xtr_f, Xte_f = flat(Xtr), flat(Xte)
        Xva_f = flat(Xva) if Xva is not None else None
        n_tokens = Xtr_f.shape[1]
        scores = fit_attentive(Xtr_f, ytr, Xva_f, yva, Xte_f, yte,
                               d=Xtr.shape[-1], n_seed=args.n_seed, heads=N_HEADS,
                               seed=args.seed)
        feature_dim = int(args.n_seed * Xtr.shape[-1])
    else:
        Xtr_f = Xtr[:, sel, :, :].reshape(Xtr.shape[0], -1)
        Xte_f = Xte[:, sel, :, :].reshape(Xte.shape[0], -1)
        n_tokens = 0
        scores = fit_sklearn(Xtr_f, ytr, Xte_f, yte, args.C, args.seed)
        feature_dim = int(Xtr_f.shape[1])

    rec = {
        "task": args.task, "eval_mode": eval_mode, "reduction": args.reduction,
        "subject": args.subject, "trial": args.trial, "fold": args.fold,
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
        "k_valid_train": int(valid_tr.sum()), "k_valid_test": int(valid_te.sum()),
        "k_sel": k_sel, "n_tokens": int(n_tokens), "feature_dim": feature_dim,
        "C": args.C, "n_seed": args.n_seed, "ckpt": args.ckpt,
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
