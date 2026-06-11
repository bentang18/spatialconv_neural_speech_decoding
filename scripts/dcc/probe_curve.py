#!/usr/bin/env python
"""Multi-checkpoint frozen-probe CURVE.

Build the Neuroprobe probe data ONCE, cache the (encoder-independent) front-end
batches, then forward EVERY checkpoint in a directory through those same cached
batches and report ``global_step -> (decoding AUROC, feature-RankMe)``.

Purpose: find the decoding-optimal SSL step. The open question (Ben, 2026-06-11):
does downstream linear-probe decoding keep IMPROVING past the SSL-loss minimum
(e.g. step >1k) even when SSL train/val loss and RankMe are rising? If so, the
SSL-loss/RankMe health gate is the WRONG early-stopping signal for transfer.

One data build (~90 GB, ~5 min) amortised across all checkpoints instead of
N separate jobs each rebuilding the 12-session Lite universe. Reuses
``probe_frozen_encoder.py`` (same dir) for the validated encoder-reconstruction,
pooling, and LogReg-AUROC logic — including the ``time_last_input=True`` fix.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_frozen_encoder as P  # noqa: E402


def collect_raw_batches(loader):
    """Pull all batches off a loader ONCE -> list of plain CPU dicts (encoder
    inputs + target). Encoder-independent, so reusable across every checkpoint
    (the expensive front-end STFT/data pipeline runs exactly once)."""
    out = []
    for batch in loader:
        d = batch.data
        rec = {
            "electrode_tokens": d["electrode_tokens"].cpu(),
            "support": d["support"].cpu(),
            "target": d["target"].cpu(),
        }
        vm = d.get("valid_mask") if hasattr(d, "get") else None
        if vm is not None:
            rec["valid_mask"] = vm.cpu()
        out.append(rec)
    return out


@torch.no_grad()
def forward_cached(encoder, batches, *, pool, m_sub_slots, device):
    """Forward the cached batches through one encoder -> (X (N,D), y (N,))."""
    feats, labels = [], []
    for rec in batches:
        x = P.encode_and_pool(
            encoder, rec, pool=pool, m_sub_slots=m_sub_slots, device=device
        )
        y = rec["target"]
        while y.ndim > 1 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        feats.append(x.float().cpu())
        labels.append(y.long().cpu())
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def rankme(Z, eps: float = 1e-12) -> float:
    """Effective rank (RankMe, Garrido et al. 2023) of feature matrix Z (N, D):
    exp(Shannon entropy of the L1-normalised singular-value spectrum). A
    feature-space collapse/health metric to correlate against decoding AUROC."""
    if Z.shape[0] < 2:
        return float("nan")
    s = np.linalg.svd(Z, compute_uv=False)
    p = s / (s.sum() + eps)
    return float(np.exp(-(p * np.log(p + eps)).sum()))


def ckpt_global_step(path: str) -> int:
    """Read global_step from a Lightning ckpt (transferable snapshots lack it)."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "global_step" in ck:
            return int(ck["global_step"])
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not read global_step from {path}: {e}")
    return -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt-dir", help="directory of checkpoints to probe")
    g.add_argument("--ckpts", nargs="+", help="explicit checkpoint paths")
    ap.add_argument("--ckpt-glob", default="*.ckpt",
                    help="glob within --ckpt-dir (default *.ckpt)")
    ap.add_argument("--task", default="volume",
                    help="single task for the curve (default volume; speech also cheap)")
    ap.add_argument("--pool", choices=("cross_attn", "mean"), default="mean",
                    help="B37 joint-SSL checkpoints use mean (freq preserved).")
    ap.add_argument("--subject-id", type=int, default=P.DEFAULT_SUBJECT_ID)
    ap.add_argument("--trial-id", type=int, default=P.DEFAULT_TRIAL_ID)
    ap.add_argument("--fold", type=int, default=P.DEFAULT_FOLD)
    ap.add_argument("--bt-root", default=None)
    ap.add_argument("--ssl-clip-len", type=float, default=P.DEFAULT_SSL_CLIP_LEN)
    ap.add_argument("--probe-clip-len", type=float, default=P.DEFAULT_PROBE_CLIP_LEN)
    ap.add_argument("--n-freq-bins", type=int, default=P.DEFAULT_N_FREQ_BINS)
    ap.add_argument("--k-parcels", type=int, default=P.DEFAULT_K_PARCELS)
    ap.add_argument("--d-model", type=int, default=P.DEFAULT_D_MODEL)
    ap.add_argument("--depth", type=int, default=P.DEFAULT_DEPTH)
    ap.add_argument("--n-heads", type=int, default=P.DEFAULT_N_HEADS)
    ap.add_argument("--m-sub-slots", type=int, default=P.DEFAULT_M_SUB_SLOTS)
    ap.add_argument("--latent-depth", type=int, default=P.DEFAULT_LATENT_DEPTH)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="0 = no dataloader subprocess (avoids copying the large "
                         "lite-universe extractor; the build is the memory hog).")
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    for k, v in P.DCC_ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)
    bt_root = args.bt_root or os.environ.get(
        "ROOT_DIR_BRAINTREEBANK", P.DCC_ENV_DEFAULTS["ROOT_DIR_BRAINTREEBANK"]
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)
    device = torch.device(args.device)

    if args.ckpts:
        ckpts = list(args.ckpts)
    else:
        ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, args.ckpt_glob)))
    if not ckpts:
        print(f"NO checkpoints found (dir={args.ckpt_dir} glob={args.ckpt_glob})")
        sys.exit(2)
    print(f"[curve] {len(ckpts)} checkpoints | task={args.task} pool={args.pool} "
          f"device={device}")

    # ---- build data ONCE + cache the front-end batches ----
    print(f"[curve] building probe data once "
          f"(task={args.task}, subj={args.subject_id}/{args.trial_id}) ...")
    loaders, front_end_view = P.build_task_loaders(
        task=args.task, subject_id=args.subject_id, trial_id=args.trial_id,
        fold=args.fold, bt_root=bt_root, clip_len=args.probe_clip_len,
        seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    train_batches = collect_raw_batches(loaders["train"])
    test_batches = collect_raw_batches(loaders["test"])
    n_tr = int(sum(b["target"].shape[0] for b in train_batches))
    n_te = int(sum(b["target"].shape[0] for b in test_batches))
    print(f"[curve] cached {len(train_batches)} train / {len(test_batches)} test "
          f"batches ({n_tr} train / {n_te} test clips)")

    # ---- loop checkpoints over the SAME cached batches ----
    rows = []
    for path in ckpts:
        step = ckpt_global_step(path)
        try:
            sd = P.load_encoder_state_dict(path)
            encoder, info = P.build_frozen_encoder(
                sd, pool=args.pool, n_freq_bins=args.n_freq_bins,
                ssl_clip_len=args.ssl_clip_len, k_parcels=args.k_parcels,
                d_model=args.d_model, depth=args.depth, n_heads=args.n_heads,
                m_sub_slots=args.m_sub_slots, latent_depth=args.latent_depth,
                front_end_view=front_end_view,
            )
            encoder = encoder.to(device)
            X_tr, y_tr = forward_cached(
                encoder, train_batches, pool=args.pool,
                m_sub_slots=args.m_sub_slots, device=device,
            )
            X_te, y_te = forward_cached(
                encoder, test_batches, pool=args.pool,
                m_sub_slots=args.m_sub_slots, device=device,
            )
            rm = rankme(X_tr)
            auroc = P.fit_probe_auroc(X_tr, y_tr, X_te, y_te, seed=args.seed)
            rows.append((step, auroc, rm, os.path.basename(path)))
            print(f"[curve] step={step:>6}  AUROC={auroc:.4f}  "
                  f"feat_rankme={rm:7.2f}  X={X_tr.shape}  {os.path.basename(path)}")
            del encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"[curve] step={step:>6}  FAILED {os.path.basename(path)}: "
                  f"{type(e).__name__}: {e}")

    rows.sort(key=lambda r: (r[0] if r[0] >= 0 else 10 ** 9))
    print("\n========== CURVE (sorted by global_step) ==========")
    print(f"{'step':>8}  {'AUROC':>7}  {'feat_rankme':>11}  file")
    for step, auroc, rm, name in rows:
        print(f"{step:>8}  {auroc:>7.4f}  {rm:>11.2f}  {name}")
    if rows:
        best = max(rows, key=lambda r: r[1])
        print(f"\n[curve] BEST decoding: step={best[0]}  AUROC={best[1]:.4f}  "
              f"(feat_rankme={best[2]:.2f})")
        print("[curve] Interpretation: if BEST step >> SSL-loss-min step, the "
              "loss/RankMe health gate under-trains for transfer.")


if __name__ == "__main__":
    main()
