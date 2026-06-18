#!/usr/bin/env python
"""M4 target trial-variance guard probe (converged-arch memo, M4 "Guard" bullet).

The one real M4 risk is the electrode-mean teacher target being too **easy**: if
``Var_trial(mean teacher feature | parcel, freq-time)`` is near zero, the M4
predictor memorises a constant per (parcel, freq-time) cell and the gradient is
weak. The memo asks to MEASURE this on a teacher ckpt before committing M4. This
is that measurement, run on DCC against a converged Lightning checkpoint.

Run on DCC (cwd = /work/ht203/repo/speech) with the project venv:

    ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank \
    EXCA_CACHE_FOLDER=/hpc/group/coganlab/ht203/cache_neuroai \
    EXCA_EXTRACTOR_CACHE_FOLDER=/work/ht203/cache/v14_extractors \
    .venv/bin/python scripts/neuroprobe/probe_m4_target_variance.py \
        /path/to/converged_checkpoint.ckpt \
        --d-model 256 --n-heads 4 \
        --frontend-layers 2 --latent-layers 2 \
        --m2-pred-dim 128 --m2-pred-layers 4 \
        --m4-pred-dim 128 --m4-pred-layers 4 \
        --spec-cache-dir /work/ht203/cache_neuroai/v14_3stft_spec_cache \
        --out reports/m4_target_variance.json

What it does (defensive, prints shapes at every step; writes NO model files):
  1. Builds the converged experiment for ONE leaderboard cell via
     build_v14_experiment(frontend="3stft", ...) with the SAME converged shape
     as the probed run (shape flags are REQUIRED — no silent run defaults).
  2. Builds the V14ConvergedSSL from the config and loads the EMA
     ``model.teacher_frontend.*`` weights from the Lightning ckpt — asserting
     every teacher PARAMETER was supplied (regenerated non-persistent buffers may
     be absent).
  3. Freezes + .eval()s the model; streams a Data-pipeline loader through the
     tested ``accumulate_m4_target_variance`` accumulator (per-clip electrode-mean
     teacher target → streaming Welford trial-variance per (parcel, freq-time)).
  4. Writes the threshold-free summary (per-cell trial-variance distribution +
     dimensionless "easiness" ratio at several cut points) to --out as JSON, so
     the human reads "too easy?" off the numbers rather than the probe deciding.

Per the memo: any escalation if too-easy is on the SUBJECT-INVARIANT axes only
(finer freq-time grid, higher parcel-mask ratio, deeper teacher target) — NEVER
the electrode axis. This probe measures; it does not decide.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="M4 teacher-target trial-variance guard probe (converged arch)",
    )
    p.add_argument("ckpt", help="converged Lightning checkpoint (.ckpt)")
    # --- converged shape (REQUIRED — must match the probed run's launch argv) --
    p.add_argument("--d-model", type=int, required=True)
    p.add_argument("--n-heads", type=int, required=True)
    p.add_argument("--frontend-layers", type=int, required=True)
    p.add_argument("--latent-layers", type=int, required=True)
    p.add_argument("--m2-pred-dim", type=int, required=True)
    p.add_argument("--m2-pred-layers", type=int, required=True)
    p.add_argument("--m4-pred-dim", type=int, required=True)
    p.add_argument("--m4-pred-layers", type=int, required=True)
    p.add_argument("--atlas", choices=("dk", "dkt"), default="dk")
    # --- the cell whose Data pipeline supplies the clips ----------------------
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--trial", type=int, default=1)
    p.add_argument("--task", default="onset")
    p.add_argument("--eval-mode", default="WithinSession")
    p.add_argument("--split", choices=("train", "val", "test"), default="train",
                   help="which split's loader to measure (default train = the "
                        "SSL corpus the M4 loss actually sees)")
    p.add_argument("--clip-len", type=float, default=1.0)
    p.add_argument("--spec-cache-dir", default=None,
                   help="the run's 3STFT spec cache (band_{slow,beta,hg} subdirs)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-clips", type=int, default=None,
                   help="cap accumulated clips (None = the whole split loader)")
    p.add_argument("--out", default="reports/m4_target_variance.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if "ROOT_DIR_BRAINTREEBANK" not in os.environ:
        sys.exit(
            "ROOT_DIR_BRAINTREEBANK is unset; export "
            "ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank"
        )
    print(f"[env] ROOT_DIR_BRAINTREEBANK={os.environ['ROOT_DIR_BRAINTREEBANK']}")
    print(f"[env] EXCA_CACHE_FOLDER={os.environ.get('EXCA_CACHE_FOLDER')}")
    print(
        f"[env] EXCA_EXTRACTOR_CACHE_FOLDER="
        f"{os.environ.get('EXCA_EXTRACTOR_CACHE_FOLDER')}"
    )
    if not Path(args.ckpt).is_file():
        sys.exit(f"ckpt not found: {args.ckpt}")
    print(f"[ckpt] {args.ckpt}")
    print(
        f"[cell] subject={args.subject} trial={args.trial} task={args.task} "
        f"mode={args.eval_mode} split={args.split}"
    )

    # Import after the env check so a missing ROOT_DIR fails fast with a clear msg.
    import speech_decoding.models  # noqa: F401  registers the model configs
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment
    from speech_decoding.models.m4_target_variance import (
        accumulate_m4_target_variance,
    )
    from speech_decoding.models.v14_converged import V14ConvergedSSL

    # ---- 1. Build the converged experiment for ONE cell ---------------------
    print("[build] build_v14_experiment(frontend='3stft', ...)")
    experiment = build_v14_experiment(
        mode="lite",
        task=args.task,
        eval_mode=args.eval_mode,
        test_subject_id=args.subject,
        test_trial_id=args.trial,
        fold_index=0,
        binary_tasks=True,
        frontend="3stft",
        atlas=args.atlas,
        d_model=args.d_model,
        n_heads=args.n_heads,
        converged_frontend_layers=args.frontend_layers,
        converged_latent_layers=args.latent_layers,
        converged_m2_pred_dim=args.m2_pred_dim,
        converged_m2_pred_layers=args.m2_pred_layers,
        converged_m4_pred_dim=args.m4_pred_dim,
        converged_m4_pred_layers=args.m4_pred_layers,
        clip_len=args.clip_len,
        spec_cache_dir=args.spec_cache_dir,
        batch_size=args.batch_size,
        num_workers=0,
        precision=None,           # full fp32 for a clean variance measurement
    )
    print(f"[build] experiment = {type(experiment).__name__}")
    cfg = experiment.brain_model_config
    print(
        f"[build] config = {type(cfg).__name__}: d_model={cfg.d_model} "
        f"n_parcels={cfg.n_parcels} n_heads={cfg.n_heads} "
        f"frontend_layers={cfg.frontend_layers} latent_layers={cfg.latent_layers} "
        f"freq_pos={cfg.freq_pos}"
    )

    # ---- 2. Build the model + load the EMA teacher frontend -----------------
    model = cfg.build(n_in_channels=1, n_outputs=1)   # n_in_channels ignored
    if not isinstance(model, V14ConvergedSSL):
        sys.exit(f"expected V14ConvergedSSL, got {type(model).__name__}")
    n_tokens = int(model.tokens_per_electrode)
    print(f"[model] V14ConvergedSSL; tokens_per_electrode={n_tokens} d={cfg.d_model}")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    prefix = "model.teacher_frontend."
    teacher_sd = {
        k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)
    }
    if not teacher_sd:
        sample = list(sd)[:20]
        sys.exit(
            f"no keys under {prefix!r} in the ckpt state_dict; first keys: {sample}"
        )
    missing, unexpected = model.teacher_frontend.load_state_dict(
        teacher_sd, strict=False,
    )
    # Rigorous check: every learnable teacher PARAMETER must have come from the
    # ckpt. Non-persistent buffers (band_id/freq_global_id/time_slot, RoPE caches)
    # are regenerated at __init__ and legitimately absent — so we check parameters,
    # not the full state_dict.
    param_names = set(dict(model.teacher_frontend.named_parameters()))
    not_loaded = sorted(p for p in param_names if p not in teacher_sd)
    print(f"[load] {len(teacher_sd)} teacher_frontend.* keys; "
          f"{len(param_names)} teacher parameters")
    print(f"[load] missing (regenerated buffers expected): {list(missing)[:8]}")
    if not_loaded or unexpected:
        sys.exit(
            f"teacher load FAILED: unsupplied parameters={not_loaded} "
            f"unexpected={list(unexpected)}"
        )
    print("[load] OK — every teacher parameter supplied by the ckpt")

    # ---- 3. Freeze + eval ---------------------------------------------------
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- 4. Stream the loader through the tested accumulator ----------------
    print(f"[data] experiment.data.build(worker_seed={experiment.seed})")
    loaders = experiment.data.build(worker_seed=experiment.seed)
    print(f"[data] splits = {list(loaders)}")
    if args.split not in loaders:
        sys.exit(f"split {args.split!r} not in loaders {list(loaders)}")
    loader = loaders[args.split]

    print(
        f"[accumulate] n_parcels={cfg.n_parcels} n_tokens={n_tokens} "
        f"d_model={cfg.d_model} max_clips={args.max_clips}"
    )
    summary = accumulate_m4_target_variance(
        model.teacher_frontend, loader,
        n_parcels=cfg.n_parcels, n_tokens=n_tokens, d_model=cfg.d_model,
        max_clips=args.max_clips,
    )

    # ---- 5. Report ----------------------------------------------------------
    n_clips = int(summary.count.sum())
    print(f"[result] clips accumulated (parcel-trials summed) = {n_clips}")
    print(f"[result] cells measured (count>=2) = {summary.n_cells_measured}")
    print(f"[result] median trial-variance = {summary.median_var:.6g}")
    print(f"[result] pooled feature variance = {summary.pooled_feature_var:.6g}")
    print(f"[result] median easiness = {summary.median_easiness:.6g}")
    print(f"[result] frac_easy_at = {summary.frac_easy_at}")

    out = {
        "ckpt": str(args.ckpt),
        "cell": {
            "subject": args.subject, "trial": args.trial, "task": args.task,
            "eval_mode": args.eval_mode, "split": args.split,
        },
        "shape": {
            "d_model": cfg.d_model, "n_parcels": cfg.n_parcels,
            "n_heads": cfg.n_heads, "frontend_layers": cfg.frontend_layers,
            "latent_layers": cfg.latent_layers, "n_tokens": n_tokens,
            "atlas": args.atlas, "clip_len": args.clip_len,
        },
        "max_clips": args.max_clips,
        "summary": summary.as_dict(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
