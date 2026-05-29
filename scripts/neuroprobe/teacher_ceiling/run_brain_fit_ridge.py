"""Brain-fit ridge sister: Whisper-L8 features → per-electrode HG envelope.

Mirror of Antonello / Vaidya / Huth fMRI-tuned ECoG: regress brain activity
from language-model features. For each electrode, fit Ridge(Whisper-L8 →
electrode HG-mean over [0, +1s] post word onset) on 80% of word anchors, score
R² on the held-out 20%.

Validates whether Whisper-L8 features carry enough auditory + linguistic
structure to predict the *brain* signal we'll be distilling INTO. Two
direction checks:
    - High per-electrode R² (typical median > 0.05 in literature) → distillation
      target is brain-aligned.
    - Top-R² electrodes localize to STG/IFG (speech cortex) → not a confound.

Uses HG-envelope summary at native 2048 Hz: per-electrode RMS over the 1-s
post-onset window (matches Neuroprobe's brain-side window). Lighter than the
full 70-150 Hz filterbank for a first-light sanity check.

Usage (per-trial smoke):
    .venv/bin/python scripts/neuroprobe/teacher_ceiling/run_brain_fit_ridge.py \\
        --subject-id 3 --trial-id 0 \\
        --features-npz /hpc/group/coganlab/ht203/cache_neuroai/whisper_ceiling/v3/sub_3_trial0.npz \\
        --layer 8 \\
        --out-dir reports/brain_fit_ridge_sub3_trial0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[brain-fit] loading Whisper features {args.features_npz}", flush=True)
    data = dict(np.load(args.features_npz, allow_pickle=False))
    X = data[f"layer_{args.layer}"].astype(np.float32)  # (W, 1280)
    word_idx = data["word_index"]
    movie_start_s = data["movie_start_s"].astype(np.float64)
    print(f"[brain-fit] features {X.shape} dtype={X.dtype}", flush=True)
    print(f"[brain-fit] {len(word_idx)} anchors", flush=True)

    print(f"[brain-fit] loading BT neural data sub_{args.subject_id} trial{args.trial_id}",
          flush=True)
    try:
        from neuroprobe.braintreebank_subject import BrainTreebankSubject
        from neuroprobe.config import SAMPLING_RATE
    except ImportError as e:
        sys.exit(f"neuroprobe not importable: {e}; set ROOT_DIR_BRAINTREEBANK")

    bt = BrainTreebankSubject(
        subject_id=args.subject_id, cache=False, coordinates_type="cortical",
    )
    raw = bt.get_all_electrode_data(args.trial_id)
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    neural = np.asarray(raw, dtype=np.float32)  # (n_ch, T)
    ch_names = list(bt.electrode_labels)
    print(f"[brain-fit] neural shape {neural.shape} @ {SAMPLING_RATE} Hz", flush=True)

    # HG-envelope substitute: window-wise RMS of raw voltage over [0, +1s].
    # Lighter than full 70-150 Hz filterbank but enough for first-light R².
    window_samples = int(args.window_s * SAMPLING_RATE)
    n_anchors = len(word_idx)
    n_channels = neural.shape[0]
    Y = np.full((n_anchors, n_channels), np.nan, dtype=np.float32)
    valid_mask = np.zeros(n_anchors, dtype=bool)
    for i, t0 in enumerate(movie_start_s):
        s0 = int(round(t0 * SAMPLING_RATE))
        s1 = s0 + window_samples
        if s0 < 0 or s1 > neural.shape[1]:
            continue
        win = neural[:, s0:s1]
        # Per-channel RMS (no mean subtraction; treats DC + AC together)
        rms = np.sqrt(np.mean(win ** 2, axis=1, dtype=np.float32))
        Y[i] = rms
        valid_mask[i] = True
    print(f"[brain-fit] {valid_mask.sum()}/{n_anchors} anchors have full neural window",
          flush=True)
    X = X[valid_mask]
    Y = Y[valid_mask]
    word_idx = word_idx[valid_mask]
    movie_start_s = movie_start_s[valid_mask]
    if len(X) < 100:
        sys.exit(f"too few valid anchors ({len(X)}); need ≥100")

    # Train/test split: chronological 80/20 to avoid leakage
    print(f"[brain-fit] fitting Ridge per electrode (n_anchors={len(X)})", flush=True)
    cut = int(0.8 * len(X))
    X_train, X_test = X[:cut], X[cut:]
    Y_train, Y_test = Y[:cut], Y[cut:]

    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)

    per_electrode = []
    Y_test_pred_all = np.empty_like(Y_test)
    for e in range(n_channels):
        y_tr = Y_train[:, e]
        y_te = Y_test[:, e]
        if not (np.all(np.isfinite(y_tr)) and np.all(np.isfinite(y_te))):
            per_electrode.append({"electrode": ch_names[e], "r2": float("nan")})
            Y_test_pred_all[:, e] = np.nan
            continue
        clf = Ridge(alpha=args.ridge_alpha, random_state=args.seed)
        clf.fit(X_train_s, y_tr)
        y_pred = clf.predict(X_test_s)
        Y_test_pred_all[:, e] = y_pred
        r2 = float(r2_score(y_te, y_pred))
        per_electrode.append({"electrode": ch_names[e], "r2": r2})

    df = pd.DataFrame(per_electrode)
    df = df.sort_values("r2", ascending=False).reset_index(drop=True)

    median_r2 = float(np.nanmedian(df["r2"]))
    p95_r2 = float(np.nanpercentile(df["r2"], 95))
    top5 = df.head(5).to_dict(orient="records")

    summary = {
        "subject_id": args.subject_id,
        "trial_id": args.trial_id,
        "features_npz": str(args.features_npz),
        "layer": args.layer,
        "n_anchors": int(len(X)),
        "n_train": int(cut),
        "n_test": int(len(X) - cut),
        "n_channels": int(n_channels),
        "window_s": args.window_s,
        "ridge_alpha": args.ridge_alpha,
        "median_r2": median_r2,
        "p95_r2": p95_r2,
        "n_electrodes_r2_above_0p05": int((df["r2"] > 0.05).sum()),
        "n_electrodes_r2_above_0p10": int((df["r2"] > 0.10).sum()),
        "top5_electrodes": top5,
    }
    print(json.dumps(summary, indent=2), flush=True)
    df.to_csv(args.out_dir / "brain_fit_ridge_per_electrode.csv", index=False)
    (args.out_dir / "brain_fit_ridge_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(args.out_dir / "Y_test_pred.npy", Y_test_pred_all)
    np.save(args.out_dir / "Y_test.npy", Y_test)
    print(f"[brain-fit] wrote {args.out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject-id", type=int, required=True)
    p.add_argument("--trial-id", type=int, required=True)
    p.add_argument("--features-npz", type=Path, required=True)
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--window-s", type=float, default=1.0)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
