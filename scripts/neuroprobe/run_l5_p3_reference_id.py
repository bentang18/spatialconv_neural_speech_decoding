"""Stage-0 L.5.P3 — reference-id positive-sanity probe.

For each BT Lite session: load the same trials twice (once with raw / R0,
once with shaft_laplacian / R4 reference) and train a binary classifier to
discriminate which reference produced each feature vector. POSITIVE sanity:
AUROC ≈ 1 expected. Surprise (and bug flag) if it isn't.

This confirms the L.2 sweep's empirical ranking (R4 wins by significant
margin) by directly verifying R0 and R4 features are not byte-identical or
trivially affine-equivalent — they encode meaningfully distinct spatial
filters.

Single sbatch job; loops 12 sessions internally.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS
from speech_decoding.views import make_upstream_helpers, preprocess_views


UPSTREAM_REPO_URL = "https://github.com/insight-neuro/neuroprobe"
UPSTREAM_COMMIT = "c7b955b0a31464f4a5eec3f3bd78ff29841d61ac"


def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bt_root: Path = args.bt_root.resolve()
    repo_dir: Path = args.neuroprobe_repo.resolve()
    sys.path.insert(0, str(repo_dir))
    os.environ["ROOT_DIR_BRAINTREEBANK"] = str(bt_root)

    from neuroprobe.braintreebank_subject import BrainTreebankSubject  # noqa: E402
    from neuroprobe.train_test_splits import subset_electrodes  # noqa: E402
    import neuroprobe.config as nconfig  # noqa: E402
    from neuroprobe.eval_utils import preprocess_stft  # noqa: E402
    from neuroprobe.eval_utils import laplacian_rereference_neural_data  # noqa: E402

    sr = int(nconfig.SAMPLING_RATE)
    win_samples = int(args.window_seconds * sr)
    sessions = list(BT_LITE_SESSIONS) if not args.sessions else _parse_sessions(args.sessions)

    stft_params = {
        "type": "stft_abs",
        "stft": {
            "nperseg": 512, "poverlap": 0.75, "window": "hann",
            "max_frequency": 150, "min_frequency": 0,
        },
        "projection": {"dim": 192, "method": "pca"},
    }
    upstream_helpers = make_upstream_helpers(
        preprocess_stft=preprocess_stft,
        laplacian_rereference_neural_data=laplacian_rereference_neural_data,
        stft_params=stft_params,
    )

    rows: list[dict[str, float | int | str]] = []
    for s_idx, (sub_id, trial_id) in enumerate(sessions):
        t_session = time.time()
        words_df_path = (
            repo_dir / "neuroprobe" / "braintreebank_features_time_alignment"
            / f"subject{sub_id}_trial{trial_id}_words_df.csv"
        )
        words_df = pd.read_csv(words_df_path)
        words_df = words_df[words_df["est_idx"].notna()].copy()

        subject = BrainTreebankSubject(sub_id, cache=True, dtype=torch.float32)
        subset_electrodes(subject, lite=True, nano=False)
        subject.load_neural_data(trial_id)
        cache = subject.neural_data_cache[trial_id]
        n_samples = cache.shape[1]
        labels = list(subject.electrode_labels)

        feats_r0: list[np.ndarray] = []
        feats_r4: list[np.ndarray] = []
        n_kept = 0
        for est_idx in words_df["est_idx"].astype(int):
            lo = est_idx
            hi = lo + win_samples
            if lo < 0 or hi > n_samples:
                continue
            window = cache[:, lo:hi].unsqueeze(0)
            f0, _ = preprocess_views(
                window, labels, ref_kind="raw", view_kind="stft_abs",
                sampling_rate=sr, upstream_helpers=upstream_helpers,
            )
            f4, _ = preprocess_views(
                window, labels, ref_kind="shaft_laplacian", view_kind="stft_abs",
                sampling_rate=sr, upstream_helpers=upstream_helpers,
            )
            feats_r0.append(f0.float().numpy().reshape(1, -1))
            feats_r4.append(f4.float().numpy().reshape(1, -1))
            n_kept += 1

        X_r0 = np.concatenate(feats_r0, axis=0)
        X_r4 = np.concatenate(feats_r4, axis=0)
        del feats_r0, feats_r4
        common_F = min(X_r0.shape[1], X_r4.shape[1])
        X = np.concatenate([X_r0[:, :common_F], X_r4[:, :common_F]], axis=0)
        y = np.array([0] * len(X_r0) + [1] * len(X_r4))
        del X_r0, X_r4
        gc.collect()

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y,
        )
        scaler = StandardScaler(copy=False)
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = LogisticRegression(random_state=args.seed, max_iter=10000, tol=1e-3)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]
        auroc = float(roc_auc_score(y_te, proba))

        rows.append({
            "subject_id": int(sub_id),
            "trial_id": int(trial_id),
            "n_trials_per_ref": int(n_kept),
            "n_features_common": int(common_F),
            "test_auroc": auroc,
        })
        print(
            f"[{s_idx + 1}/{len(sessions)}] sub{sub_id} trial{trial_id}: "
            f"AUROC={auroc:.4f}  n={n_kept}  F={common_F}  "
            f"{time.time() - t_session:.1f}s"
        )
        subject.clear_neural_data_cache(trial_id)
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "p3_reference_id.csv", index=False)
    summary = {
        "cell_id": "L.5.P3",
        "n_sessions": int(len(rows)),
        "median_auroc": float(df["test_auroc"].median()),
        "min_auroc":    float(df["test_auroc"].min()),
        "max_auroc":    float(df["test_auroc"].max()),
        "passes_sanity_threshold_0p9": bool(df["test_auroc"].min() > 0.9),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_repo_url": UPSTREAM_REPO_URL,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parse_sessions(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        sub_str, trial_str = chunk.strip().split(":")
        out.append((int(sub_str), int(trial_str)))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    env_root = os.environ.get("ROOT_DIR_BRAINTREEBANK", "").strip()
    p.add_argument("--bt-root", type=Path, default=Path(env_root) if env_root else None)
    p.add_argument(
        "--neuroprobe-repo", type=Path,
        default=(
            Path("/work/ht203/repo/neuroprobe_upstream")
            if Path("/work/ht203").exists()
            else Path(".cache/neuroprobe_upstream")
        ),
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--window-seconds", type=float, default=1.0)
    p.add_argument("--sessions", default="")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.bt_root is None:
        p.error("--bt-root or ROOT_DIR_BRAINTREEBANK is required")
    return args


if __name__ == "__main__":
    main()
