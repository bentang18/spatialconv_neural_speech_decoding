# pyright: reportMissingImports=false
"""Run Neuroprobe Stage 0 L-sweep linear baseline with normalization swap.

Mirrors the upstream `examples/eval_population.py` pipeline (Laplacian
rereference -> STFT |.| -> flatten -> [normalization] -> LogisticRegression)
but internalizes the normalization step so each L.1 cell can swap recipes
while keeping every other step byte-identical to D.0. The N1 cell
(`train_set_fixed`) reproduces the upstream linear baseline exactly.

Designed to extend to L.2/L.3/L.4 by swapping `--preprocess-type` and the
NeuralFetch `IeegExtractor` config later; for L.1 only the normalization axis
moves.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from speech_decoding.experiments import ExperimentLogger  # noqa: E402


UPSTREAM_REPO_URL = "https://github.com/insight-neuro/neuroprobe"
UPSTREAM_COMMIT = "c7b955b0a31464f4a5eec3f3bd78ff29841d61ac"

NORMALIZATION_CELLS: dict[str, str] = {
    "per_window_z":              "L.1.N0",
    "train_set_fixed":           "L.1.N1",
    "per_session_fixed":         "L.1.N2",
    "train_set_scale_only":      "L.1.N3",
    "none":                      "L.1.N4",
    "per_session_robust_mad":    "L.1.N5",
    "train_set_robust_mad":      "L.1.N6",
    "per_session_robust_scale":  "L.1.N7",
    "per_channel_train_set_z":   "L.1.N8",
}

NORMALIZATION_DESCRIPTIONS: dict[str, str] = {
    "per_window_z":             "per-sample z-score across the flattened feature dim (BrainBERT/PopT recipe)",
    "train_set_fixed":          "sklearn StandardScaler fit on training features (current upstream linear baseline)",
    "per_session_fixed":        "StandardScaler fit independently on each session's own features (transductive: test stats fit on test)",
    "train_set_scale_only":     "StandardScaler with_mean=False fit on training features (isolates demean from scale)",
    "none":                     "no normalization, raw features (sanity floor)",
    "per_session_robust_mad":   "per-feature median + 1.4826*MAD fit independently on each session (transductive: test stats fit on test)",
    "train_set_robust_mad":     "per-feature median + 1.4826*MAD fit on training set, applied to test (inductive analog of N5)",
    "per_session_robust_scale": "per-feature MAD scale only (no median subtraction) fit independently on each session",
    "per_channel_train_set_z":  "one mean/std per electrode pooled over freq+time, fit on training set (standard iEEG recipe)",
}


def main() -> None:
    args = _parse_args()
    bt_root = args.bt_root.resolve()
    repo_dir = args.neuroprobe_repo.resolve()
    if not (repo_dir / ".git").exists() and not (repo_dir / "neuroprobe").exists():
        raise FileNotFoundError(
            f"Upstream Neuroprobe repo not found at {repo_dir}. "
            f"Clone {UPSTREAM_REPO_URL} @ {UPSTREAM_COMMIT} first."
        )
    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "examples"))
    os.environ["ROOT_DIR_BRAINTREEBANK"] = str(bt_root)

    from neuroprobe.braintreebank_subject import BrainTreebankSubject  # noqa: E402
    import neuroprobe.train_test_splits as nts  # noqa: E402
    import neuroprobe.config as nconfig  # noqa: E402
    from eval_utils import (  # noqa: E402
        combine_regions,
        get_region_labels,
        laplacian_rereference_neural_data,
        preprocess_data,
        preprocess_stft,
        subset_electrodes,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cell_id = NORMALIZATION_CELLS[args.normalization]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "upstream_repo_url": UPSTREAM_REPO_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "bt_root": str(bt_root),
        "subject_id": args.subject_id,
        "trial_id": args.trial_id,
        "task": args.task,
        "split_type": args.split_type,
        "binary_tasks": args.binary_tasks,
        "backend": args.backend,
        "ref_kind": args.ref_kind if args.backend == "neuralset" else None,
        "view_kind": args.view_kind if args.backend == "neuralset" else None,
        "preprocess_type": args.preprocess_type,
        "normalization": args.normalization,
        "normalization_cell": cell_id,
        "normalization_description": NORMALIZATION_DESCRIPTIONS[args.normalization],
        "seed": args.seed,
    }

    with ExperimentLogger(
        artifact_dir=out_dir,
        block="L",
        cell=cell_id,
        run_kind="stage0_linear_baseline",
        eval_mode="binary" if args.binary_tasks else "multiclass",
        split_mode=args.split_type,
        subject_id=str(args.subject_id),
        trial_id=str(args.trial_id),
        task=args.task,
        seed=str(args.seed),
        config_json=config,
        report_dir=str(out_dir),
    ) as logger:
        diagnostics, qc_payload = run_eval(
            args=args,
            BrainTreebankSubject=BrainTreebankSubject,
            nts=nts,
            nconfig=nconfig,
            preprocess_data=preprocess_data,
            preprocess_stft=preprocess_stft,
            laplacian_rereference_neural_data=laplacian_rereference_neural_data,
            subset_electrodes=subset_electrodes,
            get_region_labels=get_region_labels,
            combine_regions=combine_regions,
        )
        diagnostics.to_csv(out_dir / "diagnostics.csv", index=False)

        summary = summarize(diagnostics)
        (out_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

        plot_signal_qc(out_dir / "signal_qc.png", qc_payload, cell_id, args)

        write_readme(out_dir, summary, cell_id, args)
        logger.set_metrics(
            summary,
            primary_metric_name="mean_test_roc_auc",
            primary_metric_value=summary["mean_test_roc_auc"],
            reference_metric_value=args.reference_mean_auroc,
        )

        print(diagnostics.to_string(index=False))
        print(json.dumps(summary, indent=2, sort_keys=True))


def run_eval(
    *,
    args: argparse.Namespace,
    BrainTreebankSubject,
    nts,
    nconfig,
    preprocess_data,
    preprocess_stft,
    laplacian_rereference_neural_data,
    subset_electrodes,
    get_region_labels,
    combine_regions,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    eval_names = args.task.split(",")
    splits_type = args.split_type
    binary_tasks = bool(args.binary_tasks)
    preprocess_type = args.preprocess_type

    preprocess_parameters = {
        "type": preprocess_type,
        "stft": {
            "nperseg": 512,
            "poverlap": 0.75,
            "window": "hann",
            "max_frequency": 150,
            "min_frequency": 0,
        },
        "projection": {"dim": 192, "method": "pca"},
    }

    upstream_helpers = None
    if args.backend == "neuralset":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from preprocess_views import make_upstream_helpers
        upstream_helpers = make_upstream_helpers(
            preprocess_stft=preprocess_stft,
            laplacian_rereference_neural_data=laplacian_rereference_neural_data,
            stft_params=preprocess_parameters,
        )

    # Anchor window: feature window is [-bins_start_before, +bins_end_after]s
    # around word onset. Default = [0, 1]s (D.0 baseline). data_idx_from is
    # computed below as (bin_start + bins_start_before) * SR; setting
    # bin_starts=[-bins_start_before], bin_ends=[bins_end_after] makes the
    # slice cover the full requested window.
    bins_start_before = float(getattr(args, "anchor_start_before", 0.0))
    bins_end_after = float(getattr(args, "anchor_end_after", 1.0))
    bin_starts, bin_ends = [-bins_start_before], [bins_end_after]

    subject = BrainTreebankSubject(args.subject_id, cache=True, dtype=torch.float32)
    subset_electrodes(subject, lite=True, nano=False)
    t0 = time.time()
    subject.load_neural_data(args.trial_id)
    print(f"[load] subject {args.subject_id} trial {args.trial_id} in {time.time() - t0:.1f}s")

    rows: list[dict[str, Any]] = []
    qc_payload: dict[str, Any] | None = None

    for eval_name in eval_names:
        if splits_type == "WithinSession":
            folds = nts.generate_splits_within_session(
                subject, args.trial_id, eval_name, dtype=torch.float32,
                output_indices=False, output_dict=False,
                start_neural_data_before_word_onset=int(bins_start_before * nconfig.SAMPLING_RATE),
                end_neural_data_after_word_onset=int(bins_end_after * nconfig.SAMPLING_RATE),
                lite=True, nano=False, binary_tasks=binary_tasks,
            )
            train_subject = subject
        elif splits_type == "CrossSession":
            folds = nts.generate_splits_cross_session(
                subject, args.trial_id, eval_name, dtype=torch.float32,
                output_indices=False, output_dict=False,
                start_neural_data_before_word_onset=int(bins_start_before * nconfig.SAMPLING_RATE),
                end_neural_data_after_word_onset=int(bins_end_after * nconfig.SAMPLING_RATE),
                lite=True, binary_tasks=binary_tasks,
            )
            train_subject = subject
        elif splits_type == "CrossSubject":
            train_subject_id = nconfig.DS_DM_TRAIN_SUBJECT_ID
            train_subject = BrainTreebankSubject(
                train_subject_id, allow_corrupted=False, cache=True, dtype=torch.float32
            )
            train_subject.set_electrode_subset(
                nconfig.NEUROPROBE_LITE_ELECTRODES[train_subject.subject_identifier]
            )
            all_subjects = {args.subject_id: subject, train_subject_id: train_subject}
            folds = nts.generate_splits_cross_subject(
                all_subjects, args.subject_id, args.trial_id, eval_name, dtype=torch.float32,
                output_indices=False, output_dict=False,
                start_neural_data_before_word_onset=int(bins_start_before * nconfig.SAMPLING_RATE),
                end_neural_data_after_word_onset=int(bins_end_after * nconfig.SAMPLING_RATE),
                lite=True, nano=False, binary_tasks=binary_tasks,
            )
        else:
            raise ValueError(f"Unknown split type: {splits_type}")

        for bin_start, bin_end in zip(bin_starts, bin_ends):
            data_idx_from = int((bin_start + bins_start_before) * nconfig.SAMPLING_RATE)
            data_idx_to = int((bin_end + bins_start_before) * nconfig.SAMPLING_RATE)

            for fold_idx, fold in enumerate(folds):
                t_fold = time.time()
                train_dataset = fold["train_dataset"]
                test_dataset = fold["test_dataset"]

                def _preprocess(item_data, electrode_labels):
                    sliced = item_data[:, data_idx_from:data_idx_to].unsqueeze(0)
                    if args.backend == "upstream":
                        return preprocess_data(
                            sliced, electrode_labels, preprocess_type, preprocess_parameters,
                        ).float().numpy()
                    from preprocess_views import preprocess_views  # noqa: PLC0415
                    feats, _ = preprocess_views(
                        sliced, list(electrode_labels),
                        ref_kind=args.ref_kind, view_kind=args.view_kind,
                        sampling_rate=int(nconfig.SAMPLING_RATE),
                        upstream_helpers=upstream_helpers,
                    )
                    return feats.float().numpy()

                X_train = np.concatenate([
                    _preprocess(item[0], train_subject.electrode_labels)
                    for item in train_dataset
                ], axis=0)
                y_train = np.array([item[1] for item in train_dataset])
                X_test = np.concatenate([
                    _preprocess(item[0], subject.electrode_labels)
                    for item in test_dataset
                ], axis=0)
                y_test = np.array([item[1] for item in test_dataset])
                gc.collect()

                if splits_type == "CrossSubject":
                    regions_train = get_region_labels(train_subject)
                    regions_test = get_region_labels(subject)
                    if args.backend == "neuralset":
                        # Re-derive regions to match post-reference virtual channels.
                        # Bipolar collapses pairs (`chA-chB`); CAR/median preserve
                        # labels 1:1; shaft_laplacian preserves labels with
                        # remove_non_laplacian=False. Without this, regions sized
                        # to the original electrode count don't index post-ref X.
                        from preprocess_views import apply_reference  # noqa: PLC0415
                        train_labels_orig = list(train_subject.electrode_labels)
                        test_labels_orig = list(subject.electrode_labels)
                        _, train_labels_ref = apply_reference(
                            torch.zeros(1, len(train_labels_orig), 1),
                            train_labels_orig,
                            args.ref_kind,
                            upstream_helpers=upstream_helpers,
                        )
                        _, test_labels_ref = apply_reference(
                            torch.zeros(1, len(test_labels_orig), 1),
                            test_labels_orig,
                            args.ref_kind,
                            upstream_helpers=upstream_helpers,
                        )
                        regions_train = _derive_virtual_regions(
                            list(train_labels_ref),
                            dict(zip(train_labels_orig, regions_train)),
                        )
                        regions_test = _derive_virtual_regions(
                            list(test_labels_ref),
                            dict(zip(test_labels_orig, regions_test)),
                        )
                    X_train, X_test, _ = combine_regions(
                        X_train, X_test, regions_train, regions_test
                    )

                n_channels = int(X_train.shape[1])
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)

                if qc_payload is None:
                    qc_payload = {
                        "pre_train_sample": _subsample(X_train, 200_000),
                        "pre_test_sample":  _subsample(X_test,  200_000),
                        "qc_task": eval_name,
                        "qc_fold": fold_idx,
                    }

                X_train, X_test = apply_normalization(
                    X_train, X_test, args.normalization, n_channels=n_channels,
                )

                if qc_payload is not None and "post_train_sample" not in qc_payload:
                    qc_payload["post_train_sample"] = _subsample(X_train, 200_000)
                    qc_payload["post_test_sample"]  = _subsample(X_test,  200_000)

                gc.collect()

                clf = LogisticRegression(random_state=args.seed, max_iter=10000, tol=1e-3)
                clf.fit(X_train, y_train)
                train_accuracy = clf.score(X_train, y_train)
                test_accuracy = clf.score(X_test, y_test)
                train_probs = clf.predict_proba(X_train)
                test_probs = clf.predict_proba(X_test)

                valid_mask = np.isin(y_test, clf.classes_)
                y_test_filtered = y_test[valid_mask]
                test_probs_filtered = test_probs[valid_mask]

                n_classes = len(clf.classes_)
                y_test_oh = np.zeros((len(y_test_filtered), n_classes))
                for i, lab in enumerate(y_test_filtered):
                    y_test_oh[i, np.where(clf.classes_ == lab)[0][0]] = 1
                y_train_oh = np.zeros((len(y_train), n_classes))
                for i, lab in enumerate(y_train):
                    y_train_oh[i, np.where(clf.classes_ == lab)[0][0]] = 1

                if n_classes > 2:
                    train_roc = roc_auc_score(y_train_oh, train_probs, multi_class="ovr", average="macro")
                    test_roc = roc_auc_score(y_test_oh, test_probs_filtered, multi_class="ovr", average="macro")
                else:
                    train_roc = roc_auc_score(y_train_oh, train_probs)
                    test_roc = roc_auc_score(y_test_oh, test_probs_filtered)

                rows.append({
                    "subject_id": args.subject_id,
                    "trial_id": args.trial_id,
                    "task": eval_name,
                    "fold": fold_idx,
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "train_accuracy": float(train_accuracy),
                    "test_accuracy": float(test_accuracy),
                    "train_roc_auc": float(train_roc),
                    "test_roc_auc": float(test_roc),
                    "n_classes": int(n_classes),
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "n_features_post_norm": int(X_train.shape[1]),
                    "normalization": args.normalization,
                    "fold_seconds": float(time.time() - t_fold),
                })
                print(
                    f"[fold {fold_idx} bin {bin_start}-{bin_end} task {eval_name}] "
                    f"test AUROC {test_roc:.4f} acc {test_accuracy:.4f}"
                )

                del X_train, X_test, y_train, y_test, train_probs, test_probs
                del y_test_filtered, test_probs_filtered, y_test_oh, y_train_oh, clf
                gc.collect()

    return pd.DataFrame(rows), qc_payload


def apply_normalization(
    X_train: Any, X_test: Any, normalization: str, *, n_channels: int = 0,
) -> tuple[Any, Any]:
    if normalization == "train_set_fixed":
        scaler = StandardScaler(copy=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif normalization == "train_set_scale_only":
        scaler = StandardScaler(copy=False, with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif normalization == "per_session_fixed":
        train_scaler = StandardScaler(copy=False).fit(X_train)
        test_scaler = StandardScaler(copy=False).fit(X_test)
        X_train = train_scaler.transform(X_train)
        X_test = test_scaler.transform(X_test)
    elif normalization == "per_session_robust_mad":
        X_train = _mad_normalize(X_train)
        X_test = _mad_normalize(X_test)
    elif normalization == "train_set_robust_mad":
        X_train, X_test = _train_set_mad(X_train, X_test)
    elif normalization == "per_session_robust_scale":
        X_train = _mad_scale_only(X_train)
        X_test = _mad_scale_only(X_test)
    elif normalization == "per_channel_train_set_z":
        X_train, X_test = _per_channel_train_set_z(X_train, X_test, n_channels)
    elif normalization == "per_window_z":
        X_train = _per_window_z(X_train)
        X_test = _per_window_z(X_test)
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    return X_train, X_test


def _derive_virtual_regions(
    virtual_labels: list[str], label_to_region: dict[str, str]
) -> np.ndarray:
    """Map post-reference virtual channel labels back to DK regions.

    Bipolar pairs are formatted ``"chA-chB"``; both contacts are within-shaft
    adjacent neighbors so they share a DK region in practice. We take the
    anode (chA) region for the pair. Non-bipolar references (raw, CAR,
    shaft_laplacian with remove_non_laplacian=False, median) preserve labels
    1:1 and look up directly. Unknown labels fall back to ``"unknown"``.
    """
    out: list[str] = []
    for label in virtual_labels:
        anode = label.split("-", 1)[0] if "-" in label else label
        out.append(str(label_to_region.get(anode, "unknown")))
    return np.array(out)


def _per_window_z(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + 1e-8)


def _mad_normalize(X: np.ndarray) -> np.ndarray:
    med = np.median(X, axis=0, keepdims=True)
    mad = np.median(np.abs(X - med), axis=0, keepdims=True) * 1.4826
    return (X - med) / (mad + 1e-8)


def _train_set_mad(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(X_train, axis=0, keepdims=True)
    mad = np.median(np.abs(X_train - med), axis=0, keepdims=True) * 1.4826
    return (X_train - med) / (mad + 1e-8), (X_test - med) / (mad + 1e-8)


def _mad_scale_only(X: np.ndarray) -> np.ndarray:
    mad = np.median(np.abs(X), axis=0, keepdims=True) * 1.4826
    return X / (mad + 1e-8)


def _per_channel_train_set_z(
    X_train: np.ndarray, X_test: np.ndarray, n_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_channels <= 0 or X_train.shape[1] % n_channels != 0:
        raise ValueError(
            f"per_channel_train_set_z needs feature_dim divisible by n_channels; "
            f"got n_channels={n_channels} feature_dim={X_train.shape[1]}"
        )
    feat_per_ch = X_train.shape[1] // n_channels
    Xtr = X_train.reshape(X_train.shape[0], n_channels, feat_per_ch)
    Xte = X_test.reshape(X_test.shape[0], n_channels, feat_per_ch)
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True)
    Xtr = (Xtr - mu) / (sd + 1e-8)
    Xte = (Xte - mu) / (sd + 1e-8)
    return Xtr.reshape(X_train.shape[0], -1), Xte.reshape(X_test.shape[0], -1)


def _subsample(X: np.ndarray, n: int) -> np.ndarray:
    flat = np.asarray(X).reshape(-1)
    if flat.size <= n:
        return flat.copy()
    rng = np.random.default_rng(42)
    idx = rng.choice(flat.size, size=n, replace=False)
    return flat[idx]


def plot_signal_qc(
    path: Path,
    qc_payload: dict[str, Any] | None,
    cell_id: str,
    args: argparse.Namespace,
) -> None:
    if qc_payload is None or "post_train_sample" not in qc_payload:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, kind in zip(axes, ("train", "test")):
        pre = qc_payload[f"pre_{kind}_sample"]
        post = qc_payload[f"post_{kind}_sample"]
        ax.hist(pre, bins=200, alpha=0.5, density=True, label=f"pre  μ={pre.mean():.2f} σ={pre.std():.2f}")
        ax.hist(post, bins=200, alpha=0.5, density=True, label=f"post μ={post.mean():.2f} σ={post.std():.2f}")
        ax.set_title(f"{kind} feature magnitudes")
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle(
        f"{cell_id} recipe={args.normalization}  "
        f"sub{args.subject_id}/trial{args.trial_id}/{qc_payload.get('qc_task', args.task)}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summarize(diagnostics: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n_rows": int(len(diagnostics)),
        "n_tasks": int(diagnostics["task"].nunique()),
        "mean_test_roc_auc": float(diagnostics["test_roc_auc"].mean()),
        "mean_test_accuracy": float(diagnostics["test_accuracy"].mean()),
        "min_test_roc_auc": float(diagnostics["test_roc_auc"].min()),
        "max_test_roc_auc": float(diagnostics["test_roc_auc"].max()),
    }


def write_readme(
    out_dir: Path,
    summary: dict[str, float | int],
    cell_id: str,
    args: argparse.Namespace,
) -> None:
    lines = [
        f"# Neuroprobe Stage 0 L.1 Linear Baseline — {cell_id}",
        "",
        f"- recipe: `{args.normalization}` ({NORMALIZATION_DESCRIPTIONS[args.normalization]})",
        f"- subject: {args.subject_id}, trial: {args.trial_id}, task: {args.task}",
        f"- split: {args.split_type}, binary: {args.binary_tasks}",
        f"- preprocess: `{args.preprocess_type}`",
        f"- mean test AUROC: {summary['mean_test_roc_auc']:.6f}",
        f"- mean test accuracy: {summary['mean_test_accuracy']:.6f}",
        "",
        "Files:",
        "- `diagnostics.csv` — per-fold per-task AUROC/accuracy",
        "- `metrics.json` — aggregate summary",
        "- `signal_qc.png` — feature-magnitude histogram pre vs post normalization",
        "- `experiment_record.json` — ExperimentLogger sidecar",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    env_root = os.environ.get("ROOT_DIR_BRAINTREEBANK", "").strip()
    p.add_argument("--bt-root", type=Path, default=Path(env_root) if env_root else None)
    p.add_argument(
        "--neuroprobe-repo",
        type=Path,
        default=(
            Path("/work/ht203/repo/neuroprobe_upstream")
            if Path("/work/ht203").exists()
            else Path(".cache/neuroprobe_upstream")
        ),
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--subject-id", type=int, required=True)
    p.add_argument("--trial-id", type=int, required=True)
    p.add_argument("--task", default="pitch")
    p.add_argument(
        "--split-type",
        choices=("WithinSession", "CrossSession", "CrossSubject"),
        required=True,
    )
    p.add_argument(
        "--binary-tasks",
        type=lambda v: v.lower() in {"1", "true", "yes"},
        default=False,
    )
    p.add_argument("--preprocess-type", default="laplacian-stft_abs")
    p.add_argument(
        "--backend",
        choices=("upstream", "neuralset"),
        default="upstream",
        help="upstream = call eval_utils.preprocess_data (default, byte-equivalent to D.0). "
             "neuralset = factor reference × view via scripts/neuroprobe/preprocess_views.py.",
    )
    p.add_argument(
        "--ref-kind",
        choices=("raw", "bipolar", "shaft_laplacian", "global_car", "shaft_car", "median"),
        default="shaft_laplacian",
        help="Spatial reference (only used when --backend=neuralset).",
    )
    p.add_argument(
        "--view-kind",
        choices=(
            "raw_voltage", "stft_abs", "hg_envelope",
            "log_stft", "hg_envelope_wide", "low_lfp",
            "multi_band_log_power", "wavelet_db4",
            "instantaneous_phase",
        ),
        default="stft_abs",
        help="Input view (only used when --backend=neuralset).",
    )
    p.add_argument(
        "--run-nuisance-probes",
        action="store_true",
        help="Also train probes on subject_id / session_id from same X (L.5 P1/P2). "
             "Writes nuisance_probe_metrics.csv to out_dir.",
    )
    p.add_argument(
        "--normalization",
        choices=tuple(NORMALIZATION_CELLS),
        required=True,
    )
    p.add_argument(
        "--anchor-start-before", type=float, default=0.0,
        help="Seconds before word onset to include in feature window. "
             "Default = 0.0 (D.0 baseline). L.4 sanity = 0.375.",
    )
    p.add_argument(
        "--anchor-end-after", type=float, default=1.0,
        help="Seconds after word onset to include in feature window. "
             "Default = 1.0 (D.0 baseline). L.4 sanity = 0.625.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reference-mean-auroc", type=float, default=None)
    args = p.parse_args()
    if args.bt_root is None:
        p.error("--bt-root or ROOT_DIR_BRAINTREEBANK is required")
    if args.split_type == "CrossSubject" and args.subject_id == 2:
        p.error("CrossSubject cannot evaluate subject 2; Neuroprobe uses subject 2 as train")
    return args


if __name__ == "__main__":
    main()
