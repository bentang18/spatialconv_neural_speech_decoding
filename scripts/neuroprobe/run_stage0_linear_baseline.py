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
import hashlib
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
    "per_session_robust_freqbin": "L.1.NA",
    "per_session_robust_pooled":  "L.1.NB",
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
    "per_session_robust_freqbin": "robust median + 1.4826*MAD per (electrode, freq-bin) pooled over time+batch, fit per session (transductive) — Nv14 per-freq-bin-z proxy",
    "per_session_robust_pooled":  "robust median + 1.4826*MAD per electrode pooled over time+freq+batch, fit per session (transductive) — pooled-z split proxy (Job-1 only)",
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

    cell_id = args.cell_id or NORMALIZATION_CELLS[args.normalization]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    notch_freqs = _parse_notch_freqs(args.notch_freqs)
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
        "notch_freqs": list(notch_freqs),
        "hpf_hz": float(args.hpf_hz),
        "pool_multi_source": bool(getattr(args, "pool_multi_source", False)),
        "region_pool_mode": str(getattr(args, "region_pool_mode", "pairwise")),
        "cell_id": cell_id,
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
    pool_multi_source = bool(getattr(args, "pool_multi_source", False))
    train_subject: Any = None
    all_subjects: dict[int, Any] = {}
    source_subjects: dict[int, Any] = {}
    source_subject_ids: list[int] = []
    fold_source_subjects: list[Any] | None = None
    fold_source_trials: list[int] | None = None

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
        from speech_decoding.views import make_upstream_helpers
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
    fb_start = getattr(args, "feature_bin_start", None)
    fb_end = getattr(args, "feature_bin_end", None)
    if fb_start is not None:
        bin_starts = [float(fb_start)]
    if fb_end is not None:
        bin_ends = [float(fb_end)]

    subject = BrainTreebankSubject(args.subject_id, cache=True, dtype=torch.float32)
    subset_electrodes(subject, lite=True, nano=False)
    keep_n = int(getattr(args, "keep_n_electrodes", 0) or 0)
    if keep_n > 0:
        full_labels = list(subject.electrode_labels)
        if keep_n < len(full_labels):
            rng = np.random.default_rng(int(getattr(args, "electrode_dropout_seed", args.seed)))
            kept_idx = rng.choice(len(full_labels), size=keep_n, replace=False)
            kept_idx.sort()
            kept_labels = [full_labels[i] for i in kept_idx]
            subject.set_electrode_subset(kept_labels)
            print(
                f"[L.6.ES] random dropout: kept {keep_n}/{len(full_labels)} electrodes "
                f"(seed={int(getattr(args, 'electrode_dropout_seed', args.seed))})"
            )
        else:
            print(
                f"[L.6.ES] keep_n {keep_n} ≥ available {len(full_labels)}; "
                f"keeping all electrodes"
            )
    if getattr(args, "shuffle_channels_per_subject", False):
        full = list(subject.electrode_labels)
        perm_seed = int(args.seed) * 1_000_000 + int(args.subject_id) * 1000
        perm = np.random.default_rng(perm_seed).permutation(len(full))
        subject.set_electrode_subset([full[i] for i in perm])
        print(
            f"[L.5.P6] subject {args.subject_id} channels shuffled "
            f"(perm_seed={perm_seed}, n={len(full)})"
        )
    t0 = time.time()
    subject.load_neural_data(args.trial_id)
    print(f"[load] subject {args.subject_id} trial {args.trial_id} in {time.time() - t0:.1f}s")

    notch_freqs = _parse_notch_freqs(getattr(args, "notch_freqs", ""))
    hpf_hz = float(getattr(args, "hpf_hz", 0.0))
    if notch_freqs or hpf_hz > 0:
        from speech_decoding.views import apply_temporal_filter_inplace  # noqa: PLC0415
        sr = int(nconfig.SAMPLING_RATE)
        t1 = time.time()
        apply_temporal_filter_inplace(
            subject.neural_data_cache[args.trial_id],
            sampling_rate=sr, notch_freqs=tuple(notch_freqs), hpf_hz=hpf_hz,
        )
        print(
            f"[L.3] filter notch={list(notch_freqs)} hpf={hpf_hz}Hz "
            f"applied to subject {args.subject_id} trial {args.trial_id} "
            f"in {time.time() - t1:.1f}s"
        )

    rows: list[dict[str, Any]] = []
    qc_payload: dict[str, Any] | None = None

    # STFT feature cache. The 15 Neuroprobe tasks share one trial and heavily
    # overlapping word-onset windows; STFT features depend only on (raw window,
    # electrode order, view), not on the task. Memoize on a content hash of the
    # sliced raw window so each unique window is transformed once, not once per
    # task. Pure speedup — same features out.
    view_sig = f"{args.backend}|{args.ref_kind}|{args.view_kind}|{preprocess_type}"
    _feat_cache: dict[bytes, np.ndarray] = {}
    _cache_stats = {"hits": 0, "misses": 0}

    # Build train/source subjects once, before the task loop. The 15 Neuroprobe
    # tasks share the same recordings, so constructing these inside the loop
    # gave each task a fresh empty neural_data_cache and re-loaded the source
    # HDF5 ~15×. Construction and the deterministic electrode subset/shuffle
    # (task-independent — keyed on seed + subject id, not eval_name) are hoisted
    # here; only generate_splits_* stays per-task. NOTE: the pool-multi-source
    # path still clears each source's neural cache per fold (OOM guard) so it
    # reloads per task — hoisting only removes its object-construction cost. The
    # non-pool CrossSubject path keeps its train subject's cache resident, so
    # its train HDF5 loads exactly once across all tasks.
    if splits_type in ("WithinSession", "CrossSession"):
        train_subject = subject
    elif splits_type == "CrossSubject" and pool_multi_source:
        # D.14: train across every NEUROPROBE_LITE source subject (excluding test).
        for src_id, _ in nconfig.NEUROPROBE_LITE_SUBJECT_TRIALS:
            if src_id == args.subject_id or src_id in source_subjects:
                continue
            src_subj = BrainTreebankSubject(
                src_id, allow_corrupted=False, cache=True, dtype=torch.float32,
            )
            src_subj.set_electrode_subset(
                nconfig.NEUROPROBE_LITE_ELECTRODES[src_subj.subject_identifier]
            )
            source_subjects[src_id] = src_subj
            source_subject_ids.append(src_id)
        if not source_subject_ids:
            raise RuntimeError("--pool-multi-source: no source subjects available.")
        all_subjects = {args.subject_id: subject, **source_subjects}
        # Each upstream fold corresponds to one source (subject_id, trial_id)
        # pair from NEUROPROBE_LITE_SUBJECT_TRIALS; recover both for ordering
        # alignment AND for per-fold cache cleanup after feature extraction.
        fold_source_subjects = [
            source_subjects[src_id]
            for src_id, _ in nconfig.NEUROPROBE_LITE_SUBJECT_TRIALS
            if src_id != args.subject_id
        ]
        fold_source_trials = [
            src_trial
            for src_id, src_trial in nconfig.NEUROPROBE_LITE_SUBJECT_TRIALS
            if src_id != args.subject_id
        ]
        print(
            f"[D.14] pool_multi_source: {len(source_subject_ids)} source subjects "
            f"{source_subject_ids} → 1 pooled train, region_pool_mode={args.region_pool_mode}"
        )
    elif splits_type == "CrossSubject":
        train_subject_id = nconfig.DS_DM_TRAIN_SUBJECT_ID
        train_subject = BrainTreebankSubject(
            train_subject_id, allow_corrupted=False, cache=True, dtype=torch.float32
        )
        train_subject.set_electrode_subset(
            nconfig.NEUROPROBE_LITE_ELECTRODES[train_subject.subject_identifier]
        )
        if getattr(args, "shuffle_channels_per_subject", False):
            t_full = list(train_subject.electrode_labels)
            t_seed = int(args.seed) * 1_000_000 + int(train_subject_id) * 1000
            t_perm = np.random.default_rng(t_seed).permutation(len(t_full))
            train_subject.set_electrode_subset([t_full[i] for i in t_perm])
            print(
                f"[L.5.P6] train subject {train_subject_id} channels shuffled "
                f"(perm_seed={t_seed}, n={len(t_full)})"
            )
        all_subjects = {args.subject_id: subject, train_subject_id: train_subject}
    else:
        raise ValueError(f"Unknown split type: {splits_type}")

    for eval_name in eval_names:
        if splits_type == "WithinSession":
            folds = nts.generate_splits_within_session(
                subject, args.trial_id, eval_name, dtype=torch.float32,
                output_indices=False, output_dict=False,
                start_neural_data_before_word_onset=int(bins_start_before * nconfig.SAMPLING_RATE),
                end_neural_data_after_word_onset=int(bins_end_after * nconfig.SAMPLING_RATE),
                lite=True, nano=False, binary_tasks=binary_tasks,
            )
        elif splits_type == "CrossSession":
            folds = nts.generate_splits_cross_session(
                subject, args.trial_id, eval_name, dtype=torch.float32,
                output_indices=False, output_dict=False,
                start_neural_data_before_word_onset=int(bins_start_before * nconfig.SAMPLING_RATE),
                end_neural_data_after_word_onset=int(bins_end_after * nconfig.SAMPLING_RATE),
                lite=True, binary_tasks=binary_tasks,
            )
        elif splits_type == "CrossSubject" and pool_multi_source:
            folds = nts.generate_splits_cross_subject(
                all_subjects, args.subject_id, args.trial_id, eval_name,
                dtype=torch.float32, output_indices=False, output_dict=False,
                start_neural_data_before_word_onset=int(bins_start_before * nconfig.SAMPLING_RATE),
                end_neural_data_after_word_onset=int(bins_end_after * nconfig.SAMPLING_RATE),
                lite=True, nano=False, binary_tasks=binary_tasks,
                include_all_train_subjects=True,
            )
            assert fold_source_subjects is not None
            if len(fold_source_subjects) != len(folds):
                raise RuntimeError(
                    f"upstream returned {len(folds)} folds but built "
                    f"{len(fold_source_subjects)} source subjects — ordering mismatch"
                )
        elif splits_type == "CrossSubject":
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

            def _preprocess(item_data, electrode_labels):
                sliced = item_data[:, data_idx_from:data_idx_to]
                key = _window_cache_key(sliced, electrode_labels, view_sig)
                cached = _feat_cache.get(key)
                if cached is not None:
                    _cache_stats["hits"] += 1
                    return cached
                _cache_stats["misses"] += 1
                batched = sliced.unsqueeze(0)
                if args.backend == "upstream":
                    feats_np = preprocess_data(
                        batched, electrode_labels, preprocess_type, preprocess_parameters,
                    ).float().numpy()
                else:
                    from speech_decoding.views import preprocess_views  # noqa: PLC0415
                    feats, _ = preprocess_views(
                        batched, list(electrode_labels),
                        ref_kind=args.ref_kind, view_kind=args.view_kind,
                        sampling_rate=int(nconfig.SAMPLING_RATE),
                        upstream_helpers=upstream_helpers,
                    )
                    feats_np = feats.float().numpy()
                _feat_cache[key] = feats_np
                return feats_np

            # Build evaluation cells. Single-source CrossSubject (and Within /
            # CrossSession) produce one cell per upstream fold; pool-multi-source
            # CrossSubject (D.14) collapses all upstream folds into one cell
            # using --region-pool-mode (test-anchored | source-union).
            eval_cells: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            if splits_type == "CrossSubject" and pool_multi_source:
                per_src_X: list[np.ndarray] = []
                per_src_y: list[np.ndarray] = []
                per_src_r: list[np.ndarray] = []
                X_test_raw: np.ndarray | None = None
                y_test_pool: np.ndarray | None = None
                regions_test_raw: np.ndarray | None = None
                assert fold_source_subjects is not None and fold_source_trials is not None
                for fi, fold in enumerate(folds):
                    train_dataset = fold["train_dataset"]
                    test_dataset = fold["test_dataset"]
                    src_subj = fold_source_subjects[fi]
                    src_trial = fold_source_trials[fi]
                    X_tr = np.concatenate(
                        [_preprocess(item[0], src_subj.electrode_labels) for item in train_dataset],
                        axis=0,
                    )
                    y_tr = np.array([item[1] for item in train_dataset])
                    r_tr = _resolve_regions_for_subject(
                        src_subj, args, upstream_helpers, get_region_labels,
                    )
                    per_src_X.append(X_tr)
                    per_src_y.append(y_tr)
                    per_src_r.append(r_tr)
                    if X_test_raw is None:
                        X_test_raw = np.concatenate(
                            [_preprocess(item[0], subject.electrode_labels) for item in test_dataset],
                            axis=0,
                        )
                        y_test_pool = np.array([item[1] for item in test_dataset])
                        regions_test_raw = _resolve_regions_for_subject(
                            subject, args, upstream_helpers, get_region_labels,
                        )
                    src_subj.clear_neural_data_cache(src_trial)
                    gc.collect()
                assert X_test_raw is not None and y_test_pool is not None and regions_test_raw is not None
                if args.region_pool_mode == "test-anchored":
                    target_regions = sorted(set(regions_test_raw.tolist()))
                elif args.region_pool_mode == "source-union":
                    target_regions = sorted(set(np.concatenate(per_src_r).tolist()))
                else:
                    raise ValueError(
                        f"--pool-multi-source requires --region-pool-mode in "
                        f"{{test-anchored, source-union}}, got {args.region_pool_mode}"
                    )
                X_train_pool_parts = [
                    _pool_to_target_regions(X, r, target_regions)
                    for X, r in zip(per_src_X, per_src_r)
                ]
                X_train_pooled = np.concatenate(X_train_pool_parts, axis=0)
                y_train_pooled = np.concatenate(per_src_y, axis=0)
                X_test_pooled = _pool_to_target_regions(X_test_raw, regions_test_raw, target_regions)
                eval_cells.append((0, X_train_pooled, y_train_pooled, X_test_pooled, y_test_pool))
                print(
                    f"[D.14] pooled cell: |target_regions|={len(target_regions)} "
                    f"({args.region_pool_mode}), n_train={len(y_train_pooled)} "
                    f"across {len(per_src_X)} source subjects, n_test={len(y_test_pool)}"
                )
                del per_src_X, per_src_r, X_train_pool_parts, X_test_raw
                gc.collect()
            else:
                for fold_idx, fold in enumerate(folds):
                    train_dataset = fold["train_dataset"]
                    test_dataset = fold["test_dataset"]
                    assert train_subject is not None
                    X_train = np.concatenate(
                        [_preprocess(item[0], train_subject.electrode_labels) for item in train_dataset],
                        axis=0,
                    )
                    y_train = np.array([item[1] for item in train_dataset])
                    X_test = np.concatenate(
                        [_preprocess(item[0], subject.electrode_labels) for item in test_dataset],
                        axis=0,
                    )
                    y_test = np.array([item[1] for item in test_dataset])
                    gc.collect()
                    if splits_type == "CrossSubject":
                        regions_train = _resolve_regions_for_subject(
                            train_subject, args, upstream_helpers, get_region_labels,
                        )
                        regions_test = _resolve_regions_for_subject(
                            subject, args, upstream_helpers, get_region_labels,
                        )
                        X_train, X_test, _ = combine_regions(
                            X_train, X_test, regions_train, regions_test
                        )
                    eval_cells.append((fold_idx, X_train, y_train, X_test, y_test))

            for fold_idx, X_train, y_train, X_test, y_test in eval_cells:
                t_fold = time.time()
                n_channels = int(X_train.shape[1])
                fa_mode = getattr(args, "feature_aggregation", "none") or "none"
                if fa_mode in ("mean", "median") and X_train.ndim == 3:
                    op = np.mean if fa_mode == "mean" else np.median
                    X_train = op(X_train, axis=2)
                    X_test = op(X_test, axis=2)
                feat_shape = tuple(X_train.shape[1:])
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)
                if fa_mode == "pca50_cross_channel":
                    from sklearn.decomposition import PCA  # noqa: PLC0415
                    n_comp = int(min(50, X_train.shape[1], max(1, X_train.shape[0] - 1)))
                    pca = PCA(n_components=n_comp, random_state=args.seed)
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)

                if qc_payload is None:
                    qc_payload = {
                        "pre_train_sample": _subsample(X_train, 200_000),
                        "pre_test_sample":  _subsample(X_test,  200_000),
                        "qc_task": eval_name,
                        "qc_fold": fold_idx,
                    }

                X_train, X_test = apply_normalization(
                    X_train, X_test, args.normalization,
                    n_channels=n_channels, feat_shape=feat_shape,
                )

                if qc_payload is not None and "post_train_sample" not in qc_payload:
                    qc_payload["post_train_sample"] = _subsample(X_train, 200_000)
                    qc_payload["post_test_sample"]  = _subsample(X_test,  200_000)

                gc.collect()

                if getattr(args, "shuffle_labels", False):
                    rng = np.random.default_rng(args.seed)
                    y_train = rng.permutation(y_train)
                    y_test = rng.permutation(y_test)

                cw_mode = getattr(args, "class_weight", "none") or "none"
                if cw_mode == "none":
                    cw: Any = None
                elif cw_mode == "balanced":
                    cw = "balanced"
                elif cw_mode == "effective_num":
                    beta = 0.999
                    cls, counts = np.unique(y_train, return_counts=True)
                    eff = (1.0 - beta ** counts) / (1.0 - beta)
                    weights = (1.0 / eff)
                    weights = weights / weights.sum() * len(cls)
                    cw = {int(c): float(w) for c, w in zip(cls, weights)}
                else:
                    raise ValueError(f"Unknown class_weight: {cw_mode}")
                clf = LogisticRegression(
                    random_state=args.seed, max_iter=10000, tol=1e-3, class_weight=cw,
                )
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

    _total = _cache_stats["hits"] + _cache_stats["misses"]
    if _total:
        cache_mb = sum(a.nbytes for a in _feat_cache.values()) / 1e6
        print(
            f"[stft-cache] {_cache_stats['hits']}/{_total} hits "
            f"({100 * _cache_stats['hits'] / _total:.1f}%); "
            f"{len(_feat_cache)} unique windows, {cache_mb:.0f} MB"
        )
    return pd.DataFrame(rows), qc_payload


def _window_cache_key(window: Any, electrode_labels: Any, view_sig: str) -> bytes:
    """Content hash of a sliced raw neural window for STFT-feature memoization.

    Two items with byte-identical sliced windows, electrode order, and view
    config produce identical features and share a cache entry. Keying on
    content (not word id) keeps this independent of upstream dataset internals.
    """
    arr = window.detach().cpu().contiguous().numpy()
    h = hashlib.blake2b(digest_size=16)
    h.update(view_sig.encode())
    h.update(repr((arr.shape, str(arr.dtype))).encode())
    h.update(repr(tuple(electrode_labels)).encode())
    h.update(arr.tobytes())
    return h.digest()


def apply_normalization(
    X_train: Any, X_test: Any, normalization: str, *, n_channels: int = 0,
    feat_shape: tuple[int, ...] | None = None,
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
    elif normalization == "per_session_robust_freqbin":
        X_train = _robust_freqbin(X_train, feat_shape)
        X_test = _robust_freqbin(X_test, feat_shape)
    elif normalization == "per_session_robust_pooled":
        X_train = _robust_pooled(X_train, feat_shape)
        X_test = _robust_pooled(X_test, feat_shape)
    elif normalization == "per_window_z":
        X_train = _per_window_z(X_train)
        X_test = _per_window_z(X_test)
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    return X_train, X_test


def _resolve_regions_for_subject(
    subject: Any,
    args: argparse.Namespace,
    upstream_helpers: Any,
    get_region_labels: Any,
) -> np.ndarray:
    """Return DK regions aligned to the post-reference virtual channels.

    For backend=upstream the upstream pipeline preserves labels 1:1 so
    `get_region_labels(subject)` is correct as-is. For backend=neuralset the
    reference may collapse pairs (bipolar) or preserve labels (CAR /
    shaft_laplacian / median); we re-derive virtual-channel regions from the
    original `electrode_labels → region` map.
    """
    base = get_region_labels(subject)
    if args.backend != "neuralset":
        return base
    from speech_decoding.views import apply_reference  # noqa: PLC0415
    labels_orig = list(subject.electrode_labels)
    _, labels_ref = apply_reference(
        torch.zeros(1, len(labels_orig), 1),
        labels_orig,
        args.ref_kind,
        upstream_helpers=upstream_helpers,
    )
    return _derive_virtual_regions(list(labels_ref), dict(zip(labels_orig, base)))


def _pool_to_target_regions(
    X: np.ndarray, regions: np.ndarray, target_regions: list[str],
) -> np.ndarray:
    """Mean-pool channels per region; zero-fill regions absent from `regions`.

    X has shape (B, C, ...) — any trailing time/feature dims preserved.
    Output shape (B, len(target_regions), ...). Channels with `regions[i] == r`
    are mean-aggregated for each target region r; if no channel matches r the
    column is filled with zeros.
    """
    if X.ndim < 2:
        raise ValueError(f"_pool_to_target_regions expects (B, C, ...), got shape {X.shape}")
    out_shape = (X.shape[0], len(target_regions)) + X.shape[2:]
    out = np.zeros(out_shape, dtype=X.dtype)
    for j, r in enumerate(target_regions):
        mask = regions == r
        if mask.any():
            out[:, j, ...] = X[:, mask, ...].mean(axis=1)
    return out


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


def _robust_freqbin(X: np.ndarray, feat_shape: tuple[int, ...] | None) -> np.ndarray:
    """Robust z per (electrode, freq-bin), pooled over time + batch (Nv14 proxy).

    feat_shape is the per-window feature shape (C, [T], F) with frequency as the
    last axis — the STFT layout from `preprocess_stft` is (B, C, n_timebins,
    n_freqs). Statistic is fit on X itself (transductive / per-session); called
    separately on train and test.
    """
    if feat_shape is None or len(feat_shape) < 2:
        raise ValueError(
            f"per_session_robust_freqbin needs feat_shape (C, [T], F); got {feat_shape}"
        )
    n_freqs = int(feat_shape[-1])
    n_channels = int(feat_shape[0])
    Xr = X.reshape(X.shape[0], n_channels, -1, n_freqs)  # (B, C, M, F)
    med = np.median(Xr, axis=(0, 2), keepdims=True)       # (1, C, 1, F)
    mad = np.median(np.abs(Xr - med), axis=(0, 2), keepdims=True) * 1.4826
    Xr = (Xr - med) / (mad + 1e-8)
    return Xr.reshape(X.shape[0], -1)


def _robust_pooled(X: np.ndarray, feat_shape: tuple[int, ...] | None) -> np.ndarray:
    """Robust z per electrode, pooled over time + freq + batch (pooled-z proxy).

    Removes per-recording per-channel scale only, preserving each recording's
    cross-frequency shape. Transductive / per-session: fit on X itself.
    """
    if feat_shape is None or len(feat_shape) < 2:
        raise ValueError(
            f"per_session_robust_pooled needs feat_shape (C, ...); got {feat_shape}"
        )
    n_channels = int(feat_shape[0])
    Xr = X.reshape(X.shape[0], n_channels, -1)            # (B, C, F)
    med = np.median(Xr, axis=(0, 2), keepdims=True)       # (1, C, 1)
    mad = np.median(np.abs(Xr - med), axis=(0, 2), keepdims=True) * 1.4826
    Xr = (Xr - med) / (mad + 1e-8)
    return Xr.reshape(X.shape[0], -1)


def _parse_notch_freqs(spec: str) -> tuple[float, ...]:
    if not spec:
        return ()
    return tuple(float(x.strip()) for x in spec.split(",") if x.strip())


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
    notch = getattr(args, "notch_freqs", "") or "(none)"
    hpf = getattr(args, "hpf_hz", 0.0)
    lines = [
        f"# Neuroprobe Stage 0 Linear Baseline — {cell_id}",
        "",
        f"- normalization: `{args.normalization}` ({NORMALIZATION_DESCRIPTIONS[args.normalization]})",
        f"- subject: {args.subject_id}, trial: {args.trial_id}, task: {args.task}",
        f"- split: {args.split_type}, binary: {args.binary_tasks}",
        f"- preprocess: `{args.preprocess_type}` (backend={args.backend})",
        f"- ref_kind: `{getattr(args, 'ref_kind', '-')}`  view_kind: `{getattr(args, 'view_kind', '-')}`",
        f"- notch: `{notch}` Hz  hpf: `{hpf}` Hz",
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
             "neuralset = factor reference × view via speech_decoding.views.",
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
            "hg_envelope_70_90", "hg_envelope_90_120", "hg_envelope_120_150",
            "low_gamma_30_70", "vhg_150_300", "mua_300_500",
            "envelope_30_500", "envelope_70_500", "envelope_30_150",
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
        "--shuffle-labels",
        action="store_true",
        help="L.5.P11 label-permutation null: independently permute y_train and "
             "y_test before LogReg fit (seeded by --seed). Held-out AUROC must "
             "be ≤ empirical-chance + 0.02 across 3 seeds. Anything above flags "
             "label-into-features leakage in the pipeline.",
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
    p.add_argument(
        "--notch-freqs", default="",
        help="Comma-separated notch-filter frequencies in Hz (e.g. '60,120,180'). "
             "Applied per-channel via zero-phase IIR notch on the FULL session "
             "voltage before trial slicing. Empty = no notch (L.3.F0 default).",
    )
    p.add_argument(
        "--hpf-hz", type=float, default=0.0,
        help="High-pass filter cutoff in Hz (e.g. 0.5, 1.0). 4th-order Butterworth, "
             "applied per-channel via zero-phase filtfilt on the FULL session voltage. "
             "0.0 = no HPF (L.3.F0/F1 default). HPF on 1-s windowed slices is "
             "mathematically broken at <2 Hz cutoffs — must be applied pre-window.",
    )
    p.add_argument(
        "--cell-id", default=None,
        help="Optional override for the L-cell ID written to ExperimentLogger "
             "(e.g. 'L.3.F1'). Defaults to the L.1 cell derived from --normalization.",
    )
    p.add_argument(
        "--keep-n-electrodes", type=int, default=0,
        help="L.6.ES random electrode dropout: after Lite cap, restrict to a "
             "random subset of size keep_n (sorted by original index). 0 = no "
             "dropout. If keep_n exceeds available, all are kept.",
    )
    p.add_argument(
        "--electrode-dropout-seed", type=int, default=None,
        help="Seed for L.6.ES random electrode subset draw. Defaults to --seed.",
    )
    p.add_argument(
        "--shuffle-channels-per-subject", action="store_true",
        help="L.5.P6: independently permute electrode order per subject before "
             "reference is applied. CrossSubject only; intent is to verify that "
             "cross-subject task accuracy drops substantially when channel "
             "identity correspondence is broken.",
    )
    p.add_argument(
        "--feature-bin-start", type=float, default=None,
        help="L.5.P5: override start of feature slice (seconds, in onset-anchored "
             "frame). Default = -anchor_start_before. Use to slice a sub-window "
             "of the loaded context — e.g., load [0, 6]s and slice [5, 6]s.",
    )
    p.add_argument(
        "--feature-bin-end", type=float, default=None,
        help="L.5.P5: override end of feature slice (seconds). Default = "
             "anchor_end_after.",
    )
    p.add_argument(
        "--class-weight", choices=("none", "balanced", "effective_num"),
        default="none",
        help="L.6.CB: LogisticRegression class_weight. balanced = inverse "
             "frequency. effective_num = Cui 2019 (β=0.999).",
    )
    p.add_argument(
        "--feature-aggregation",
        choices=("none", "mean", "median", "pca50_cross_channel"),
        default="none",
        help="L.6.FA: aggregation applied to (B, C, F) features before LogReg. "
             "mean / median collapse the time/feature axis per channel; "
             "pca50_cross_channel reduces flattened (B, C*F) → 50 components.",
    )
    p.add_argument(
        "--pool-multi-source", action="store_true",
        help="D.14: pool training across every NEUROPROBE_LITE source subject "
             "(excluding the held-out test subject) into a single classifier "
             "instead of one fold per source. Requires --split-type CrossSubject. "
             "Test subject may be subject 2 in this mode.",
    )
    p.add_argument(
        "--region-pool-mode",
        choices=("pairwise", "test-anchored", "source-union"),
        default="pairwise",
        help="DK region pooling for CrossSubject. pairwise = upstream "
             "intersection mean-pool (single-source default). test-anchored = "
             "D.14 default; pool to test subject's DK regions, mean over source "
             "channels per region, zero-fill missing. source-union = D.14 sister; "
             "pool to ⋃ source DK regions, zero-fill missing in test.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reference-mean-auroc", type=float, default=None)
    args = p.parse_args()
    if args.bt_root is None:
        p.error("--bt-root or ROOT_DIR_BRAINTREEBANK is required")
    if args.pool_multi_source and args.split_type != "CrossSubject":
        p.error("--pool-multi-source requires --split-type CrossSubject")
    if args.pool_multi_source and args.region_pool_mode == "pairwise":
        p.error(
            "--pool-multi-source requires --region-pool-mode in {test-anchored, source-union}"
        )
    if (
        args.split_type == "CrossSubject"
        and args.subject_id == 2
        and not args.pool_multi_source
    ):
        p.error("CrossSubject cannot evaluate subject 2; Neuroprobe uses subject 2 as train")
    return args


if __name__ == "__main__":
    main()
