"""DCC entry point: run LogReg probes over per-trial Whisper feature NPZs.

For every Neuroprobe Lite continuous task + face_num, fit:
    - within-trial (random 80/20)
    - cross-session (train trial A, test trial B same subject)
    - cross-subject (LOSO)
at every Whisper layer + a concat-of-all-layers sister.

Output: one CSV per task with (subject, trial, layer, split, auroc, n_train, n_test, seed).

Usage:
    .venv/bin/python scripts/neuroprobe/teacher_ceiling/run_probe_ceiling.py \\
        --features-dir /hpc/group/coganlab/ht203/cache_neuroai/whisper_ceiling/v3 \\
        --out-dir reports/neuroprobe_whisper_ceiling_2026_05_28
"""
from __future__ import annotations

import os

# Pin BLAS to a single thread per process so joblib's process-level parallelism
# doesn't oversubscribe CPUs (8 workers × 8 OMP threads → 64 threads on 8 cores
# is a 5-10× wall-clock regression). MUST be set before numpy/sklearn import.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import csv
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

# Belt + suspenders — also set in probe.py so loky workers inherit the filter
# via re-import; this line covers warnings emitted from main-process paths.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from speech_decoding.whisper_ceiling.probe import (
    DEFAULT_TASKS,
    ProbeResult,
    binary_labels_face_num,
    binary_labels_from_continuous,
    fit_probe_cross_session,
    fit_probe_cross_subject,
    fit_probe_within_trial,
    fit_probe_within_trial_with_scores,
    topk_layers_by_within_auroc,
)


def _n_jobs() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


# Single-slot per-worker cache. With joblib loky workers persistent across
# delayed() calls, this means each worker loads a given trial NPZ at most once
# even when it gets assigned multiple (sid, tid, task) units in sequence.
_WORKER_TRIAL_CACHE: dict[str, dict] = {}


def _load_trial_in_worker(npz_path: str) -> dict:
    cached = _WORKER_TRIAL_CACHE.get(npz_path)
    if cached is not None:
        return cached
    _WORKER_TRIAL_CACHE.clear()
    data = dict(np.load(npz_path, allow_pickle=False))
    _WORKER_TRIAL_CACHE[npz_path] = data
    return data

NEUROPROBE_LITE_SUBJECT_TRIALS = [
    (1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
    (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1),
]


def load_trial_npz(npz_path: Path) -> dict:
    """Load a per-trial features NPZ.

    Returns dict with keys:
        layers: np.ndarray of layer indices
        layer_<L>: (W, d) features
        word_index, movie_start_s: (W,)
        label_<col>: (W,) per known label column
    """
    data = dict(np.load(npz_path, allow_pickle=False))
    return data


def stack_concat(data: dict, layers: list[int]) -> np.ndarray:
    return np.concatenate([data[f"layer_{L}"].astype(np.float32) for L in layers], axis=1)


def mean_all(data: dict, layers: list[int]) -> np.ndarray:
    return np.mean(
        np.stack([data[f"layer_{L}"].astype(np.float32) for L in layers], axis=0),
        axis=0,
    )


def features_for_key(data: dict, layers: list[int], key) -> np.ndarray:
    """Resolve a layer-merge key into a (W, ?) feature matrix.

    Supported keys:
        int                 - single layer's mean-pooled features
        "concat_all"        - concat every hooked layer
        "mean_all"          - elementwise mean across hooked layers
        ("concat_top", k, [layer_indices]) - concat top-k layers (data-driven)
    """
    if isinstance(key, int):
        return data[f"layer_{key}"].astype(np.float32)
    if key == "concat_all":
        return stack_concat(data, layers)
    if key == "mean_all":
        return mean_all(data, layers)
    if isinstance(key, tuple) and key[0] == "concat_top":
        sel = list(key[2])
        return stack_concat(data, sel)
    raise ValueError(f"unknown feature key: {key!r}")


def labels_for_task(data: dict, task: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (kept_indices_into_features, binary_labels)."""
    if task == "face_num":
        return binary_labels_face_num(data["label_face_num"])
    label_col = DEFAULT_TASKS.get(task)
    if label_col is None:
        return np.array([]), np.array([])
    key = f"label_{label_col}"
    if key not in data:
        return np.array([]), np.array([])
    return binary_labels_from_continuous(data[key])


def run_within_trial(
    trial_data: dict,
    layers: list[int],
    task: str,
    layer_keys: list,
    seed: int = 42,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    kept, y = labels_for_task(trial_data, task)
    if len(kept) < 20:
        return results
    per_layer_scores: dict[int, np.ndarray] = {}
    y_test_shared: np.ndarray | None = None
    for key in layer_keys:
        X = features_for_key(trial_data, layers, key)[kept]
        result, y_test, norm = fit_probe_within_trial_with_scores(
            X, y, task=task, layer=key, seed=seed,
        )
        results.append(result)
        # Cache scores for the late-fusion ensemble. All single-layer fits share
        # the same (labels, seed) stratified split, so their test slices align
        # row-for-row and can be averaged into an ensemble with zero refits.
        if isinstance(key, int) and y_test is not None and norm is not None:
            per_layer_scores[key] = norm
            y_test_shared = y_test
    if per_layer_scores and y_test_shared is not None:
        ensemble = np.mean(
            np.stack(list(per_layer_scores.values()), axis=0), axis=0,
        )
        try:
            lf_auc = float(roc_auc_score(y_test_shared, ensemble))
        except ValueError:
            lf_auc = float("nan")
        n_test = len(y_test_shared)
        n_train = max(0, len(y) - n_test)
        results.append(ProbeResult(
            task, "late_fusion", "within_trial", lf_auc, n_train, n_test, seed,
        ))
    else:
        results.append(ProbeResult(
            task, "late_fusion", "within_trial", float("nan"), 0, 0, seed,
        ))
    return results


def collect_trial_paths(features_dir: Path) -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    for path in sorted(features_dir.glob("sub_*_trial*.npz")):
        m = re.match(r"sub_(\d+)_trial(\d+)\.npz", path.name)
        if not m:
            continue
        sid, tid = int(m.group(1)), int(m.group(2))
        out[(sid, tid)] = str(path)
    return out


def _pass1_worker(
    sid: int, tid: int, npz_path: str, task: str,
    layers: list[int], layer_keys: list, seed: int,
) -> tuple[int, int, str, list[ProbeResult], float]:
    t0 = time.time()
    data = _load_trial_in_worker(npz_path)
    results = run_within_trial(data, layers, task, layer_keys, seed=seed)
    return sid, tid, task, results, time.time() - t0


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[probe] indexing features from {args.features_dir}", flush=True)
    trial_paths = collect_trial_paths(args.features_dir)
    if not trial_paths:
        raise SystemExit(f"no NPZ files found in {args.features_dir}")
    print(f"[probe] indexed {len(trial_paths)} trials", flush=True)

    # Peek one NPZ to learn the layer list (assumed uniform across trials).
    peek_path = next(iter(trial_paths.values()))
    with np.load(peek_path, allow_pickle=False) as peek:
        layers = list(int(x) for x in peek["layers"])
    print(f"[probe] hooked layers = {layers}", flush=True)

    base_layer_keys: list = list(layers) + ["concat_all", "mean_all"]
    # `face_num` is already in DEFAULT_TASKS; don't double-add (was costing
    # ~200s of duplicate fits per trial and writing duplicate CSV rows).
    tasks = list(DEFAULT_TASKS.keys())
    if args.tasks:
        tasks = [t for t in tasks if t in args.tasks]

    rows: list[dict] = []

    # === WITHIN-TRIAL ===
    # Pass NPZ paths through the joblib closure, not the materialized feature
    # dicts. With ~700 MB per trial × 12 trials, the previous closure was
    # pickling ~8 GB of features into every loky worker on startup; now each
    # worker mmaps only the trial(s) it gets assigned, cached single-slot.
    n_jobs = _n_jobs()
    units = [
        (sid, tid, path, task)
        for (sid, tid), path in trial_paths.items()
        for task in tasks
    ]
    print(
        f"[probe] PASS 1: WITHIN-TRIAL ({len(units)} trial×task units, "
        f"n_jobs={n_jobs})",
        flush=True,
    )

    pass1 = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_pass1_worker)(sid, tid, path, task, layers, base_layer_keys, args.seed)
        for (sid, tid, path, task) in units
    )
    assert pass1 is not None
    for sid, tid, task, results, elapsed in pass1:
        print(
            f"[probe]   sub_{sid} trial{tid} task={task} "
            f"({len(results)} fits, {elapsed:.1f}s)",
            flush=True,
        )
        for r in results:
            rows.append({
                "subject_id": sid, "trial_id": tid,
                "task": r.task, "layer": str(r.layer), "split": r.split,
                "auroc": r.auroc, "n_train": r.n_train, "n_test": r.n_test,
                "seed": r.seed,
            })

    # Downstream passes (top-3 concat, cross-session, cross-subject) still
    # operate on the in-memory feature dicts. PASS 1 has finished, so loading
    # now no longer pickles through any joblib closures.
    print(
        f"[probe] materializing {len(trial_paths)} trial feature dicts "
        f"for downstream passes",
        flush=True,
    )
    trial_features: dict[tuple[int, int], dict] = {
        key: dict(np.load(path, allow_pickle=False))
        for key, path in trial_paths.items()
    }

    # Pool per-task per-layer within-trial AUROCs across trials -> pick top-3
    print(f"[probe] PASS 2: deriving top-3 layers per task and re-probing", flush=True)
    pooled_within: dict[str, list[ProbeResult]] = {}
    for row in rows:
        if row["split"] != "within_trial":
            continue
        if row["layer"] in ("concat_all", "mean_all", "late_fusion"):
            continue
        try:
            layer_int = int(row["layer"])
        except (TypeError, ValueError):
            continue
        pooled_within.setdefault(row["task"], []).append(
            ProbeResult(
                row["task"], layer_int, row["split"], row["auroc"],
                row["n_train"], row["n_test"], row["seed"],
            )
        )
    topk_per_task: dict[str, list[int]] = {}
    for task in tasks:
        recs = pooled_within.get(task, [])
        # Mean per-layer AUROC across trials, then top-3
        per_layer_mean: dict[int, list[float]] = {}
        for r in recs:
            if isinstance(r.layer, int) and np.isfinite(r.auroc):
                per_layer_mean.setdefault(r.layer, []).append(r.auroc)
        mean_results = [
            ProbeResult(task, L, "within_trial", float(np.mean(vals)),
                        sum(1 for _ in vals), 0, args.seed)
            for L, vals in per_layer_mean.items()
        ]
        topk_per_task[task] = topk_layers_by_within_auroc(mean_results, k=3)
    print(f"[probe] top-3 layers per task: {topk_per_task}", flush=True)

    # Run concat-top-3 sister per (trial, task)
    for (sid, tid), data in trial_features.items():
        for task in tasks:
            top3 = topk_per_task.get(task, [])
            if len(top3) < 1:
                continue
            kept, y = labels_for_task(data, task)
            if len(kept) < 20:
                continue
            key = ("concat_top", 3, tuple(top3))
            X = features_for_key(data, layers, key)[kept]
            r = fit_probe_within_trial(X, y, task=task,
                                       layer=f"concat_top3_{'_'.join(map(str, top3))}",
                                       seed=args.seed)
            rows.append({
                "subject_id": sid, "trial_id": tid,
                "task": r.task, "layer": str(r.layer), "split": r.split,
                "auroc": r.auroc, "n_train": r.n_train, "n_test": r.n_test,
                "seed": r.seed,
            })

    # Update layer_keys to include the concat_top_3 sister for downstream splits
    layer_keys: list = list(base_layer_keys)
    layer_keys_per_task: dict[str, list] = {
        task: list(base_layer_keys) + [("concat_top", 3, tuple(topk_per_task.get(task, [])))]
        for task in tasks
    }

    # === CROSS-SESSION (same subject, different trial pairs) ===
    subjects = sorted({s for (s, _) in trial_features})
    css_units = [
        (sid, ta, tb)
        for sid in subjects
        for trials in [sorted([t for (s, t) in trial_features if s == sid])]
        if len(trials) >= 2
        for (ta, tb) in [(trials[0], trials[1]), (trials[1], trials[0])]
    ]
    print(f"[probe] CROSS-SESSION ({len(css_units)} subject×direction)", flush=True)
    for sid in subjects:
        trials = sorted([t for (s, t) in trial_features if s == sid])
        if len(trials) < 2:
            continue
        for ta, tb in [(trials[0], trials[1]), (trials[1], trials[0])]:
            t_css = time.time()
            data_a = trial_features[(sid, ta)]
            data_b = trial_features[(sid, tb)]
            for task in tasks:
                kept_a, y_a = labels_for_task(data_a, task)
                kept_b, y_b = labels_for_task(data_b, task)
                if len(kept_a) < 20 or len(kept_b) < 20:
                    continue
                for key in layer_keys_per_task.get(task, layer_keys):
                    X_a = features_for_key(data_a, layers, key)[kept_a]
                    X_b = features_for_key(data_b, layers, key)[kept_b]
                    if isinstance(key, tuple) and key[0] == "concat_top":
                        key_label = f"concat_top3_{'_'.join(map(str, key[2]))}"
                    else:
                        key_label = str(key)
                    r = fit_probe_cross_session(
                        features_by_trial={ta: X_a, tb: X_b},
                        labels_by_trial={ta: y_a, tb: y_b},
                        train_trial=ta, test_trial=tb,
                        task=task, layer=key_label, seed=args.seed,
                    )
                    rows.append({
                        "subject_id": sid,
                        "trial_id": -1,  # cross-session bridges trials
                        "train_trial": ta, "test_trial": tb,
                        "task": r.task, "layer": key_label, "split": r.split,
                        "auroc": r.auroc, "n_train": r.n_train, "n_test": r.n_test,
                        "seed": r.seed,
                    })
            print(
                f"[probe]   sub_{sid} {ta}->{tb} cross-session "
                f"({time.time() - t_css:.1f}s)",
                flush=True,
            )

    # === CROSS-SUBJECT (LOSO) ===
    print(
        f"[probe] CROSS-SUBJECT probes (LOSO) — "
        f"{len(subjects)} subjects × {len(tasks)} tasks",
        flush=True,
    )
    features_by_subject: dict = {sid: {} for sid in subjects}
    labels_by_subject: dict = {sid: {} for sid in subjects}
    for task in tasks:
        t_cs = time.time()
        keys_for_task = layer_keys_per_task.get(task, layer_keys)
        for sid in subjects:
            X_layers: dict = {repr(k): [] for k in keys_for_task}
            y_concat = []
            for (s, t), data in trial_features.items():
                if s != sid:
                    continue
                kept, y = labels_for_task(data, task)
                if len(kept) < 5:
                    continue
                y_concat.append(y)
                for key in keys_for_task:
                    X_layers[repr(key)].append(features_for_key(data, layers, key)[kept])
            if not y_concat:
                continue
            features_by_subject[sid][task] = {
                repr(k): np.concatenate(X_layers[repr(k)], axis=0) for k in keys_for_task
            }
            labels_by_subject[sid][task] = np.concatenate(y_concat)

        for held_out in subjects:
            if task not in features_by_subject[held_out]:
                continue
            for key in keys_for_task:
                key_repr = repr(key)
                feats_by_subj = {
                    s: features_by_subject[s][task][key_repr] for s in subjects
                    if task in features_by_subject[s]
                }
                labs_by_subj = {
                    s: labels_by_subject[s][task] for s in subjects
                    if task in labels_by_subject[s]
                }
                if isinstance(key, tuple) and key[0] == "concat_top":
                    key_label = f"concat_top3_{'_'.join(map(str, key[2]))}"
                else:
                    key_label = str(key)
                r = fit_probe_cross_subject(
                    feats_by_subj, labs_by_subj,
                    held_out_subject=held_out,
                    task=task, layer=key_label, seed=args.seed,
                )
                rows.append({
                    "subject_id": held_out, "trial_id": -2,
                    "task": r.task, "layer": key_label, "split": r.split,
                    "auroc": r.auroc, "n_train": r.n_train, "n_test": r.n_test,
                    "seed": r.seed,
                })
        print(
            f"[probe]   task={task} LOSO done ({time.time() - t_cs:.1f}s)",
            flush=True,
        )

    # Write CSV
    out_csv = args.out_dir / "ceiling_results.csv"
    fieldnames = sorted({k for row in rows for k in row})
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[probe] wrote {len(rows)} rows -> {out_csv}", flush=True)

    summary = {
        "n_trials": len(trial_features),
        "n_subjects": len(subjects),
        "tasks": tasks,
        "layers": layer_keys,
        "n_rows": len(rows),
    }
    (args.out_dir / "ceiling_summary.json").write_text(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tasks", type=str, nargs="*", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
