"""Phase-2 sweep: contiguous-band merge centered on each task's top layer.

Phase 1 (`run_probe_ceiling.py`) identifies the best single layer per task.
Phase 2 takes that winner and probes contiguous-band merges (e.g., L8 winner →
test merge of L[6..10], L[5..11], L[4..12]) to see whether nearby-layer pooling
beats the single-best layer. Adjacent transformer layers tend to encode similar
features; a contiguous-band concat captures that without the noise of
distant layers.

For each task and each band-width in {3, 5, 7, 9}:
    - concat features over [center - half, center + half], clipped to [0, 31]
    - probe within_trial / cross_session / cross_subject

Output: separate CSV `band_sweep_results.csv` to keep clean of Phase 1 results.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Use upstream-style relative imports through the probe module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from speech_decoding.whisper_ceiling.probe import (  # noqa: E402
    DEFAULT_TASKS,
    ProbeResult,
    binary_labels_face_num,
    binary_labels_from_continuous,
    fit_probe_cross_session,
    fit_probe_cross_subject,
    fit_probe_within_trial,
)


BAND_WIDTHS = (3, 5, 7, 9)


def load_features(features_dir: Path) -> dict[tuple[int, int], dict]:
    out = {}
    for path in sorted(features_dir.glob("sub_*_trial*.npz")):
        m = re.match(r"sub_(\d+)_trial(\d+)\.npz", path.name)
        if not m:
            continue
        sid, tid = int(m.group(1)), int(m.group(2))
        out[(sid, tid)] = dict(np.load(path, allow_pickle=False))
    return out


def top_layer_per_task(phase1_csv: Path) -> dict[str, int]:
    """Read Phase-1 CSV, return per-task argmax layer over mean within-trial AUROC."""
    per_task: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(phase1_csv) as fh:
        for row in csv.DictReader(fh):
            if row.get("split") != "within_trial":
                continue
            try:
                layer = int(row["layer"])
            except (ValueError, TypeError):
                continue
            try:
                auroc = float(row["auroc"])
            except (ValueError, TypeError):
                continue
            if not np.isfinite(auroc):
                continue
            per_task[row["task"]][layer].append(auroc)
    out: dict[str, int] = {}
    for task, layer_aurocs in per_task.items():
        means = {L: float(np.mean(vals)) for L, vals in layer_aurocs.items()}
        out[task] = max(means.items(), key=lambda kv: kv[1])[0]
    return out


def band_layers(center: int, width: int, n_layers: int) -> list[int]:
    half = width // 2
    lo = max(0, center - half)
    hi = min(n_layers - 1, center + half)
    return list(range(lo, hi + 1))


def labels_for_task(data: dict, task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "face_num":
        return binary_labels_face_num(data["label_face_num"])
    label_col = DEFAULT_TASKS.get(task)
    if label_col is None:
        return np.array([]), np.array([])
    key = f"label_{label_col}"
    if key not in data:
        return np.array([]), np.array([])
    return binary_labels_from_continuous(data[key])


def concat_band(data: dict, layers: list[int], kept: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [data[f"layer_{L}"][kept].astype(np.float32) for L in layers], axis=1
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[band] reading Phase-1 winners from {args.phase1_csv}", flush=True)
    centers = top_layer_per_task(args.phase1_csv)
    print(f"[band] per-task winning center layer: {centers}", flush=True)

    print(f"[band] loading features from {args.features_dir}", flush=True)
    trial_features = load_features(args.features_dir)
    if not trial_features:
        sys.exit(f"no NPZ files found in {args.features_dir}")
    n_layers = len(next(iter(trial_features.values()))["layers"])
    subjects = sorted({s for (s, _) in trial_features})

    tasks = list(centers.keys())
    if args.tasks:
        tasks = [t for t in tasks if t in args.tasks]

    rows: list[dict] = []

    for task in tasks:
        center = centers[task]
        for width in BAND_WIDTHS:
            band = band_layers(center, width, n_layers)
            label = f"band_w{width}_c{center}_L{'-'.join(map(str, band))}"
            print(f"[band] task={task} center={center} width={width} band={band}", flush=True)

            # within_trial
            for (sid, tid), data in trial_features.items():
                kept, y = labels_for_task(data, task)
                if len(kept) < 20:
                    continue
                X = concat_band(data, band, kept)
                r = fit_probe_within_trial(X, y, task=task, layer=label, seed=args.seed)
                rows.append(_row(sid, tid, r))

            # cross_session
            for sid in subjects:
                trials = sorted([t for (s, t) in trial_features if s == sid])
                if len(trials) < 2:
                    continue
                for ta, tb in [(trials[0], trials[1]), (trials[1], trials[0])]:
                    da, db = trial_features[(sid, ta)], trial_features[(sid, tb)]
                    ka, ya = labels_for_task(da, task)
                    kb, yb = labels_for_task(db, task)
                    if len(ka) < 20 or len(kb) < 20:
                        continue
                    Xa = concat_band(da, band, ka)
                    Xb = concat_band(db, band, kb)
                    r = fit_probe_cross_session(
                        features_by_trial={ta: Xa, tb: Xb},
                        labels_by_trial={ta: ya, tb: yb},
                        train_trial=ta, test_trial=tb,
                        task=task, layer=label, seed=args.seed,
                    )
                    rows.append(_row(sid, -1, r, train_trial=ta, test_trial=tb))

            # cross_subject (LOSO)
            feats_by_subj: dict[int, np.ndarray] = {}
            labs_by_subj: dict[int, np.ndarray] = {}
            for sid in subjects:
                Xs, ys = [], []
                for (s, t), data in trial_features.items():
                    if s != sid:
                        continue
                    kept, y = labels_for_task(data, task)
                    if len(kept) < 5:
                        continue
                    Xs.append(concat_band(data, band, kept))
                    ys.append(y)
                if not ys:
                    continue
                feats_by_subj[sid] = np.concatenate(Xs, axis=0)
                labs_by_subj[sid] = np.concatenate(ys)
            for held_out in subjects:
                if held_out not in feats_by_subj:
                    continue
                r = fit_probe_cross_subject(
                    feats_by_subj, labs_by_subj,
                    held_out_subject=held_out,
                    task=task, layer=label, seed=args.seed,
                )
                rows.append(_row(held_out, -2, r))

    # Write band-sweep CSV
    out_csv = args.out_dir / "band_sweep_results.csv"
    fieldnames = sorted({k for row in rows for k in row})
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[band] wrote {len(rows)} rows -> {out_csv}", flush=True)

    summary = {
        "n_tasks": len(tasks),
        "centers": centers,
        "band_widths": list(BAND_WIDTHS),
        "n_rows": len(rows),
    }
    (args.out_dir / "band_sweep_summary.json").write_text(json.dumps(summary, indent=2))


def _row(sid: int, tid: int, r: ProbeResult, train_trial=None, test_trial=None) -> dict:
    return {
        "subject_id": sid, "trial_id": tid,
        "train_trial": train_trial, "test_trial": test_trial,
        "task": r.task, "layer": str(r.layer), "split": r.split,
        "auroc": r.auroc, "n_train": r.n_train, "n_test": r.n_test,
        "seed": r.seed,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase1-csv", type=Path, required=True,
                   help="ceiling_results.csv from run_probe_ceiling.py")
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tasks", type=str, nargs="*", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
