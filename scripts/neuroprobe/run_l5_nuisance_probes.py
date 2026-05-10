# pyright: reportMissingImports=false
"""L.5.P2 within-subject session-id nuisance probe on the L.2 winner view.

Per-subject probe: for each subject with ≥ 2 sessions in BT Lite, fit a linear
classifier on word-level features from the L.2 winner view (R4xI2 + N1 =
shaft_laplacian + stft_abs + train_set_fixed) to decode session-id from
features. Macro AUROC across subjects + per-subject kill verdict.

Why P2 only (P1 dropped):
  P1 (subject-id) is trivially decodable from the raw view — different subjects
  have different channel sets so feature width itself reveals identity. The
  meaningful test is whether the L.2 winner *within a subject* drifts across
  sessions in a way a linear classifier can exploit; that is the leak v14 must
  be invariant to.

Per `docs/neuroprobe/stage_0.md` L.5 spec — kill the view if any subject's
held-out P2 macro AUROC > 0.95.

Output:
  metrics.json           — per-subject + macro P2 + kill flags
  nuisance_probe_metrics.csv — one row per subject probe with diagnostics
  experiment_record.json — ExperimentLogger sidecar
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from speech_decoding.experiments import ExperimentLogger
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS


UPSTREAM_REPO_URL = "https://github.com/insight-neuro/neuroprobe"
UPSTREAM_COMMIT = "c7b955b0a31464f4a5eec3f3bd78ff29841d61ac"
KILL_AUROC = 0.95


def main() -> None:
    args = _parse_args()
    bt_root = args.bt_root.resolve()
    repo_dir = args.neuroprobe_repo.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "examples"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.environ["ROOT_DIR_BRAINTREEBANK"] = str(bt_root)

    from neuroprobe.braintreebank_subject import BrainTreebankSubject  # noqa: E402
    import neuroprobe.config as nconfig  # noqa: E402
    from eval_utils import (  # noqa: E402
        laplacian_rereference_neural_data,
        preprocess_stft,
        subset_electrodes,
    )
    from preprocess_views import make_upstream_helpers, preprocess_views  # noqa: E402

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sessions = _parse_sessions(args.sessions) or list(BT_LITE_SESSIONS)
    sr = int(nconfig.SAMPLING_RATE)
    win_samples = int(args.window_seconds * sr)

    stft_params = {
        "type": "laplacian-stft_abs",
        "stft": {
            "nperseg": 512,
            "poverlap": 0.75,
            "window": "hann",
            "max_frequency": 150,
            "min_frequency": 0,
        },
        "projection": {"dim": 192, "method": "pca"},
    }
    upstream_helpers = make_upstream_helpers(
        preprocess_stft=preprocess_stft,
        laplacian_rereference_neural_data=laplacian_rereference_neural_data,
        stft_params=stft_params,
    )

    config = {
        "upstream_repo_url": UPSTREAM_REPO_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "bt_root": str(bt_root),
        "ref_kind": args.ref_kind,
        "view_kind": args.view_kind,
        "normalization": "train_set_fixed",
        "cell_id": "L.5.P2",
        "window_seconds": args.window_seconds,
        "sessions": [list(s) for s in sessions],
        "seed": args.seed,
        "kill_auroc": KILL_AUROC,
    }

    with ExperimentLogger(
        artifact_dir=out_dir,
        block="L",
        cell="L.5.P2",
        run_kind="stage0_nuisance_probes",
        eval_mode="binary",
        split_mode="random_80_20_per_subject",
        subject_id="per_subject",
        trial_id="per_subject",
        task="session_id_within_subject",
        seed=str(args.seed),
        config_json=config,
        report_dir=str(out_dir),
    ) as logger:
        feats_by_subject: dict[int, list[np.ndarray]] = defaultdict(list)
        labels_by_subject: dict[int, list[int]] = defaultdict(list)

        for s_idx, (sub_id, trial_id) in enumerate(sessions):
            t_session = time.time()
            words_df_path = (
                repo_dir / "neuroprobe" / "braintreebank_features_time_alignment"
                / f"subject{sub_id}_trial{trial_id}_words_df.csv"
            )
            if not words_df_path.exists():
                raise FileNotFoundError(f"Missing words_df: {words_df_path}")
            words_df = pd.read_csv(words_df_path)
            words_df = words_df[words_df["est_idx"].notna()].copy()

            subject = BrainTreebankSubject(sub_id, cache=True, dtype=torch.float32)
            subset_electrodes(subject, lite=True, nano=False)
            subject.load_neural_data(trial_id)
            cache = subject.neural_data_cache[trial_id]  # (C, T)
            n_samples = cache.shape[1]

            session_feats: list[np.ndarray] = []
            n_kept = 0
            for est_idx in words_df["est_idx"].astype(int):
                lo = est_idx
                hi = lo + win_samples
                if lo < 0 or hi > n_samples:
                    continue
                window = cache[:, lo:hi].unsqueeze(0)  # (1, C, T)
                feats, _ = preprocess_views(
                    window, list(subject.electrode_labels),
                    ref_kind=args.ref_kind, view_kind=args.view_kind,
                    sampling_rate=sr, upstream_helpers=upstream_helpers,
                )
                session_feats.append(feats.float().numpy().reshape(1, -1))
                n_kept += 1

            X_session = np.concatenate(session_feats, axis=0)
            del session_feats
            feats_by_subject[sub_id].append(X_session)
            labels_by_subject[sub_id].extend([trial_id] * n_kept)
            print(
                f"[{s_idx + 1}/{len(sessions)}] sub{sub_id} trial{trial_id}: "
                f"{n_kept}/{len(words_df)} words, F={X_session.shape[1]}, "
                f"{time.time() - t_session:.1f}s"
            )
            subject.clear_neural_data_cache(trial_id)
            gc.collect()

        rows: list[dict[str, float | int | str]] = []
        for sub_id in sorted(feats_by_subject):
            X_subj = np.concatenate(feats_by_subject[sub_id], axis=0)
            y_subj = np.array(labels_by_subject[sub_id])
            n_classes = int(len(np.unique(y_subj)))
            if n_classes < 2:
                print(f"[skip sub{sub_id}] only {n_classes} session(s) — skipping P2.")
                continue
            row = run_probe(
                f"L.5.P2.sub{sub_id}", "session_id", X_subj, y_subj, seed=args.seed,
            )
            row["subject_id"] = sub_id
            rows.append(row)
            del X_subj
            gc.collect()
            print(json.dumps(row, indent=2, sort_keys=True))

        if not rows:
            raise RuntimeError(
                "No subjects with ≥ 2 sessions found — P2 probe needs at least one "
                "subject with multiple sessions."
            )

        diagnostics = pd.DataFrame(rows)
        diagnostics.to_csv(out_dir / "nuisance_probe_metrics.csv", index=False)

        per_subject_aurocs = [float(r["test_macro_auroc"]) for r in rows]
        per_subject_balaccs = [float(r["test_balanced_accuracy"]) for r in rows]
        kill_subjects = [
            int(r["subject_id"])
            for r, a in zip(rows, per_subject_aurocs)
            if a > KILL_AUROC
        ]
        summary = {
            "p2_per_subject": [
                {
                    "subject_id": int(r["subject_id"]),
                    "n_classes": int(r["n_classes"]),
                    "n_train": int(r["n_train"]),
                    "n_test": int(r["n_test"]),
                    "n_features": int(r["n_features"]),
                    "auroc": float(r["test_macro_auroc"]),
                    "balacc": float(r["test_balanced_accuracy"]),
                    "chance": float(r["chance_balanced_accuracy"]),
                    "kill": bool(float(r["test_macro_auroc"]) > KILL_AUROC),
                }
                for r in rows
            ],
            "p2_macro_auroc": float(np.mean(per_subject_aurocs)),
            "p2_macro_balacc": float(np.mean(per_subject_balaccs)),
            "p2_max_auroc": float(np.max(per_subject_aurocs)),
            "p2_min_auroc": float(np.min(per_subject_aurocs)),
            "p2_kill_any": bool(len(kill_subjects) > 0),
            "p2_kill_subjects": kill_subjects,
            "n_subjects_probed": int(len(rows)),
            "kill_auroc_threshold": KILL_AUROC,
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        write_readme(out_dir, summary, args)

        logger.set_metrics(
            summary,
            primary_metric_name="p2_macro_auroc",
            primary_metric_value=summary["p2_macro_auroc"],
        )

        print(json.dumps(summary, indent=2, sort_keys=True))


def run_probe(
    probe_id: str, label_name: str, X: np.ndarray, y: np.ndarray, *, seed: int,
) -> dict[str, float | int | str]:
    n_classes = int(len(np.unique(y)))
    chance = 1.0 / n_classes
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    scaler = StandardScaler(copy=False)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(random_state=seed, max_iter=10000, tol=1e-3)
    clf.fit(X_tr, y_tr)
    test_proba = clf.predict_proba(X_te)
    test_pred = clf.predict(X_te)

    classes = list(clf.classes_)
    if n_classes == 2:
        # Binary: roc_auc_score expects 1-D scores for the positive class.
        pos_idx = classes.index(max(classes))
        test_auroc = float(roc_auc_score(y_te, test_proba[:, pos_idx]))
    else:
        y_te_oh = np.zeros((len(y_te), len(classes)), dtype=np.float32)
        for i, lab in enumerate(y_te):
            y_te_oh[i, classes.index(lab)] = 1.0
        test_auroc = float(
            roc_auc_score(y_te_oh, test_proba, multi_class="ovr", average="macro")
        )
    test_balacc = float(balanced_accuracy_score(y_te, test_pred))

    return {
        "probe_id": probe_id,
        "label": label_name,
        "n_classes": int(n_classes),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_features": int(X.shape[1]),
        "test_balanced_accuracy": test_balacc,
        "test_macro_auroc": test_auroc,
        "chance_balanced_accuracy": float(chance),
    }


def _parse_sessions(spec: str) -> list[tuple[int, int]] | None:
    if not spec:
        return None
    out: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        sub_str, trial_str = chunk.strip().split(":")
        out.append((int(sub_str), int(trial_str)))
    return out


def write_readme(
    out_dir: Path, summary: dict, args: argparse.Namespace,
) -> None:
    lines = [
        "# Stage 0 L.5.P2 — within-subject session-id nuisance probe",
        "",
        f"- ref_kind: `{args.ref_kind}`  view_kind: `{args.view_kind}`",
        f"- normalization: `train_set_fixed` (per-probe StandardScaler refit)",
        f"- per-subject probe (binary: trial-id within subject)",
        f"- N subjects probed: {summary['n_subjects_probed']}",
        "",
        "## Headline",
        "",
        f"- macro AUROC across subjects: **{summary['p2_macro_auroc']:.4f}**",
        f"- max AUROC: {summary['p2_max_auroc']:.4f}, min: {summary['p2_min_auroc']:.4f}",
        f"- kill threshold: {summary['kill_auroc_threshold']:.2f}",
        f"- KILL (any subject > threshold): {summary['p2_kill_any']}",
        f"- subjects exceeding kill threshold: {summary['p2_kill_subjects'] or 'none'}",
        "",
        "## Per-subject",
        "",
        "| subject | n_classes | balacc | chance | macro AUROC | kill |",
        "|---|---|---|---|---|---|",
    ]
    for s in summary["p2_per_subject"]:
        lines.append(
            f"| {s['subject_id']} | {s['n_classes']} | {s['balacc']:.4f} | "
            f"{s['chance']:.4f} | {s['auroc']:.4f} | "
            f"{'YES' if s['kill'] else 'no'} |"
        )
    lines += [
        "",
        "Files:",
        "- `nuisance_probe_metrics.csv` — per-subject probe diagnostics",
        "- `metrics.json` — aggregate summary + per-subject + kill flags",
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
    p.add_argument("--ref-kind", default="shaft_laplacian")
    p.add_argument("--view-kind", default="stft_abs")
    p.add_argument("--window-seconds", type=float, default=1.0,
                   help="Word-anchored window length (Neuroprobe default = 1.0 s).")
    p.add_argument("--sessions", default="",
                   help="Comma-separated subject:trial pairs; default = all 12 BT Lite sessions.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.bt_root is None:
        p.error("--bt-root or ROOT_DIR_BRAINTREEBANK is required")
    return args


if __name__ == "__main__":
    main()
