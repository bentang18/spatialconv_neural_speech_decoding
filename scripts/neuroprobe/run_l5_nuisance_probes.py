# pyright: reportMissingImports=false
"""L.5.P1 + L.5.P2 nuisance probes on the L.2 winner view (R4xI2 + N1).

Decode subject-ID (P1) and session-ID (P2) from word-level features pooled
across all 12 BT Lite sessions, on the L.2 winner cell (shaft_laplacian +
stft_abs + train_set_fixed normalization).

Per `docs/neuroprobe/stage_0.md` L.5 spec — kill criterion: drop view if
held-out AUROC > 0.95. Stratified random 80/20 split (not LOSO) — these
probes ask "can I decode subject from features?" treating subject/session
as multi-class classification, the standard nuisance-probe protocol used by
the V0 QC report (2026-05-01) at the upstream baseline.

Output:
  metrics.json       — held-out balanced acc + macro AUROC + chance for P1, P2
  nuisance_probe_metrics.csv — one row per probe with full diagnostics
  experiment_record.json — ExperimentLogger sidecar
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from speech_decoding.experiments import ExperimentLogger
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS


UPSTREAM_REPO_URL = "https://github.com/insight-neuro/neuroprobe"
UPSTREAM_COMMIT = "c7b955b0a31464f4a5eec3f3bd78ff29841d61ac"


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
        "cell_id_p1": "L.5.P1",
        "cell_id_p2": "L.5.P2",
        "window_seconds": args.window_seconds,
        "sessions": [list(s) for s in sessions],
        "seed": args.seed,
    }

    with ExperimentLogger(
        artifact_dir=out_dir,
        block="L",
        cell="L.5.P1+P2",
        run_kind="stage0_nuisance_probes",
        eval_mode="multiclass",
        split_mode="random_80_20",
        subject_id="pooled",
        trial_id="pooled",
        task="subject_id+session_id",
        seed=str(args.seed),
        config_json=config,
        report_dir=str(out_dir),
    ) as logger:
        feats_per_session: list[np.ndarray] = []
        subject_ids: list[int] = []
        session_ids: list[int] = []

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
            feats_per_session.append(X_session)
            subject_ids.extend([sub_id] * n_kept)
            session_ids.extend([s_idx] * n_kept)
            print(
                f"[{s_idx + 1}/{len(sessions)}] sub{sub_id} trial{trial_id}: "
                f"{n_kept}/{len(words_df)} words, F={X_session.shape[1]}, "
                f"{time.time() - t_session:.1f}s"
            )
            subject.clear_neural_data_cache(trial_id)
            gc.collect()

        X_all = np.concatenate(feats_per_session, axis=0)
        del feats_per_session
        y_subj = np.array(subject_ids)
        y_sess = np.array(session_ids)
        gc.collect()

        print(
            f"\n[pooled] X_all={X_all.shape} subjects={len(np.unique(y_subj))} "
            f"sessions={len(np.unique(y_sess))}"
        )

        rows: list[dict[str, float | int | str]] = []
        for probe_id, label_name, y in (
            ("L.5.P1", "subject_id", y_subj),
            ("L.5.P2", "session_id", y_sess),
        ):
            row = run_probe(probe_id, label_name, X_all, y, seed=args.seed)
            rows.append(row)
            print(json.dumps(row, indent=2, sort_keys=True))

        diagnostics = pd.DataFrame(rows)
        diagnostics.to_csv(out_dir / "nuisance_probe_metrics.csv", index=False)

        p1_auroc = float(rows[0]["test_macro_auroc"])  # type: ignore[arg-type]
        p2_auroc = float(rows[1]["test_macro_auroc"])  # type: ignore[arg-type]
        summary = {
            "p1_subject_id_auroc": p1_auroc,
            "p1_subject_id_balacc": float(rows[0]["test_balanced_accuracy"]),  # type: ignore[arg-type]
            "p1_chance": float(rows[0]["chance_balanced_accuracy"]),  # type: ignore[arg-type]
            "p1_kill": bool(p1_auroc > 0.95),
            "p2_session_id_auroc": p2_auroc,
            "p2_session_id_balacc": float(rows[1]["test_balanced_accuracy"]),  # type: ignore[arg-type]
            "p2_chance": float(rows[1]["chance_balanced_accuracy"]),  # type: ignore[arg-type]
            "p2_kill": bool(p2_auroc > 0.95),
            "n_words_pooled": int(X_all.shape[0]),
            "n_features": int(X_all.shape[1]),
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        write_readme(out_dir, summary, args)

        logger.set_metrics(
            summary,
            primary_metric_name="p1_subject_id_auroc",
            primary_metric_value=summary["p1_subject_id_auroc"],
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
        "# Stage 0 L.5.P1 + L.5.P2 — nuisance probes on L.2 winner view",
        "",
        f"- ref_kind: `{args.ref_kind}`  view_kind: `{args.view_kind}`",
        f"- normalization: `train_set_fixed` (per-probe StandardScaler refit)",
        f"- pooled across {len([1 for _ in (BT_LITE_SESSIONS if not args.sessions else args.sessions.split(','))])} sessions",
        f"- N words pooled: {summary['n_words_pooled']}, F = {summary['n_features']}",
        "",
        "## P1 — subject-id from features",
        "",
        f"- held-out balanced accuracy: {summary['p1_subject_id_balacc']:.4f} (chance {summary['p1_chance']:.4f})",
        f"- held-out macro AUROC: {summary['p1_subject_id_auroc']:.4f}",
        f"- KILL (AUROC > 0.95): {summary['p1_kill']}",
        "",
        "## P2 — session-id from features",
        "",
        f"- held-out balanced accuracy: {summary['p2_session_id_balacc']:.4f} (chance {summary['p2_chance']:.4f})",
        f"- held-out macro AUROC: {summary['p2_session_id_auroc']:.4f}",
        f"- KILL (AUROC > 0.95): {summary['p2_kill']}",
        "",
        "Files:",
        "- `nuisance_probe_metrics.csv` — per-probe diagnostics",
        "- `metrics.json` — aggregate summary + kill flags",
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
