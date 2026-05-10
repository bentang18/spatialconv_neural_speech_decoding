"""V0.x — CrossSession stimulus-overlap audit.

Per (subject, task), the Neuroprobe CrossSession protocol uses one trial as
test and one or more other trials as train. If the same word/exemplar appears
in both train and test sets, the linear classifier can pattern-match stimulus
identity rather than brain response — making the AUROC partially a stimulus
recognition score rather than a brain-decoding score.

Two modes:
  * ``--mode upper-bound``: laptop-runnable. Computes overlap fraction over
    *all* unique words in each trial's words_df, ignoring task-specific label
    selection. Upper bound on per-task overlap, fast, no BT data required —
    only the upstream package's bundled
    ``braintreebank_features_time_alignment/`` CSVs.
  * ``--mode per-task``: DCC-only. Instantiates the upstream
    ``BrainTreebankSubjectTrialBenchmarkDataset`` per (subject, trial, task)
    and computes overlap on the actual rows the dataset selects (label
    filter + ``rng.choice`` cap). Requires BT data + neural-data caches.

Output: ``stimulus_overlap.csv`` + ``README.md`` summary.

Usage:
    # laptop, upper-bound
    .venv/bin/python scripts/neuroprobe/audit_stimulus_overlap_cross_session.py \\
        --mode upper-bound \\
        --neuroprobe-repo .cache/neuroprobe_upstream \\
        --out-dir reports/neuroprobe_stage0_v0x_stimulus_overlap_$(date +%Y_%m_%d)

    # DCC, per-task
    .venv/bin/python scripts/neuroprobe/audit_stimulus_overlap_cross_session.py \\
        --mode per-task \\
        --bt-root /work/ht203/data/braintreebank \\
        --neuroprobe-repo /work/ht203/repo/neuroprobe_upstream \\
        --out-dir reports/neuroprobe_stage0_v0x_stimulus_overlap_$(date +%Y_%m_%d)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from speech_decoding.studies.braintreebank.labels import NEUROPROBE_TASKS
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS


WORD_COLUMN = "full_word"


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = args.neuroprobe_repo.resolve()
    df_dir = repo / "neuroprobe" / "braintreebank_features_time_alignment"
    if not df_dir.exists():
        raise SystemExit(f"Expected words_df dir at {df_dir}")

    sessions_by_subject: dict[int, list[int]] = defaultdict(list)
    for sub_id, trial_id in BT_LITE_SESSIONS:
        sessions_by_subject[sub_id].append(trial_id)

    rows: list[dict] = []
    if args.mode == "upper-bound":
        rows = _audit_upper_bound(df_dir, sessions_by_subject)
    elif args.mode == "per-task":
        rows = _audit_per_task(args, sessions_by_subject)
    else:
        raise SystemExit(f"unknown mode {args.mode!r}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "stimulus_overlap.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(df)} rows)")

    _write_readme(out_dir, df, args)


def _audit_upper_bound(
    df_dir: Path, sessions_by_subject: dict[int, list[int]]
) -> list[dict]:
    """Per (subject, test_trial), overlap over all unique words in train/test."""
    rows: list[dict] = []
    for sub_id, trials in sessions_by_subject.items():
        if len(trials) < 2:
            continue
        for test_trial in trials:
            train_trial = [t for t in trials if t != test_trial][0]
            train_words = _load_words(df_dir, sub_id, train_trial)
            test_words = _load_words(df_dir, sub_id, test_trial)
            rows.append(_overlap_row(
                subject_id=sub_id,
                train_trial=train_trial,
                test_trial=test_trial,
                task="ALL_WORDS_UPPER_BOUND",
                train_words=train_words,
                test_words=test_words,
            ))
    return rows


def _audit_per_task(
    args: argparse.Namespace, sessions_by_subject: dict[int, list[int]]
) -> list[dict]:
    """Per (subject, test_trial, task), overlap over the dataset's selected rows.

    Requires BT data + the upstream dataset machinery. We instantiate the
    train + test datasets via the same generate_splits_cross_session call
    as the L wrapper, then map each item back to its underlying ``full_word``
    via the dataset's ``label_indices`` and ``all_words_df``.
    """
    bt_root = args.bt_root
    if bt_root is None:
        raise SystemExit("--bt-root required for per-task mode")
    repo_dir = args.neuroprobe_repo.resolve()
    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "examples"))
    os.environ["ROOT_DIR_BRAINTREEBANK"] = str(bt_root)

    import torch  # noqa: PLC0415

    from neuroprobe.braintreebank_subject import BrainTreebankSubject  # noqa: PLC0415
    import neuroprobe.train_test_splits as nts  # noqa: PLC0415
    import neuroprobe.config as nconfig  # noqa: PLC0415

    rows: list[dict] = []
    for sub_id, trials in sessions_by_subject.items():
        if len(trials) < 2:
            continue
        subject = BrainTreebankSubject(sub_id, cache=True, dtype=torch.float32)
        subject.set_electrode_subset(
            nconfig.NEUROPROBE_LITE_ELECTRODES[subject.subject_identifier]
        )
        for test_trial in trials:
            subject.load_neural_data(test_trial)
            for task in NEUROPROBE_TASKS:
                try:
                    folds = nts.generate_splits_cross_session(
                        subject, test_trial, task, dtype=torch.float32,
                        output_indices=False, output_dict=False,
                        lite=True, binary_tasks=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[skip] sub{sub_id} trial{test_trial} task={task}: {exc}")
                    continue
                fold = folds[0]
                train_words = _selected_words(fold["train_dataset"], task)
                test_words = _selected_words(fold["test_dataset"], task)
                train_trial = _trial_id_of(fold["train_dataset"])
                rows.append(_overlap_row(
                    subject_id=sub_id,
                    train_trial=train_trial,
                    test_trial=test_trial,
                    task=task,
                    train_words=train_words,
                    test_words=test_words,
                ))
                print(
                    f"sub{sub_id} test_trial={test_trial} task={task} "
                    f"overlap={rows[-1]['overlap_fraction']:.3f}"
                )
    return rows


def _selected_words(dataset, task: str) -> list[str]:
    """Pull each item's underlying full_word via label_indices + all_words_df.

    `dataset` is a `Subset` wrapper around `BrainTreebankSubjectTrialBenchmarkDataset`
    (CrossSession test/val) OR the raw dataset (CrossSession train). We pull
    the underlying dataset and the indices it selected.
    """
    base = getattr(dataset, "dataset", dataset)
    indices = getattr(dataset, "indices", range(len(base)))
    label_indices: dict[int, np.ndarray] = base.label_indices
    n_classes = base.n_classes
    words_df = base.all_words_df
    nonverbal_df = getattr(base, "nonverbal_df", None)
    out: list[str] = []
    for ds_idx in indices:
        current_label = (ds_idx + 1) % n_classes
        word_index = int(label_indices[current_label][ds_idx // n_classes])
        if task in ("onset", "speech") and current_label == 0 and nonverbal_df is not None:
            row = nonverbal_df.iloc[word_index]
            out.append(f"NONVERBAL_{int(row.get('original_index', word_index))}")
        else:
            row = words_df.iloc[word_index]
            text = row.get(WORD_COLUMN, None)
            if text is None or (isinstance(text, float) and np.isnan(text)):
                text = f"WORDIDX_{int(row.get('original_index', word_index))}"
            out.append(str(text).lower().strip())
    return out


def _trial_id_of(dataset) -> int:
    base = getattr(dataset, "dataset", dataset)
    return int(base.trial_id)


def _load_words(df_dir: Path, sub_id: int, trial_id: int) -> list[str]:
    path = df_dir / f"subject{sub_id}_trial{trial_id}_words_df.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if WORD_COLUMN not in df.columns:
        raise SystemExit(
            f"expected column {WORD_COLUMN!r} in {path}; got {list(df.columns)}"
        )
    return df[WORD_COLUMN].astype(str).str.lower().str.strip().tolist()


def _overlap_row(
    *,
    subject_id: int,
    train_trial: int,
    test_trial: int,
    task: str,
    train_words: list[str],
    test_words: list[str],
) -> dict:
    train_set = set(train_words)
    test_set = set(test_words)
    if not test_set:
        overlap_frac = 0.0
    else:
        overlap_frac = len(test_set & train_set) / len(test_set)
    return {
        "subject_id": subject_id,
        "train_trial": train_trial,
        "test_trial": test_trial,
        "task": task,
        "n_train_unique": len(train_set),
        "n_test_unique": len(test_set),
        "n_overlap_unique": len(test_set & train_set),
        "overlap_fraction": overlap_frac,
        "n_train_total": len(train_words),
        "n_test_total": len(test_words),
    }


def _write_readme(out_dir: Path, df: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        f"# V0.x — CrossSession stimulus-overlap audit ({args.mode})",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"Sessions: {len(BT_LITE_SESSIONS)} BT Lite (subject, trial) pairs.",
        f"Mode: `{args.mode}` "
        + (
            "(unique-word overlap across full per-trial words_df; upper bound)"
            if args.mode == "upper-bound"
            else "(per-task; uses dataset label_indices selection)"
        ),
        "",
        "## Per-task summary (median across subjects)",
        "",
    ]
    if not df.empty:
        agg = (
            df.groupby("task")["overlap_fraction"]
            .agg(["median", "min", "max", "count"])
            .reset_index()
            .sort_values("median", ascending=False)
        )
        lines.append("| task | median | min | max | n_pairs |")
        lines.append("|---|---|---|---|---|")
        for _, r in agg.iterrows():
            lines.append(
                f"| {r['task']} | {r['median']:.3f} | {r['min']:.3f} | "
                f"{r['max']:.3f} | {int(r['count'])} |"
            )
        lines += ["", "## Flagged (overlap > 0.50)", ""]
        flagged = df[df["overlap_fraction"] > 0.50]
        if flagged.empty:
            lines.append("No (subject, task) pairs exceed 50% overlap.")
        else:
            lines.append("| subject | test_trial | task | overlap | n_test |")
            lines.append("|---|---|---|---|---|")
            for _, r in flagged.iterrows():
                lines.append(
                    f"| {int(r['subject_id'])} | {int(r['test_trial'])} | "
                    f"{r['task']} | {r['overlap_fraction']:.3f} | "
                    f"{int(r['n_test_unique'])} |"
                )
    else:
        lines.append("(no rows)")
    lines += [
        "",
        "## How to read",
        "",
        "`overlap_fraction = |unique(test_words) ∩ unique(train_words)| / |unique(test_words)|`. "
        "If 1.0, every test word also appears in train and the linear classifier can match "
        "stimulus identity rather than brain response. CrossSession protocol shuffles train/test "
        "by session, not by stimulus; movie repeats can re-expose words across sessions.",
        "",
        "**v14 contract**: tasks with median overlap > 0.50 must be flagged in stage_0.md V6 "
        "as stimulus-recognition-confounded. L.2/L.3 winners must report numbers separately "
        "for flagged vs unflagged tasks.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))
    print(f"wrote {out_dir / 'README.md'}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=("upper-bound", "per-task"), required=True,
        help="upper-bound: laptop-runnable from words_df CSVs only. "
             "per-task: DCC; uses dataset label_indices selection.",
    )
    p.add_argument(
        "--neuroprobe-repo", type=Path,
        default=Path(".cache/neuroprobe_upstream"),
    )
    p.add_argument(
        "--bt-root", type=Path, default=None,
        help="BT data root; required for --mode=per-task.",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path(
            f"reports/neuroprobe_stage0_v0x_stimulus_overlap_{datetime.now():%Y_%m_%d}"
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
