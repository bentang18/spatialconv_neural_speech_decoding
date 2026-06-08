#!/usr/bin/env python
"""Neuroprobe split-policy parity — TIER 2 (DCC, against constructed upstream).

Ben's requirement (2026-06-08): our P4 eval has **100% parity on all counts**
with the Neuroprobe leaderboard, for all three modes — WithinSession,
CrossSession, CrossSubject. Tier 1 (``test_neuroprobe_parity.py``, local) proves
the split *arithmetic* against the real ``sklearn.KFold`` + upstream's verbatim
val/test halving. THIS script closes the remaining gap **end-to-end against the
actual upstream code**: it constructs the real
``BrainTreebankSubjectTrialBenchmarkDataset`` via upstream
``generate_splits_{within_session,cross_session,cross_subject}`` (the exact
leaderboard entry points), enumerates every realized item per split, and asserts
our ``Wang2024Treebank`` + ``BTWordEvents`` chain reproduces, **byte-for-byte and
in order**, the same per-split sequence of:

    (window_from, window_to, label)

— where ``window_from/window_to`` are neural-clock sample indices (the 1-second
leaderboard clip) and ``label`` is the per-task class. Upstream is queried with
``output_indices=True`` so it returns clip windows without reading the h5
voltage (lightweight); it still reads ``transcripts/*/features.csv`` in
``__init__``, so this MUST run on DCC with ``ROOT_DIR_BRAINTREEBANK`` set.

ELECTRODE-SET parity (the Lite-120 subset) is checked when ``--check-electrodes``
is passed AND our chain has electrode subsetting wired; until then it is reported
as ``PENDING`` (the split/window/label parity above is electrode-independent and
verified regardless).

Exit code: 0 iff every (mode, session, task) cell is an EXACT match.

Usage (DCC):
    .venv/bin/python scripts/neuroprobe/verify_parity.py \
        --modes within,csession,csubject --tasks speech
    # all 15 tasks, all Lite sessions:
    .venv/bin/python scripts/neuroprobe/verify_parity.py --tasks all
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# --- upstream import (pinned clone) -----------------------------------------
# DCC: /work/ht203/repo/neuroprobe_upstream ; laptop: .cache/neuroprobe_upstream
_UPSTREAM_CANDIDATES = [
    os.environ.get("NEUROPROBE_UPSTREAM_DIR", ""),
    "/work/ht203/repo/neuroprobe_upstream",
    str(Path(__file__).resolve().parents[2] / ".cache" / "neuroprobe_upstream"),
]


def _add_upstream_to_path() -> str:
    for cand in _UPSTREAM_CANDIDATES:
        if cand and (Path(cand) / "neuroprobe" / "train_test_splits.py").exists():
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand
    raise SystemExit(
        "Could not locate the pinned neuroprobe_upstream clone. Set "
        "NEUROPROBE_UPSTREAM_DIR or run on DCC where "
        "/work/ht203/repo/neuroprobe_upstream exists."
    )


SAMPLING_RATE = 2048
CLIP_DURATION_S = 1.0  # leaderboard 1-second window (before=0, after=1)


def _our_split_windows(
    *,
    eval_mode: str,
    task: str,
    test_subject_id: int,
    test_trial_id: int,
    train_subject_id: int,
    train_trial_id: int,
    binary_tasks: bool,
    fold_index: int,
) -> dict[str, list[tuple[int, int, int]]]:
    """Run OUR Wang2024Treebank + BTWordEvents chain; return, per split, the
    ordered list of (window_from, window_to, label) neural-sample clips."""
    import neuralset as ns

    from speech_decoding.studies.braintreebank.word_events import BTWordEvents

    study = ns.Study(
        name="Wang2024Treebank",
        path=Path(os.environ["ROOT_DIR_BRAINTREEBANK"]),
        mode="lite",
        infra_timelines={"cluster": None},
    )
    word_events = BTWordEvents(
        tasks=(task,),
        binary_tasks=binary_tasks,
        lite=True,
        nano=False,
        balance=True,
        eval_mode=eval_mode,  # "WithinSession" | "CrossSession" | "CrossSubject"
        fold_index=fold_index,
        test_subject_id=test_subject_id,
        test_trial_id=test_trial_id,
        train_subject_id=train_subject_id,
        train_trial_id=train_trial_id,
    )
    chain = ns.Chain(steps=[study, word_events])
    events = chain.run()
    words = events.loc[events["type"] == "Word"]

    out: dict[str, list[tuple[int, int, int]]] = {}
    dur = int(round(CLIP_DURATION_S * SAMPLING_RATE))
    for split in ("train", "val", "test"):
        rows = words.loc[words["split"] == split]
        clips: list[tuple[int, int, int]] = []
        for _, r in rows.iterrows():
            wf = int(round(float(r["start"]) * SAMPLING_RATE))
            clips.append((wf, wf + dur, int(r["label"])))
        out[split] = clips
    return out


def _upstream_dataset_windows(ds) -> list[tuple[int, int, int]]:
    """Enumerate a (possibly Subset/ConcatDataset) upstream dataset in item order
    → [(window_from, window_to, label), ...] using output_indices=True items."""
    clips: list[tuple[int, int, int]] = []
    for i in range(len(ds)):
        item = ds[i]
        (wf, wt) = item["data"]
        clips.append((int(wf), int(wt), int(item["label"])))
    return clips


def _upstream_split_windows(
    *,
    eval_mode: str,
    task: str,
    test_subject_id: int,
    test_trial_id: int,
    binary_tasks: bool,
    fold_index: int,
):
    """Build the upstream split via the real generate_splits_* entry point and
    return (per-split windows dict, electrode_labels list)."""
    import torch

    from neuroprobe.braintreebank_subject import BrainTreebankSubject
    import neuroprobe.train_test_splits as tts

    common = dict(
        eval_name=task,
        dtype=torch.float32,
        lite=True,
        binary_tasks=binary_tasks,
        output_indices=True,
        output_dict=True,
    )
    test_subject = BrainTreebankSubject(
        subject_id=test_subject_id, cache=False, coordinates_type="cortical"
    )

    if eval_mode == "WithinSession":
        folds = tts.generate_splits_within_session(
            test_subject, test_trial_id, **common
        )
        chosen = folds[fold_index]
    elif eval_mode == "CrossSession":
        chosen = tts.generate_splits_cross_session(
            test_subject, test_trial_id, **common
        )[0]
    elif eval_mode == "CrossSubject":
        all_subjects = {test_subject_id: test_subject}
        # cross_subject default trains on (2, 4); construct that subject too.
        from neuroprobe.config import DS_DM_TRAIN_SUBJECT_ID

        all_subjects[DS_DM_TRAIN_SUBJECT_ID] = BrainTreebankSubject(
            subject_id=DS_DM_TRAIN_SUBJECT_ID, cache=False, coordinates_type="cortical"
        )
        chosen = tts.generate_splits_cross_subject(
            all_subjects, test_subject_id, test_trial_id, **common
        )[0]
    else:
        raise ValueError(f"unknown eval_mode {eval_mode!r}")

    windows = {
        "train": _upstream_dataset_windows(chosen["train_dataset"]),
        "val": _upstream_dataset_windows(chosen["val_dataset"]),
        "test": _upstream_dataset_windows(chosen["test_dataset"]),
    }
    # electrode_labels is copied onto every split dataset (Lite-120 in lite order)
    elabels = list(getattr(chosen["test_dataset"], "electrode_labels", []))
    return windows, elabels


def _our_electrode_labels(test_subject_id: int) -> list[str] | None:
    """Our chain's eval electrode set for the subject, or None if subsetting is
    not yet wired (full montage → not directly comparable to Lite-120)."""
    try:
        from speech_decoding.studies.braintreebank.anatomy import lite_voltage_order
    except ImportError:
        return None
    try:
        return list(lite_voltage_order(os.environ["ROOT_DIR_BRAINTREEBANK"], test_subject_id))
    except Exception:
        return None


def _compare(label: str, ours: list, theirs: list) -> list[str]:
    """Return a list of human-readable mismatch lines (empty == exact match)."""
    errs: list[str] = []
    if len(ours) != len(theirs):
        errs.append(f"{label}: COUNT ours={len(ours)} upstream={len(theirs)}")
        return errs  # length mismatch — element diff would be noise
    for i, (a, b) in enumerate(zip(ours, theirs)):
        if a != b:
            errs.append(f"{label}: item {i} ours={a} upstream={b}")
            if len(errs) >= 6:
                errs.append(f"{label}: … (further diffs suppressed)")
                break
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", default="within,csession,csubject",
                    help="comma list of within|csession|csubject")
    ap.add_argument("--tasks", default="speech",
                    help="comma list of task names, or 'all' for all 15")
    ap.add_argument("--binary-tasks", action="store_true", default=True)
    ap.add_argument("--multiclass", dest="binary_tasks", action="store_false")
    ap.add_argument("--check-electrodes", action="store_true",
                    help="also assert Lite-120 electrode-set parity (needs subsetting wired)")
    args = ap.parse_args()

    upstream_dir = _add_upstream_to_path()
    print(f"[parity] upstream clone: {upstream_dir}")
    if "ROOT_DIR_BRAINTREEBANK" not in os.environ:
        raise SystemExit("ROOT_DIR_BRAINTREEBANK must be set (DCC).")

    from neuroprobe.config import (
        DS_DM_TRAIN_SUBJECT_ID,
        DS_DM_TRAIN_TRIAL_ID,
        NEUROPROBE_LITE_SUBJECT_TRIALS,
        NEUROPROBE_TASKS,
    )

    mode_map = {
        "within": "WithinSession",
        "csession": "CrossSession",
        "csubject": "CrossSubject",
    }
    modes = [mode_map[m.strip()] for m in args.modes.split(",") if m.strip()]
    tasks = NEUROPROBE_TASKS if args.tasks == "all" else [
        t.strip() for t in args.tasks.split(",") if t.strip()
    ]

    n_pass = n_fail = 0
    failures: list[str] = []

    for eval_mode in modes:
        if eval_mode == "CrossSubject":
            # one test subject per Lite trial that isn't the fixed train subject
            cells = [(s, t) for (s, t) in NEUROPROBE_LITE_SUBJECT_TRIALS
                     if s != DS_DM_TRAIN_SUBJECT_ID]
        else:
            cells = list(NEUROPROBE_LITE_SUBJECT_TRIALS)

        for (subj, trial) in cells:
            n_folds = 2 if eval_mode == "WithinSession" else 1
            for fold in range(n_folds):
                for task in tasks:
                    tag = (f"{eval_mode} S{subj}T{trial}"
                           f"{f' fold{fold}' if n_folds > 1 else ''} {task}")
                    our_windows = None
                    try:
                        up_windows, up_elabels = _upstream_split_windows(
                            eval_mode=eval_mode, task=task,
                            test_subject_id=subj, test_trial_id=trial,
                            binary_tasks=args.binary_tasks, fold_index=fold,
                        )
                        our_windows = _our_split_windows(
                            eval_mode=eval_mode, task=task,
                            test_subject_id=subj, test_trial_id=trial,
                            train_subject_id=DS_DM_TRAIN_SUBJECT_ID,
                            train_trial_id=DS_DM_TRAIN_TRIAL_ID,
                            binary_tasks=args.binary_tasks, fold_index=fold,
                        )
                        cell_errs: list[str] = []
                        for split in ("train", "val", "test"):
                            cell_errs += _compare(
                                f"{tag} [{split}]",
                                our_windows[split], up_windows[split],
                            )
                        if args.check_electrodes:
                            ours_e = _our_electrode_labels(subj)
                            if ours_e is None:
                                cell_errs.append(
                                    f"{tag} [electrodes]: PENDING (subsetting not wired)")
                            elif set(ours_e) != set(up_elabels):
                                only_ours = sorted(set(ours_e) - set(up_elabels))[:5]
                                only_up = sorted(set(up_elabels) - set(ours_e))[:5]
                                cell_errs.append(
                                    f"{tag} [electrodes]: SET mismatch "
                                    f"|ours|={len(ours_e)} |up|={len(up_elabels)} "
                                    f"ours_only={only_ours} up_only={only_up}")
                    except Exception:
                        cell_errs = [f"{tag}: EXCEPTION\n{traceback.format_exc()}"]

                    if cell_errs and not all("PENDING" in e for e in cell_errs):
                        n_fail += 1
                        failures.extend(cell_errs)
                        print(f"[FAIL] {tag}")
                        for e in cell_errs:
                            print(f"       {e}")
                    else:
                        n_pass += 1
                        note = " (electrodes PENDING)" if any(
                            "PENDING" in e for e in cell_errs) else ""
                        counts = (
                            "/".join(str(len(our_windows[s]))
                                     for s in ("train", "val", "test"))
                            if our_windows is not None else "?/?/?"
                        )
                        print(f"[ OK ] {tag}  train/val/test={counts}{note}")

    print("\n=== PARITY SUMMARY ===")
    print(f"cells passed: {n_pass}")
    print(f"cells failed: {n_fail}")
    if n_fail:
        print("\nFIRST FAILURES:")
        for line in failures[:20]:
            print(line)
        return 1
    print("ABSOLUTE PARITY VERIFIED for all checked (mode, session, task) cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
