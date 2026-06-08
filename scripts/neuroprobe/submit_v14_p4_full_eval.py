"""V14 Phase-4 full leaderboard eval — one Slurm array via job_array.

Loads a frozen P3b encoder snapshot (``--ckpt``) and runs the B35 frozen-PMA →
mean → linear readout probe over the **full Neuroprobe-Lite matrix**:

    modes  × tasks × Lite (subject, trial) cells  [× KFold folds]
    ───────────────────────────────────────────────────────────────
    WithinSession  15 × 12 cells × 2 folds = 360
    CrossSession   15 × 12 cells           = 180
    CrossSubject   15 × 10 cells           = 150   (drop the fixed train subject 2)
    ───────────────────────────────────────────────────────────────
    total                                  = 690 probe cells

Each cell reuses ``build_v14_experiment(phase4_frozen_probe=True, ...)`` with the
SAME canonical P4 config the ``--chain`` driver pins (``clip_len=1.0``,
``neural_lag_s=0.0``, ``collapse_guard=False``, ``electrode_set="lite"``) plus the
standalone ``pretrained_ckpt`` handoff. exca's config-keyed UID gives every cell
its own cache dir under one infra folder; ``infra.job_array`` dispatches them as a
single Slurm array. Re-running is idempotent (cached cells return instantly).

The eval matrix MIRRORS ``scripts/neuroprobe/verify_parity.py`` exactly (same
modes/cells/tasks/fold enumeration), which has proven 690/690 absolute Neuroprobe
parity. The leaderboard contract + electrode-set gating are locked in
``memory/project_v14_p4_eval_neuroprobe_parity_lock_2026_06_08.md``.

Paired collector (sweep discipline — no sweep without a collector):

    # aggregate the per-cell experiment_record.json sidecars into a flat CSV
    scripts/neuroprobe/collect_experiment_records.py \\
        --artifact-root <exca_folder>/v14_p4_full_eval_v1 \\
        --out-csv docs/experiments/v14_p4_full_eval_2026_06_08.csv
    # then pivot to a mode×task leaderboard
    scripts/neuroprobe/analyze_v14_p4_full_eval.py \\
        --runs-csv docs/experiments/v14_p4_full_eval_2026_06_08.csv

DCC invocation (isolated clone speech_bt — never the shared speech clone):

    scripts/dcc/dispatch -m scripts.neuroprobe.submit_v14_p4_full_eval \\
        --cluster slurm --clone speech_bt

Smoke (laptop, no BT data + no Slurm):

    ROOT_DIR_BRAINTREEBANK=/tmp/dummy EXCA_CACHE_FOLDER=/tmp/exca \\
    .venv/bin/python -m scripts.neuroprobe.submit_v14_p4_full_eval --dry-run
"""

from __future__ import annotations

import argparse
import os
import typing as tp
from pathlib import Path

import speech_decoding.models  # noqa: F401  # registers V14ParcelPerceiver
from speech_decoding.experiments.dispatch_v14 import (
    DEFAULT_N_EPOCHS,
    DEFAULT_P4_EARLY_STOP_PATIENCE,
    build_v14_experiment,
)
from speech_decoding.studies.braintreebank._neuroprobe_lite_tables import (
    NEUROPROBE_LITE_SUBJECT_TRIALS,
)

GRID_VERSION = "1"
EXP_NAME = f"v14_p4_full_eval_v{GRID_VERSION}"

# The 15 Neuroprobe leaderboard tasks (upstream NEUROPROBE_TASKS, pinned c7b955b).
# Hardcoded so the submitter imports without ROOT_DIR_BRAINTREEBANK (the upstream
# config.py reads that env at import). Drift-guarded by verify_parity.py, which
# pulls the live NEUROPROBE_TASKS.
NEUROPROBE_TASKS: tuple[str, ...] = (
    "onset", "speech", "volume", "delta_volume", "pitch", "word_index",
    "word_gap", "gpt2_surprisal", "word_head_pos", "word_part_speech",
    "word_length", "global_flow", "local_flow", "frame_brightness", "face_num",
)

# CrossSubject trains on the fixed (subject 2, trial 4) cell upstream — so a
# CrossSubject test cell can be any Lite cell EXCEPT subject 2.
DS_DM_TRAIN_SUBJECT_ID = 2

# Capstone P3b snapshot (job 47829858, the last long run) on the persistent
# cache. Overridable with --ckpt. The "..." in the dir name is exca's UID
# truncation, not a glob — it is the literal on-disk path.
DEFAULT_P3B_CKPT = (
    "/hpc/group/coganlab/ht203/cache_neuroai/"
    "speech_decoding.experiments.v14_phase3.V14Phase3Experiment.run,1/"
    "phase=3b,n_epochs=100,max_steps=1500,target_field=label,gradient_clip_val=1,"
    "limit_val_batches=32,precision=bf16-mixed,val_check_...981-d61c0ba5/"
    "checkpoints/best.ckpt"
)

_MODE_ALIASES = {
    "within": "WithinSession",
    "csession": "CrossSession",
    "csubject": "CrossSubject",
}
# Lite WithinSession uses KFold(n_splits=2) — score both held-out folds.
_LITE_N_FOLDS = 2


def enumerate_cells(modes: list[str], tasks: list[str]) -> list[dict[str, tp.Any]]:
    """All (eval_mode, task, test_subject, test_trial, fold) cells — mirrors
    verify_parity.py's matrix exactly."""
    cells: list[dict[str, tp.Any]] = []
    for eval_mode in modes:
        if eval_mode == "CrossSubject":
            pairs = [
                (s, t) for (s, t) in NEUROPROBE_LITE_SUBJECT_TRIALS
                if s != DS_DM_TRAIN_SUBJECT_ID
            ]
        else:
            pairs = list(NEUROPROBE_LITE_SUBJECT_TRIALS)
        n_folds = _LITE_N_FOLDS if eval_mode == "WithinSession" else 1
        for task in tasks:
            for (subj, trial) in pairs:
                for fold in range(n_folds):
                    cells.append({
                        "eval_mode": eval_mode, "task": task,
                        "test_subject_id": subj, "test_trial_id": trial,
                        "fold_index": fold,
                    })
    return cells


def cell_label(cell: dict[str, tp.Any]) -> str:
    base = (f"{cell['eval_mode']}_{cell['task']}_"
            f"S{cell['test_subject_id']}T{cell['test_trial_id']}")
    if cell["eval_mode"] == "WithinSession":
        base += f"_f{cell['fold_index']}"
    return base


def build_cell_experiment(*, cell: dict[str, tp.Any], args: argparse.Namespace,
                          base_folder: Path):
    """One P4 frozen-probe Experiment for a single matrix cell. Canonical chain
    P4 config + standalone pretrained_ckpt handoff."""
    return build_v14_experiment(
        bt_root=args.bt_root,
        mode=args.mode,                       # "lite" = the leaderboard eval universe
        task=cell["task"],
        eval_mode=cell["eval_mode"],
        test_subject_id=cell["test_subject_id"],
        test_trial_id=cell["test_trial_id"],
        fold_index=cell["fold_index"],
        # --- canonical P4 readout (mirrors _build_v14_chain's p4) ---
        phase4_frozen_probe=True,
        pretrained_ckpt=args.ckpt,
        clip_len=1.0,
        neural_lag_s=0.0,
        collapse_guard=False,
        electrode_set="lite",
        n_epochs=args.n_epochs,
        early_stopping_patience=(
            args.p4_early_stop_patience if args.p4_early_stop_patience > 0 else None
        ),
        seed=args.seed,
        exca_folder=str(base_folder / EXP_NAME),
        cluster=args.cluster,
        # P4 is a frozen-encoder CPU probe (no GPU requested), so it leaves the
        # coganlab-gpu cards free for nano intuition runs. The frozen encoder
        # forward still runs each step, so it needs real RAM — the submitit 2 GB
        # default OOM-thrashes. Route to the 'common' CPU partition.
        mem_gb=args.mem_gb,
        cpus_per_task=args.cpus_per_task,
        slurm_partition=args.slurm_partition,
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="V14 Phase-4 full Neuroprobe-Lite leaderboard eval — one Slurm array.",
    )
    p.add_argument("--modes", default="within,csession,csubject",
                   help="Comma list of within|csession|csubject.")
    p.add_argument("--tasks", default="all",
                   help="Comma list of Neuroprobe task names, or 'all' (15 tasks).")
    p.add_argument("--ckpt", default=DEFAULT_P3B_CKPT,
                   help="Frozen P3b encoder snapshot to load (Experiment.pretrained_ckpt).")
    p.add_argument("--mode", choices=("nano", "lite", "full"), default="lite",
                   help="P4 eval clip universe. 'lite' = the 12 Neuroprobe Lite "
                        "eval sessions (the leaderboard); 'nano' = tiny smoke.")
    p.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    p.add_argument("--p4-early-stop-patience", dest="p4_early_stop_patience",
                   type=int, default=DEFAULT_P4_EARLY_STOP_PATIENCE,
                   help="Early-stop patience (epochs) on the probe val_loss; <=0 disables.")
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--dry-run", action="store_true",
                   help="Build all per-cell Experiments + print the matrix; do not submit.")
    p.add_argument("--max-workers", type=int, default=64,
                   help="Slurm array parallelism cap (job_array max_workers).")
    p.add_argument("--mem-gb", dest="mem_gb", type=float, default=16.0,
                   help="Per-cell RAM. Frozen-encoder forward + 169 MB ckpt OOMs "
                        "at the submitit 2 GB default; 16 GB is the safe floor.")
    p.add_argument("--cpus-per-task", dest="cpus_per_task", type=int, default=2,
                   help="Per-cell CPUs (CPU-only probe).")
    p.add_argument("--slurm-partition", dest="slurm_partition", default="common",
                   help="Slurm partition. Default 'common' (CPU) keeps P4 off the "
                        "GPU nodes so coganlab-gpu stays free for nano runs.")
    p.add_argument("--bt-root", default=None)
    p.add_argument("--exca-folder", default=None,
                   help="Base cache folder. Falls back to $EXCA_CACHE_FOLDER. On "
                        "DCC use /hpc/group/coganlab/ht203/cache_neuroai/ — NOT "
                        "/work/ht203/ (75-day purge).")
    p.add_argument("--cluster", default=None,
                   help="exca TaskInfra cluster ('slurm' on DCC, omit for local).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    modes = [_MODE_ALIASES[m.strip()] for m in args.modes.split(",") if m.strip()]
    tasks = (list(NEUROPROBE_TASKS) if args.tasks == "all"
             else [t.strip() for t in args.tasks.split(",") if t.strip()])
    unknown = [t for t in tasks if t not in NEUROPROBE_TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}; valid: {NEUROPROBE_TASKS}")

    base_folder_str = args.exca_folder or os.environ.get("EXCA_CACHE_FOLDER")
    if base_folder_str is None:
        raise RuntimeError(
            "EXCA_CACHE_FOLDER must be set or --exca-folder passed explicitly."
        )
    base_folder = Path(base_folder_str)

    cells = enumerate_cells(modes, tasks)
    by_mode: dict[str, int] = {}
    for c in cells:
        by_mode[c["eval_mode"]] = by_mode.get(c["eval_mode"], 0) + 1

    print(f"V14 Phase-4 full eval — {EXP_NAME}")
    print(f"  modes={modes}  tasks={len(tasks)}  → {len(cells)} probe cells")
    for m, n in by_mode.items():
        print(f"    {m}: {n}")
    print(f"  ckpt={args.ckpt}")
    print(f"  mode={args.mode}  n_epochs={args.n_epochs}  "
          f"p4_patience={args.p4_early_stop_patience}  seed={args.seed}")
    print(f"  base_folder={base_folder / EXP_NAME}")

    if not args.dry_run and not Path(args.ckpt).exists():
        raise SystemExit(
            f"P3b checkpoint not found: {args.ckpt}\n"
            "Pass --ckpt or run with --dry-run on a machine without the cache."
        )

    experiments = [
        build_cell_experiment(cell=cell, args=args, base_folder=base_folder)
        for cell in cells
    ]

    if args.dry_run:
        for cell, xp in zip(cells[:6], experiments[:6]):
            print(f"  [dry-run] {cell_label(cell)}  uid={xp.infra.uid()}")
        if len(cells) > 6:
            print(f"  [dry-run] ... +{len(cells) - 6} more cells")
        return 0

    base_infra = experiments[0].infra
    with base_infra.job_array(max_workers=args.max_workers) as tasks_q:
        for xp in experiments:
            tasks_q.append(xp)
    print(f"V14 P4 full-eval dispatch: {len(experiments)} cells submitted as one array.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
