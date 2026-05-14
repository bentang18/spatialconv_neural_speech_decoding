"""V14 first-pass DCC dispatch entrypoint.

Composes the v14 NeuralTrain Experiment: BT Wang2024Treebank study + DK-hard
support extractor + V14ParcelPerceiver + DETR readout, with first-pass defaults
locked in ``memory/project_v14_encoder_design_2026_05_13.md``.

DCC invocation (via ``scripts/dcc/dispatch``):

    scripts/dcc/dispatch -m speech_decoding.experiments.dispatch_v14 \\
        --mode lite --eps 1e-2 --m-sub-slots 4 --d-model 128 --depth 6

Smoke-test (laptop, no BT data):

    .venv/bin/python -m speech_decoding.experiments.dispatch_v14 --dry-run

Default electrode-tokens extractor is :class:`LogStftView` (N1 × R2 × I2L × F1),
default support is :class:`V14DKHardSupportExtractor` (K=80 DK, ``c_max=120``
padded), default valid-mask is :class:`ElectrodeValidMask` (``c_max=120``).
Caller can pass ``electrode_tokens_extractor=...`` to override the default
(e.g. for the P2 defensive sister run using the linear-baseline recipe).
"""

from __future__ import annotations

import argparse
import os
import typing as tp
from pathlib import Path

import neuralset as ns

import speech_decoding.models  # noqa: F401  # registers V14ParcelPerceiver with BaseModelConfig
from speech_decoding.experiments import Data, Experiment
from speech_decoding.extractors.dk_support import V14DKHardSupportExtractor
from speech_decoding.extractors.valid_mask import ElectrodeValidMask
from speech_decoding.extractors.view import LogStftView
from speech_decoding.studies.braintreebank.anatomy import (
    DEFAULT_SUPPORT_BIAS_EPS,
    V14_DK_PARCEL_LABELS,
)
from speech_decoding.studies.braintreebank.manifest import V14_TRAIN_SUBJECT_IDS
from speech_decoding.studies.braintreebank.study import Wang2024Treebank
from speech_decoding.studies.braintreebank.word_events import BTWordEvents


# First-pass defaults, locked in the encoder design memo.
DEFAULT_D_MODEL = 128
DEFAULT_DEPTH = 6
DEFAULT_N_HEADS = 4
DEFAULT_M_SUB_SLOTS = 4
DEFAULT_K_PARCELS = len(V14_DK_PARCEL_LABELS)  # 80
DEFAULT_N_FREQ_BINS = 38   # ≤150 Hz with the locked STFT nperseg=512 @ 2 kHz
DEFAULT_N_TIME_BINS = 17   # 1-second window with overlap=0.75
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_EPOCHS = 100
# Ship-first task default — `speech` is the highest-signal binary task
# requiring zero transcript enrichment (Sentence Onset = 0.780 CS-SOTA,
# Speech = 0.751; full ship-first set is {onset, speech, delta_volume,
# word_index} per the v2 paper's CS-above-chance four).
DEFAULT_TASK = "speech"
DEFAULT_EVAL_MODE = "CrossSession"
DEFAULT_TEST_SUBJECT_ID = 2
DEFAULT_TEST_TRIAL_ID = 4
DEFAULT_C_MAX = 256  # BT raw electrode max — Wang2024Treebank emits full set,
                     # the NEUROPROBE_LITE_ELECTRODES filter is not yet a chain
                     # step (TODO: add as EventsTransform). 120 was the Lite
                     # cap which assumed upstream filtering; un-filtered BT
                     # subjects can carry 130-200+ electrodes.


def build_v14_experiment(
    *,
    bt_root: str | None = None,
    mode: tp.Literal["nano", "lite", "full"] = "lite",
    task: str = DEFAULT_TASK,
    eval_mode: tp.Literal["CrossSession", "CrossSubject"] = DEFAULT_EVAL_MODE,
    test_subject_id: int = DEFAULT_TEST_SUBJECT_ID,
    test_trial_id: int = DEFAULT_TEST_TRIAL_ID,
    binary_tasks: bool = True,
    electrode_tokens_extractor: tp.Any | None = None,
    eps: float = DEFAULT_SUPPORT_BIAS_EPS,
    d_model: int = DEFAULT_D_MODEL,
    depth: int = DEFAULT_DEPTH,
    n_heads: int = DEFAULT_N_HEADS,
    m_sub_slots: int = DEFAULT_M_SUB_SLOTS,
    n_freq_bins: int = DEFAULT_N_FREQ_BINS,
    n_time_bins: int = DEFAULT_N_TIME_BINS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_epochs: int = DEFAULT_N_EPOCHS,
    seed: int = 33,
    exca_folder: str | None = None,
    cluster: str | None = None,
    fast_dev_run: bool | int = False,
) -> Experiment:
    """Compose a v14 first-pass Experiment ready for ``.run()`` dispatch.

    The ``electrode_tokens_extractor`` arg is REQUIRED for real runs and must
    emit per-event ``(n_channels, n_time_bins, n_freq_bins)`` STFT tokens
    following the v14 preprocessing recipe (``N1 × R2 × I2L × F1``).

    Word events are appended downstream of :class:`Wang2024Treebank` via
    :class:`BTWordEvents` (``ns.Chain``) so per-trial ``words_df`` /
    ``nonverbal_df`` only load when ``study.run()`` materialises the chain.
    """
    bt_root = bt_root or os.environ.get("ROOT_DIR_BRAINTREEBANK")
    if bt_root is None:
        raise RuntimeError(
            "ROOT_DIR_BRAINTREEBANK must be set or bt_root passed explicitly"
        )

    if electrode_tokens_extractor is None:
        electrode_tokens_extractor = LogStftView(
            event_types="Ieeg",
            car="shaft",
            notch_filter=60.0,
            scaler="StandardScaler",
            channel_order="original",
            c_max=DEFAULT_C_MAX,
        )

    study = Wang2024Treebank(
        path=Path(bt_root), mode=mode,
        infra_timelines={"cluster": None},
    )
    word_events = BTWordEvents(
        tasks=(task,),
        binary_tasks=binary_tasks,
        lite=(mode == "lite"),
        nano=(mode == "nano"),
        eval_mode=eval_mode,
        test_subject_id=test_subject_id,
        test_trial_id=test_trial_id,
        bt_root=bt_root,
    )
    chain = ns.Chain(steps=[study, word_events])

    dk_extractor = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=bt_root, unknown_label_policy="skip",
        c_max=DEFAULT_C_MAX,
    )
    valid_mask_extractor = ElectrodeValidMask(
        event_types="Ieeg", bt_root=bt_root, c_max=DEFAULT_C_MAX,
        unknown_label_policy="skip",
    )

    data = Data(
        study=chain,
        segmenter={
            "extractors": {
                "electrode_tokens": electrode_tokens_extractor,
                "support": dk_extractor,
                "valid_mask": valid_mask_extractor,
                "target": {
                    "name": "EventField",
                    "event_types": "Word",
                    "event_field": "label",
                    "aggregation": "trigger",
                },
            },
            "trigger_query": "type == 'Word'",
            "start": 0.0,
            "duration": 1.0,
        },
        batch_size=batch_size,
    )

    exca_folder = exca_folder or os.environ.get("EXCA_CACHE_FOLDER")
    infra_cfg: dict[str, tp.Any] = {}
    if exca_folder is not None:
        infra_cfg["folder"] = exca_folder
    if cluster is not None:
        infra_cfg["cluster"] = cluster

    return Experiment(
        data=data,
        infra=infra_cfg,
        target_field="label",
        brain_model_config={
            "name": "V14ParcelPerceiver",
            "n_freq_bins": n_freq_bins,
            "n_time_bins": n_time_bins,
            "k_parcels": DEFAULT_K_PARCELS,
            "d_model": d_model,
            "n_heads": n_heads,
            "depth_self_attn": depth,
            "m_sub_slots": m_sub_slots,
            "eps": eps,
            "time_last_input": True,
        },
        loss={"name": "CrossEntropyLoss"},
        optim={"optimizer": {"name": "Adam", "lr": 1e-3}},
        metrics=[
            {
                "name": "Accuracy",
                "log_name": "acc",
                "kwargs": {"task": "multiclass", "num_classes": 2},
            }
        ],
        n_epochs=n_epochs,
        seed=seed,
        x_name=("electrode_tokens", "support", "valid_mask"),
        accelerator="auto",
        devices="auto",
        fast_dev_run=fast_dev_run,
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="V14 first-pass DCC dispatch (BT cohort, K=80 DK parcels)."
    )
    p.add_argument("--mode", choices=("nano", "lite", "full"), default="lite")
    p.add_argument("--task", default=DEFAULT_TASK,
                   help="Neuroprobe task name (event field for the target).")
    p.add_argument("--eval-mode", choices=("CrossSession", "CrossSubject"),
                   default=DEFAULT_EVAL_MODE,
                   help="Split policy (CrossSession = submission gate, "
                        "CrossSubject = scientific generalization).")
    p.add_argument("--test-subject-id", type=int, default=DEFAULT_TEST_SUBJECT_ID)
    p.add_argument("--test-trial-id", type=int, default=DEFAULT_TEST_TRIAL_ID)
    p.add_argument("--binary-tasks", action="store_true", default=True,
                   help="(default) Binary label derivation per Neuroprobe leaderboard. "
                        "Pass --no-binary-tasks to switch to 3-class multiclass.")
    p.add_argument("--no-binary-tasks", dest="binary_tasks", action="store_false")
    p.add_argument("--eps", type=float, default=DEFAULT_SUPPORT_BIAS_EPS,
                   help="Anatomy-prior strength for log(support+eps).")
    p.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--m-sub-slots", type=int, default=DEFAULT_M_SUB_SLOTS)
    p.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--cluster", default=None,
                   help="Exca TaskInfra cluster ('slurm' or None for local).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved config without dispatching.")
    p.add_argument("--fast-dev-run", action="store_true",
                   help="Lightning fast-dev-run: 1 batch train+val+test, no checkpoints.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(f"V14 dispatch — cohort subject_ids = {V14_TRAIN_SUBJECT_IDS} (9 subjects, S5 excluded)")
    print(f"  mode={args.mode} task={args.task} binary_tasks={args.binary_tasks} seed={args.seed}")
    print(f"  eval_mode={args.eval_mode} test=({args.test_subject_id},{args.test_trial_id})")
    print(f"  d_model={args.d_model} depth={args.depth} n_heads={args.n_heads} "
          f"M={args.m_sub_slots} eps={args.eps}")
    print(f"  K=80 DK parcels, batch_size={args.batch_size}, n_epochs={args.n_epochs}")

    if args.dry_run:
        print("  (dry-run: not building Experiment; "
              "default electrode-tokens extractor = LogStftView)")
        return 0

    xp = build_v14_experiment(
        mode=args.mode, task=args.task, seed=args.seed,
        eval_mode=args.eval_mode,
        test_subject_id=args.test_subject_id,
        test_trial_id=args.test_trial_id,
        binary_tasks=args.binary_tasks,
        eps=args.eps, d_model=args.d_model, depth=args.depth,
        n_heads=args.n_heads, m_sub_slots=args.m_sub_slots,
        batch_size=args.batch_size, n_epochs=args.n_epochs,
        cluster=args.cluster, fast_dev_run=args.fast_dev_run,
    )
    result = xp.run()
    print(f"V14 dispatch result: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
