"""Dump per-trial native sample counts → a tiny JSON the study reads instead of h5.

The ONLY h5 touch in BrainTreebank timeline building is
``Wang2024Treebank._trial_duration_seconds``, which opens each trial's raw h5 to
read its native sample count. An h5-free redeploy (``--spec-only`` on DeltaAI,
which ships only the |STFT| spec cache) therefore cannot build its event
timelines. This script precomputes that one quantity on DCC (h5 present) into

    {"<subject_id>_<trial_id>": native_n_samples}

which ``dispatch_v14 --trial-durations PATH`` loads into the study so
``_trial_duration_seconds`` returns from the map. The read here MIRRORS the study
read exactly (first electrode's ``data`` group length), so the JSON-sourced
duration is byte-identical to the h5 path — duration is uid-invariant, so caches
and clips are unchanged.

Run on DCC:
    .venv/bin/python -m scripts.neuroprobe.dump_trial_durations \
        --mode pretrain --mode lite --out /work/ht203/v14_trial_durations.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py

from speech_decoding.studies.braintreebank.study import _SESSIONS_BY_MODE


def _native_n_samples(subject_id: int, trial_id: int) -> int:
    from neuroprobe.braintreebank_subject import BrainTreebankSubject
    from neuroprobe.config import ROOT_DIR

    bt = BrainTreebankSubject(
        subject_id=subject_id, cache=False, coordinates_type="cortical",
    )
    trial_path = Path(os.fspath(ROOT_DIR)) / f"sub_{subject_id}_trial{trial_id:03}.h5"
    first_label = bt.electrode_labels[0]
    first_key = bt.h5_neural_data_keys[first_label]
    with h5py.File(trial_path, "r") as h5:
        return int(h5["data"][first_key].shape[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", action="append", required=True,
        choices=sorted(_SESSIONS_BY_MODE),
        help="Corpus mode(s) to cover; repeatable. Use the deploy corpora "
             "(e.g. --mode pretrain --mode lite). The union of their sessions is "
             "emitted (a superset is harmless — the study looks up only its keys).",
    )
    p.add_argument("--out", required=True, help="Output JSON path.")
    args = p.parse_args()

    sessions: set[tuple[int, int]] = set()
    for mode in args.mode:
        sessions.update(_SESSIONS_BY_MODE[mode])

    durations: dict[str, int] = {}
    for subject_id, trial_id in sorted(sessions):
        n = _native_n_samples(subject_id, trial_id)
        durations[f"{subject_id}_{trial_id}"] = n
        print(f"  ({subject_id},{trial_id}) -> {n} native samples")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(durations, indent=1, sort_keys=True))
    print(f"wrote {len(durations)} trial durations -> {out}")


if __name__ == "__main__":
    main()
