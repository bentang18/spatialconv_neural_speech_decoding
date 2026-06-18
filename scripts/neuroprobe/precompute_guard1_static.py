"""GUARD-1 STATIC precompute (#207) — scan each BT session's full-montage voltage
with the three locked detectors and dump a per-session signature JSON.

This is the DCC scan that feeds the #208 collector (``collect_guard1_static.py``).
For each (subject, trial) it reproduces the detectors' input domain — the model's
PRE-CAR filtered voltage over the FULL montage — and runs ``guard1.static_bad_mask``:

    bt_load_raw(drop_static=False)        # FULL montage (incl. the contacts we may
      → native -> 2048 Hz resample        #   drop), so the set is re-derivable
      → notch(60) + HPF(0.5)               # the model's per-channel filters
      → static_bad_mask(filt, fs)          # spike | noisy | dead, per the locked bars

Why ``drop_static=False`` and NO shaft-CAR:
  * Full montage — scanning a montage from which the bad contacts were already
    removed is circular; the detectors must SEE the contacts they decide to drop,
    and ``noisy``/``dead`` are cohort-relative so the peer set must be complete.
  * Pre-CAR — the static drop happens pre-CAR in the loader precisely so a bad
    contact cannot poison its shaft's CAR reference; the decision must therefore
    be made on pre-CAR voltage (a bad contact's pathology is its own, not the
    shaft mean's). The detectors are voltage-domain by design (the Guard-1 spec).

The per-session signature (``guard1.session_signature``) carries ``clip_frac`` +
``rmad`` (the diagnostics the collector keys the locked thresholds off) plus the
fired detector label-lists for provenance. The collector re-derives the drop from
these via ``classify_from_signature`` — so the bars live in ONE place and a JSON
collation classifies byte-identically to this live scan (round-trip tested in
``test_guard1.py::test_session_signature_roundtrips_through_collector``).

READ-ONLY w.r.t. the dataset/cache. DCC-only: needs ``ROOT_DIR_BRAINTREEBANK`` +
BT voltage. Array-parallel via ``SLURM_ARRAY_TASK_ID`` (one session per task).

Usage (DCC; scripts/ is not a package, so run by PATH not ``-m``)
-----
    # one session (DCC only — laptop has no BT voltage)
    ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank \
      .venv/bin/python scripts/neuroprobe/precompute_guard1_static.py \
        --out-dir reports/guard1_static/detections --session 9 0

    # full v14 corpus, one Slurm array task per session (25 sessions -> 0-24)
    #   sbatch --array=0-24 ... \
    #     "ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank \
    #      .venv/bin/python scripts/neuroprobe/precompute_guard1_static.py \
    #        --out-dir reports/guard1_static/detections"
"""

from __future__ import annotations

import argparse
import gc
import json
import os

import mne
import numpy as np

mne.set_log_level("ERROR")

from speech_decoding.studies.braintreebank import guard1
from speech_decoding.studies.braintreebank.loader import bt_load_raw
from speech_decoding.studies.braintreebank.manifest import (
    BT_FULL_SESSIONS,
    V14_EXCLUDED_SUBJECT_IDS,
)

NOTCH_HZ: float = 60.0
HPF_HZ: float = 0.5


def scan_session(subject_id: int, trial_id: int) -> dict:
    """Scan one (subject, trial) session and return its signature payload."""
    from neuroprobe.braintreebank_subject import BrainTreebankSubject

    bt = BrainTreebankSubject(
        subject_id=subject_id, cache=False, coordinates_type="cortical"
    )
    # FULL montage (drop_static=False) + native->2048 resample + label clean, all
    # from the single bt_load_raw source so the S9 1024->2048 resample is not
    # re-derived here. The STATIC drop is the ONLY thing skipped.
    data, ch_names, sfreq = bt_load_raw(
        bt, trial_id=trial_id, subject_id=subject_id, drop_static=False
    )
    ch_names = list(ch_names)
    sfreq = float(sfreq)

    # notch + HPF (float64), NO shaft-CAR (detectors are pre-CAR by design).
    info = mne.create_info(
        ch_names=ch_names, sfreq=sfreq, ch_types="seeg", verbose=False
    )
    raw = mne.io.RawArray(np.asarray(data, dtype=np.float64), info, verbose=False)
    raw.load_data()
    raw.notch_filter(NOTCH_HZ, phase="zero", filter_length="auto", verbose=False)
    raw.filter(HPF_HZ, None, verbose=False)
    filt = np.asarray(raw.get_data(), dtype=np.float64)  # (C, N), pre-CAR
    del raw, data
    gc.collect()

    sig = guard1.static_bad_mask(filt, sfreq)
    payload = guard1.session_signature(subject_id, trial_id, ch_names, sig)
    payload["fs"] = sfreq
    payload["n_elec"] = int(filt.shape[0])
    payload["n_samples"] = int(filt.shape[1])
    return payload


def _resolve_sessions(arg_sessions: str | None) -> list[tuple[int, int]]:
    """Sessions to scan: ``--sessions "S:T,S:T,..."`` override, else BT_FULL minus
    the v14-excluded subjects (S5: lesion, no h5, not in cohort). The static set is
    per-subject, so every loaded trial of each cohort subject is scanned."""
    if arg_sessions:
        out: list[tuple[int, int]] = []
        for tok in arg_sessions.split(","):
            s, t = tok.split(":")
            out.append((int(s), int(t)))
        return out
    excluded = set(V14_EXCLUDED_SUBJECT_IDS)
    return [(s, t) for (s, t) in BT_FULL_SESSIONS if s not in excluded]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True,
                    help="per-session signature output dir (= collector --detections-dir)")
    ap.add_argument("--session", nargs=2, type=int, metavar=("SUBJECT", "TRIAL"),
                    help="scan exactly this one session (overrides array indexing)")
    ap.add_argument("--sessions", default=None,
                    help='comma list "S:T,S:T,..." to scan (array indexes into it)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.session is not None:
        todo = [(args.session[0], args.session[1])]
    else:
        sessions = _resolve_sessions(args.sessions)
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        todo = [sessions[int(task)]] if task is not None else sessions

    for subject_id, trial_id in todo:
        payload = scan_session(subject_id, trial_id)
        out_path = os.path.join(args.out_dir, f"btbank{subject_id}_t{trial_id}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(
            f"[btbank{subject_id}_t{trial_id}] {payload['n_elec']} elec  "
            f"{payload['n_samples'] / payload['fs'] / 60:.1f}min  "
            f"spike={payload['spike']}  noisy={payload['noisy']}  "
            f"dead={payload['dead']}  -> static={payload['static']}  -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
