"""GUARD-1 STATIC precompute for the Cogan D-cohort — the Cogan sibling of
``precompute_guard1_static.py`` (BT). Scans each Cogan BIDS run's FULL neural
montage with the three locked detectors (``guard1.static_bad_mask``) and dumps a
per-run signature JSON that the collector re-derives the static drop from.

Parity with the BT scan is BY CONSTRUCTION — the detector core (``guard1.py``)
and its Ben-locked thresholds (spike |z(dV/dt)|>100 over >1% of 5s clips, noisy
cohort-z(rmad)>8, dead ratio<0.35) are SHARED, and the input domain is reproduced
identically:

    cogan_load_raw(edf, channels_tsv, extra_bad=())   # FULL neural montage,
      → native -> 2048 resample                        #   resampled to 2048
      → mne.RawArray(float64)
      → notch(power_line_hz) + HPF(0.5)                # model's per-channel filters
      → static_bad_mask(filt, fs)                       # spike | noisy | dead
      → session_signature(...)                          # collector re-derives from this

Why ``extra_bad=()`` (the analogue of BT's ``drop_static=False``) and NO shaft-CAR:
  * FULL montage — the detectors must SEE the contacts they decide to drop, and
    ``noisy``/``dead`` are cohort-relative so the peer set must be complete.
    ``cogan_load_raw`` with empty ``extra_bad`` returns every real neural contact
    (physio/scalp refs and non-neural types are still dropped — they are never in
    the cohort, matching BT dropping trigger/corrupted).
  * Pre-CAR — the static drop is baked BEFORE shaft-CAR precisely so a bad contact
    cannot poison its shaft's CAR reference; the decision must therefore be made on
    pre-CAR voltage. The detectors are voltage-domain by design (the Guard-1 spec).

Notch = per-run ``power_line_hz`` from the manifest (60 Hz for all Cogan/Duke runs,
US mains) — matches BT's single-60 scan, so the cohort-relative bars port cleanly.
European corpora (50 Hz) will read their own ``power_line_hz`` here.

READ-ONLY w.r.t. the dataset/cache. DCC-only: needs the Cogan BIDS EDF tree + the
run manifest (``build_cogan_manifest.py``). Array-parallel via ``SLURM_ARRAY_TASK_ID``
(one run per task, indexing the manifest row order).

Usage (DCC; scripts/ is not a package, so run by PATH not ``-m``)
-----
    # one run (by manifest 0-based row index)
    .venv/bin/python scripts/neuroprobe/precompute_guard1_static_cogan.py \
        --manifest /work/ht203/cogan_v3/manifest/cogan_manifest.csv \
        --out-dir  /work/ht203/cogan_v3/guard1/detections --row 0

    # full corpus, one Slurm array task per run (938 runs -> 0-937)
    #   sbatch --array=0-937 ... \
    #     ".venv/bin/python scripts/neuroprobe/precompute_guard1_static_cogan.py \
    #        --manifest .../cogan_manifest.csv --out-dir .../guard1/detections"
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os

import numpy as np

HPF_HZ: float = 0.5


def read_manifest(manifest_path: str) -> list[dict]:
    """Manifest CSV -> ordered list of per-run dicts (row order = array order).

    Keeps only the fields the scan needs: ``subject_id`` (= ``global_subject_id``),
    ``trial_id``, ``edf_path``, ``channels_tsv_path``, ``power_line_hz``.
    """
    rows: list[dict] = []
    with open(manifest_path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                {
                    "subject_id": int(r["global_subject_id"]),
                    "trial_id": int(r["trial_id"]),
                    "edf_path": r["edf_path"],
                    "channels_tsv_path": r["channels_tsv_path"],
                    "power_line_hz": float(r["power_line_hz"]),
                }
            )
    return rows


def resolve_rows(manifest: list[dict], row_arg: int | None) -> list[dict]:
    """Rows to scan: ``--row N`` picks one, else ``SLURM_ARRAY_TASK_ID`` indexes the
    manifest, else the whole manifest (serial)."""
    if row_arg is not None:
        return [manifest[row_arg]]
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    return [manifest[int(task)]] if task is not None else manifest


def scan_run(row: dict) -> dict:
    """Scan one Cogan run and return its signature payload."""
    import mne

    mne.set_log_level("ERROR")
    from speech_decoding.studies.braintreebank import guard1
    from speech_decoding.studies.cogan_dcohort.loader import cogan_load_raw

    data, ch_names, sfreq = cogan_load_raw(
        row["edf_path"], row["channels_tsv_path"], extra_bad=()
    )
    ch_names = list(ch_names)
    sfreq = float(sfreq)

    # notch(power_line) + HPF (float64), NO shaft-CAR (detectors are pre-CAR).
    info = mne.create_info(
        ch_names=ch_names, sfreq=sfreq, ch_types="seeg", verbose=False
    )
    raw = mne.io.RawArray(np.asarray(data, dtype=np.float64), info, verbose=False)
    raw.load_data()
    raw.notch_filter(
        row["power_line_hz"], phase="zero", filter_length="auto", verbose=False
    )
    raw.filter(HPF_HZ, None, verbose=False)
    filt = np.asarray(raw.get_data(), dtype=np.float64)  # (C, N), pre-CAR
    del raw, data
    gc.collect()

    sig = guard1.static_bad_mask(filt, sfreq)
    payload = guard1.session_signature(
        row["subject_id"], row["trial_id"], ch_names, sig
    )
    payload["fs"] = sfreq
    payload["n_elec"] = int(filt.shape[0])
    payload["n_samples"] = int(filt.shape[1])
    payload["power_line_hz"] = row["power_line_hz"]
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="cogan_manifest.csv path")
    ap.add_argument("--out-dir", required=True,
                    help="per-run signature output dir (= collector --detections-dir)")
    ap.add_argument("--row", type=int, default=None,
                    help="scan exactly this one 0-based manifest row (overrides array)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = read_manifest(args.manifest)
    todo = resolve_rows(manifest, args.row)

    for row in todo:
        payload = scan_run(row)
        sid, tid = payload["subject"], payload["trial"]
        out_path = os.path.join(args.out_dir, f"cogan{sid}_t{tid}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(
            f"[cogan{sid}_t{tid}] {payload['n_elec']} elec  "
            f"{payload['n_samples'] / payload['fs'] / 60:.1f}min  "
            f"notch={payload['power_line_hz']}Hz  "
            f"spike={payload['spike']}  noisy={payload['noisy']}  "
            f"dead={payload['dead']}  -> static={payload['static']}  -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
