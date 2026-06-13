"""Golden-snapshot regression over the RAW VOLTAGE CLIP the model ingests
(task #19, voltage tier; ledger ``reports/bt_alignment/data_pipeline_bug_ledger.md``).

The metadata tier (``test_golden_metadata.py``) freezes everything derivable
from anatomy alone (electrode order, parcel assignment, coverage) WITHOUT
touching voltage. This tier freezes the next layer down: the actual
voltage-clip bytes the front-end sees, for a fixed handful of word clips. It
exercises the parts that anatomy can't:

* ``bt_load_raw`` — including the **S9 resample** path (native 1024 Hz →
  2048 Hz via ``resample_poly``). A regression that drops/changes the resample,
  or that re-orders the loaded channels, moves these hashes.
* the **neural→voltage windowing**: ``onset_sample = round(est_idx/native * 2048)``.
  This is the exact crop production applies (``start = est_idx/native`` seconds,
  5 s window on the 2048 Hz stream, the production SSL clip). The S9-class failure — slicing a native
  ``est_idx`` against the wrong rate — lands the window on the wrong samples and
  changes the clip hash, even though nothing crashes.

WHY CAPTURE AT THE RAW-CLIP LAYER (not the front-end features): ``view.py`` /
the STFT front-end is live FE-2STFT WIP and changes often — a golden over its
output would break on every legitimate FE tweak. The raw windowed voltage is
the STABLE contract upstream of the FE, so this golden only moves when the data
path itself moves (which is exactly the S9-class bug surface we are locking).

RUN-WHERE-DATA-LIVES: this needs the real distributed h5 (``sub_1_trial000.h5``
etc.) and an importable neuroprobe — i.e. DCC, where ``ROOT_DIR_BRAINTREEBANK``
is set. It SKIPS cleanly anywhere the env var is unset or the h5 is absent
(GitHub CI, laptops). The golden is captured ON DCC against the real data —
which also validates the production cache-build path — and only the tiny hashes
are committed. Raw BT voltage is NEVER committed.

Regenerate (on DCC, against the real data) after an INTENTIONAL change::

    .venv/bin/python -m speech_decoding.studies.braintreebank.test_golden_voltage --update

then review the JSON diff before committing.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

# (subject_id, trial_id) targets. Subject 1 = native-2048 baseline (no resample);
# subject 9 = native-1024 → exercises the resample_poly S9 path (the S9-class case).
_VOLTAGE_TARGETS: tuple[tuple[int, int], ...] = ((1, 0), (9, 0))
_N_CLIPS = 8  # first N word events of the trial
_TARGET_RATE = 2048  # global rate every subject is resampled UP to
# Production SSL clip length: P1/P2/P3 ingest 5 s windows (dispatch DEFAULT_CLIP_LEN_S
# = 5.0; neural_lag_s defaults 0.0), so the window is [est_idx/native, +5 s]. Mirror
# that exact span so the golden locks the clip the model actually sees, not a proxy.
_CLIP_S = 5.0
_CLIP_LEN = round(_CLIP_S * _TARGET_RATE)  # 10240 samples on the 2048 Hz stream

_GOLDEN_PATH = Path(__file__).resolve().parent / "_golden_voltage.json"


def _sha(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _bt_root() -> Path | None:
    """Resolve the real BT data root from the env, or None (→ skip).

    Reads the env directly rather than importing ``neuroprobe.config`` so this
    module stays import-safe / collectable on laptops where the env is unset.
    """
    root = os.environ.get("ROOT_DIR_BRAINTREEBANK")
    if not root:
        return None
    p = Path(root)
    return p if p.is_dir() else None


def _h5_path(bt_root: Path, subject_id: int, trial_id: int) -> Path:
    return bt_root / f"sub_{subject_id}_trial{trial_id:03}.h5"


def _clip_sha(clip: np.ndarray) -> str:
    """Resample-noise-robust content hash of a voltage clip.

    Rounds to 2 dp before hashing: BT voltages are O(10-100) µV and the
    ``resample_poly`` path introduces only ~1e-3 absolute noise, so 2 dp kills
    cross-node float jitter while remaining FAR finer than any S9-class bug
    (wrong rate / wrong window / wrong channel order → wholly different samples).
    Subject 1 (no resample) hashes its exact h5 bytes regardless.
    """
    q = np.round(np.ascontiguousarray(clip, dtype=np.float64), 2).astype(np.float32)
    return _sha(np.ascontiguousarray(q).tobytes())


def _capture_target(bt_root: Path, subject_id: int, trial_id: int) -> dict:
    """Production-path raw-voltage-clip snapshot for one (subject, trial)."""
    # Lazy imports: these pull in neuroprobe (needs the env) — only when running.
    from neuroprobe.braintreebank_subject import BrainTreebankSubject

    from speech_decoding.studies.braintreebank.loader import bt_load_raw
    from speech_decoding.studies.braintreebank.manifest import (
        bt_subject_native_rate_hz,
    )
    from speech_decoding.studies.braintreebank.word_events import (
        _load_words_and_nonverbal,
    )

    bt = BrainTreebankSubject(
        subject_id=subject_id,
        cache=False,
        coordinates_type="cortical",
    )
    data, ch_names, sfreq = bt_load_raw(bt, trial_id=trial_id, subject_id=subject_id)
    data = np.asarray(data)
    native = int(bt_subject_native_rate_hz(subject_id))

    words_df, _ = _load_words_and_nonverbal(
        subject_id, trial_id, bt_root=bt_root, enrich=False
    )
    head = words_df.head(_N_CLIPS)

    clips = []
    for _, row in head.iterrows():
        est_idx = float(row["est_idx"])
        # The EXACT production crop: start = est_idx/native seconds, sliced on the
        # 2048 Hz post-resample stream. The native-rate division is the S9 guard.
        onset = int(round(est_idx / native * _TARGET_RATE))
        end = onset + _CLIP_LEN
        if onset < 0 or end > data.shape[1]:
            # First few clips sit near the start; a clip running off the end would
            # mean a windowing regression — record it explicitly rather than hide.
            clips.append(
                {
                    "est_idx": round(est_idx, 3),
                    "onset_sample": onset,
                    "word": str(row.get("full_word", "")),
                    "out_of_bounds": True,
                }
            )
            continue
        clip = data[:, onset:end]
        clips.append(
            {
                "est_idx": round(est_idx, 3),
                "onset_sample": onset,
                "word": str(row.get("full_word", "")),
                "clip_shape": list(clip.shape),
                "clip_sha": _clip_sha(clip),
            }
        )

    return {
        "native_rate": native,
        "sfreq": int(round(float(sfreq))),
        "n_ch": int(data.shape[0]),
        "n_samples_total": int(data.shape[1]),
        "ch_names_sha": _sha("\n".join(map(str, ch_names)).encode()),
        "clips": clips,
    }


def _build_golden(bt_root: Path) -> dict:
    golden: dict = {"_schema": 1, "n_clips": _N_CLIPS, "clip_len": _CLIP_LEN, "targets": {}}
    for sid, tid in _VOLTAGE_TARGETS:
        if not _h5_path(bt_root, sid, tid).is_file():
            continue  # subject/trial not present on this machine
        golden["targets"][f"sub{sid}_trial{tid}"] = _capture_target(bt_root, sid, tid)
    return golden


def _require_target(subject_id: int, trial_id: int) -> Path:
    bt_root = _bt_root()
    if bt_root is None:
        pytest.skip("ROOT_DIR_BRAINTREEBANK unset or not a dir (real BT data absent)")
    if not _h5_path(bt_root, subject_id, trial_id).is_file():
        pytest.skip(f"real h5 absent: {_h5_path(bt_root, subject_id, trial_id)}")
    return bt_root


def _load_golden() -> dict:
    if not _GOLDEN_PATH.is_file():
        pytest.skip(f"golden reference absent: {_GOLDEN_PATH} (capture on DCC with --update)")
    return json.loads(_GOLDEN_PATH.read_text())


@pytest.mark.parametrize("subject_id,trial_id", _VOLTAGE_TARGETS)
def test_voltage_clip_matches_golden(subject_id: int, trial_id: int) -> None:
    """First ``_N_CLIPS`` raw voltage clips (windowed at production onsets) ==
    frozen golden — locks bt_load_raw (incl. S9 resample), channel order, and
    the native-rate windowing math against silent drift."""
    bt_root = _require_target(subject_id, trial_id)
    golden = _load_golden()
    key = f"sub{subject_id}_trial{trial_id}"
    if key not in golden.get("targets", {}):
        pytest.skip(f"{key} not in golden")
    expected = golden["targets"][key]
    actual = _capture_target(bt_root, subject_id, trial_id)
    assert actual == expected, (
        f"{key}: raw voltage clip golden drifted.\n"
        f"  expected={json.dumps(expected, sort_keys=True)[:400]}\n"
        f"  actual  ={json.dumps(actual, sort_keys=True)[:400]}\n"
        f"If this change is INTENTIONAL, regenerate on DCC with --update and review."
    )


if __name__ == "__main__":
    import sys

    if "--update" not in sys.argv:
        print("usage: python -m ...test_golden_voltage --update  (run on DCC, env set)")
        raise SystemExit(2)
    bt_root = _bt_root()
    if bt_root is None:
        raise SystemExit("cannot update: ROOT_DIR_BRAINTREEBANK unset or not a dir")
    golden = _build_golden(bt_root)
    n = len(golden["targets"])
    if n == 0:
        raise SystemExit(f"cannot update: no target h5 found under {bt_root}")
    _GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"wrote {_GOLDEN_PATH} for {n} target(s) ({_GOLDEN_PATH.stat().st_size} bytes)")
