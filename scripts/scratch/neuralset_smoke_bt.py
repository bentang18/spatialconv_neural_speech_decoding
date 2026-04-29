"""Phase 0 smoke: NeuralSet API contract on a synthetic BraintreebankIeeg.

Goal: confirm that BraintreebankIeeg + IeegExtractor + Pulse + Segmenter return a
coherent batch shaped (4, n_ch, 2048) at 2048 Hz raw voltage. Validates the
plumbing — Segmenter slicing, channel-order pinning, Ieeg subclass override of
_read() — without depending on BT h5 data being on the laptop.

Self-contained: no `from speech_decoding...` imports. Survives the reorg
unchanged; deleted in Phase 3 once `studies/braintreebank/study.py` lands.

Run with the Phase-0 scratch venv:
    /tmp/neuralset_scratch/.venv/bin/python scripts/scratch/neuralset_smoke_bt.py
"""

from __future__ import annotations

import os
from pathlib import Path

# Keep exca cache out of the repo + project venv; tmp is fine for a smoke test.
os.environ.setdefault("CACHE_FOLDER", "/tmp/neuralset_smoke_cache")

import mne
import numpy as np
import pandas as pd
import torch

import neuralset as ns
from neuralset.events.etypes import Event, Ieeg
from neuralset.events.utils import standardize_events
from neuralset.extractors.base import BaseStatic

SUBJECT = "btbank2"
TRIAL = "trial_4"
N_CHANNELS = 128
SFREQ = 2048.0
RAW_DURATION_S = 20.0
TRIGGER_TIMES = [2.0, 5.0, 8.0, 11.0]
SEG_DURATION = 1.0
EXPECTED_SHAPE = (len(TRIGGER_TIMES), N_CHANNELS, int(SFREQ * SEG_DURATION))


class BraintreebankIeeg(Ieeg):
    """h5-backed Ieeg event for BrainTreebank — synthetic stub for Phase 0.

    `_read()` returns raw voltage at native 2048 Hz, no re-reference, matching
    Neuroprobe `__getitem__`. Re-ref + HG envelope are Stage-1 ablation cells,
    not loader behavior. Phase-3 production version replaces the synthetic
    array with a `BrainTreebankSubject.get_electrode_data()` call.
    """

    subject_num: int = 0

    def _read(self) -> mne.io.RawArray:
        rng = np.random.default_rng(self.subject_num)
        n_samples = int(RAW_DURATION_S * SFREQ)
        data = rng.standard_normal((N_CHANNELS, n_samples)).astype(np.float32) * 50e-6
        ch_names = [f"E{i:03d}" for i in range(N_CHANNELS)]
        info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="seeg")
        return mne.io.RawArray(data, info, verbose=False)


class V14ParcelMetadataExtractor(BaseStatic):
    """Per-channel (parcel_id, support, fsaverage_xyz) metadata for an Ieeg event.

    Output shape (n_channels, 5):
        [:, 0]   = Tier-1 parcel id (int, 0..n_tier1-1)
        [:, 1]   = support weight (float, [0,1])
        [:, 2:5] = fsaverage_xyz (mm)

    Synthetic cache in this smoke; Phase-3 production version reads from
    `support_cache_dir` keyed by `event.subject` (parity with the
    `ChannelPositions(neuro=ieeg)` pattern at NS extractors/test_neuro.py:898).
    """

    event_types: str = "Ieeg"

    def get_static(self, event: Event) -> torch.Tensor:
        rng = np.random.default_rng(hash(event.subject) & 0xFFFFFFFF)
        parcel_id = rng.integers(0, 15, size=N_CHANNELS).astype(np.float32)
        support = rng.uniform(0.0, 1.0, size=N_CHANNELS).astype(np.float32)
        xyz = rng.uniform(-60.0, 60.0, size=(N_CHANNELS, 3)).astype(np.float32)
        out = np.concatenate([parcel_id[:, None], support[:, None], xyz], axis=1)
        return torch.from_numpy(out)


def build_events() -> pd.DataFrame:
    ieeg_row = {
        "type": "BraintreebankIeeg",
        "start": 0.0,
        "duration": RAW_DURATION_S,
        "timeline": TRIAL,
        "subject": SUBJECT,
        "subject_num": 2,
        "filepath": f"synthetic:bt_{SUBJECT}_{TRIAL}",
        "frequency": SFREQ,
    }
    word_rows = [
        {
            "type": "Word",
            "start": float(t),
            "duration": 0.3,
            "timeline": TRIAL,
            "text": f"word_{i}",
        }
        for i, t in enumerate(TRIGGER_TIMES)
    ]
    return pd.DataFrame([ieeg_row, *word_rows])


def main() -> None:
    Path(os.environ["CACHE_FOLDER"]).mkdir(parents=True, exist_ok=True)

    events = standardize_events(build_events())
    print(f"events: {len(events)} rows ({(events.type == 'Word').sum()} Word triggers)")

    ieeg = ns.extractors.IeegExtractor(
        event_types="Ieeg",
        picks=("seeg",),
        apply_hilbert=False,
        filter=None,
        frequency=SFREQ,
        channel_order="original",
        scaler=None,
    )
    pulse = ns.extractors.Pulse(event_types="Word")
    metadata = V14ParcelMetadataExtractor(event_types="Ieeg")

    segmenter = ns.Segmenter(
        start=0.0,
        duration=SEG_DURATION,
        trigger_query="type=='Word'",
        extractors={"neural": ieeg, "metadata": metadata, "stim": pulse},
        drop_incomplete=True,
    )

    dset = segmenter.apply(events)
    print(f"segments: {len(dset)}")
    dset.prepare()

    loader = torch.utils.data.DataLoader(
        dset, batch_size=len(TRIGGER_TIMES), collate_fn=dset.collate_fn
    )
    batch = next(iter(loader))

    neural = batch.data["neural"].data
    metadata_t = batch.data["metadata"].data
    stim = batch.data["stim"].data
    print(f"neural:   shape={tuple(neural.shape)} dtype={neural.dtype}")
    print(f"metadata: shape={tuple(metadata_t.shape)} dtype={metadata_t.dtype}")
    print(f"stim:     shape={tuple(stim.shape)} dtype={stim.dtype}")

    assert tuple(neural.shape) == EXPECTED_SHAPE, (
        f"neural shape {tuple(neural.shape)} != expected {EXPECTED_SHAPE}"
    )
    assert torch.isfinite(neural).all(), "non-finite values in neural batch"
    expected_meta = (len(TRIGGER_TIMES), N_CHANNELS, 5)
    assert tuple(metadata_t.shape) == expected_meta, (
        f"metadata shape {tuple(metadata_t.shape)} != expected {expected_meta}"
    )
    parcel_ids = metadata_t[..., 0]
    support = metadata_t[..., 1]
    assert (parcel_ids >= 0).all() and (parcel_ids < 15).all(), "parcel ids out of range"
    assert (support >= 0).all() and (support <= 1).all(), "support out of [0,1]"
    assert tuple(stim.shape)[0] == len(TRIGGER_TIMES), "stim batch dim mismatch"

    print("\nOK — NeuralSet API contract holds for BraintreebankIeeg + V14ParcelMetadataExtractor.")


if __name__ == "__main__":
    main()
