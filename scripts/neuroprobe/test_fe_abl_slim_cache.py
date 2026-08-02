"""Slim enc0 cache — the invariants that keep a smaller cache the SAME cache.

A slim cache that drops rows, reorders units, or carries a band layout that does not describe
its own payload yields a complete, plausible ablation grid answering a different question. Each
test here is one of those silent failures made loud.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.neuroprobe.fe_abl_slim_cache import KEEP_KEYS, LITE_SESSIONS, slim
from scripts.neuroprobe.v3_board_readout import LITE_SESSIONS as READOUT_SESSIONS
from scripts.neuroprobe.v3_board_readout import BOARD_TASKS

BL, FD, WIDTH = (4, 16, 32), (7, 6, 7), 348


def _full(n=16, n_parcels=3, n_elec=5, width=WIDTH):
    rng = np.random.default_rng(0)
    def raw(u):
        return {"raw": torch.from_numpy(
            rng.normal(size=(n, u, width)).astype(np.float32)).to(torch.float16)}
    return {
        "subject_id": 3, "trial_id": 1, "ckpt_tag": "t",
        "present_parcels": np.arange(n_parcels), "parcel_canon": np.zeros(n_elec, dtype=np.int64),
        "band_lengths": np.asarray(BL), "band_fdims": np.asarray(FD),
        "clip_starts": np.arange(n), "labels": {t: np.zeros(n) for t in BOARD_TASKS},
        "ws_split": {}, "cs_split": {}, "n_windows": n,
        "feats": {"enc0": raw(n_parcels), "enc0_elec": raw(n_elec),
                  "enc12_elec": raw(n_elec), "enc12": raw(n_parcels)},
    }


def test_session_order_matches_the_readout() -> None:
    """--index must mean the same session in both jobs, or shards pair the wrong cells."""
    assert LITE_SESSIONS == READOUT_SESSIONS


def test_enc0_payload_survives_bit_exactly_and_encoder_taps_are_dropped() -> None:
    full = _full()
    out = slim(full, elec_labels=np.array(["a", "b", "c", "d", "e"]))
    assert set(out["feats"]) == {"enc0", "enc0_elec"}
    for tap in ("enc0", "enc0_elec"):
        assert torch.equal(out["feats"][tap]["raw"], full["feats"][tap]["raw"])
        assert out["feats"][tap]["raw"].dtype == torch.float16
    for k in KEEP_KEYS:
        got, want = out[k], full[k]
        assert np.array_equal(got, want) if isinstance(want, np.ndarray) else got == want


def test_elec_labels_are_embedded_so_csession_needs_no_sidecar() -> None:
    out = slim(_full(), elec_labels=np.array(["a", "b", "c", "d", "e"]))
    assert len(out["elec_labels"]) == out["feats"]["enc0_elec"]["raw"].shape[1]


def test_a_layout_that_does_not_describe_the_payload_is_refused() -> None:
    full = _full()
    full["band_fdims"] = np.asarray((7, 7, 7))     # sums past 348
    with pytest.raises(SystemExit, match="does not describe this cache"):
        slim(full)


def test_row_count_disagreeing_with_n_windows_is_refused() -> None:
    full = _full()
    full["n_windows"] = 15                          # payload has 16
    with pytest.raises(SystemExit, match="rows != n_windows"):
        slim(full)


def test_wrong_sidecar_length_is_refused() -> None:
    with pytest.raises(SystemExit, match="!= enc0_elec electrodes"):
        slim(_full(), elec_labels=np.array(["a", "b"]))


def test_missing_enc0_is_refused() -> None:
    full = _full()
    del full["feats"]["enc0"]
    with pytest.raises(SystemExit, match="not an enc0-bearing board cache"):
        slim(full)


def test_slim_record_drives_the_readout_unchanged() -> None:
    """The schema contract: a slim record must fit the readout's own cell function."""
    from scripts.neuroprobe.test_v3_board_readout import _rec
    from scripts.neuroprobe.v3_board_readout import _ws_cell

    rec = _rec(n=64, n_parcels=3, feat=WIDTH)
    rec.update({"subject_id": 3, "trial_id": 1, "ckpt_tag": "t", "n_windows": 64,
                "parcel_canon": np.zeros(3, dtype=np.int64), "clip_starts": np.arange(64),
                "band_lengths": np.asarray(BL), "band_fdims": np.asarray(FD)})
    rec["feats"]["enc0_elec"] = rec["feats"]["enc0"]
    out = slim(rec)
    got = _ws_cell(out, BOARD_TASKS[0], ("enc0", "fm:hga:enc0"))["cells"]
    assert {"enc0|std", "fm:hga:enc0|std"} <= set(got)
