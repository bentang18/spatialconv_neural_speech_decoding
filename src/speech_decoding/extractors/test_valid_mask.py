"""Tests for ElectrodeValidMask.

Emits per-event ``(c_max,) bool`` valid-mask from BT depth-wm.csv electrode
count. Required because NeuralSet's ``MneRaw._get_timed_array`` zero-pads
to channel-union dim but does NOT emit a sibling mask. The v14 encoder
consumes ``valid_mask`` to set ``-inf`` cross-attn bias for padding slots.

Cohort C_MAX default = 384 (CQ12 / B14 lock 2026-05-23 PM; covers D-cohort
max 366, AJILE12 ~200, BT 256, SWEC 128 with headroom). See ElectrodeValidMask.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speech_decoding.extractors.valid_mask import ElectrodeValidMask


def _write_depth_wm(
    bt_root: Path, subject_id: int, rows: list[tuple[str, str]]
) -> Path:
    path = bt_root / "localization" / f"sub_{subject_id}" / "depth-wm.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("Electrode,DesikanKilliany\n")
        for electrode, label in rows:
            f.write(f"{electrode},{label}\n")
    return path


def test_valid_mask_shape_and_dtype(tmp_path: Path) -> None:
    _write_depth_wm(
        tmp_path, subject_id=1,
        rows=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "ctx-rh-bankssts"),
            ("E3", "Left-Hippocampus"),
        ],
    )
    ext = ElectrodeValidMask(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=120,
    )
    out = ext.get_static(SimpleNamespace(subject="1"))  # type: ignore[arg-type]

    assert isinstance(out, torch.Tensor)
    assert out.shape == (120,)
    assert out.dtype == torch.bool


def test_valid_mask_first_n_true_rest_false(tmp_path: Path) -> None:
    """First ``n_electrodes`` slots True; remaining slots False."""
    _write_depth_wm(
        tmp_path, subject_id=2,
        rows=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "ctx-rh-bankssts"),
            ("E3", "Left-Hippocampus"),
            ("E4", "ctx-lh-precentral"),
        ],
    )
    ext = ElectrodeValidMask(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=120,
    )
    out = ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]

    assert out[:4].all().item() is True
    assert (~out[4:]).all().item() is True


def test_valid_mask_handles_c_max_below_electrode_count(tmp_path: Path) -> None:
    """Raise when an event has more electrodes than ``c_max`` — silent
    truncation would corrupt downstream alignment with support + tokens."""
    _write_depth_wm(
        tmp_path, subject_id=3,
        rows=[("E1", "ctx-lh-superiortemporal"), ("E2", "ctx-rh-bankssts")],
    )
    ext = ElectrodeValidMask(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=1,
    )
    with pytest.raises(ValueError, match="exceeds c_max"):
        ext.get_static(SimpleNamespace(subject="3"))  # type: ignore[arg-type]


def test_valid_mask_skip_policy_filters_btbank4_inf_lat_vent(tmp_path: Path) -> None:
    """``unknown_label_policy='skip'`` drops electrodes with labels outside
    K=80 (e.g. btbank4 ``Left-Inf-Lat-Vent``) so the mask aligns with the
    DK-support extractor when both run in 'skip' mode."""
    _write_depth_wm(
        tmp_path, subject_id=4,
        rows=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "Left-Inf-Lat-Vent"),  # outside K=80
            ("E3", "ctx-rh-bankssts"),
        ],
    )
    ext = ElectrodeValidMask(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=10,
        unknown_label_policy="skip",
    )
    out = ext.get_static(SimpleNamespace(subject="4"))  # type: ignore[arg-type]
    assert out[:2].all().item() is True
    assert (~out[2:]).all().item() is True


def test_valid_mask_raise_policy_default_on_unknown_label(tmp_path: Path) -> None:
    """Default ``unknown_label_policy='raise'`` surfaces unknown labels loudly.
    Matches V14DKHardSupportExtractor default to keep two extractors aligned."""
    _write_depth_wm(
        tmp_path, subject_id=5,
        rows=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "Left-Inf-Lat-Vent"),
        ],
    )
    ext = ElectrodeValidMask(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=10,
    )
    assert ext.unknown_label_policy == "raise"
    with pytest.raises(KeyError, match="absent from parcel vocabulary"):
        ext.get_static(SimpleNamespace(subject="5"))  # type: ignore[arg-type]


def test_valid_mask_subject_coercion_btbank_prefix(tmp_path: Path) -> None:
    """Accepts ``"btbank<N>"`` per the same coercion rule as DK support extractor."""
    _write_depth_wm(
        tmp_path, subject_id=7,
        rows=[("E1", "ctx-lh-precentral")],
    )
    ext = ElectrodeValidMask(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=5,
    )
    out = ext.get_static(SimpleNamespace(subject="btbank7"))  # type: ignore[arg-type]
    assert out.shape == (5,)
    assert out[0].item() is True
    assert (~out[1:]).all().item() is True


def test_valid_mask_missing_depth_wm_raises(tmp_path: Path) -> None:
    ext = ElectrodeValidMask(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=10,
    )
    with pytest.raises(FileNotFoundError, match="depth-wm.csv"):
        ext.get_static(SimpleNamespace(subject="99"))  # type: ignore[arg-type]
