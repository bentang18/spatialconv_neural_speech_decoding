"""Tests for ElectrodeValidMask.

Emits per-event ``(c_max,) bool`` valid-mask aligned to the VOLTAGE electrode
order (``BrainTreebankSubject.electrode_labels`` — same order as the DK support
extractor and the front-end tokens). Slot ``c`` is True iff voltage electrode
``c`` is mapped to a DK parcel (equivalently ``support[c]`` is nonzero); voltage
electrodes with no anatomy row / an out-of-vocab label are False **at their true
position**, and trailing padding slots are False. The encoder consumes
``effective_support = support * valid_mask`` and ``drop_electrode = ~valid_mask``,
so per-row alignment with ``support`` is load-bearing (C1/C2 fix).

Cohort C_MAX default = 384 (CQ12 / B14 lock 2026-05-23 PM; covers D-cohort
max 366, AJILE12 ~200, BT 256, SWEC 128 with headroom). See ElectrodeValidMask.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speech_decoding.extractors.valid_mask import ElectrodeValidMask


def _write_bt(
    bt_root: Path,
    subject_id: int,
    *,
    voltage: list[str],
    anatomy: list[tuple[str, str]],
) -> None:
    labels_path = (
        bt_root / "electrode_labels" / f"sub_{subject_id}" / "electrode_labels.json"
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(list(voltage)))

    csv_path = bt_root / "localization" / f"sub_{subject_id}" / "depth-wm.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as f:
        f.write("Electrode,DesikanKilliany\n")
        for electrode, label in anatomy:
            f.write(f"{electrode},{label}\n")


def test_valid_mask_shape_and_dtype(tmp_path: Path) -> None:
    _write_bt(
        tmp_path, 1,
        voltage=["E1", "E2", "E3"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "ctx-rh-bankssts"),
            ("E3", "Left-Hippocampus"),
        ],
    )
    ext = ElectrodeValidMask(event_types="Ieeg", bt_root=str(tmp_path), c_max=120)
    out = ext.get_static(SimpleNamespace(subject="1"))  # type: ignore[arg-type]

    assert isinstance(out, torch.Tensor)
    assert out.shape == (120,)
    assert out.dtype == torch.bool


def test_valid_mask_mapped_true_padding_false(tmp_path: Path) -> None:
    """All voltage electrodes mapped -> first ``n_voltage`` True, rest False."""
    _write_bt(
        tmp_path, 2,
        voltage=["E1", "E2", "E3", "E4"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "ctx-rh-bankssts"),
            ("E3", "Left-Hippocampus"),
            ("E4", "ctx-lh-precentral"),
        ],
    )
    ext = ElectrodeValidMask(event_types="Ieeg", bt_root=str(tmp_path), c_max=120)
    out = ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]

    assert out[:4].all().item() is True
    assert (~out[4:]).all().item() is True


def test_valid_mask_interior_unmapped_false_in_place(tmp_path: Path) -> None:
    """C1 regression: an interior out-of-vocab electrode is False at its true
    voltage index, and the electrodes after it stay True (no re-pack)."""
    _write_bt(
        tmp_path, 2,
        voltage=["E1", "E2", "E3", "E4"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "Left-Inf-Lat-Vent"),  # interior, out-of-vocab
            ("E3", "ctx-rh-bankssts"),
            ("E4", "ctx-lh-precentral"),
        ],
    )
    ext = ElectrodeValidMask(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=10,
        unmapped_policy="zero",
    )
    out = ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]

    assert out[0].item() is True
    assert out[1].item() is False  # E2 unmapped, in place
    assert out[2].item() is True   # E3 still True at index 2 (not shifted to 1)
    assert out[3].item() is True
    assert (~out[4:]).all().item() is True


def test_valid_mask_aligns_with_support_row_for_row(tmp_path: Path) -> None:
    """``valid_mask[c]`` must equal ``support[c].any()`` for every voltage row —
    the encoder relies on this (``effective_support = support * valid_mask``)."""
    from speech_decoding.extractors.dk_support import V14DKHardSupportExtractor

    voltage = ["E1", "E2", "E3", "E4", "E5"]
    anatomy = [
        ("E1", "ctx-lh-superiortemporal"),
        ("E2", "Left-Inf-Lat-Vent"),   # out-of-vocab -> unmapped
        ("E3", "ctx-rh-bankssts"),
        # E4 has no anatomy row -> unmapped
        ("E5", "ctx-lh-precentral"),
    ]
    _write_bt(tmp_path, 2, voltage=voltage, anatomy=anatomy)

    c_max = 8
    vm = ElectrodeValidMask(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=c_max,
        unmapped_policy="zero",
    ).get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]
    sup = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=c_max,
        unmapped_policy="zero",
    ).get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]

    support_nonzero = sup.sum(dim=1) > 0
    torch.testing.assert_close(vm, support_nonzero)
    assert vm.tolist() == [True, False, True, False, True, False, False, False]


def test_valid_mask_c_max_below_count_raises(tmp_path: Path) -> None:
    _write_bt(
        tmp_path, 3,
        voltage=["E1", "E2"],
        anatomy=[("E1", "ctx-lh-superiortemporal"), ("E2", "ctx-rh-bankssts")],
    )
    ext = ElectrodeValidMask(event_types="Ieeg", bt_root=str(tmp_path), c_max=1)
    with pytest.raises(ValueError, match="exceeds c_max"):
        ext.get_static(SimpleNamespace(subject="3"))  # type: ignore[arg-type]


def test_valid_mask_raise_policy_default_on_unknown_label(tmp_path: Path) -> None:
    """Default ``unmapped_policy='raise'`` surfaces unknown labels loudly —
    matches V14DKHardSupportExtractor default to keep the two extractors aligned."""
    _write_bt(
        tmp_path, 5,
        voltage=["E1", "E2"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "Left-Inf-Lat-Vent"),
        ],
    )
    ext = ElectrodeValidMask(event_types="Ieeg", bt_root=str(tmp_path), c_max=10)
    assert ext.unmapped_policy == "raise"
    with pytest.raises(KeyError, match="absent from parcel vocabulary"):
        ext.get_static(SimpleNamespace(subject="5"))  # type: ignore[arg-type]


def test_valid_mask_subject_coercion_btbank_prefix(tmp_path: Path) -> None:
    _write_bt(tmp_path, 7, voltage=["E1"], anatomy=[("E1", "ctx-lh-precentral")])
    ext = ElectrodeValidMask(event_types="Ieeg", bt_root=str(tmp_path), c_max=5)
    out = ext.get_static(SimpleNamespace(subject="btbank7"))  # type: ignore[arg-type]
    assert out.shape == (5,)
    assert out[0].item() is True
    assert (~out[1:]).all().item() is True


def test_electrode_set_defaults_to_all(tmp_path: Path) -> None:
    """Default ``electrode_set='all'`` — byte-identical to the pre-Lite extractor."""
    _write_bt(tmp_path, 6, voltage=["E1"], anatomy=[("E1", "ctx-lh-precentral")])
    ext = ElectrodeValidMask(event_types="Ieeg", bt_root=str(tmp_path), c_max=4)
    assert ext.electrode_set == "all"
    out = ext.get_static(SimpleNamespace(subject="6"))  # type: ignore[arg-type]
    assert out[0].item() is True


def test_electrode_set_lite_aligns_to_lite_montage(tmp_path: Path, monkeypatch) -> None:
    """``electrode_set='lite'`` aligns the mask to the PRE-CAR Lite montage
    (``lite_voltage_order``), NOT a pool-side AND-in. The loader subsets voltage
    rows to the Lite set before CAR, so the mask has ONE row per Lite electrode
    (packed, in voltage order): non-Lite electrodes are GONE (not False-in-place).
    Every parcel-mapped Lite row is valid."""
    import speech_decoding.studies.braintreebank._neuroprobe_lite_tables as lt

    _write_bt(
        tmp_path, 2,
        voltage=["E1", "E2", "E3", "E4"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "ctx-rh-bankssts"),
            ("E3", "Left-Hippocampus"),
            ("E4", "ctx-lh-precentral"),
        ],
    )
    # Lite keeps E1 and E3 only (drop the interior E2 and the trailing E4).
    monkeypatch.setattr(lt, "NEUROPROBE_LITE_ELECTRODES", {"btbank2": ["E1", "E3"]})

    out = ElectrodeValidMask(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=8, electrode_set="lite",
    ).get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]

    # Lite montage = [E1, E3] (packed, voltage order); both mapped -> valid.
    # E2/E4 are not rows at all (dropped pre-CAR). Rows 2.. are pad-False.
    assert out.tolist() == [True, True, False, False, False, False, False, False]


def test_electrode_set_lite_unmapped_stays_false(tmp_path: Path, monkeypatch) -> None:
    """An unmapped (no parcel) electrode IN the Lite montage occupies its row but
    is False — the Lite subset only removes non-Lite electrodes, it never
    resurrects an invalid slot."""
    import speech_decoding.studies.braintreebank._neuroprobe_lite_tables as lt

    _write_bt(
        tmp_path, 2,
        voltage=["E1", "E2", "E3"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "Left-Inf-Lat-Vent"),  # out-of-vocab -> unmapped
            ("E3", "ctx-rh-bankssts"),
        ],
    )
    # Lite lists E2 (unmapped) and E3 (mapped); E1 dropped by Lite (non-Lite).
    monkeypatch.setattr(lt, "NEUROPROBE_LITE_ELECTRODES", {"btbank2": ["E2", "E3"]})

    out = ElectrodeValidMask(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=6, electrode_set="lite",
        unmapped_policy="zero",
    ).get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]

    # Lite montage = [E2, E3] (E1 dropped pre-CAR). Row0 E2 unmapped -> False;
    # row1 E3 mapped -> True. Rows 2.. pad-False.
    assert out.tolist() == [False, True, False, False, False, False]


def test_electrode_set_lite_cache_uid_distinct_from_all(tmp_path: Path) -> None:
    """The Lite extractor must not collide with the 'all' extractor in exca's
    config-keyed cache (different electrode_set -> different serialized config)."""
    all_ext = ElectrodeValidMask(event_types="Ieeg", bt_root=str(tmp_path), c_max=8)
    lite_ext = ElectrodeValidMask(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=8, electrode_set="lite",
    )
    assert all_ext.model_dump() != lite_ext.model_dump()
    assert all_ext.model_dump()["electrode_set"] == "all"


def test_valid_mask_missing_depth_wm_raises(tmp_path: Path) -> None:
    labels_path = tmp_path / "electrode_labels" / "sub_99" / "electrode_labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(["E1"]))  # voltage ok, anatomy missing
    ext = ElectrodeValidMask(event_types="Ieeg", bt_root=str(tmp_path), c_max=10)
    with pytest.raises(FileNotFoundError, match="depth-wm.csv"):
        ext.get_static(SimpleNamespace(subject="99"))  # type: ignore[arg-type]
