"""Tests for V14DKHardSupportExtractor.

The extractor emits ``(n_voltage, K=80)`` one-hot support over the canonical v14
DK parcel vocabulary, consumed by the v14 encoder's hard block-diagonal
per-parcel pool (the one-hot assignment IS the routing; B36 replaced the
``log(support+eps)`` cross-attn bias).

**Alignment contract (C1/C2):** row ``c`` describes the same physical electrode
as ``electrode_tokens[c]`` — i.e. the VOLTAGE order
``BrainTreebankSubject.electrode_labels`` (cleaned ``electrode_labels.json``
order, corrupted/trigger/missing-coord electrodes removed), NOT the independent
``depth-wm.csv`` row order. Voltage electrodes with no anatomy row or an
out-of-vocab DK label get a zero support row **in place** under
``unmapped_policy="zero"`` (no re-pack — positions never shift).

Tests cover the voltage-order contract, interior-unmapped handling,
corrupted/trigger filtering, strict-mode raises, and subject-id coercion. A
guarded section validates ``voltage_electrode_order`` against the real upstream
``BrainTreebankSubject`` over the vendored fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from speech_decoding.extractors.dk_support import V14DKHardSupportExtractor
from speech_decoding.studies.braintreebank.anatomy import V14_DK_PARCEL_LABELS

_PARCEL_INDEX = {label: i for i, label in enumerate(V14_DK_PARCEL_LABELS)}


def _write_electrode_labels(
    bt_root: Path, subject_id: int, labels: list[str]
) -> Path:
    """Write ``electrode_labels.json`` (voltage channel order)."""
    path = bt_root / "electrode_labels" / f"sub_{subject_id}" / "electrode_labels.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(labels)))
    return path


def _write_depth_wm(
    bt_root: Path, subject_id: int, rows: list[tuple[str, str]]
) -> Path:
    """rows = [(electrode_label, dk_label), ...]; depth-wm row order is arbitrary."""
    path = bt_root / "localization" / f"sub_{subject_id}" / "depth-wm.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("Electrode,DesikanKilliany\n")
        for electrode, label in rows:
            f.write(f"{electrode},{label}\n")
    return path


def _write_corrupted(bt_root: Path, mapping: dict[str, list[str]]) -> Path:
    path = bt_root / "corrupted_elec.json"
    path.write_text(json.dumps(mapping))
    return path


def _write_bt(
    bt_root: Path,
    subject_id: int,
    *,
    voltage: list[str],
    anatomy: list[tuple[str, str]],
) -> None:
    _write_electrode_labels(bt_root, subject_id, voltage)
    _write_depth_wm(bt_root, subject_id, anatomy)


def test_dk_extractor_emits_one_hot_support_80(tmp_path: Path) -> None:
    _write_bt(
        tmp_path, 1,
        voltage=["E1", "E2", "E3"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "ctx-rh-bankssts"),
            ("E3", "Left-Hippocampus"),
        ],
    )

    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    out = ext.get_static(SimpleNamespace(subject="1"))  # type: ignore[arg-type]

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 80)
    assert out.dtype == torch.float32
    np.testing.assert_array_equal(out.sum(dim=1).numpy(), np.ones(3, dtype=np.float32))
    assert out[0, _PARCEL_INDEX["ctx-lh-superiortemporal"]] == 1.0
    assert out[1, _PARCEL_INDEX["ctx-rh-bankssts"]] == 1.0
    assert out[2, _PARCEL_INDEX["Left-Hippocampus"]] == 1.0


def test_dk_extractor_follows_voltage_order_not_depth_wm(tmp_path: Path) -> None:
    """Row order follows ``electrode_labels.json`` (voltage), even when the
    ``depth-wm.csv`` rows are in a different order — this is the C2 fix."""
    _write_bt(
        tmp_path, 6,
        voltage=["E_first", "E_second", "E_third"],
        # depth-wm deliberately in a DIFFERENT order
        anatomy=[
            ("E_third", "Right-Amygdala"),
            ("E_first", "ctx-rh-precentral"),
            ("E_second", "ctx-lh-postcentral"),
        ],
    )
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    out = ext.get_static(SimpleNamespace(subject="6"))  # type: ignore[arg-type]

    assert out.argmax(dim=1).tolist() == [
        _PARCEL_INDEX["ctx-rh-precentral"],   # E_first
        _PARCEL_INDEX["ctx-lh-postcentral"],  # E_second
        _PARCEL_INDEX["Right-Amygdala"],      # E_third
    ]


def test_dk_extractor_drops_corrupted_and_trigger_from_voltage(tmp_path: Path) -> None:
    """Voltage order excludes corrupted (``corrupted_elec.json``) and trigger
    (``DC*``/``TRIG*``) channels — exactly as upstream
    ``BrainTreebankSubject.electrode_labels`` does (C2)."""
    _write_bt(
        tmp_path, 8,
        voltage=["E1", "DC1", "E2", "TRIG3", "E3"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "ctx-rh-bankssts"),
            ("E3", "Left-Hippocampus"),
        ],
    )
    _write_corrupted(tmp_path, {"sub_8": ["E3"]})  # E3 corrupted -> dropped too
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    out = ext.get_static(SimpleNamespace(subject="8"))  # type: ignore[arg-type]

    # DC1 (trigger), TRIG3 (trigger), E3 (corrupted) dropped -> only E1, E2 remain
    assert out.shape == (2, 80)
    assert out[0, _PARCEL_INDEX["ctx-lh-superiortemporal"]] == 1.0
    assert out[1, _PARCEL_INDEX["ctx-rh-bankssts"]] == 1.0


def test_dk_extractor_cleans_bt_star_hash_electrode_suffixes(tmp_path: Path) -> None:
    _write_bt(
        tmp_path, 1,
        voltage=["LT2bHb3*", "F3a#1"],
        anatomy=[
            ("LT2bHb3", "ctx-lh-superiortemporal"),
            ("F3a1", "ctx-rh-insula"),
        ],
    )
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    out = ext.get_static(SimpleNamespace(subject="1"))  # type: ignore[arg-type]
    assert out.shape == (2, 80)
    assert out[0, _PARCEL_INDEX["ctx-lh-superiortemporal"]] == 1.0
    assert out[1, _PARCEL_INDEX["ctx-rh-insula"]] == 1.0


def test_dk_extractor_strict_raises_on_label_outside_v14_vocab(tmp_path: Path) -> None:
    """``Left-Inf-Lat-Vent`` appears in BT btbank4 but is intentionally excluded
    from K=80 — strict default (``unmapped_policy='raise'``) must raise, not
    silently zero the row."""
    _write_bt(
        tmp_path, 2,
        voltage=["LT2bHb3", "LT2bHb4", "LT2bHb5"],
        anatomy=[
            ("LT2bHb3", "Left-Inf-Lat-Vent"),
            ("LT2bHb4", "Left-Inf-Lat-Vent"),
            ("LT2bHb5", "Left-Hippocampus"),
        ],
    )
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    assert ext.unmapped_policy == "raise"
    with pytest.raises(KeyError, match="absent from parcel vocabulary"):
        ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]


def test_dk_extractor_strict_raises_on_voltage_without_anatomy_row(tmp_path: Path) -> None:
    """A voltage electrode with no ``depth-wm.csv`` row is a hard error under the
    strict default — silently zeroing it would hide a localization gap."""
    _write_bt(
        tmp_path, 2,
        voltage=["E1", "E2"],
        anatomy=[("E1", "ctx-lh-superiortemporal")],  # E2 has no anatomy row
    )
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    with pytest.raises(KeyError, match="missing BT anatomy rows"):
        ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]


def test_dk_extractor_zero_policy_keeps_interior_unmapped_in_place(tmp_path: Path) -> None:
    """C1 regression: an INTERIOR out-of-vocab electrode gets a zero row at its
    true voltage index — rows after it are NOT shifted (no re-pack)."""
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
    ext = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), unmapped_policy="zero",
    )
    out = ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]

    assert out.shape == (4, 80)  # NOT (3, 80) — nothing dropped
    assert out[1].sum().item() == 0.0  # E2 zeroed in place
    # E3/E4 stay at indices 2/3 (not shifted up to 1/2)
    assert out[0, _PARCEL_INDEX["ctx-lh-superiortemporal"]] == 1.0
    assert out[2, _PARCEL_INDEX["ctx-rh-bankssts"]] == 1.0
    assert out[3, _PARCEL_INDEX["ctx-lh-precentral"]] == 1.0


def test_dk_extractor_zero_policy_handles_missing_anatomy_row(tmp_path: Path) -> None:
    """Under ``unmapped_policy='zero'`` a voltage electrode absent from
    ``depth-wm.csv`` also becomes a zero row in place (no raise, no shift)."""
    _write_bt(
        tmp_path, 2,
        voltage=["E1", "E2", "E3"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E3", "ctx-rh-bankssts"),  # E2 has no anatomy row
        ],
    )
    ext = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), unmapped_policy="zero",
    )
    out = ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]
    assert out.shape == (3, 80)
    assert out[1].sum().item() == 0.0
    assert out[0, _PARCEL_INDEX["ctx-lh-superiortemporal"]] == 1.0
    assert out[2, _PARCEL_INDEX["ctx-rh-bankssts"]] == 1.0


def test_dk_extractor_c_max_pads_in_voltage_order(tmp_path: Path) -> None:
    _write_bt(
        tmp_path, 2,
        voltage=["E1", "E2"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "ctx-rh-bankssts"),
        ],
    )
    ext = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=5,
    )
    out = ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]
    assert out.shape == (5, 80)
    assert out[:2].sum().item() == 2.0
    assert out[2:].sum().item() == 0.0  # padding rows all-zero


def test_dk_extractor_c_max_below_count_raises(tmp_path: Path) -> None:
    _write_bt(
        tmp_path, 2,
        voltage=["E1", "E2", "E3"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "ctx-rh-bankssts"),
            ("E3", "Left-Hippocampus"),
        ],
    )
    ext = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=2,
    )
    with pytest.raises(ValueError, match="exceeds c_max"):
        ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]


def test_dk_extractor_raises_on_missing_electrode_labels(tmp_path: Path) -> None:
    # depth-wm present but no electrode_labels.json -> voltage order undefined
    _write_depth_wm(tmp_path, 99, rows=[("E1", "ctx-lh-superiortemporal")])
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="electrode_labels"):
        ext.get_static(SimpleNamespace(subject="99"))  # type: ignore[arg-type]


def test_dk_extractor_raises_on_missing_depth_wm(tmp_path: Path) -> None:
    _write_electrode_labels(tmp_path, 99, ["E1"])  # voltage ok, anatomy missing
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="depth-wm.csv"):
        ext.get_static(SimpleNamespace(subject="99"))  # type: ignore[arg-type]


def test_dk_extractor_handles_integer_event_subject(tmp_path: Path) -> None:
    _write_bt(
        tmp_path, 7,
        voltage=["E1"], anatomy=[("E1", "ctx-lh-superiortemporal")],
    )
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    out_str = ext.get_static(SimpleNamespace(subject="7"))  # type: ignore[arg-type]
    out_int = ext.get_static(SimpleNamespace(subject=7))  # type: ignore[arg-type]
    torch.testing.assert_close(out_str, out_int)


def test_dk_extractor_accepts_btbank_prefixed_subject(tmp_path: Path) -> None:
    _write_bt(tmp_path, 2, voltage=["E1"], anatomy=[("E1", "ctx-rh-insula")])
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    out_plain = ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]
    out_prefixed = ext.get_static(SimpleNamespace(subject="btbank2"))  # type: ignore[arg-type]
    torch.testing.assert_close(out_plain, out_prefixed)


def test_dk_extractor_accepts_study_qualified_subject(tmp_path: Path) -> None:
    _write_bt(tmp_path, 2, voltage=["E1"], anatomy=[("E1", "ctx-rh-insula")])
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    out_plain = ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]
    out_qualified = ext.get_static(  # type: ignore[arg-type]
        SimpleNamespace(subject="Wang2024Treebank/btbank2"),
    )
    torch.testing.assert_close(out_plain, out_qualified)


# --- single-electrode-parcel exclusion (#154, Ben 2026-06-13) --------------- #
def test_exclude_single_electrode_parcels_drops_lone_electrode(tmp_path: Path) -> None:
    """A parcel covered by exactly one electrode → that electrode's support row is
    zeroed (and it becomes invalid); a 2-electrode parcel is untouched."""
    _write_bt(
        tmp_path, 1,
        voltage=["E1", "E2", "E3"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal"),  # parcel A
            ("E2", "ctx-lh-superiortemporal"),  # parcel A (2 electrodes → kept)
            ("E3", "ctx-lh-insula"),            # parcel B (1 electrode → dropped)
        ],
    )
    base = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), unmapped_policy="zero",
    )
    excl = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), unmapped_policy="zero",
        exclude_single_electrode_parcels=True,
    )
    out_base = base.get_static(SimpleNamespace(subject=1))  # type: ignore[arg-type]
    out_excl = excl.get_static(SimpleNamespace(subject=1))  # type: ignore[arg-type]
    a = _PARCEL_INDEX["ctx-lh-superiortemporal"]
    b = _PARCEL_INDEX["ctx-lh-insula"]
    # baseline: all three electrodes mapped
    assert out_base[0, a] == 1.0 and out_base[1, a] == 1.0 and out_base[2, b] == 1.0
    # excluded: the 2-electrode parcel A rows stay; the lone parcel-B row is zeroed
    assert out_excl[0, a] == 1.0 and out_excl[1, a] == 1.0
    assert out_excl[2].sum() == 0.0, "lone-electrode parcel-B row not zeroed"
    # parcel A still covered by 2; parcel B now uncovered
    assert out_excl[:, a].sum() == 2.0
    assert out_excl[:, b].sum() == 0.0


def test_exclude_single_electrode_parcels_off_by_default(tmp_path: Path) -> None:
    _write_bt(
        tmp_path, 1,
        voltage=["E1", "E2"],
        anatomy=[("E1", "ctx-lh-superiortemporal"), ("E2", "ctx-lh-insula")],
    )
    ext = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), unmapped_policy="zero",
    )
    out = ext.get_static(SimpleNamespace(subject=1))  # type: ignore[arg-type]
    # both are single-electrode parcels but exclusion is OFF → both kept
    assert out[0].sum() == 1.0 and out[1].sum() == 1.0


def test_exclude_single_electrode_keeps_support_valid_mask_row_aligned(
    tmp_path: Path,
) -> None:
    """Under exclusion the support extractor's zeroed lone-electrode row and the
    ElectrodeValidMask's False flag must describe the SAME electrode row — the
    load-bearing ``effective_support = support * valid_mask`` invariant (``valid[c]``
    True iff support row ``c`` is nonzero) for EVERY electrode. Both extractors
    derive from one memoized ``aligned_voltage_support`` call, but nothing pinned
    that they stay row-aligned once the single-electrode drop fires. C1/C2 contract
    under the #154 drop. (audit gap, 2026-06-13)"""
    from speech_decoding.extractors.valid_mask import ElectrodeValidMask

    _write_bt(
        tmp_path, 1,
        voltage=["E1", "E2", "E3"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal"),  # parcel A (2 electrodes → kept)
            ("E2", "ctx-lh-superiortemporal"),
            ("E3", "ctx-lh-insula"),            # parcel B (1 electrode → dropped)
        ],
    )
    kw = dict(
        event_types="Ieeg", bt_root=str(tmp_path), unmapped_policy="zero",
        exclude_single_electrode_parcels=True,
    )
    support = V14DKHardSupportExtractor(**kw).get_static(
        SimpleNamespace(subject=1))  # type: ignore[arg-type]
    valid = ElectrodeValidMask(**kw).get_static(
        SimpleNamespace(subject=1))  # type: ignore[arg-type]
    n = support.shape[0]
    support_nonzero = support.sum(dim=1) > 0  # (n,) bool
    # The invariant the encoder relies on: valid[c] iff support row c survives.
    assert torch.equal(valid[:n], support_nonzero)
    # E3 (lone parcel-B electrode) is dropped on BOTH sides, in lockstep.
    assert support[2].sum() == 0.0 and not bool(valid[2])
    # The 2-electrode parcel A rows stay valid on both sides.
    assert bool(valid[0]) and bool(valid[1])
    # Padding slots past the real electrodes are False.
    assert not valid[n:].any()


# --- DKT atlas column (#155) ------------------------------------------------ #
def _write_dkt_depth_wm(
    bt_root: Path, subject_id: int, rows: list[tuple[str, str]]
) -> None:
    """Write a depth-wm.csv carrying the native DKT column (+ a DK column so the
    file is realistic; the extractor reads only the selected ``label_column``)."""
    path = bt_root / "localization" / f"sub_{subject_id}" / "depth-wm.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("Electrode,DesikanKilliany,DKT\n")
        for electrode, dkt_label in rows:
            f.write(f"{electrode},Unknown,{dkt_label}\n")


def test_dkt_atlas_routes_over_k74_vocabulary(tmp_path: Path) -> None:
    from speech_decoding.studies.braintreebank.anatomy import V14_DKT_PARCEL_LABELS
    _write_electrode_labels(tmp_path, 1, ["E1", "E2"])
    _write_dkt_depth_wm(tmp_path, 1, [
        ("E1", "ctx-lh-superiortemporal"),
        ("E2", "Left-Hippocampus"),
    ])
    ext = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), unmapped_policy="zero",
        label_column="DKT", parcel_labels=V14_DKT_PARCEL_LABELS,
    )
    out = ext.get_static(SimpleNamespace(subject=1))  # type: ignore[arg-type]
    assert out.shape == (2, 74), "DKT support must be K=74 wide"
    dkt_index = {l: i for i, l in enumerate(V14_DKT_PARCEL_LABELS)}
    assert out[0, dkt_index["ctx-lh-superiortemporal"]] == 1.0
    assert out[1, dkt_index["Left-Hippocampus"]] == 1.0
