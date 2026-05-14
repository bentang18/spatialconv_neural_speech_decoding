"""Tests for V14DKHardSupportExtractor.

The extractor reads BT-shipped ``localization/sub_<id>/depth-wm.csv`` and emits
``(n_electrodes, K=80)`` one-hot support over the canonical v14 DK parcel
vocabulary, consumed by the v14 encoder cross-attn ``log(support+eps)`` bias.

Tests cover happy path on synthetic anatomy, strict-mode behavior on labels
that fall outside the K=80 vocabulary (e.g. BT btbank4 has ``Left-Inf-Lat-Vent``
electrodes that the v14 vocabulary intentionally excludes), and IO-error paths.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from speech_decoding.extractors.dk_support import V14DKHardSupportExtractor
from speech_decoding.studies.braintreebank.anatomy import V14_DK_PARCEL_LABELS


def _write_depth_wm(
    bt_root: Path, subject_id: int, rows: list[tuple[str, str]]
) -> Path:
    """rows = [(electrode_label, dk_label), ...]"""
    path = bt_root / "localization" / f"sub_{subject_id}" / "depth-wm.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("Electrode,DesikanKilliany\n")
        for electrode, label in rows:
            f.write(f"{electrode},{label}\n")
    return path


def test_dk_extractor_emits_one_hot_support_80(tmp_path: Path) -> None:
    _write_depth_wm(
        tmp_path,
        subject_id=1,
        rows=[
            ("E1", "ctx-lh-superiortemporal"),
            ("E2", "ctx-rh-bankssts"),
            ("E3", "Left-Hippocampus"),
        ],
    )

    ext = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=str(tmp_path),
    )
    event = SimpleNamespace(subject="1")
    out = ext.get_static(event)  # type: ignore[arg-type]

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 80)
    assert out.dtype == torch.float32

    # exactly one parcel hit per electrode
    np.testing.assert_array_equal(out.sum(dim=1).numpy(), np.ones(3, dtype=np.float32))

    parcel_index = {label: i for i, label in enumerate(V14_DK_PARCEL_LABELS)}
    assert out[0, parcel_index["ctx-lh-superiortemporal"]] == 1.0
    assert out[1, parcel_index["ctx-rh-bankssts"]] == 1.0
    assert out[2, parcel_index["Left-Hippocampus"]] == 1.0


def test_dk_extractor_preserves_depth_wm_electrode_order(tmp_path: Path) -> None:
    """Output row order matches the CSV row order (which downstream alignment uses)."""
    rows = [
        ("E_third", "ctx-lh-postcentral"),
        ("E_first", "ctx-rh-precentral"),
        ("E_second", "Right-Amygdala"),
    ]
    _write_depth_wm(tmp_path, subject_id=2, rows=rows)
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    out = ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]

    parcel_index = {label: i for i, label in enumerate(V14_DK_PARCEL_LABELS)}
    expected = [
        parcel_index["ctx-lh-postcentral"],
        parcel_index["ctx-rh-precentral"],
        parcel_index["Right-Amygdala"],
    ]
    assert out.argmax(dim=1).tolist() == expected


def test_dk_extractor_cleans_bt_star_hash_electrode_suffixes(tmp_path: Path) -> None:
    _write_depth_wm(
        tmp_path,
        subject_id=1,
        rows=[("LT2bHb3*", "ctx-lh-superiortemporal"), ("F3a#1", "ctx-rh-insula")],
    )
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    out = ext.get_static(SimpleNamespace(subject="1"))  # type: ignore[arg-type]
    assert out.shape == (2, 80)


def test_dk_extractor_strict_raises_on_label_outside_v14_vocab(tmp_path: Path) -> None:
    """``Left-Inf-Lat-Vent`` appears in BT btbank4 but is intentionally excluded
    from K=80 — the extractor must raise, not silently zero-out the row.
    Cohort-loading layer decides the policy (drop electrode or relax vocabulary)."""
    _write_depth_wm(
        tmp_path,
        subject_id=4,
        rows=[
            ("LT2bHb3", "Left-Inf-Lat-Vent"),
            ("LT2bHb4", "Left-Inf-Lat-Vent"),
            ("LT2bHb5", "Left-Hippocampus"),
        ],
    )
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    with pytest.raises(KeyError, match="absent from parcel vocabulary"):
        ext.get_static(SimpleNamespace(subject="4"))  # type: ignore[arg-type]


def test_dk_extractor_lenient_skip_unmapped(tmp_path: Path) -> None:
    """With ``unknown_label_policy='skip'``, electrodes whose label falls outside
    K=80 are dropped from the output. Loud diagnostic via logged record count."""
    _write_depth_wm(
        tmp_path,
        subject_id=4,
        rows=[
            ("LT2bHb3", "Left-Inf-Lat-Vent"),
            ("LT2bHb4", "Left-Hippocampus"),
            ("LT2bHb5", "Left-Inf-Lat-Vent"),
            ("LT2bHb6", "ctx-lh-superiortemporal"),
        ],
    )
    ext = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=str(tmp_path),
        unknown_label_policy="skip",
    )
    out = ext.get_static(SimpleNamespace(subject="4"))  # type: ignore[arg-type]
    assert out.shape == (2, 80)
    assert out.sum().item() == 2.0


def test_dk_extractor_raises_on_missing_depth_wm(tmp_path: Path) -> None:
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="depth-wm.csv"):
        ext.get_static(SimpleNamespace(subject="99"))  # type: ignore[arg-type]


def test_dk_extractor_handles_integer_event_subject(tmp_path: Path) -> None:
    """BT subjects come in as ints in some NeuralSet pipelines; extractor should
    coerce. (BT subjects are ``btbank<N>``; some pipelines pass the int part.)"""
    _write_depth_wm(
        tmp_path, subject_id=7,
        rows=[("E1", "ctx-lh-superiortemporal")],
    )
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    out_str = ext.get_static(SimpleNamespace(subject="7"))  # type: ignore[arg-type]
    out_int = ext.get_static(SimpleNamespace(subject=7))  # type: ignore[arg-type]
    torch.testing.assert_close(out_str, out_int)


def test_dk_extractor_accepts_btbank_prefixed_subject(tmp_path: Path) -> None:
    """When pipelines pass ``btbank<N>``, extractor coerces to the BT path layout."""
    _write_depth_wm(
        tmp_path, subject_id=2,
        rows=[("E1", "ctx-rh-insula")],
    )
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    out_plain = ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]
    out_prefixed = ext.get_static(SimpleNamespace(subject="btbank2"))  # type: ignore[arg-type]
    torch.testing.assert_close(out_plain, out_prefixed)


def test_dk_extractor_accepts_study_qualified_subject(tmp_path: Path) -> None:
    """Wang2024Treebank emits ``Wang2024Treebank/btbank<N>`` as the canonical
    event.subject; extractor must strip the study prefix and coerce the tail."""
    _write_depth_wm(
        tmp_path, subject_id=2,
        rows=[("E1", "ctx-rh-insula")],
    )
    ext = V14DKHardSupportExtractor(event_types="Ieeg", bt_root=str(tmp_path))
    out_plain = ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]
    out_qualified = ext.get_static(  # type: ignore[arg-type]
        SimpleNamespace(subject="Wang2024Treebank/btbank2"),
    )
    torch.testing.assert_close(out_plain, out_qualified)
