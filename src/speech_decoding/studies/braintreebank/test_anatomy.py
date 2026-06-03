from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from speech_decoding.studies.braintreebank.anatomy import (
    DEFAULT_SUPPORT_BIAS_EPS,
    V14_DK_PARCEL_LABELS,
    aligned_voltage_support,
    build_hard_public_bt_label_support,
    bt_label_vocabulary,
    load_public_bt_anatomy,
    support_attention_bias,
    voltage_electrode_order,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BT_CACHE = _REPO_ROOT / ".cache" / "braintreebank"
_NEUROPROBE_UPSTREAM = _REPO_ROOT / ".cache" / "neuroprobe_upstream"
_VENDORED_SUBJECTS = tuple(range(1, 11))


def test_load_public_bt_anatomy_cleans_electrode_labels(tmp_path: Path) -> None:
    path = tmp_path / "localization" / "sub_1"
    path.mkdir(parents=True)
    (path / "depth-wm.csv").write_text(
        "Electrode,DesikanKilliany,Hemisphere\n"
        "A*1,superiortemporal,L\n"
        "B#2,insula,L\n"
    )

    anatomy = load_public_bt_anatomy(tmp_path, 1)

    assert anatomy["Subject"].tolist() == ["sub_1", "sub_1"]
    assert anatomy["Electrode"].tolist() == ["A1", "B2"]
    assert anatomy["DesikanKilliany"].tolist() == ["superiortemporal", "insula"]


def test_bt_label_vocabulary_can_include_hemisphere() -> None:
    table = pd.DataFrame(
        {
            "Electrode": ["L1", "R1", "L2"],
            "Hemisphere": ["L", "R", "L"],
            "DesikanKilliany": ["insula", "insula", "superiortemporal"],
        }
    )

    assert bt_label_vocabulary([table]) == ("insula", "superiortemporal")
    assert bt_label_vocabulary([table], include_hemisphere=True) == (
        "L:insula",
        "L:superiortemporal",
        "R:insula",
    )


def test_build_hard_public_bt_label_support_is_one_hot_in_channel_order() -> None:
    anatomy = pd.DataFrame(
        {
            "Electrode": ["E1", "E2", "E3"],
            "DesikanKilliany": ["insula", "superiortemporal", "insula"],
        }
    )

    result = build_hard_public_bt_label_support(
        ["E3", "E1", "E2"],
        anatomy,
        ["insula", "superiortemporal"],
    )

    assert result.kind == "hard_public_bt_label"
    assert result.electrode_labels == ("E3", "E1", "E2")
    assert result.parcel_labels == ("insula", "superiortemporal")
    np.testing.assert_array_equal(
        result.support,
        np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_build_hard_public_bt_label_support_can_split_by_hemisphere() -> None:
    anatomy = pd.DataFrame(
        {
            "Electrode": ["L1", "R1"],
            "Hemisphere": ["L", "R"],
            "DesikanKilliany": ["insula", "insula"],
        }
    )

    result = build_hard_public_bt_label_support(
        ["L1", "R1"],
        anatomy,
        ["L:insula", "R:insula"],
        include_hemisphere=True,
    )

    np.testing.assert_array_equal(
        result.support,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_build_hard_public_bt_label_support_emits_valid_mask() -> None:
    anatomy = pd.DataFrame(
        {"Electrode": ["E1", "E2"], "DesikanKilliany": ["insula", "superiortemporal"]}
    )
    result = build_hard_public_bt_label_support(
        ["E1", "E2"], anatomy, ["insula", "superiortemporal"],
    )
    np.testing.assert_array_equal(result.valid, np.array([True, True]))


def test_build_hard_public_bt_label_support_rejects_missing_electrode() -> None:
    anatomy = pd.DataFrame(
        {"Electrode": ["E1"], "DesikanKilliany": ["insula"]}
    )

    with pytest.raises(KeyError, match="missing BT anatomy rows"):
        build_hard_public_bt_label_support(["E1", "E2"], anatomy, ["insula"])


def test_build_hard_public_bt_label_support_rejects_unknown_label() -> None:
    anatomy = pd.DataFrame(
        {"Electrode": ["E1"], "DesikanKilliany": ["unknown_region"]}
    )

    with pytest.raises(KeyError, match="absent from parcel vocabulary"):
        build_hard_public_bt_label_support(["E1"], anatomy, ["insula"])


def test_build_hard_support_zero_policy_keeps_unmapped_in_place() -> None:
    """``unmapped_policy='zero'`` -> zero row + valid=False at the true index for
    both a missing-anatomy electrode (E2) and an out-of-vocab label (E4); the
    surrounding rows keep their positions (no re-pack)."""
    anatomy = pd.DataFrame(
        {
            "Electrode": ["E1", "E3", "E4"],
            "DesikanKilliany": ["insula", "superiortemporal", "unknown_region"],
        }
    )
    result = build_hard_public_bt_label_support(
        ["E1", "E2", "E3", "E4"],
        anatomy,
        ["insula", "superiortemporal"],
        unmapped_policy="zero",
    )
    np.testing.assert_array_equal(
        result.support,
        np.array(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32
        ),
    )
    np.testing.assert_array_equal(
        result.valid, np.array([True, False, True, False])
    )


# --- Real-data guards over the vendored fixtures (skip if absent) ------------


def _require_vendored(subject_id: int) -> None:
    labels = _BT_CACHE / "electrode_labels" / f"sub_{subject_id}" / "electrode_labels.json"
    if not labels.exists():
        pytest.skip(f"vendored BT fixtures absent: {labels}")


def _upstream_electrode_labels_or_skip(subject_id: int) -> tuple[str, ...]:
    """Real ``BrainTreebankSubject.electrode_labels`` from the vendored upstream
    clone, pointed at the vendored fixtures. Skips if either is unavailable."""
    _require_vendored(subject_id)
    if not _NEUROPROBE_UPSTREAM.exists():
        pytest.skip(f"vendored neuroprobe_upstream absent: {_NEUROPROBE_UPSTREAM}")
    # config.ROOT_DIR is read from the env at import time; force it to the
    # vendored fixtures before the first neuroprobe import.
    os.environ["ROOT_DIR_BRAINTREEBANK"] = str(_BT_CACHE)
    if str(_NEUROPROBE_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_NEUROPROBE_UPSTREAM))
    try:
        from neuroprobe.braintreebank_subject import BrainTreebankSubject
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"neuroprobe upstream not importable: {exc}")
    subject = BrainTreebankSubject(
        subject_id, cache=False, coordinates_type="cortical"
    )
    return tuple(subject.electrode_labels)


@pytest.mark.parametrize("subject_id", _VENDORED_SUBJECTS)
def test_voltage_order_matches_upstream(subject_id: int) -> None:
    """Drift guard: our replicated ``voltage_electrode_order`` must equal the
    real upstream ``BrainTreebankSubject.electrode_labels`` (order + set) for
    every vendored subject. Catches any divergence in the corrupted / trigger /
    missing-coordinate filter copied from the pinned upstream."""
    expected = _upstream_electrode_labels_or_skip(subject_id)
    actual = voltage_electrode_order(str(_BT_CACHE), subject_id)
    assert actual == expected


def test_aligned_voltage_support_sub4_interior_unmapped() -> None:
    """sub_4 has 2 voltage contacts (Inf-Lat-Vent) outside K=80 that sit in the
    interior of the voltage order. ``unmapped_policy='zero'`` must zero exactly
    those rows + set valid=False at their true positions, leaving all others
    mapped (real-data C1 regression)."""
    _require_vendored(4)
    result = aligned_voltage_support(
        str(_BT_CACHE), 4,
        parcel_labels=V14_DK_PARCEL_LABELS,
        unmapped_policy="zero",
    )
    n_voltage = len(result.electrode_labels)
    n_mapped = int(result.valid.sum())
    assert n_voltage == 183
    assert n_mapped == 181  # exactly 2 unmapped

    unmapped_idx = np.flatnonzero(~result.valid)
    assert unmapped_idx.tolist() == [
        result.electrode_labels.index("LT2bHb3"),
        result.electrode_labels.index("LT2bHb4"),
    ]
    # the unmapped rows are interior (not a trailing block) and zeroed
    assert unmapped_idx.max() < n_voltage - 1
    assert result.support[unmapped_idx].sum() == 0.0
    # valid[c] <=> support[c] nonzero, for every row
    np.testing.assert_array_equal(result.valid, result.support.sum(axis=1) > 0)



def test_support_attention_bias_is_log_support_plus_eps() -> None:
    support = np.array([[1.0, 0.0]], dtype=np.float32)

    bias = support_attention_bias(support, eps=1e-3)

    np.testing.assert_allclose(
        bias,
        np.log(np.array([[1.001, 0.001]], dtype=np.float32)),
        rtol=1e-6,
    )


def test_support_attention_bias_rejects_negative_support() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        support_attention_bias(np.array([[-1.0]], dtype=np.float32))


def test_support_attention_bias_default_eps_is_v14_prior_strength() -> None:
    assert DEFAULT_SUPPORT_BIAS_EPS == 1e-2
    bias = support_attention_bias(np.array([[1.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(
        bias,
        np.log(np.array([[1.01, 0.01]], dtype=np.float32)),
        rtol=1e-6,
    )
