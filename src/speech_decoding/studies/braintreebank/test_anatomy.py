from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from speech_decoding.studies.braintreebank.anatomy import (
    DEFAULT_SUPPORT_BIAS_EPS,
    build_hard_public_bt_label_support,
    bt_label_vocabulary,
    load_public_bt_anatomy,
    support_attention_bias,
)


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
