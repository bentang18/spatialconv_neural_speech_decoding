"""Tests for V14ElectrodeCoordsExtractor.

Emits per-event ``(c_max, 3)`` float32 native ``(L, I, P)`` coords aligned to the
VOLTAGE electrode order (same order as the DK support / valid-mask extractors and
the front-end tokens). Row ``c`` is the coordinate of the same physical electrode
as ``support[c]`` / ``valid[c]``. The load-bearing invariant vs the support/valid
extractors: coords are PARCEL-INDEPENDENT — an unmapped (out-of-vocab) electrode
still carries its true coordinate at its true row (it is NOT zeroed like support),
because the relational PE needs every kept electrode's position regardless of
parcel membership. Trailing padding rows are zeros.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speech_decoding.extractors.electrode_coords import V14ElectrodeCoordsExtractor

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BT_CACHE = _REPO_ROOT / ".cache" / "braintreebank"
_HAS_BT = (_BT_CACHE / "localization" / "sub_1" / "depth-wm.csv").exists()


def _write_bt(
    bt_root: Path,
    subject_id: int,
    *,
    voltage: list[str],
    anatomy: list[tuple[str, str, float, float, float]],
) -> None:
    """Write synthetic BT voltage-order + depth-wm.csv (with L,I,P coord columns)."""
    labels_path = (
        bt_root / "electrode_labels" / f"sub_{subject_id}" / "electrode_labels.json"
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(list(voltage)))

    csv_path = bt_root / "localization" / f"sub_{subject_id}" / "depth-wm.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as f:
        f.write("Electrode,DesikanKilliany,L,I,P\n")
        for electrode, label, ll, ii, pp in anatomy:
            f.write(f"{electrode},{label},{ll},{ii},{pp}\n")


def test_coords_shape_and_dtype(tmp_path: Path) -> None:
    _write_bt(
        tmp_path, 1,
        voltage=["E1", "E2", "E3"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal", 1.0, 2.0, 3.0),
            ("E2", "ctx-rh-bankssts", 4.0, 5.0, 6.0),
            ("E3", "Left-Hippocampus", 7.0, 8.0, 9.0),
        ],
    )
    ext = V14ElectrodeCoordsExtractor(event_types="Ieeg", bt_root=str(tmp_path), c_max=120)
    out = ext.get_static(SimpleNamespace(subject="1"))  # type: ignore[arg-type]

    assert isinstance(out, torch.Tensor)
    assert out.shape == (120, 3)
    assert out.dtype == torch.float32


def test_coords_values_and_padding_zero(tmp_path: Path) -> None:
    """Coords land at their true voltage row; trailing rows are zero-padded."""
    _write_bt(
        tmp_path, 2,
        voltage=["E1", "E2", "E3"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal", 10.0, 11.0, 12.0),
            ("E2", "ctx-rh-bankssts", 20.0, 21.0, 22.0),
            ("E3", "ctx-lh-precentral", 30.0, 31.0, 32.0),
        ],
    )
    ext = V14ElectrodeCoordsExtractor(event_types="Ieeg", bt_root=str(tmp_path), c_max=6)
    out = ext.get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]

    torch.testing.assert_close(out[0], torch.tensor([10.0, 11.0, 12.0]))
    torch.testing.assert_close(out[1], torch.tensor([20.0, 21.0, 22.0]))
    torch.testing.assert_close(out[2], torch.tensor([30.0, 31.0, 32.0]))
    assert (out[3:] == 0).all().item()


def test_coords_unmapped_electrode_keeps_its_coordinate(tmp_path: Path) -> None:
    """THE load-bearing invariant: an interior out-of-vocab electrode (unmapped,
    valid=False in the support/valid extractors) STILL carries its true coordinate
    at its true row — coords are parcel-INDEPENDENT (not zeroed like support)."""
    from speech_decoding.extractors.valid_mask import ElectrodeValidMask

    voltage = ["E1", "E2", "E3", "E4"]
    anatomy = [
        ("E1", "ctx-lh-superiortemporal", 1.0, 1.0, 1.0),
        ("E2", "Left-Inf-Lat-Vent", 2.0, 2.0, 2.0),  # out-of-vocab -> unmapped
        ("E3", "ctx-rh-bankssts", 3.0, 3.0, 3.0),
        ("E4", "ctx-lh-precentral", 4.0, 4.0, 4.0),
    ]
    _write_bt(tmp_path, 2, voltage=voltage, anatomy=anatomy)

    c_max = 8
    coords = V14ElectrodeCoordsExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=c_max,
    ).get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]
    valid = ElectrodeValidMask(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=c_max,
        unmapped_policy="zero",
    ).get_static(SimpleNamespace(subject="2"))  # type: ignore[arg-type]

    # E2 is invalid (unmapped) but its coordinate is present + correct, in place.
    assert valid[1].item() is False
    torch.testing.assert_close(coords[1], torch.tensor([2.0, 2.0, 2.0]))
    # And every other row still names the same electrode as valid/support.
    torch.testing.assert_close(coords[0], torch.tensor([1.0, 1.0, 1.0]))
    torch.testing.assert_close(coords[2], torch.tensor([3.0, 3.0, 3.0]))
    torch.testing.assert_close(coords[3], torch.tensor([4.0, 4.0, 4.0]))


def test_coords_c_max_below_count_raises(tmp_path: Path) -> None:
    _write_bt(
        tmp_path, 3,
        voltage=["E1", "E2"],
        anatomy=[
            ("E1", "ctx-lh-superiortemporal", 0.0, 0.0, 0.0),
            ("E2", "ctx-rh-bankssts", 1.0, 1.0, 1.0),
        ],
    )
    ext = V14ElectrodeCoordsExtractor(event_types="Ieeg", bt_root=str(tmp_path), c_max=1)
    with pytest.raises(ValueError, match="exceeds c_max"):
        ext.get_static(SimpleNamespace(subject="3"))  # type: ignore[arg-type]


def test_coords_missing_coordinate_columns_raises(tmp_path: Path) -> None:
    """depth-wm.csv without L,I,P columns fails loud — the relational PE needs them."""
    labels_path = tmp_path / "electrode_labels" / "sub_9" / "electrode_labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(["E1"]))
    csv_path = tmp_path / "localization" / "sub_9" / "depth-wm.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("Electrode,DesikanKilliany\nE1,ctx-lh-precentral\n")  # no L,I,P

    ext = V14ElectrodeCoordsExtractor(event_types="Ieeg", bt_root=str(tmp_path), c_max=5)
    with pytest.raises(KeyError, match="missing coordinate columns"):
        ext.get_static(SimpleNamespace(subject="9"))  # type: ignore[arg-type]


def test_coords_missing_depth_wm_raises(tmp_path: Path) -> None:
    labels_path = tmp_path / "electrode_labels" / "sub_99" / "electrode_labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(["E1"]))  # voltage ok, anatomy missing
    ext = V14ElectrodeCoordsExtractor(event_types="Ieeg", bt_root=str(tmp_path), c_max=10)
    with pytest.raises(FileNotFoundError, match="depth-wm.csv"):
        ext.get_static(SimpleNamespace(subject="99"))  # type: ignore[arg-type]


def test_coords_subject_coercion_btbank_prefix(tmp_path: Path) -> None:
    _write_bt(
        tmp_path, 7, voltage=["E1"],
        anatomy=[("E1", "ctx-lh-precentral", 5.0, 6.0, 7.0)],
    )
    ext = V14ElectrodeCoordsExtractor(event_types="Ieeg", bt_root=str(tmp_path), c_max=5)
    out = ext.get_static(SimpleNamespace(subject="btbank7"))  # type: ignore[arg-type]
    assert out.shape == (5, 3)
    torch.testing.assert_close(out[0], torch.tensor([5.0, 6.0, 7.0]))
    assert (out[1:] == 0).all().item()


def test_electrode_set_default_and_cache_uid(tmp_path: Path) -> None:
    """Default electrode_set='all'; the 'lite' variant serializes to a distinct
    exca config so the two caches never collide."""
    all_ext = V14ElectrodeCoordsExtractor(event_types="Ieeg", bt_root=str(tmp_path), c_max=8)
    lite_ext = V14ElectrodeCoordsExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=8, electrode_set="lite",
    )
    assert all_ext.electrode_set == "all"
    assert all_ext.model_dump() != lite_ext.model_dump()


@pytest.mark.skipif(not _HAS_BT, reason="BT anatomy cache not present")
@pytest.mark.parametrize("subject_id", [1, 2, 3, 4])
def test_coords_row_identity_real_data(subject_id: int) -> None:
    """On real BT data, extractor row ``c`` matches ``aligned_voltage_coords[c]``
    and the support extractor's row order — the end-to-end alignment invariant."""
    from speech_decoding.extractors.dk_support import V14DKHardSupportExtractor
    from speech_decoding.studies.braintreebank.anatomy import aligned_voltage_coords

    bt_root = str(_BT_CACHE)
    c_max = 384
    coords = V14ElectrodeCoordsExtractor(
        event_types="Ieeg", bt_root=bt_root, c_max=c_max,
    ).get_static(SimpleNamespace(subject=str(subject_id)))  # type: ignore[arg-type]
    support = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=bt_root, c_max=c_max, unmapped_policy="zero",
    ).get_static(SimpleNamespace(subject=str(subject_id)))  # type: ignore[arg-type]

    native = torch.from_numpy(aligned_voltage_coords(bt_root, subject_id))
    n = native.shape[0]
    assert n == support.shape[0] or n <= c_max
    torch.testing.assert_close(coords[:n], native)
    assert (coords[n:] == 0).all().item()
