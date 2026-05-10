"""Tests for the Tier-1 per-electrode support cache (docs/plans/v14-core.md A1).

Unit tests drive the writer with a synthetic support matrix — no Box mount, no
atlas, no network. An optional integration test checks that a pre-built cache
on disk passes the same invariants for each core patient.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from speech_decoding.atlas.support import (
    TIER1_BNA_INDICES_1BASED,
    TIER1_COLUMNS,
    PatientQC,
    lookup_support_for_kept_channels,
    qc_row_for_patient,
    read_support_cache,
    sanitize_parcel_name,
    write_qc_report,
    write_support_cache,
)
from speech_decoding.atlas.tokens import DEFAULT_BASE_PARCELS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORT_CACHE_DIR = PROJECT_ROOT / "data" / "atlas" / "support_cache"
CORE_PATIENTS = ("S14", "S26", "S33", "S62")


def test_tier1_columns_match_default_base_parcels() -> None:
    assert len(TIER1_COLUMNS) == len(DEFAULT_BASE_PARCELS) == 15
    for col, (name, _idx) in zip(TIER1_COLUMNS, DEFAULT_BASE_PARCELS):
        assert col == f"support_{sanitize_parcel_name(name)}"
    # BNA indices unchanged by sanitization
    assert TIER1_BNA_INDICES_1BASED == tuple(idx for _, idx in DEFAULT_BASE_PARCELS)


def test_sanitize_parcel_name_replaces_slash() -> None:
    assert sanitize_parcel_name("A1/2/3ulhf") == "A1_2_3ulhf"
    assert sanitize_parcel_name("A4hf") == "A4hf"
    assert sanitize_parcel_name("A9/46v") == "A9_46v"


def test_write_support_cache_round_trip(tmp_path: Path) -> None:
    names = ("e0", "e1", "e2")
    support = np.array(
        [[0.0] * 15, [1.0] * 15, np.arange(15, dtype=np.float32) * 5.0],
        dtype=np.float32,
    )
    out_path = tmp_path / "fake_support_tier1.csv"
    write_support_cache(out_path, names, support)

    text = out_path.read_text().splitlines()
    assert text[0] == ",".join(("name",) + TIER1_COLUMNS)
    assert len(text) == 1 + len(names)
    row1 = text[2].split(",")
    assert row1[0] == "e1"
    assert all(float(v) == pytest.approx(1.0) for v in row1[1:])

    # Column count: name + 15 parcel columns.
    assert len(text[0].split(",")) == 16
    for data_row in text[1:]:
        assert len(data_row.split(",")) == 16


def test_write_support_cache_rejects_shape_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="support shape"):
        write_support_cache(tmp_path / "bad.csv", ("e0", "e1"), np.zeros((3, 15), dtype=np.float32))


def test_read_support_cache_round_trip(tmp_path: Path) -> None:
    names = ("a", "b")
    support = np.array([[1.0] * 15, [2.0] * 15], dtype=np.float32)
    path = tmp_path / "rt.csv"
    write_support_cache(path, names, support)
    got_names, got_support = read_support_cache(path)
    assert got_names == names
    np.testing.assert_allclose(got_support, support, rtol=0, atol=1e-5)
    assert got_support.dtype == np.float32


def test_lookup_support_for_kept_channels_reorders(tmp_path: Path) -> None:
    names = ("a", "b", "c")
    support = np.array(
        [[1.0] * 15, [2.0] * 15, [3.0] * 15],
        dtype=np.float32,
    )
    path = tmp_path / "lk.csv"
    write_support_cache(path, names, support)
    out = lookup_support_for_kept_channels(path, ("c", "a"))
    np.testing.assert_allclose(out[0], 3.0)
    np.testing.assert_allclose(out[1], 1.0)


def test_lookup_support_raises_on_missing(tmp_path: Path) -> None:
    names = ("a",)
    support = np.zeros((1, 15), dtype=np.float32)
    path = tmp_path / "miss.csv"
    write_support_cache(path, names, support)
    with pytest.raises(KeyError, match="missing electrodes"):
        lookup_support_for_kept_channels(path, ("a", "missing"))


def test_qc_row_for_patient_counts() -> None:
    # 4 electrodes: first all-zero, second Tier-1-max=50, third mass<1, fourth max=90
    support = np.array(
        [
            [0.0] * 15,
            [50.0] + [0.0] * 14,
            [0.5, 0.3] + [0.0] * 13,
            [90.0] + [0.0] * 14,
        ],
        dtype=np.float32,
    )
    qc = qc_row_for_patient("FAKE", support)
    assert isinstance(qc, PatientQC)
    assert qc.patient == "FAKE"
    assert qc.n_electrodes == 4
    assert qc.argmax_in_tier1 == 3  # three rows with max > 0
    assert qc.low_tier1_mass == 2   # e0 sum=0, e2 sum=0.8 → both < 1
    assert sum(qc.hist_counts) == 4


def test_write_qc_report_contains_headers(tmp_path: Path) -> None:
    support = np.zeros((2, 15), dtype=np.float32)
    rows = [qc_row_for_patient("S14", support)]
    qc_path = tmp_path / "qc.md"
    write_qc_report(qc_path, rows)
    text = qc_path.read_text()
    assert "# Tier-1 support cache QC report" in text
    assert "| patient | n_elec | argmax_in_tier1" in text
    assert "| S14 |" in text


@pytest.mark.parametrize("patient", CORE_PATIENTS)
def test_support_cache_on_disk_for_core_patients(patient: str) -> None:
    """Integration: a prebuilt cache file exists and passes the declared contract.

    Skipped until a support cache has been built for the patient.
    """

    path = SUPPORT_CACHE_DIR / f"{patient}_support_tier1.csv"
    if not path.exists():
        pytest.skip(f"support cache not built for {patient}")

    lines = path.read_text().splitlines()
    header = lines[0].split(",")
    assert header[0] == "name"
    assert tuple(header[1:]) == TIER1_COLUMNS
    assert len(header) == 16

    names: set[str] = set()
    for row_idx, line in enumerate(lines[1:], start=2):
        fields = line.split(",")
        assert len(fields) == 16, f"{path}:{row_idx}: expected 16 columns"
        name = fields[0]
        assert name and name not in names, f"{path}:{row_idx}: missing or duplicate name {name!r}"
        names.add(name)
        for col_idx, v in enumerate(fields[1:], start=1):
            value = float(v)
            assert not np.isnan(value), f"{path}:{row_idx} col {col_idx}: NaN"
            assert 0.0 <= value <= 100.0 + 1e-3, (
                f"{path}:{row_idx} col {col_idx}: value {value} outside [0, 100]"
            )
