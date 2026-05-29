"""Tests for the Tier-1 per-electrode support cache (docs/plans/v14-core.md A1).

Unit tests drive the writer with a synthetic support matrix — no Box mount, no
atlas, no network. An optional integration test checks that a pre-built cache
on disk passes the same invariants for each core patient.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import torch

from speech_decoding.atlas.support import (
    TIER1_BNA_INDICES_1BASED,
    TIER1_COLUMNS,
    PatientQC,
    compute_gated_log_support_bias,
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


# ---------------------------------------------------------------------------
# B29 Item 12 — λ_anat per-clip gate on log(support+ε)
# ---------------------------------------------------------------------------


def _toy_support(B: int = 4, C: int = 3, K: int = 5) -> torch.Tensor:
    rng = np.random.default_rng(0)
    arr = rng.random((B, C, K)).astype(np.float32)
    return torch.from_numpy(arr)


def test_b29_compute_gated_log_support_bias_scalar_lambda_default() -> None:
    """``lambda_anat=1.0`` (default) is just ``log(support + ε)``."""
    support = _toy_support()
    eps = 1e-2
    out = compute_gated_log_support_bias(support, eps=eps, lambda_anat=1.0)
    expected = torch.log(support + eps)
    torch.testing.assert_close(out, expected)


def test_b29_compute_gated_log_support_bias_scalar_zero_zeros_output() -> None:
    """``lambda_anat=0`` collapses the anatomy bias to zero (uniform-over-electrodes)."""
    support = _toy_support()
    out = compute_gated_log_support_bias(support, eps=1e-2, lambda_anat=0.0)
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_b29_compute_gated_log_support_bias_per_clip_tensor_broadcasts() -> None:
    """A ``(B,)`` ``lambda_anat`` scales each clip independently — anatomy-rich
    clips keep the full bias while SWEC clips zero out."""
    B, C, K = 4, 3, 5
    support = _toy_support(B, C, K)
    lambda_anat = torch.tensor([1.0, 0.0, 1.0, 0.0])
    out = compute_gated_log_support_bias(support, eps=1e-2, lambda_anat=lambda_anat)
    log_support = torch.log(support + 1e-2)
    torch.testing.assert_close(out[0], log_support[0])      # anatomy-rich
    torch.testing.assert_close(out[2], log_support[2])      # anatomy-rich
    torch.testing.assert_close(out[1], torch.zeros_like(log_support[1]))  # SWEC
    torch.testing.assert_close(out[3], torch.zeros_like(log_support[3]))  # SWEC


def test_b29_compute_gated_log_support_bias_rejects_wrong_tensor_shape() -> None:
    support = _toy_support(B=4)
    with pytest.raises(ValueError, match="must have shape"):
        compute_gated_log_support_bias(
            support, eps=1e-2,
            lambda_anat=torch.tensor([1.0, 0.0]),  # wrong B
        )


def test_b_cr_2_compute_gated_log_support_bias_accepts_b1_collated_shape() -> None:
    """NeuralSet collates per-event ``(1,)`` TimedArrays into
    ``(B, 1)``. The helper must squeeze the trailing singleton so the
    encoder/atlas pair don't reject every realistic dataloader batch.
    """
    B, C, K = 4, 3, 5
    support = _toy_support(B, C, K)
    lambda_b1 = torch.tensor([[1.0], [0.0], [1.0], [0.0]])  # (B, 1)
    out = compute_gated_log_support_bias(support, eps=1e-2, lambda_anat=lambda_b1)
    log_support = torch.log(support + 1e-2)
    torch.testing.assert_close(out[0], log_support[0])
    torch.testing.assert_close(out[2], log_support[2])
    torch.testing.assert_close(out[1], torch.zeros_like(log_support[1]))
    torch.testing.assert_close(out[3], torch.zeros_like(log_support[3]))


def test_b_bug_5_compute_gated_log_support_bias_accepts_0d_tensor() -> None:
    """A 0-d Tensor (e.g. from a warmup schedule wrapper) is the
    scalar branch — broadcast to all clips.
    """
    support = _toy_support()
    eps = 1e-2
    out = compute_gated_log_support_bias(
        support, eps=eps, lambda_anat=torch.tensor(0.7),
    )
    expected = 0.7 * torch.log(support + eps)
    torch.testing.assert_close(out, expected)


def test_b_bug_5_compute_gated_log_support_bias_rejects_negative_0d_tensor() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        compute_gated_log_support_bias(
            _toy_support(), eps=1e-2, lambda_anat=torch.tensor(-0.5),
        )


def test_b29_compute_gated_log_support_bias_rejects_negative_scalar() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        compute_gated_log_support_bias(
            _toy_support(), eps=1e-2, lambda_anat=-0.1,
        )


def test_b29_compute_gated_log_support_bias_rejects_negative_tensor() -> None:
    support = _toy_support(B=4)
    with pytest.raises(ValueError, match="must be >= 0"):
        compute_gated_log_support_bias(
            support, eps=1e-2,
            lambda_anat=torch.tensor([1.0, -0.1, 1.0, 0.5]),
        )


def test_b29_compute_gated_log_support_bias_rejects_non_3d_support() -> None:
    bad = torch.zeros(3, 5)  # missing batch dim
    with pytest.raises(ValueError, match="must be"):
        compute_gated_log_support_bias(bad, eps=1e-2, lambda_anat=1.0)
