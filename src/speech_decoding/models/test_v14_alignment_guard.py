"""L2 — always-on electrode-row data-integrity guard tests.

``assert_electrode_alignment_integrity`` runs on every encoder forward. It must:
fire on a corrupted/desynced support or valid_mask lane, AND never false-positive
on the legitimate paths (Lite electrode set, masked-JEPA zeroing, valid_mask=None).
See reports/bt_alignment/electrode_desync_damage_2026_06_09.md.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_encoder import assert_electrode_alignment_integrity

# L2 is a pre-dispatch gate: a desynced support/valid lane must abort the
# dispatch before any DCC spend. Run by `scripts/dcc/dispatch`.
pytestmark = pytest.mark.must_pass_before_dispatch


def _one_hot(rows: list[int], k: int = 5) -> torch.Tensor:
    """(1, C, K) one-hot support; row value -1 means an all-zero (unmapped) row."""
    sup = torch.zeros(1, len(rows), k)
    for c, p in enumerate(rows):
        if p >= 0:
            sup[0, c, p] = 1.0
    return sup


def test_guard_passes_on_aligned_one_hot_support() -> None:
    sup = _one_hot([0, 1, 2, -1])  # last row unmapped (pad)
    valid = torch.tensor([[True, True, True, False]])
    assert_electrode_alignment_integrity(sup, valid)  # no raise


def test_guard_passes_on_lite_excluded_mapped_electrode() -> None:
    """Lite set: a parcel-mapped electrode (support one-hot) may be valid=False
    (non-Lite). support stays one-hot; the guard must NOT fire."""
    sup = _one_hot([0, 1, 2, 3])  # all mapped
    valid = torch.tensor([[True, False, True, False]])  # 2 excluded by Lite
    assert_electrode_alignment_integrity(sup, valid)  # no raise


def test_guard_passes_when_valid_mask_none() -> None:
    sup = _one_hot([0, 1, 2])
    assert_electrode_alignment_integrity(sup, None)  # no raise


def test_guard_raises_on_non_one_hot_support() -> None:
    """A row assigning two parcels = corrupted support lane."""
    sup = _one_hot([0, 1, 2, 3])
    sup[0, 1, 4] = 1.0  # row 1 now sums to 2
    valid = torch.tensor([[True, True, True, True]])
    with pytest.raises(ValueError, match="one-hot-or-zero"):
        assert_electrode_alignment_integrity(sup, valid)


def test_guard_raises_on_valid_electrode_with_empty_support() -> None:
    """valid=True on an unmapped (empty-support) row = support/valid_mask desync —
    the exact DP2 zero-fed-as-valid signature in the anatomy lane."""
    sup = _one_hot([0, 1, -1, 3])  # row 2 empty
    valid = torch.tensor([[True, True, True, True]])  # but row 2 marked valid
    with pytest.raises(ValueError, match="empty support"):
        assert_electrode_alignment_integrity(sup, valid)


def test_guard_raises_on_shape_mismatch() -> None:
    sup = _one_hot([0, 1, 2, 3])
    valid = torch.tensor([[True, True, True]])  # C=3 != support C=4
    with pytest.raises(ValueError, match="row desync"):
        assert_electrode_alignment_integrity(sup, valid)
