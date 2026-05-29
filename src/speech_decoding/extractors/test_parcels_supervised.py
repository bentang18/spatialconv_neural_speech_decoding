"""MASK-02 (B03 mask-discipline lock 2026-05-25 PM): per-subject
``parcels_supervised`` extractor and slot-mask helpers.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.extractors.parcels_supervised import (
    compute_parcels_supervised_from_support,
    parcels_supervised_to_slot_mask,
)


def test_compute_parcels_supervised_from_dense_support() -> None:
    """Argmax-style support: each electrode one-hot at a parcel. The set of
    parcels with ≥1 electrode is exactly the union of one-hot indices."""
    C, K = 6, 10
    support = torch.zeros(C, K)
    support[0, 2] = 1.0
    support[1, 2] = 1.0
    support[2, 5] = 1.0
    support[3, 5] = 1.0
    support[4, 8] = 1.0
    support[5, 8] = 1.0  # no coverage at parcels 0, 1, 3, 4, 6, 7, 9
    assert compute_parcels_supervised_from_support(support) == {2, 5, 8}


def test_compute_parcels_supervised_from_soft_support() -> None:
    """A parcel is "supervised" iff support is strictly positive somewhere;
    soft (BNA-style) support still counts."""
    C, K = 4, 5
    support = torch.zeros(C, K)
    support[0, 0] = 0.7
    support[0, 1] = 0.3
    support[1, 2] = 1.0
    # rest of electrodes empty
    assert compute_parcels_supervised_from_support(support) == {0, 1, 2}


def test_compute_parcels_supervised_honors_valid_mask() -> None:
    """Padded electrodes (valid_mask=False) must not contribute coverage."""
    C, K = 4, 5
    support = torch.zeros(C, K)
    support[0, 0] = 1.0  # real → parcel 0 covered
    support[3, 4] = 1.0  # padded electrode → must NOT count
    valid_mask = torch.tensor([True, True, True, False])
    out = compute_parcels_supervised_from_support(support, valid_mask=valid_mask)
    assert out == {0}, f"valid_mask=False must drop electrode 3; got {out}"


def test_compute_parcels_supervised_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="2-D"):
        compute_parcels_supervised_from_support(torch.zeros(2, 3, 4))


def test_parcels_supervised_to_slot_mask_expands_over_subslots() -> None:
    """A parcel in the set enables all M of its sub-slots."""
    mask = parcels_supervised_to_slot_mask({1, 3}, k_parcels=5, m_sub_slots=4)
    assert mask.shape == (5 * 4,)
    assert mask.dtype == torch.bool
    expected_indices = {1 * 4 + s for s in range(4)} | {3 * 4 + s for s in range(4)}
    got_indices = {int(i) for i in torch.nonzero(mask, as_tuple=False).flatten().tolist()}
    assert got_indices == expected_indices


def test_parcels_supervised_to_slot_mask_swec_fallback_is_all_true() -> None:
    """SWEC fallback (empty / None) supervises all 320 slots — anatomy-blind."""
    K, M = 80, 4
    mask_empty = parcels_supervised_to_slot_mask(set(), k_parcels=K, m_sub_slots=M)
    mask_none = parcels_supervised_to_slot_mask(None, k_parcels=K, m_sub_slots=M)
    assert mask_empty.all() and mask_empty.shape == (K * M,)
    assert mask_none.all() and mask_none.shape == (K * M,)


def test_parcels_supervised_to_slot_mask_rejects_out_of_range_id() -> None:
    with pytest.raises(ValueError, match="invalid id"):
        parcels_supervised_to_slot_mask({99}, k_parcels=10, m_sub_slots=4)
