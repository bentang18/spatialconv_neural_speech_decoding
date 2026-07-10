"""v14_converged_v3 Phase 1 — shaft/index sidecar (TDD).

The sidecar turns a session's ordered surviving electrode labels + hard atlas
support into the per-contact geometry the v3 encoder/masking need (memo
project-v14-converged-v3-sensor-architecture, "shaft/index sidecar"):

  - shaft_id  : contiguous group id for L1 block-diagonal attention + mask ∝ count
  - depth     : CANONICAL along-shaft contact index off the FULL shaft, drop-gaps
                PRESERVED — this is the clinical contact number (parse_shaft[1]),
                NOT a dense re-numbering. The memo's "keep the gap, do NOT renumber"
                rule: a dropped LTG5 leaves LTG4→4 and LTG6→6 at distance 2.
  - parcel_id : DKT/DK hard tag = support.argmax(-1), the cross-subject query identity.

All three are row-aligned to the voltage electrode order (the survivors).
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.sidecar import (
    SensorSidecar,
    build_sidecar,
)


def _onehot(parcel_ids: list[int], k: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(parcel_ids), num_classes=k).float()


def test_shaft_ids_are_contiguous_first_appearance() -> None:
    labels = ["LTA1", "LTA2", "LTB1", "LTA3", "LTB2"]
    sc = build_sidecar(labels, parcel_id=torch.zeros(5, dtype=torch.long))
    # LTA→0 (first seen), LTB→1 (second seen); revisits reuse the id.
    assert sc.shaft_id.tolist() == [0, 0, 1, 0, 1]
    assert sc.n_shafts == 2


def test_depth_preserves_drop_gaps() -> None:
    # LTG3 dropped as a bad contact → survivors keep clinical numbers 1,2,4,5.
    labels = ["LTG1", "LTG2", "LTG4", "LTG5"]
    sc = build_sidecar(labels, parcel_id=torch.zeros(4, dtype=torch.long))
    assert sc.depth.tolist() == [1, 2, 4, 5]
    # The crux: LTG2 ↔ LTG4 relative distance MUST be 2 (the hole), not 1.
    assert (sc.depth[2] - sc.depth[1]).item() == 2


def test_depth_is_not_densely_renumbered() -> None:
    # A dense renumber (the forbidden _bt_contact_pos behavior) would give 0,1,2,3.
    labels = ["OFa2", "OFa5", "OFa9"]
    sc = build_sidecar(labels, parcel_id=torch.zeros(3, dtype=torch.long))
    assert sc.depth.tolist() == [2, 5, 9]
    assert sc.depth.tolist() != [0, 1, 2]


def test_multidigit_stem_peels_only_trailing_number() -> None:
    # Stem itself contains digits ("LT3aHa"); only the trailing run is the depth.
    labels = ["LT3aHa12", "LT3aHa14"]
    sc = build_sidecar(labels, parcel_id=torch.zeros(2, dtype=torch.long))
    assert sc.shaft_id.tolist() == [0, 0]  # same shaft
    assert sc.depth.tolist() == [12, 14]


def test_parcel_id_from_support_argmax() -> None:
    labels = ["LTA1", "LTA2", "LTB1"]
    support = _onehot([3, 3, 7], k=10)
    sc = build_sidecar(labels, support=support)
    assert sc.parcel_id.tolist() == [3, 3, 7]
    assert sc.parcel_id.dtype == torch.long


def test_label_without_contact_number_is_rejected() -> None:
    # Every real sEEG/grid contact ends in a number; a bare label = mislabel → fail loud.
    with pytest.raises(ValueError):
        build_sidecar(["LTA1", "TRIG"], parcel_id=torch.zeros(2, dtype=torch.long))


def test_requires_support_or_parcel_id() -> None:
    with pytest.raises(ValueError):
        build_sidecar(["LTA1", "LTA2"])


def test_full_sidecar_shapes_dtypes_and_alignment() -> None:
    labels = ["LTA1", "LTA2", "LTA4", "LTB1", "LTB2"]
    support = _onehot([1, 1, 1, 5, 5], k=8)
    sc = build_sidecar(labels, support=support)
    n = len(labels)
    assert isinstance(sc, SensorSidecar)
    for t in (sc.shaft_id, sc.depth, sc.parcel_id):
        assert t.shape == (n,)
        assert t.dtype == torch.long
    assert sc.labels == tuple(labels)
    assert sc.n_shafts == 2
    # Row alignment: shaft boundary at index 3 coincides with the label prefix change.
    assert sc.shaft_id.tolist() == [0, 0, 0, 1, 1]
    assert sc.depth.tolist() == [1, 2, 4, 1, 2]


def test_length_mismatch_between_labels_and_parcel_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_sidecar(["LTA1", "LTA2"], parcel_id=torch.zeros(3, dtype=torch.long))
