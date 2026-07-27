"""viz_lobe_audit runs standalone on a compute node, so its lobe map is a copy.

A copy that drifts is worse than no copy: the audit would silently report coverage for an
anatomy nobody uses. These tests are the only thing keeping the two in sync.
"""
from __future__ import annotations

from scripts.neuroprobe.viz_lobe_audit import PARCEL_LOBE_KEYS, UNKNOWN_LOBE, _collapse_hemi
from speech_decoding.studies.braintreebank.anatomy import parcel_lobe_keys
from speech_decoding.studies.braintreebank import anatomy


def test_inlined_copy_matches_canonical_map() -> None:
    assert PARCEL_LOBE_KEYS == parcel_lobe_keys()


def test_inlined_copy_is_index_aligned_with_the_parcel_vocabulary() -> None:
    # a parcel id indexes the map directly, and the reserved unknown id is the last slot
    assert len(PARCEL_LOBE_KEYS) == len(anatomy.V14_DKT_PARCEL_LABELS) + 1
    assert PARCEL_LOBE_KEYS[-1] == UNKNOWN_LOBE
    assert PARCEL_LOBE_KEYS.count(UNKNOWN_LOBE) == 1


def test_collapse_hemi_drops_the_side_but_keeps_unknown_whole() -> None:
    assert _collapse_hemi("lh-superior") == "superior"
    assert _collapse_hemi("rh-mtl") == "mtl"
    assert _collapse_hemi(UNKNOWN_LOBE) == UNKNOWN_LOBE


def test_collapse_hemi_pairs_every_lobe_across_hemispheres() -> None:
    # hemi-pooled coverage is only meaningful if the two sides use the same lobe names
    lh = {k.split("-", 1)[1] for k in PARCEL_LOBE_KEYS if k.startswith("lh-")}
    rh = {k.split("-", 1)[1] for k in PARCEL_LOBE_KEYS if k.startswith("rh-")}
    assert lh == rh
    assert lh == {"frontal", "temporal", "parietal", "occipital",
                  "cingulate", "insula", "mtl", "subcortical"}
