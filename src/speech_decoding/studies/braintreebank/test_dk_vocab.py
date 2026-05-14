"""Tests for the canonical K=80 FreeSurfer DK parcel vocabulary used by v14.

The vocabulary is fixed by the atlas standard, not by the BT cohort. Cortical
labels match the 68 hemis-distinct FreeSurfer aparc strings; subcortical labels
match the 12 standard FreeSurfer aseg subcortical-of-interest strings (6 regions
times left/right).

Vocabulary string format matches what BT `depth-wm.csv:DesikanKilliany` emits
(verified 2026-05-13 in ``reports/neuroprobe_stage0_public_bt_hard_labels_2026_05_05``).
"""

from __future__ import annotations

import pytest

from speech_decoding.studies.braintreebank.anatomy import (
    V14_DK_PARCEL_LABELS,
    V14_DK_PARCEL_LABELS_CORTICAL,
    V14_DK_PARCEL_LABELS_SUBCORTICAL,
    parse_dk_label,
)


def test_v14_dk_parcel_labels_has_canonical_length_80() -> None:
    assert len(V14_DK_PARCEL_LABELS) == 80
    assert len(V14_DK_PARCEL_LABELS_CORTICAL) == 68
    assert len(V14_DK_PARCEL_LABELS_SUBCORTICAL) == 12
    assert (
        tuple(V14_DK_PARCEL_LABELS_CORTICAL) + tuple(V14_DK_PARCEL_LABELS_SUBCORTICAL)
        == V14_DK_PARCEL_LABELS
    )


def test_v14_dk_parcel_labels_all_unique() -> None:
    assert len(set(V14_DK_PARCEL_LABELS)) == 80


def test_v14_dk_cortical_uses_freesurfer_aparc_string_format() -> None:
    """Every cortical label is ``ctx-{lh,rh}-<base>`` matching BT depth-wm.csv."""
    for label in V14_DK_PARCEL_LABELS_CORTICAL:
        assert label.startswith(("ctx-lh-", "ctx-rh-")), f"bad cortical label: {label}"


def test_v14_dk_cortical_has_34_per_hemisphere() -> None:
    lh = [s for s in V14_DK_PARCEL_LABELS_CORTICAL if s.startswith("ctx-lh-")]
    rh = [s for s in V14_DK_PARCEL_LABELS_CORTICAL if s.startswith("ctx-rh-")]
    assert len(lh) == 34
    assert len(rh) == 34
    lh_bases = {s.removeprefix("ctx-lh-") for s in lh}
    rh_bases = {s.removeprefix("ctx-rh-") for s in rh}
    assert lh_bases == rh_bases, "lh/rh DK base sets must match"


def test_v14_dk_cortical_contains_known_bt_observed_regions() -> None:
    """Sanity-check against the 2026-05-13 BT DK audit ground truth."""
    observed_in_bt = {
        "superiortemporal", "rostralmiddlefrontal", "superiorfrontal",
        "lateralorbitofrontal", "insula", "middletemporal", "precentral",
        "parstriangularis", "caudalmiddlefrontal", "parsopercularis",
        "postcentral", "medialorbitofrontal", "supramarginal",
        "inferiortemporal", "rostralanteriorcingulate", "bankssts",
        "inferiorparietal", "caudalanteriorcingulate", "transversetemporal",
        "temporalpole",
    }
    lh_bases = {s.removeprefix("ctx-lh-") for s in V14_DK_PARCEL_LABELS_CORTICAL
                if s.startswith("ctx-lh-")}
    missing = observed_in_bt - lh_bases
    assert not missing, f"observed BT regions missing from K=80 cortical: {missing}"


def test_v14_dk_subcortical_uses_freesurfer_aseg_string_format() -> None:
    """Every subcortical label is ``{Left,Right}-<Region>`` matching BT depth-wm.csv."""
    for label in V14_DK_PARCEL_LABELS_SUBCORTICAL:
        assert label.startswith(("Left-", "Right-")), f"bad aseg label: {label}"


def test_v14_dk_subcortical_has_6_bilateral_regions() -> None:
    left = [s for s in V14_DK_PARCEL_LABELS_SUBCORTICAL if s.startswith("Left-")]
    right = [s for s in V14_DK_PARCEL_LABELS_SUBCORTICAL if s.startswith("Right-")]
    assert len(left) == 6
    assert len(right) == 6
    left_bases = {s.removeprefix("Left-") for s in left}
    right_bases = {s.removeprefix("Right-") for s in right}
    assert left_bases == right_bases, "Left/Right aseg base sets must match"


def test_v14_dk_subcortical_includes_v14_target_regions() -> None:
    """Hippocampus, Amygdala, Putamen must be present (BT observes all three)."""
    bases = {s.removeprefix("Left-").removeprefix("Right-")
             for s in V14_DK_PARCEL_LABELS_SUBCORTICAL}
    for region in ("Hippocampus", "Amygdala", "Putamen"):
        assert region in bases, f"v14 target subcortical region missing: {region}"


def test_parse_dk_label_cortical() -> None:
    kind, hemi, base = parse_dk_label("ctx-lh-superiortemporal")
    assert kind == "cortical"
    assert hemi == "lh"
    assert base == "superiortemporal"


def test_parse_dk_label_subcortical() -> None:
    kind, hemi, base = parse_dk_label("Left-Hippocampus")
    assert kind == "subcortical"
    assert hemi == "lh"
    assert base == "Hippocampus"

    kind, hemi, base = parse_dk_label("Right-Amygdala")
    assert kind == "subcortical"
    assert hemi == "rh"
    assert base == "Amygdala"


def test_parse_dk_label_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="unrecognised DK label"):
        parse_dk_label("superiortemporal")  # no hemisphere prefix
    with pytest.raises(ValueError, match="unrecognised DK label"):
        parse_dk_label("ctx-zh-bankssts")  # bad hemi prefix
