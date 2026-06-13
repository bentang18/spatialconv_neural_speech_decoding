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
    V14_DKT_PARCEL_LABELS,
    V14_DKT_PARCEL_LABELS_CORTICAL,
    V14_DKT_PARCEL_LABELS_SUBCORTICAL,
    atlas_spec,
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


# --- DKT (Desikan-Killiany-Tourville) vocabulary ---------------------------- #
_DKT_DROPPED = {"bankssts", "frontalpole", "temporalpole"}


def test_v14_dkt_parcel_labels_has_canonical_length_74() -> None:
    assert len(V14_DKT_PARCEL_LABELS) == 74
    assert len(V14_DKT_PARCEL_LABELS_CORTICAL) == 62
    assert len(V14_DKT_PARCEL_LABELS_SUBCORTICAL) == 12
    assert (
        tuple(V14_DKT_PARCEL_LABELS_CORTICAL)
        + tuple(V14_DKT_PARCEL_LABELS_SUBCORTICAL)
        == V14_DKT_PARCEL_LABELS
    )


def test_v14_dkt_all_unique() -> None:
    assert len(set(V14_DKT_PARCEL_LABELS)) == 74


def test_v14_dkt_cortical_is_dk_minus_three_dropped_bases() -> None:
    """DKT = DK aparc bases minus {bankssts, frontalpole, temporalpole}, per hemi."""
    dk_bases = {s.removeprefix("ctx-lh-") for s in V14_DK_PARCEL_LABELS_CORTICAL
                if s.startswith("ctx-lh-")}
    dkt_bases = {s.removeprefix("ctx-lh-") for s in V14_DKT_PARCEL_LABELS_CORTICAL
                 if s.startswith("ctx-lh-")}
    assert dkt_bases == dk_bases - _DKT_DROPPED
    assert len(dkt_bases) == 31
    # the dropped three never appear under DKT (either hemi)
    for label in V14_DKT_PARCEL_LABELS_CORTICAL:
        base = label.removeprefix("ctx-lh-").removeprefix("ctx-rh-")
        assert base not in _DKT_DROPPED


def test_v14_dkt_subcortical_identical_to_dk() -> None:
    """DKT only re-parcellates cortex; the aseg subcortical 12 are unchanged."""
    assert V14_DKT_PARCEL_LABELS_SUBCORTICAL == V14_DK_PARCEL_LABELS_SUBCORTICAL


def test_atlas_spec_pairs_column_with_vocabulary() -> None:
    """atlas_spec is the only sanctioned (column, vocab) source — they move together."""
    dk_col, dk_vocab = atlas_spec("dk")
    assert dk_col == "DesikanKilliany"
    assert dk_vocab == V14_DK_PARCEL_LABELS
    dkt_col, dkt_vocab = atlas_spec("dkt")
    assert dkt_col == "DKT"
    assert dkt_vocab == V14_DKT_PARCEL_LABELS
    # case-insensitive
    assert atlas_spec("DKT") == atlas_spec("dkt")


def test_atlas_spec_rejects_unknown_atlas() -> None:
    with pytest.raises(ValueError, match="unknown atlas"):
        atlas_spec("destrieux")
