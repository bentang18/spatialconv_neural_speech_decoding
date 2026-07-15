"""Tests for the RAM ind.region -> DKT parcel_fn (synthetic tsv, torch-free core)."""

from __future__ import annotations

import pytest

from speech_decoding.studies.braintreebank.anatomy import atlas_spec
from speech_decoding.studies.ram_cohort.parcels import (
    best_regions_for_rid,
    make_ram_parcel_fn,
    read_electrodes_regions,
    region_to_parcel_id,
    resolve_parcel_ids,
)

_HEADER = "name\tx\ty\tz\tsize\tgroup\themisphere\ttype\tind.region\tdas.region\n"


def _lti():
    _lcol, plabels = atlas_spec("dkt")
    return {lab: i for i, lab in enumerate(plabels)}, len(plabels)


def _write_tsv(path, rows, bom=False):
    enc = "utf-8-sig" if bom else "utf-8"
    with open(path, "w", encoding=enc) as fh:
        fh.write(_HEADER)
        for name, hemi, region in rows:
            fh.write(f"{name}\t0\t0\t0\t-999\tG\t{hemi}\tdepth\t{region}\tn/a\n")


def test_read_electrodes_regions_bom(tmp_path):
    p = tmp_path / "e.tsv"
    _write_tsv(p, [("LAF1", "L", "superiortemporal"), ("LAF2", "R", "n/a")], bom=True)
    got = read_electrodes_regions(str(p))
    assert got == {"LAF1": ("L", "superiortemporal"), "LAF2": ("R", "n/a")}


def test_region_to_parcel_id_cortical():
    lti, uid = _lti()
    assert region_to_parcel_id("L", "superiortemporal", lti, uid, "sentinel") == lti["ctx-lh-superiortemporal"]
    assert region_to_parcel_id("R", "precentral", lti, uid, "sentinel") == lti["ctx-rh-precentral"]
    # case-insensitive base
    assert region_to_parcel_id("L", "SuperiorTemporal", lti, uid, "sentinel") == lti["ctx-lh-superiortemporal"]


def test_dk_policy_dropped_bases():
    lti, uid = _lti()
    for base in ("bankssts", "temporalpole", "frontalpole"):
        assert region_to_parcel_id("L", base, lti, uid, "sentinel") == uid
    # neighbour reassignment
    assert region_to_parcel_id("L", "bankssts", lti, uid, "neighbor") == lti["ctx-lh-superiortemporal"]
    assert region_to_parcel_id("L", "temporalpole", lti, uid, "neighbor") == lti["ctx-lh-superiortemporal"]
    assert region_to_parcel_id("R", "frontalpole", lti, uid, "neighbor") == lti["ctx-rh-superiorfrontal"]


def test_sentinel_fallbacks():
    lti, uid = _lti()
    assert region_to_parcel_id("L", "n/a", lti, uid, "sentinel") == uid
    assert region_to_parcel_id("L", "", lti, uid, "sentinel") == uid
    assert region_to_parcel_id("X", "superiortemporal", lti, uid, "sentinel") == uid  # bad hemi
    assert region_to_parcel_id("L", "notaregion", lti, uid, "sentinel") == uid        # out-of-vocab


def test_resolve_preserves_order_and_counts():
    lti, uid = _lti()
    regions = {"A1": ("L", "superiortemporal"), "A2": ("R", "n/a")}
    labels = ["A2", "A1", "MISSING"]  # order preserved; MISSING has no row -> sentinel
    ids, n_valid = resolve_parcel_ids(regions, labels, lti, uid, "sentinel")
    assert ids == [uid, lti["ctx-lh-superiortemporal"], uid]
    assert n_valid == 1


def test_best_regions_picks_most_localized(tmp_path):
    d = tmp_path / "sidecars"
    (d / "dsA").mkdir(parents=True)
    (d / "dsB").mkdir(parents=True)
    # Same rid, two sessions: dsB is better-localized -> chosen.
    _write_tsv(d / "dsA" / "sub-R1001P_ses-0_electrodes.tsv",
               [("A1", "L", "n/a"), ("A2", "L", "n/a")])
    _write_tsv(d / "dsB" / "sub-R1001P_ses-1_electrodes.tsv",
               [("A1", "L", "superiortemporal"), ("A2", "R", "precentral")])
    best = best_regions_for_rid(str(d), "R1001P")
    assert best == {"A1": ("L", "superiortemporal"), "A2": ("R", "precentral")}


def test_best_regions_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        best_regions_for_rid(str(tmp_path), "R9999X")


def test_make_ram_parcel_fn_rejects_bad_policy(tmp_path):
    # Validated BEFORE the torch import, so this raises without torch present.
    with pytest.raises(ValueError):
        make_ram_parcel_fn(str(tmp_path), {"R1001P": 3000}, dk_policy="bogus")
