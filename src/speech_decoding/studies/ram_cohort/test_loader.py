"""Synthetic unit tests for the RAM voltage loader (no EDF on disk)."""

from __future__ import annotations

import numpy as np
import pytest

from speech_decoding.studies.ram_cohort.loader import (
    NEURAL_TYPES,
    TARGET_RATE_HZ,
    parse_contact,
    ram_car_groups,
    read_channels_tsv,
    resample_to,
    select_neural,
)


def _write_channels_tsv(tmp_path, rows, bom=False):
    """rows = list of (name, type, group, sfreq); returns the file path."""
    p = tmp_path / "channels.tsv"
    hdr = "name\ttype\tunits\tlow_cutoff\thigh_cutoff\tgroup\tsampling_frequency\tdescription\tnotch\n"
    body = "".join(
        f"{n}\t{t}\tuV\tn/a\tn/a\t{g}\t{sf}\tn/a\t60\n" for (n, t, g, sf) in rows
    )
    data = hdr + body
    p.write_bytes(("﻿" + data).encode("utf-8") if bom else data.encode("utf-8"))
    return str(p)


# ---- parse_contact ---------------------------------------------------------

@pytest.mark.parametrize(
    "name,expect",
    [
        ("LAF1", ("LAF", 1)),
        ("LTG64", ("LTG", 64)),
        ("R2SF3", ("R2SF", 3)),      # stem may embed a digit
        ("LP1_1", ("LP1_", 1)),      # compound name: parse stem != group (see below)
        ("EKG", None),              # no trailing index -> physio ref
        ("", None),
    ],
)
def test_parse_contact(name, expect):
    assert parse_contact(name) == expect


# ---- read_channels_tsv -----------------------------------------------------

def test_read_channels_tsv_columns(tmp_path):
    p = _write_channels_tsv(tmp_path, [("LAF1", "ECOG", "LAF", "1000"),
                                       ("LD1", "SEEG", "LD", "1000")])
    meta = read_channels_tsv(p)
    assert meta["LAF1"] == {"type": "ECOG", "group": "LAF", "sampling_frequency": "1000"}
    assert meta["LD1"]["type"] == "SEEG"


def test_read_channels_tsv_bom(tmp_path):
    p = _write_channels_tsv(tmp_path, [("LAF1", "ECOG", "LAF", "1000")], bom=True)
    meta = read_channels_tsv(p)         # must not KeyError on '﻿name'
    assert meta["LAF1"]["type"] == "ECOG"


# ---- select_neural ---------------------------------------------------------

def test_select_neural_whitelist_and_index(tmp_path):
    names = ["LAF1", "LD1", "EKGL", "TRIG", "Cz"]
    meta = read_channels_tsv(_write_channels_tsv(tmp_path, [
        ("LAF1", "ECOG", "LAF", "1000"),   # keep
        ("LD1", "SEEG", "LD", "1000"),     # keep
        ("EKGL", "ECOG", "n/a", "1000"),   # drop: physio ref mistyped ECOG, no index
        ("TRIG", "TRIG", "n/a", "1000"),   # drop: non-neural type
        ("Cz", "SEEG", "n/a", "1000"),     # drop: scalp ref, no trailing index
    ]))
    data = np.arange(5 * 8, dtype=np.float32).reshape(5, 8)
    out, kept = select_neural(names, data, meta)
    assert kept == ["LAF1", "LD1"]
    assert out.shape == (2, 8)
    np.testing.assert_array_equal(out[0], data[0])
    np.testing.assert_array_equal(out[1], data[1])


def test_select_neural_extra_bad_drops_micro_and_guard1(tmp_path):
    names = ["LAF1", "LAF2", "MICRO1"]
    meta = read_channels_tsv(_write_channels_tsv(tmp_path, [
        ("LAF1", "ECOG", "LAF", "1000"),
        ("LAF2", "ECOG", "LAF", "1000"),
        ("MICRO1", "SEEG", "MICRO", "1000"),   # micro typed SEEG w/ index -> only extra_bad catches it
    ]))
    data = np.zeros((3, 4), dtype=np.float32)
    _, kept = select_neural(names, data, meta, extra_bad={"MICRO1", "LAF2"})
    assert kept == ["LAF1"]


def test_select_neural_no_survivors_raises(tmp_path):
    meta = read_channels_tsv(_write_channels_tsv(tmp_path, [("TRIG", "TRIG", "n/a", "1000")]))
    with pytest.raises(ValueError, match="no neural channels"):
        select_neural(["TRIG"], np.zeros((1, 4), dtype=np.float32), meta)


def test_select_neural_shape_mismatch_raises(tmp_path):
    meta = read_channels_tsv(_write_channels_tsv(tmp_path, [("LAF1", "ECOG", "LAF", "1000")]))
    with pytest.raises(ValueError, match="!= n ch_names"):
        select_neural(["LAF1"], np.zeros((2, 4), dtype=np.float32), meta)


# ---- ram_car_groups (the #94-feeding correction) ---------------------------

def test_ram_car_groups_uses_group_column_not_name_parse(tmp_path):
    # LP1_1/LP1_2 parse to stem 'LP1_' but the authoritative group is 'LP'.
    names = ["LAF1", "LP1_1", "LP1_2", "LP2_1"]
    meta = read_channels_tsv(_write_channels_tsv(tmp_path, [
        ("LAF1", "ECOG", "LAF", "1000"),
        ("LP1_1", "ECOG", "LP", "1000"),
        ("LP1_2", "ECOG", "LP", "1000"),
        ("LP2_1", "ECOG", "LP", "1000"),
    ]))
    groups = ram_car_groups(names, meta)
    assert groups == ["LAF", "LP", "LP", "LP"]        # not 'LP1_', 'LP2_'
    # and it disagrees with the naive name-parse exactly on the compound names
    naive = [parse_contact(n)[0] for n in names]
    assert naive == ["LAF", "LP1_", "LP1_", "LP2_"]


def test_ram_car_groups_falls_back_to_stem_when_group_missing(tmp_path):
    names = ["LAF1", "LAF2"]
    meta = read_channels_tsv(_write_channels_tsv(tmp_path, [
        ("LAF1", "ECOG", "", "1000"),
        ("LAF2", "ECOG", "", "1000"),
    ]))
    assert ram_car_groups(names, meta) == ["LAF", "LAF"]


# ---- resample_to -----------------------------------------------------------

def test_resample_identity():
    x = np.random.default_rng(0).standard_normal((3, 100)).astype(np.float32)
    out = resample_to(x, 2048.0, 2048.0)
    np.testing.assert_array_equal(out, x)
    assert out.dtype == np.float32


@pytest.mark.parametrize("native,n_in", [(1000, 1000), (500, 500), (1600, 1600), (2000, 2000)])
def test_resample_rate_lengths(native, n_in):
    x = np.zeros((2, n_in), dtype=np.float32)
    out = resample_to(x, float(native), TARGET_RATE_HZ)
    expected = int(round(n_in * TARGET_RATE_HZ / native))
    assert abs(out.shape[1] - expected) <= 1
    assert out.dtype == np.float32


def test_resample_rejects_nonpositive():
    with pytest.raises(ValueError, match="non-positive"):
        resample_to(np.zeros((1, 4), dtype=np.float32), 0.0)


def test_constants():
    assert TARGET_RATE_HZ == 2048.0
    assert NEURAL_TYPES == frozenset({"ECOG", "SEEG"})
