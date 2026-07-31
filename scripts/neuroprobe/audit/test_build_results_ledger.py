"""Tests for the board_ft ledger reader, focused on the epoch-ensemble decoders.

The ledger exists to stop numbers from being quoted without provenance, so the thing worth testing
is not "does it emit rows" but the two ways an ensemble row could silently LIE: emitting a rule that
only some folds carry (a partial average masquerading as a unit), and emitting a fold-level number
where every other family is fold-averaged.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "brl", Path(__file__).resolve().parent / "build_results_ledger.py")
assert _SPEC and _SPEC.loader
BRL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BRL)

RULES = ("ens_all", "ens_valge0", "ens_top3", "ens_top1")


def _rec(fold, **kw):
    r = {"regime": "ws", "cell": "S1T1", "task": "onset", "fold": fold,
         "test_frozen_vallam": 0.90, "test_c": 0.92}
    r.update(kw)
    return r


@pytest.fixture
def arm(tmp_path, monkeypatch):
    """Writes one arm dir and points the reader at it; returns a writer for its one shard."""
    sd = tmp_path / "board_ft" / "pbs50_cd45k_ens__k1"
    sd.mkdir(parents=True)
    monkeypatch.setattr(BRL, "R6", tmp_path)

    def write(recs):
        json.dump(recs, open(sd / "ft_ws_S1T1.json", "w"))
        return {r["decoder"]: r for r in BRL._rows_board_ft()}
    return write


def test_an_arm_without_ensemble_fields_emits_exactly_the_two_original_decoders(arm):
    rows = arm([_rec(0), _rec(1)])
    assert set(rows) == {"ridge", "ridge_ft_k1"}


def test_ensemble_rules_become_their_own_decoders(arm):
    rows = arm([_rec(f, **{r: 0.93 for r in RULES}) for f in (0, 1)])
    assert set(rows) == {"ridge", "ridge_ft_k1"} | {f"ridge_ft_k1_{r}" for r in RULES}


def test_ensemble_rows_are_fold_averaged_like_every_other_family(arm):
    # If this regressed to "last fold wins" the value would be 0.98, not the mean.
    rows = arm([_rec(0, ens_all=0.88), _rec(1, ens_all=0.98)])
    assert rows["ridge_ft_k1_ens_all"]["value"] == pytest.approx(0.93)
    assert rows["ridge"]["value"] == pytest.approx(0.90)


def test_a_rule_missing_from_one_fold_is_dropped_rather_than_averaged_over_the_folds_that_have_it(arm):
    # THE DEFECT THIS GUARDS: averaging 0.88 alone would emit a one-fold number in a column every
    # consumer reads as fold-averaged, and it would sit next to a correctly-averaged `ridge` row.
    rows = arm([_rec(0, ens_all=0.88), _rec(1)])
    assert "ridge_ft_k1_ens_all" not in rows
    assert "ridge_ft_k1" in rows


def test_a_nan_rule_is_dropped_because_nan_would_poison_the_macro(arm):
    rows = arm([_rec(0, ens_all=float("nan")), _rec(1, ens_all=0.98)])
    assert "ridge_ft_k1_ens_all" not in rows


def test_rules_are_dropped_independently_of_each_other(arm):
    rows = arm([_rec(0, ens_all=0.88, ens_top1=0.92), _rec(1, ens_top1=0.92)])
    assert "ridge_ft_k1_ens_all" not in rows
    assert rows["ridge_ft_k1_ens_top1"]["value"] == pytest.approx(0.92)


def test_ens_top1_carries_the_same_value_as_ridge_ft_which_is_the_alarm_this_row_exists_for(arm):
    # ens_top1 averages the single selected epoch, so it is a copy of test_c BY CONSTRUCTION.
    # A ledger groupby showing these two apart means the curve and the selection came apart.
    rows = arm([_rec(f, **{r: 0.92 for r in RULES}) for f in (0, 1)])
    assert rows["ridge_ft_k1_ens_top1"]["value"] == pytest.approx(rows["ridge_ft_k1"]["value"])


def test_ensemble_rows_keep_the_arm_regime_and_unit_of_the_row_they_derive_from(arm):
    rows = arm([_rec(f, **{r: 0.93 for r in RULES}) for f in (0, 1)])
    for name in ("ridge_ft_k1", "ridge_ft_k1_ens_all"):
        r = rows[name]
        assert (r["arm_tag"], r["regime"], r["cell"], r["task"], r["tap"], r["split"]) == (
            "pbs50_cd45k_ens", "ws", "S1T1", "onset", "enc12", "test")


def test_every_emitted_row_has_exactly_the_ledger_fields(arm):
    rows = arm([_rec(f, **{r: 0.93 for r in RULES}) for f in (0, 1)])
    for r in rows.values():
        assert set(r) == set(BRL.FIELDS)


# --- the frozen-board reader: score-level ensemble columns ------------------------------------
#
# `_rows_board` splits a shard column on "|" into tap|norm and stamps decoder="ridge". That is
# right for a plain tap ("enc12|std") and WRONG for a score-level ensemble, which is a different
# DECODER over the same features. Left unfixed, a groupby on tap would sprout phantom taps named
# "ens:auto" and a groupby on decoder would show every ensemble hiding inside "ridge".

@pytest.fixture
def board(tmp_path, monkeypatch):
    """Writes one frozen-board shard and returns its rows keyed by (tap, decoder)."""
    sd = tmp_path / "board" / "shards_ens_cs"
    sd.mkdir(parents=True)
    monkeypatch.setattr(BRL, "R6", tmp_path)

    def write(cols):
        json.dump({"kind": "cs", "name": "sub3",
                   "cells": {"onset": {"cells": cols}}}, open(sd / "cs_sub3.json", "w"))
        return {(r["tap"], r["decoder"]): r for r in BRL._rows_board()
                if r["split"] == "test"}
    return write


def test_a_plain_tap_column_is_untouched(board):
    rows = board({"enc12|std": {"test": 0.61, "val": 0.60}})
    assert set(rows) == {("enc12", "ridge")}
    assert rows[("enc12", "ridge")]["norm"] == "std"
    assert rows[("enc12", "ridge")]["value"] == 0.61


def test_a_lambda_ensemble_keeps_its_tap_and_moves_the_rule_into_decoder(board):
    # "lam3:enc12" names its own tap, so the tap survives and only the rule relocates -- which is
    # what makes `groupby(tap).decoder` a legible comparison of rules AT a fixed tap.
    rows = board({"lam3:enc12|std": {"test": 0.6092}})
    assert set(rows) == {("enc12", "ridge_lam3")}


def test_a_tap_ensemble_reports_tap_multi_because_its_members_are_a_set(board):
    # 🔴 The one thing that must NOT happen is inventing tap="ens:auto": a tap is a layer, and
    # this column has no single layer behind it. "multi" says exactly that, and keeps it from
    # being averaged into a per-depth ladder.
    rows = board({"ens:auto|std": {"test": 0.6133}, "ens:top2|std": {"test": 0.6129}})
    assert set(rows) == {("multi", "ridge_ens_auto"), ("multi", "ridge_ens_top2")}


def test_depth_concat_is_also_a_decoder_not_a_tap(board):
    # Concat is the closed/negative axis, but its columns share the "rule:base" shape, so it has
    # to land somewhere honest too -- and its base IS a set of layers.
    rows = board({"cat:enc9+enc12|std": {"test": 0.6088}})
    assert set(rows) == {("multi", "ridge_cat_enc9+enc12")}


def test_time_pooling_keeps_its_tap_because_pooling_is_applied_to_one_tap(board):
    rows = board({"gpool:enc12|std": {"test": 0.6719}, "bpool:enc12|std": {"test": 0.6711}})
    assert set(rows) == {("enc12", "ridge_gpool"), ("enc12", "ridge_bpool")}


def test_the_ensemble_and_its_control_stay_in_the_same_shard_so_the_delta_is_paired(board):
    # The whole reason these columns are trustworthy is that the control is recomputed IN-PATH.
    # If a future edit routed them to different families the pairing would be lost silently.
    rows = board({"enc12|std": {"test": 0.6094}, "ens:auto|std": {"test": 0.6133}})
    fams = {r["family"] for r in rows.values()}
    cells = {r["cell"] for r in rows.values()}
    assert fams == {"board"} and cells == {"sub3"}
