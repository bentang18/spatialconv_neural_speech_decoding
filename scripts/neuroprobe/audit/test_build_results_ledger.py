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
