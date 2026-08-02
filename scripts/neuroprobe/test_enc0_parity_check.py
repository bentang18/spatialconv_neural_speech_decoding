"""_same must answer, not raise, on every shape the record's non-tap fields actually take.

The risk it guards is asymmetric: a false "same" licenses a wrong experiment at full confidence,
while a false "differs" only costs a re-run. So the cases below pin the false-same direction
hardest -- nested dicts, integer label vectors, and float splits that differ below allclose.
"""
from __future__ import annotations

import numpy as np
import torch

from enc0_parity_check import _same


def test_nested_dicts_of_arrays_compare_elementwise():
    a = {"onset": {"train": np.array([0, 1, 1]), "test": np.array([2])}}
    assert _same(a, {"onset": {"train": np.array([0, 1, 1]), "test": np.array([2])}})
    assert not _same(a, {"onset": {"train": np.array([0, 1, 0]), "test": np.array([2])}})


def test_differing_keys_are_not_same():
    assert not _same({"a": 1}, {"a": 1, "b": 2})
    assert not _same({"a": 1}, {"b": 1})


def test_dict_vs_non_dict_is_not_same_and_does_not_raise():
    assert not _same({"a": 1}, np.array([1]))
    assert not _same(np.array([1]), {"a": 1})


def test_integer_labels_are_exact_not_tolerant():
    """Labels are class indices: 1 vs 2 is a different trial, never 'close enough'."""
    assert not _same(np.array([1, 2, 3]), np.array([1, 2, 4]))
    assert _same(np.array([1, 2, 3]), np.array([1, 2, 3]))


def test_float_fields_use_allclose_but_still_catch_real_drift():
    assert _same(np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-12]))
    assert not _same(np.array([1.0, 2.0]), np.array([1.0, 2.5]))


def test_shape_mismatch_is_not_same():
    assert not _same(np.array([1, 2, 3]), np.array([1, 2]))
    assert not _same(np.array([[1, 2]]), np.array([1, 2]))


def test_tensors_compare_bit_exactly():
    assert _same(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert not _same(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.001]))
    assert not _same(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))


def test_string_arrays_compare(  ):
    """present_parcels / elec labels are string arrays -- kind 'U', not float."""
    assert _same(np.array(["LSTG", "RSTG"]), np.array(["LSTG", "RSTG"]))
    assert not _same(np.array(["LSTG", "RSTG"]), np.array(["RSTG", "LSTG"]))
