"""The coords dump reproduces the DKT tag rule instead of importing it, so it can run in an
environment without neuraltrain. A reproduced rule can drift, and a drifted rule here does
not crash -- it silently paints the wrong electrodes, which is the one failure a brain figure
cannot advertise. So the copy is pinned to the original's OUTPUT, not merely eyeballed.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

# imported at module scope, BEFORE any test stubs out the anatomy module: dispatch_v3 pulls
# in v14_encoder, which imports the real anatomy eagerly and would fail against the stub.
# make_bt_parcel_fn itself imports anatomy lazily, so it still sees the stub when called.
from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn

from scripts.neuroprobe.viz_coords_dump import make_parcel_fn

LABELS = ["LT1", "LT2", "LT3", "RF1"]
PLABELS = ["superiortemporal", "middletemporal", "parsopercularis"]


class _Support:
    def __init__(self, support, electrode_labels):
        self.support = np.asarray(support)
        self.electrode_labels = list(electrode_labels)


def _stub_anatomy(monkeypatch, support):
    """Stand in for the BT anatomy module so the rule is tested, not the CSVs on disk."""
    mod = types.ModuleType("speech_decoding.studies.braintreebank.anatomy")
    mod.atlas_spec = lambda atlas: ("DKT", list(PLABELS))          # type: ignore[attr-defined]
    mod.aligned_voltage_support = (                                 # type: ignore[attr-defined]
        lambda root, subj, **kw: _Support(support, LABELS))
    monkeypatch.setitem(sys.modules, "speech_decoding.studies.braintreebank.anatomy", mod)
    return mod


SUPPORT = np.asarray([
    [0.2, 0.8, 0.0],      # LT1 -> parcel 1 (argmax)
    [1.0, 0.0, 0.0],      # LT2 -> parcel 0
    [0.0, 0.0, 0.0],      # LT3 -> no support anywhere -> the reserved unknown id
    [0.0, 0.1, 0.9],      # RF1 -> parcel 2
])


def test_the_tag_is_the_argmax_of_the_support_row(monkeypatch) -> None:
    _stub_anatomy(monkeypatch, SUPPORT)
    got = make_parcel_fn("/bt")(3, 0, LABELS)
    np.testing.assert_array_equal(got, [1, 0, len(PLABELS), 2])
    assert got.dtype == np.int64


def test_an_unsupported_electrode_gets_the_reserved_id_not_parcel_zero(monkeypatch) -> None:
    """argmax of an all-zero row is 0, which is a REAL parcel. Collapsing 'outside every
    DKT parcel' onto parcel 0 would quietly merge two anatomies into one colour."""
    _stub_anatomy(monkeypatch, SUPPORT)
    got = make_parcel_fn("/bt")(3, 0, LABELS)
    assert got[2] == len(PLABELS) and got[2] != 0


def test_the_output_follows_the_requested_label_order(monkeypatch) -> None:
    """The rows are matched to coordinates positionally, so a function that returned the
    anatomy file's order rather than the caller's would permute the whole figure."""
    _stub_anatomy(monkeypatch, SUPPORT)
    fn = make_parcel_fn("/bt")
    np.testing.assert_array_equal(fn(3, 0, ["RF1", "LT2"]), [2, 0])


def test_a_label_absent_from_the_voltage_order_raises(monkeypatch) -> None:
    _stub_anatomy(monkeypatch, SUPPORT)
    with pytest.raises(KeyError, match="absent"):
        make_parcel_fn("/bt")(3, 0, ["LT1", "NOPE"])


def test_the_copied_rule_matches_dispatch_v3s_original(monkeypatch) -> None:
    """The drift guard. make_bt_parcel_fn is the rule the ENCODE used to write
    parcel_canon; if these two ever disagree the dump's runtime [check] would start
    refusing to write, and the reason would be this copy, not the data."""
    _stub_anatomy(monkeypatch, SUPPORT)
    original = make_bt_parcel_fn("/bt")(3, 0, LABELS).numpy()
    np.testing.assert_array_equal(make_parcel_fn("/bt")(3, 0, LABELS), original)
