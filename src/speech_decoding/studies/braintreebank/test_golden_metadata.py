"""Golden-snapshot regression fixtures for BT metadata (task #19; ledger
``reports/bt_alignment/data_pipeline_bug_ledger.md``).

Freezes the EXACT production output of the metadata-derived quantities the model
routes on, for all 10 vendored BT subjects, in both
``exclude_single_electrode_parcels`` modes:

* ``order_sha``  — sha1 of the per-subject VOLTAGE electrode order
  (``voltage_electrode_order``). This is the electrode-row desync invariant
  (L2/L4/L6): the channel order the loader feeds the front-end, and the row
  order ``support``/``valid_mask`` must line up with. A silent re-order is the
  canonical S9-class bug.
* ``assign_sha`` — sha1 of the dtype-independent per-electrode DK parcel
  assignment (parcel index, or ``-1`` when ``valid=False``). Catches any drift
  in anatomy parsing, the K=80 vocabulary, or the single-electrode-parcel /
  ventricle drop policy.
* ``valid_sum`` / ``n_elec`` — coverage counts (e.g. sub_4's 2 ventricle
  contacts under exclude=False, +5 single-electrode parcels under exclude=True).

WHY THIS, ON TOP OF THE DIFFERENTIAL ORACLE: ``test_anatomy.py`` checks our
electrode order against the *live* upstream neuroprobe package (which must be
importable, and which could itself drift). This is instead a self-contained
FROZEN snapshot — it catches ANY drift from the known-good state, including a
coordinated change where both our code and upstream move together, using only
the vendored ``.cache/braintreebank`` metadata (no voltage, no neuroprobe
import). Values are computed via the PRODUCTION path
(``aligned_voltage_support`` / ``voltage_electrode_order``), so a golden value
encodes real pipeline behavior rather than a reimplementation.

Raw BT data is never committed — only the tiny hashes/counts in
``_golden_metadata.json`` (a few hundred bytes per subject). The metadata files
themselves live under the gitignored ``.cache/braintreebank`` (run on a machine
where they are present; the module skips otherwise).

Regenerate after an INTENTIONAL change::

    .venv/bin/python -m speech_decoding.studies.braintreebank.test_golden_metadata --update

then review the JSON diff before committing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from speech_decoding.studies.braintreebank.anatomy import (
    V14_DK_PARCEL_LABELS,
    aligned_voltage_support,
    atlas_spec,
    voltage_electrode_order,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BT_CACHE = _REPO_ROOT / ".cache" / "braintreebank"
_GOLDEN_PATH = Path(__file__).resolve().parent / "_golden_metadata.json"
_VENDORED_SUBJECTS = tuple(range(1, 11))
_EXCLUDE_MODES = (False, True)


def _sha(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _capture_subject(bt_root: Path, subject_id: int, exclude: bool) -> dict:
    """Production-path metadata snapshot for one (subject, exclude) pair.

    Returns dtype/order-stable scalars + hashes so the golden JSON is tiny and a
    float-dtype change in ``support`` never spuriously fails (we hash the integer
    parcel-assignment semantics, not the raw one-hot bytes).
    """
    order = voltage_electrode_order(str(bt_root), subject_id)
    res = aligned_voltage_support(
        str(bt_root),
        subject_id,
        parcel_labels=V14_DK_PARCEL_LABELS,
        unmapped_policy="zero",
        label_column=atlas_spec("dk")[0],
        exclude_single_electrode_parcels=exclude,
    )
    support = np.asarray(res.support)
    valid = np.asarray(res.valid).astype(bool)
    if support.shape[0] != len(order):
        raise AssertionError(
            f"subject {subject_id}: support rows {support.shape[0]} != "
            f"voltage order {len(order)} (row-alignment broken)"
        )
    # dtype-independent semantic encoding: per-electrode parcel index, -1 if invalid.
    assign = np.where(valid, support.argmax(axis=1), -1).astype(np.int32)
    return {
        "n_elec": len(order),
        "K": int(support.shape[1]),
        "valid_sum": int(valid.sum()),
        "order_sha": _sha("\n".join(order).encode()),
        "assign_sha": _sha(np.ascontiguousarray(assign).tobytes()),
    }


def _build_golden(bt_root: Path) -> dict:
    """Recompute the full golden structure from the vendored metadata."""
    dk_col, dk_labels = atlas_spec("dk")
    dkt_col, dkt_labels = atlas_spec("dkt")
    golden = {
        "_schema": 1,
        "bt_root_basename": bt_root.name,
        "atlas": {
            "dk": {"label_column": dk_col, "K": len(dk_labels)},
            "dkt": {"label_column": dkt_col, "K": len(dkt_labels)},
        },
        "subjects": {},
    }
    for sid in _VENDORED_SUBJECTS:
        golden["subjects"][str(sid)] = {
            f"exclude_{str(ex).lower()}": _capture_subject(bt_root, sid, ex)
            for ex in _EXCLUDE_MODES
        }
    return golden


def _require_cache() -> Path:
    if not (_BT_CACHE / "localization").is_dir():
        pytest.skip(f"vendored BT fixtures absent: {_BT_CACHE}")
    return _BT_CACHE


def _load_golden() -> dict:
    if not _GOLDEN_PATH.is_file():
        pytest.skip(f"golden reference absent: {_GOLDEN_PATH} (run with --update)")
    return json.loads(_GOLDEN_PATH.read_text())


def test_atlas_spec_locked() -> None:
    """DK / DKT (column, K) pairs are frozen — a vocab/column desync is the
    single-source-of-truth failure ``atlas_spec`` exists to prevent."""
    golden = _load_golden()
    for atlas, spec in golden["atlas"].items():
        col, labels = atlas_spec(atlas)
        assert col == spec["label_column"], f"{atlas}: label_column drifted"
        assert len(labels) == spec["K"], f"{atlas}: K drifted"


@pytest.mark.parametrize("subject_id", _VENDORED_SUBJECTS)
@pytest.mark.parametrize("exclude", _EXCLUDE_MODES)
def test_subject_metadata_matches_golden(subject_id: int, exclude: bool) -> None:
    """Per-subject electrode order + parcel assignment + coverage == frozen golden."""
    bt_root = _require_cache()
    golden = _load_golden()
    key = f"exclude_{str(exclude).lower()}"
    if str(subject_id) not in golden["subjects"]:
        pytest.skip(f"subject {subject_id} not in golden")
    expected = golden["subjects"][str(subject_id)][key]
    actual = _capture_subject(bt_root, subject_id, exclude)
    assert actual == expected, (
        f"subject {subject_id} ({key}): metadata drifted from golden.\n"
        f"  expected={expected}\n  actual  ={actual}\n"
        f"If this change is INTENTIONAL, regenerate with --update and review the diff."
    )


def test_sub4_ventricle_and_single_electrode_drops() -> None:
    """Explicit lock on the LG15 edge case: sub_4 drops 2 ventricle contacts under
    exclude=False, and strictly more under exclude=True (single-electrode parcels)."""
    golden = _load_golden()
    s4 = golden["subjects"].get("4")
    if s4 is None:
        pytest.skip("sub_4 not in golden")
    n = s4["exclude_false"]["n_elec"]
    v_false = s4["exclude_false"]["valid_sum"]
    v_true = s4["exclude_true"]["valid_sum"]
    assert v_false < n, "sub_4 should drop the out-of-vocab ventricle contacts"
    assert v_true < v_false, "exclude=True should drop additional single-electrode parcels"


if __name__ == "__main__":
    import sys

    if "--update" not in sys.argv:
        print("usage: python -m ...test_golden_metadata --update")
        raise SystemExit(2)
    if not (_BT_CACHE / "localization").is_dir():
        raise SystemExit(f"cannot update: vendored BT fixtures absent at {_BT_CACHE}")
    golden = _build_golden(_BT_CACHE)
    _GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    n = len(golden["subjects"])
    print(f"wrote {_GOLDEN_PATH} for {n} subjects ({_GOLDEN_PATH.stat().st_size} bytes)")
