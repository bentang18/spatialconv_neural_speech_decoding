from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from speech_decoding.studies.braintreebank.anatomy import (
    DEFAULT_SUPPORT_BIAS_EPS,
    V14_DK_PARCEL_LABELS,
    _MIN_PARCEL_VALID_FRACTION,
    aligned_voltage_support,
    build_hard_public_bt_label_support,
    bt_label_vocabulary,
    clean_bt_electrode_label,
    extra_bad_electrodes,
    lite_voltage_mask,
    lite_voltage_order,
    load_public_bt_anatomy,
    support_attention_bias,
    voltage_electrode_order,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BT_CACHE = _REPO_ROOT / ".cache" / "braintreebank"
_NEUROPROBE_UPSTREAM = _REPO_ROOT / ".cache" / "neuroprobe_upstream"
_VENDORED_SUBJECTS = tuple(range(1, 11))


def test_load_public_bt_anatomy_cleans_electrode_labels(tmp_path: Path) -> None:
    path = tmp_path / "localization" / "sub_1"
    path.mkdir(parents=True)
    (path / "depth-wm.csv").write_text(
        "Electrode,DesikanKilliany,Hemisphere\n"
        "A*1,superiortemporal,L\n"
        "B#2,insula,L\n"
    )

    anatomy = load_public_bt_anatomy(tmp_path, 1)

    assert anatomy["Subject"].tolist() == ["sub_1", "sub_1"]
    assert anatomy["Electrode"].tolist() == ["A1", "B2"]
    assert anatomy["DesikanKilliany"].tolist() == ["superiortemporal", "insula"]


def test_bt_label_vocabulary_can_include_hemisphere() -> None:
    table = pd.DataFrame(
        {
            "Electrode": ["L1", "R1", "L2"],
            "Hemisphere": ["L", "R", "L"],
            "DesikanKilliany": ["insula", "insula", "superiortemporal"],
        }
    )

    assert bt_label_vocabulary([table]) == ("insula", "superiortemporal")
    assert bt_label_vocabulary([table], include_hemisphere=True) == (
        "L:insula",
        "L:superiortemporal",
        "R:insula",
    )


def test_build_hard_public_bt_label_support_is_one_hot_in_channel_order() -> None:
    anatomy = pd.DataFrame(
        {
            "Electrode": ["E1", "E2", "E3"],
            "DesikanKilliany": ["insula", "superiortemporal", "insula"],
        }
    )

    result = build_hard_public_bt_label_support(
        ["E3", "E1", "E2"],
        anatomy,
        ["insula", "superiortemporal"],
    )

    assert result.kind == "hard_public_bt_label"
    assert result.electrode_labels == ("E3", "E1", "E2")
    assert result.parcel_labels == ("insula", "superiortemporal")
    np.testing.assert_array_equal(
        result.support,
        np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_build_hard_public_bt_label_support_can_split_by_hemisphere() -> None:
    anatomy = pd.DataFrame(
        {
            "Electrode": ["L1", "R1"],
            "Hemisphere": ["L", "R"],
            "DesikanKilliany": ["insula", "insula"],
        }
    )

    result = build_hard_public_bt_label_support(
        ["L1", "R1"],
        anatomy,
        ["L:insula", "R:insula"],
        include_hemisphere=True,
    )

    np.testing.assert_array_equal(
        result.support,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_build_hard_public_bt_label_support_emits_valid_mask() -> None:
    anatomy = pd.DataFrame(
        {"Electrode": ["E1", "E2"], "DesikanKilliany": ["insula", "superiortemporal"]}
    )
    result = build_hard_public_bt_label_support(
        ["E1", "E2"], anatomy, ["insula", "superiortemporal"],
    )
    np.testing.assert_array_equal(result.valid, np.array([True, True]))


def test_build_hard_public_bt_label_support_rejects_missing_electrode() -> None:
    anatomy = pd.DataFrame(
        {"Electrode": ["E1"], "DesikanKilliany": ["insula"]}
    )

    with pytest.raises(KeyError, match="missing BT anatomy rows"):
        build_hard_public_bt_label_support(["E1", "E2"], anatomy, ["insula"])


def test_build_hard_public_bt_label_support_rejects_unknown_label() -> None:
    anatomy = pd.DataFrame(
        {"Electrode": ["E1"], "DesikanKilliany": ["unknown_region"]}
    )

    with pytest.raises(KeyError, match="absent from parcel vocabulary"):
        build_hard_public_bt_label_support(["E1"], anatomy, ["insula"])


def test_build_hard_support_zero_policy_keeps_unmapped_in_place() -> None:
    """``unmapped_policy='zero'`` -> zero row + valid=False at the true index for
    both a missing-anatomy electrode (E2) and an out-of-vocab label (E4); the
    surrounding rows keep their positions (no re-pack)."""
    anatomy = pd.DataFrame(
        {
            "Electrode": ["E1", "E3", "E4"],
            "DesikanKilliany": ["insula", "superiortemporal", "unknown_region"],
        }
    )
    result = build_hard_public_bt_label_support(
        ["E1", "E2", "E3", "E4"],
        anatomy,
        ["insula", "superiortemporal"],
        unmapped_policy="zero",
    )
    np.testing.assert_array_equal(
        result.support,
        np.array(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32
        ),
    )
    np.testing.assert_array_equal(
        result.valid, np.array([True, False, True, False])
    )


# --- Real-data guards over the vendored fixtures (skip if absent) ------------


def _require_vendored(subject_id: int) -> None:
    labels = _BT_CACHE / "electrode_labels" / f"sub_{subject_id}" / "electrode_labels.json"
    if not labels.exists():
        pytest.skip(f"vendored BT fixtures absent: {labels}")


def _upstream_electrode_labels_or_skip(subject_id: int) -> tuple[str, ...]:
    """Real ``BrainTreebankSubject.electrode_labels`` from the vendored upstream
    clone, pointed at the vendored fixtures. Skips if either is unavailable."""
    _require_vendored(subject_id)
    if not _NEUROPROBE_UPSTREAM.exists():
        pytest.skip(f"vendored neuroprobe_upstream absent: {_NEUROPROBE_UPSTREAM}")
    # config.ROOT_DIR is read from the env at import time; force it to the
    # vendored fixtures before the first neuroprobe import.
    os.environ["ROOT_DIR_BRAINTREEBANK"] = str(_BT_CACHE)
    if str(_NEUROPROBE_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_NEUROPROBE_UPSTREAM))
    try:
        from neuroprobe.braintreebank_subject import BrainTreebankSubject
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"neuroprobe upstream not importable: {exc}")
    subject = BrainTreebankSubject(
        subject_id, cache=False, coordinates_type="cortical"
    )
    return tuple(subject.electrode_labels)


@pytest.mark.must_pass_before_dispatch
@pytest.mark.parametrize("subject_id", _VENDORED_SUBJECTS)
def test_voltage_order_matches_upstream(subject_id: int) -> None:
    """Drift guard: our replicated ``voltage_electrode_order`` must equal the
    real upstream ``BrainTreebankSubject.electrode_labels`` (order + set) for
    every vendored subject, MINUS the deliberate v14 flaky-contact exclusion
    (``extra_bad_electrodes``). Subtracting the intentional drop keeps the guard
    sensitive to UNintended divergence in the corrupted / trigger /
    missing-coordinate filter while permitting the v14 static exclusion."""
    bad = extra_bad_electrodes(subject_id)
    expected = tuple(
        e for e in _upstream_electrode_labels_or_skip(subject_id)
        if clean_bt_electrode_label(e) not in bad
    )
    actual = voltage_electrode_order(str(_BT_CACHE), subject_id)
    assert actual == expected


_STATIC_EXCLUDED_SUBJECTS = (2, 4, 7, 8, 9)


def test_extra_bad_electrodes_dict_is_the_locked_11() -> None:
    """Pin the finalized Ben-approved static-exclusion list (2026-06-14) so an
    accidental edit fails loud. The 11 contacts: their subjects, labels, and total
    count. ``extra_bad_electrodes`` returns the cleaned set per subject and an
    empty set for any subject not listed."""
    import speech_decoding.studies.braintreebank.anatomy as anatomy

    assert anatomy._BT_V14_EXTRA_BAD_ELECTRODES == {
        2: ("LT2aA2", "LT3a8"),
        4: ("LT2aA11", "LT2bHb12", "LF3cIc10", "LF3aOFa16"),
        7: ("LF3bOFb1",),
        8: ("F3cIc8", "T3H12"),
        9: ("P2e6", "P2e8"),
    }
    total = sum(len(v) for v in anatomy._BT_V14_EXTRA_BAD_ELECTRODES.values())
    assert total == 11
    # the literal parametrize list below must track the dict keys
    assert tuple(sorted(anatomy._BT_V14_EXTRA_BAD_ELECTRODES)) == _STATIC_EXCLUDED_SUBJECTS
    assert extra_bad_electrodes(4) == frozenset(
        {"LT2aA11", "LT2bHb12", "LF3cIc10", "LF3aOFa16"}
    )
    assert extra_bad_electrodes(99) == frozenset()  # unlisted subject → no drop


@pytest.mark.must_pass_before_dispatch
@pytest.mark.parametrize("subject_id", _STATIC_EXCLUDED_SUBJECTS)
def test_static_excluded_contacts_are_real_and_actually_dropped(
    subject_id: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TDD guard against a typo'd static-exclusion label: a label that matches no
    real contact would drop NOTHING and silently leave the flaky electrode on the
    encoder's input. For each listed subject, every named contact must (a) be
    PRESENT in the real montage before the drop and (b) be ABSENT after — proving
    the label is genuine and the exclusion bites. The post-drop order also shrinks
    by exactly the number of named contacts (no over- or under-drop)."""
    import speech_decoding.studies.braintreebank.anatomy as anatomy

    _require_vendored(subject_id)
    named = extra_bad_electrodes(subject_id)

    # Pre-drop baseline: clear the static dict so voltage_electrode_order keeps the
    # flaky contacts. Each named contact must appear here (else it's a typo).
    monkeypatch.setattr(anatomy, "_BT_V14_EXTRA_BAD_ELECTRODES", {})
    full_order = voltage_electrode_order(str(_BT_CACHE), subject_id)
    missing = named - set(full_order)
    assert not missing, f"sub_{subject_id}: static labels not in montage: {missing}"

    # Post-drop: restore the dict; assert each named contact is gone and the count
    # dropped by exactly len(named).
    monkeypatch.undo()
    dropped_order = voltage_electrode_order(str(_BT_CACHE), subject_id)
    assert named.isdisjoint(dropped_order)
    assert len(dropped_order) == len(full_order) - len(named)


@pytest.mark.must_pass_before_dispatch
def test_voltage_order_raises_on_duplicate_labels(tmp_path: Path) -> None:
    """L4 fail-loud single-source guard: a duplicate electrode label surviving
    cleaning/drop is ambiguous row identity (the DP2 collision precondition) — row
    c would name two physical contacts — and must RAISE, not silently propagate to
    support / valid_mask / the front-end scatter."""
    import json

    d = tmp_path / "electrode_labels" / "sub_77"
    d.mkdir(parents=True)
    (d / "electrode_labels.json").write_text(json.dumps(["LA1", "LA2", "LA1"]))
    with pytest.raises(ValueError, match="duplicate electrode"):
        voltage_electrode_order(str(tmp_path), 77)


@pytest.mark.must_pass_before_dispatch
def test_aligned_voltage_support_propagates_duplicate_guard(tmp_path: Path) -> None:
    """The single-source guard must protect the whole support/valid path:
    ``aligned_voltage_support`` derives order from ``voltage_electrode_order``
    first, so an ambiguous montage raises before any anatomy is loaded."""
    import json

    d = tmp_path / "electrode_labels" / "sub_77"
    d.mkdir(parents=True)
    (d / "electrode_labels.json").write_text(json.dumps(["LA1", "LA1"]))
    with pytest.raises(ValueError, match="duplicate electrode"):
        aligned_voltage_support(str(tmp_path), 77)


@pytest.mark.must_pass_before_dispatch
@pytest.mark.parametrize("subject_id", _VENDORED_SUBJECTS)
def test_voltage_order_unique_for_all_vendored_subjects(subject_id: int) -> None:
    """The dup guard must not trip on any real subject (0/10 have ambiguous
    labels) — the live cohort passes the single-source ambiguity check."""
    _require_vendored(subject_id)
    order = voltage_electrode_order(str(_BT_CACHE), subject_id)
    assert len(set(order)) == len(order)


@pytest.mark.must_pass_before_dispatch
def test_extra_bad_electrodes_dropped_from_voltage_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v14 flaky-contact static exclusion: an electrode listed in
    ``_BT_V14_EXTRA_BAD_ELECTRODES`` is removed from ``voltage_electrode_order``
    (so support/valid_mask lose it), the surviving rows keep their relative order,
    and ``extra_bad_electrodes`` returns the cleaned set. The mark (``*``) must be
    matched after cleaning, not literally."""
    import json

    import speech_decoding.studies.braintreebank.anatomy as anatomy

    d = tmp_path / "electrode_labels" / "sub_55"
    d.mkdir(parents=True)
    (d / "electrode_labels.json").write_text(
        json.dumps(["LA1", "LA2*", "LA3", "LA4"])
    )

    # baseline: nothing excluded -> full cleaned order
    assert voltage_electrode_order(str(tmp_path), 55) == ("LA1", "LA2", "LA3", "LA4")
    assert extra_bad_electrodes(55) == frozenset()

    # inject a bad contact (raw spelling; helper cleans before matching)
    monkeypatch.setitem(anatomy._BT_V14_EXTRA_BAD_ELECTRODES, 55, ("LA2*",))
    assert extra_bad_electrodes(55) == frozenset({"LA2"})
    assert voltage_electrode_order(str(tmp_path), 55) == ("LA1", "LA3", "LA4")
    # an unrelated subject is untouched
    assert extra_bad_electrodes(56) == frozenset()


@pytest.mark.must_pass_before_dispatch
def test_extra_bad_exclusion_keeps_loader_and_voltage_order_in_lockstep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DP4 lockstep: the SAME bad contact must drop from BOTH the front-end voltage
    (``bt_load_raw``) and ``voltage_electrode_order`` (support/valid_mask), leaving
    their channel orders byte-identical. A desync here routes voltages into the
    wrong parcels."""
    import json
    from dataclasses import dataclass

    import speech_decoding.studies.braintreebank.anatomy as anatomy
    from speech_decoding.studies.braintreebank.loader import bt_load_raw

    @dataclass
    class _FakeBT:
        data: np.ndarray
        electrode_labels: list[str]

        def get_all_electrode_data(self, trial_id: int) -> np.ndarray:
            return self.data

    labels = ["LA1", "LA2", "LA3", "LA4"]
    d = tmp_path / "electrode_labels" / "sub_55"
    d.mkdir(parents=True)
    (d / "electrode_labels.json").write_text(json.dumps(labels))

    # distinct per-row signal so a wrong-row drop is detectable
    rows = np.arange(len(labels), dtype=np.float32)[:, None] * np.ones((1, 64), np.float32)
    monkeypatch.setitem(anatomy._BT_V14_EXTRA_BAD_ELECTRODES, 55, ("LA3",))

    data, ch_names, _ = bt_load_raw(
        _FakeBT(data=rows.copy(), electrode_labels=list(labels)),
        trial_id=1, subject_id=55,
    )
    order = voltage_electrode_order(str(tmp_path), 55)

    assert ch_names == ["LA1", "LA2", "LA4"]
    assert tuple(ch_names) == order  # byte-identical -> support/valid stay aligned
    # the dropped row (LA3, value 2.0) is gone; survivors keep their values
    np.testing.assert_array_equal(data[:, 0], np.array([0.0, 1.0, 3.0], np.float32))


def test_aligned_voltage_support_sub4_interior_unmapped() -> None:
    """sub_4 has 2 voltage contacts (Inf-Lat-Vent) outside K=80 that sit in the
    interior of the voltage order. ``unmapped_policy='zero'`` must zero exactly
    those rows + set valid=False at their true positions, leaving all others
    mapped (real-data C1 regression)."""
    _require_vendored(4)
    result = aligned_voltage_support(
        str(_BT_CACHE), 4,
        parcel_labels=V14_DK_PARCEL_LABELS,
        unmapped_policy="zero",
    )
    n_voltage = len(result.electrode_labels)
    n_mapped = int(result.valid.sum())
    # 183 raw voltage contacts minus the 4 v14 flaky-contact static drops
    # (LT2aA11, LT2bHb12, LF3cIc10, LF3aOFa16 — all previously mapped) = 179.
    # The 2 interior-unmapped Inf-Lat-Vent contacts (LT2bHb3/4) are untouched.
    assert n_voltage == 179
    assert n_mapped == 177  # still exactly 2 unmapped

    unmapped_idx = np.flatnonzero(~result.valid)
    assert unmapped_idx.tolist() == [
        result.electrode_labels.index("LT2bHb3"),
        result.electrode_labels.index("LT2bHb4"),
    ]
    # the unmapped rows are interior (not a trailing block) and zeroed
    assert unmapped_idx.max() < n_voltage - 1
    assert result.support[unmapped_idx].sum() == 0.0
    # valid[c] <=> support[c] nonzero, for every row
    np.testing.assert_array_equal(result.valid, result.support.sum(axis=1) > 0)


@pytest.mark.must_pass_before_dispatch
def test_aligned_voltage_support_fails_loud_on_coverage_collapse(
    tmp_path: Path,
) -> None:
    """LG15: an S9-class namespace divergence (depth-wm.csv electrode labels stop
    matching the electrode_labels.json voltage namespace) drops most/all
    electrodes to valid=False under ``unmapped_policy='zero'``. The per-subject
    coverage floor must convert that silent collapse into a loud failure."""
    import json

    labels_dir = tmp_path / "electrode_labels" / "sub_88"
    labels_dir.mkdir(parents=True)
    (labels_dir / "electrode_labels.json").write_text(
        json.dumps(["X1", "X2", "X3", "X4"])
    )
    loc_dir = tmp_path / "localization" / "sub_88"
    loc_dir.mkdir(parents=True)
    # depth-wm.csv names a DISJOINT electrode namespace (Y* vs the X* montage):
    # every voltage electrode misses its anatomy row -> 0/4 valid.
    (loc_dir / "depth-wm.csv").write_text(
        "Electrode,DesikanKilliany,Hemisphere\n"
        "Y1,insula,L\nY2,superiortemporal,L\nY3,insula,R\nY4,insula,R\n"
    )

    with pytest.raises(ValueError, match=r"below the 0\.50 floor"):
        aligned_voltage_support(str(tmp_path), 88, unmapped_policy="zero")
    # message names the ledger entry for traceability
    with pytest.raises(ValueError, match="LG15"):
        aligned_voltage_support(str(tmp_path), 88, unmapped_policy="zero")


@pytest.mark.must_pass_before_dispatch
@pytest.mark.parametrize("subject_id", _VENDORED_SUBJECTS)
def test_parcel_support_coverage_holds_for_all_vendored_subjects(
    subject_id: int,
) -> None:
    """Real-cohort negative control for LG15: every live subject maps well above
    the floor (measured 2026-06-13: 9/10 at 1.000, sub_4 at 0.989). Pinning >=0.98
    catches even a partial coverage regression, not just a catastrophic collapse;
    the production guard floor stays conservative at 0.5 to never false-fire."""
    _require_vendored(subject_id)
    result = aligned_voltage_support(
        str(_BT_CACHE), subject_id, unmapped_policy="zero"
    )
    n_total = int(result.valid.shape[0])
    fraction = int(result.valid.sum()) / n_total
    assert fraction >= 0.98, (
        f"sub_{subject_id}: parcel coverage {fraction:.3f} dropped below the "
        f"measured live range — possible depth-wm.csv namespace drift (LG15)"
    )
    assert fraction >= _MIN_PARCEL_VALID_FRACTION



# --- Neuroprobe-Lite electrode-set parity (L1 local + L2 upstream drift) -----


@pytest.mark.parametrize("subject_id", _VENDORED_SUBJECTS)
def test_lite_voltage_mask_aligns_and_subsets(subject_id: int) -> None:
    """L1 (local): ``lite_voltage_mask`` is over the SAME voltage order as
    ``voltage_electrode_order`` (row-for-row), and the realized Lite order
    set-equals the Lite list intersected with the montage — reproducing
    upstream's ``[full.index(e) for e in lite if e in full]`` subset as a set."""
    _require_vendored(subject_id)
    from speech_decoding.studies.braintreebank._neuroprobe_lite_tables import (
        NEUROPROBE_LITE_ELECTRODES,
    )

    order = voltage_electrode_order(str(_BT_CACHE), subject_id)
    mask = lite_voltage_mask(str(_BT_CACHE), subject_id)
    assert mask.shape == (len(order),)

    lite_labels = [
        clean_bt_electrode_label(e)
        for e in NEUROPROBE_LITE_ELECTRODES[f"btbank{subject_id}"]
    ]
    lite_set = set(lite_labels)
    # mask[c] True iff voltage electrode c is in the Lite set, at its true index.
    expected_mask = np.array([e in lite_set for e in order], dtype=bool)
    np.testing.assert_array_equal(mask, expected_mask)

    realized = lite_voltage_order(str(_BT_CACHE), subject_id)
    # Set-parity with upstream's Lite-order subset (intersection of lite list
    # with the montage); order differs (we keep voltage order — pool-invariant).
    upstream_subset = [e for e in lite_labels if e in set(order)]
    assert set(realized) == set(upstream_subset)
    assert len(realized) == len(set(realized))  # no dupes


def test_vendored_lite_table_matches_upstream() -> None:
    """L2 (drift guard): the vendored ``NEUROPROBE_LITE_ELECTRODES`` must equal
    the pinned upstream ``neuroprobe.config.NEUROPROBE_LITE_ELECTRODES`` exactly.
    Skips off-DCC / when the upstream clone is absent."""
    if not _NEUROPROBE_UPSTREAM.exists():
        pytest.skip(f"vendored neuroprobe_upstream absent: {_NEUROPROBE_UPSTREAM}")
    os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", str(_BT_CACHE))
    if str(_NEUROPROBE_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_NEUROPROBE_UPSTREAM))
    try:
        from neuroprobe.config import NEUROPROBE_LITE_ELECTRODES as UPSTREAM
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"neuroprobe upstream config not importable: {exc}")
    from speech_decoding.studies.braintreebank._neuroprobe_lite_tables import (
        NEUROPROBE_LITE_ELECTRODES as VENDORED,
        UPSTREAM_PIN,
    )

    # Compare as plain dicts of lists (upstream may use tuples/lists).
    vend = {k: list(v) for k, v in VENDORED.items()}
    up = {k: list(v) for k, v in UPSTREAM.items()}
    assert vend == up, (
        f"vendored Lite table drifted from upstream pin {UPSTREAM_PIN}; "
        "regenerate _neuroprobe_lite_tables.py"
    )


def test_support_attention_bias_is_log_support_plus_eps() -> None:
    support = np.array([[1.0, 0.0]], dtype=np.float32)

    bias = support_attention_bias(support, eps=1e-3)

    np.testing.assert_allclose(
        bias,
        np.log(np.array([[1.001, 0.001]], dtype=np.float32)),
        rtol=1e-6,
    )


def test_support_attention_bias_rejects_negative_support() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        support_attention_bias(np.array([[-1.0]], dtype=np.float32))


def test_support_attention_bias_default_eps_is_v14_prior_strength() -> None:
    assert DEFAULT_SUPPORT_BIAS_EPS == 1e-2
    bias = support_attention_bias(np.array([[1.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(
        bias,
        np.log(np.array([[1.01, 0.01]], dtype=np.float32)),
        rtol=1e-6,
    )
