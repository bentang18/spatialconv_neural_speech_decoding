from __future__ import annotations

import json
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
    aligned_voltage_coords,
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
_HAS_BT = _BT_CACHE.exists() and (_BT_CACHE / "electrode_labels" / "sub_1").exists()


def _depth_wm_coord_map(subject_id: int) -> dict[str, tuple[float, float, float]]:
    """Ground-truth native (L, I, P) per cleaned electrode label, read raw."""
    dw = pd.read_csv(_BT_CACHE / "localization" / f"sub_{subject_id}" / "depth-wm.csv")
    return {
        clean_bt_electrode_label(str(e)): (float(l), float(i), float(p))
        for e, l, i, p in zip(dw["Electrode"], dw["L"], dw["I"], dw["P"])
    }


def _write_synthetic_bt(tmp_path: Path, labels, coord_rows, *, corrupted=()) -> None:
    """Minimal BT root: electrode_labels.json + optional corrupted + depth-wm.csv."""
    lab_dir = tmp_path / "electrode_labels" / "sub_99"
    lab_dir.mkdir(parents=True)
    (lab_dir / "electrode_labels.json").write_text(json.dumps(list(labels)))
    if corrupted:
        (tmp_path / "corrupted_elec.json").write_text(json.dumps({"sub_99": list(corrupted)}))
    loc = tmp_path / "localization" / "sub_99"
    loc.mkdir(parents=True)
    header = "Electrode,L,I,P,DesikanKilliany\n"
    body = "".join(f"{e},{l},{i},{p},insula\n" for (e, l, i, p) in coord_rows)
    (loc / "depth-wm.csv").write_text(header + body)


@pytest.mark.skipif(not _HAS_BT, reason="BrainTreebank cache absent")
@pytest.mark.parametrize("subject_id", (1, 2, 3, 4))
def test_aligned_voltage_coords_row_identity(subject_id: int) -> None:
    """coords[c] is the native (L,I,P) of voltage_electrode_order[c] — the invariant."""
    order = voltage_electrode_order(_BT_CACHE, subject_id)
    coords = aligned_voltage_coords(_BT_CACHE, subject_id)
    assert coords.shape == (len(order), 3)
    assert coords.dtype == np.float32
    assert np.isfinite(coords).all()  # full coverage: every survivor has a coord row
    truth = _depth_wm_coord_map(subject_id)
    for c in {0, len(order) // 2, len(order) - 1}:
        assert tuple(coords[c]) == truth[order[c]], f"row {c} desync"


@pytest.mark.skipif(not _HAS_BT, reason="BrainTreebank cache absent")
def test_aligned_voltage_support_with_coords_row_aligned() -> None:
    """with_coords=True attaches coords aligned to support rows, present even where
    valid=False (subject 4 has unmapped ventricle contacts)."""
    supp = aligned_voltage_support(
        _BT_CACHE, 4, trial_id=2, unmapped_policy="zero", with_coords=True
    )
    assert supp.coords is not None
    assert supp.coords.shape == (len(supp.electrode_labels), 3)
    assert np.isfinite(supp.coords).all()
    # coords are independent of parcel validity — an unmapped (valid=False) row
    # still carries its physical coordinate.
    if (~supp.valid).any():
        inv = int(np.flatnonzero(~supp.valid)[0])
        assert np.isfinite(supp.coords[inv]).all()
    truth = _depth_wm_coord_map(4)
    step = max(1, len(supp.electrode_labels) // 5)
    for c in range(0, len(supp.electrode_labels), step):
        assert tuple(supp.coords[c]) == truth[supp.electrode_labels[c]]


def test_aligned_voltage_coords_fails_loud_on_missing(tmp_path: Path) -> None:
    """A voltage-order survivor with no depth-wm.csv row is a desync → ValueError."""
    _write_synthetic_bt(
        tmp_path,
        labels=["E1", "E2", "E3"],
        coord_rows=[("E1", 10, 20, 30), ("E2", 11, 21, 31)],  # E3 has no coord row
    )
    with pytest.raises(ValueError, match="desync"):
        aligned_voltage_coords(tmp_path, 99)


def test_aligned_voltage_coords_drops_corrupted_consistently(tmp_path: Path) -> None:
    """A corrupted electrode drops from BOTH the voltage order and coords, in order."""
    _write_synthetic_bt(
        tmp_path,
        labels=["E1", "E2", "E3"],
        coord_rows=[("E1", 10, 20, 30), ("E2", 11, 21, 31), ("E3", 12, 22, 32)],
        corrupted=["E2"],
    )
    order = voltage_electrode_order(tmp_path, 99)
    coords = aligned_voltage_coords(tmp_path, 99)
    assert order == ("E1", "E3")
    assert coords.shape == (2, 3)
    assert tuple(coords[0]) == (10.0, 20.0, 30.0)  # E1
    assert tuple(coords[1]) == (12.0, 22.0, 32.0)  # E3 (E2 skipped, no shift)


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


# Per-session GUARD-1 STATIC drop (baked 2026-06-18, #213). The (subject, trial)
# sessions that drop ≥1 contact — the parametrize list for the realness guard.
# Tracked against the baked per-session dict by
# ``test_extra_bad_electrodes_per_session_is_the_baked_guard1_set``.
_STATIC_DROPPING_SESSIONS = (
    (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2), (2, 6),
    (3, 0), (3, 1), (3, 2),
    (4, 1), (4, 2),
    (6, 0), (6, 1), (6, 4),
    (7, 0), (7, 1),
    (8, 0),
    (9, 0),
    (10, 0), (10, 1),
)


def test_extra_bad_electrodes_per_session_is_the_baked_guard1_set() -> None:
    """Pin the baked GUARD-1 per-session static-exclusion set (#213, 2026-06-18) so
    an accidental edit fails loud. STATIC is now PER-SESSION ``(subject, trial)``:
    25 scanned sessions, 81 dropped contacts, 21 sessions dropping ≥1 (the other 4
    carry an explicit ``()`` clean-session entry). The legacy per-subject "locked-11"
    dict is RETIRED + empty, so a call with no per-session entry (``trial_id=None``
    or an unscanned session) drops nothing."""
    import speech_decoding.studies.braintreebank.anatomy as anatomy

    # legacy per-subject fallback is retired + empty
    assert anatomy._BT_V14_EXTRA_BAD_ELECTRODES == {}

    ps = anatomy._BT_V14_EXTRA_BAD_ELECTRODES_PER_SESSION
    assert len(ps) == 25
    assert sum(len(v) for v in ps.values()) == 81
    assert tuple(sorted(k for k, v in ps.items() if v)) == _STATIC_DROPPING_SESSIONS

    # representative entries: drift across a subject's sessions + explicit clean entry
    assert ps[(2, 2)] == ("LT3d7", "RT1aIa7", "RT1aIa8")
    assert ps[(6, 4)] == ("T2A8",)
    assert ps[(9, 0)] == ("P2e5", "P2e6", "P2e7", "P2e8", "T1c5")
    assert ps[(2, 3)] == ()  # scanned-clean session → explicit empty entry

    # extra_bad_electrodes returns the cleaned per-session set
    assert extra_bad_electrodes(2, 2) == frozenset({"LT3d7", "RT1aIa7", "RT1aIa8"})
    assert extra_bad_electrodes(2, 3) == frozenset()  # explicit clean session
    # no trial / unscanned session → legacy (empty) fallback → no drop
    assert extra_bad_electrodes(2) == frozenset()
    assert extra_bad_electrodes(99, 0) == frozenset()


@pytest.mark.must_pass_before_dispatch
@pytest.mark.parametrize("subject_id,trial_id", _STATIC_DROPPING_SESSIONS)
def test_static_excluded_contacts_are_real_and_actually_dropped(
    subject_id: int, trial_id: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TDD guard against a typo'd static-exclusion label: a label that matches no
    real contact would drop NOTHING and silently leave the flaky electrode on the
    encoder's input. For each per-session ``(subject, trial)`` drop, every named
    contact must (a) be PRESENT in the real montage before the drop and (b) be
    ABSENT after — proving the label is genuine and the exclusion bites. The
    post-drop order also shrinks by exactly the number of named contacts (no over-
    or under-drop). The montage is per-subject (trial-independent); the per-session
    DROP is what varies, so we thread ``trial_id`` through both ends."""
    import speech_decoding.studies.braintreebank.anatomy as anatomy

    _require_vendored(subject_id)
    named = extra_bad_electrodes(subject_id, trial_id)
    assert named, f"sub_{subject_id} trial_{trial_id}: expected ≥1 named drop"

    # Pre-drop baseline: clear BOTH dicts so voltage_electrode_order keeps the flaky
    # contacts. Each named contact must appear here (else it's a typo).
    monkeypatch.setattr(anatomy, "_BT_V14_EXTRA_BAD_ELECTRODES", {})
    monkeypatch.setattr(anatomy, "_BT_V14_EXTRA_BAD_ELECTRODES_PER_SESSION", {})
    full_order = voltage_electrode_order(str(_BT_CACHE), subject_id, trial_id)
    missing = named - set(full_order)
    assert not missing, (
        f"sub_{subject_id} trial_{trial_id}: static labels not in montage: {missing}"
    )

    # Post-drop: restore the dicts; assert each named contact is gone and the count
    # dropped by exactly len(named).
    monkeypatch.undo()
    dropped_order = voltage_electrode_order(str(_BT_CACHE), subject_id, trial_id)
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
def test_extra_bad_per_session_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-session STATIC: a contact listed in the per-session override for
    ``(subject, trial)`` drops ONLY for that trial; a different trial of the same
    subject is untouched. This is the per-session-drift granularity (each session
    decides its own bad set — no cross-trial aggregation heuristic)."""
    import json

    import speech_decoding.studies.braintreebank.anatomy as anatomy

    d = tmp_path / "electrode_labels" / "sub_55"
    d.mkdir(parents=True)
    (d / "electrode_labels.json").write_text(json.dumps(["LA1", "LA2", "LA3", "LA4"]))

    # empty per-session override + empty per-subject -> no drop for any trial
    monkeypatch.setattr(anatomy, "_BT_V14_EXTRA_BAD_ELECTRODES", {})
    monkeypatch.setattr(anatomy, "_BT_V14_EXTRA_BAD_ELECTRODES_PER_SESSION", {})
    assert extra_bad_electrodes(55, 0) == frozenset()
    assert voltage_electrode_order(str(tmp_path), 55, 0) == ("LA1", "LA2", "LA3", "LA4")

    # per-session override: drop LA2 in trial 0 only (raw spelling cleaned on match)
    monkeypatch.setitem(
        anatomy._BT_V14_EXTRA_BAD_ELECTRODES_PER_SESSION, (55, 0), ("LA2*",)
    )
    assert extra_bad_electrodes(55, 0) == frozenset({"LA2"})
    assert extra_bad_electrodes(55, 1) == frozenset()  # other trial untouched
    assert voltage_electrode_order(str(tmp_path), 55, 0) == ("LA1", "LA3", "LA4")
    assert voltage_electrode_order(str(tmp_path), 55, 1) == ("LA1", "LA2", "LA3", "LA4")


@pytest.mark.must_pass_before_dispatch
def test_extra_bad_trial_none_falls_back_to_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``trial_id=None`` (laptop audits / drift guard / pre-bake transition)
    returns the per-subject fallback set, and an explicit trial with NO override
    entry ALSO falls back to per-subject — so threading the trial through the
    production path is byte-identical to today while the override is empty (the
    cache/golden-preserving transition). An explicit trial WITH an override entry
    takes precedence (per-session replaces, does not union with, the subject set)."""
    import speech_decoding.studies.braintreebank.anatomy as anatomy

    monkeypatch.setattr(anatomy, "_BT_V14_EXTRA_BAD_ELECTRODES", {55: ("LB1",)})
    monkeypatch.setattr(
        anatomy, "_BT_V14_EXTRA_BAD_ELECTRODES_PER_SESSION", {(55, 0): ("LA2",)}
    )
    assert extra_bad_electrodes(55) == frozenset({"LB1"})  # no trial -> subject
    assert extra_bad_electrodes(55, 0) == frozenset({"LA2"})  # override wins
    assert extra_bad_electrodes(55, 9) == frozenset({"LB1"})  # no entry -> subject


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


@pytest.mark.must_pass_before_dispatch
def test_extra_bad_per_session_loader_voltage_order_lockstep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DP4 lockstep under PER-SESSION drop: the same trial's bad contact drops from
    BOTH ``bt_load_raw(trial_id=t)`` and ``voltage_electrode_order(..., trial_id=t)``,
    and a DIFFERENT trial keeps it — front-end voltage and support/valid_mask stay
    row-aligned per session (a desync routes voltages into the wrong parcels)."""
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
    rows = np.arange(len(labels), dtype=np.float32)[:, None] * np.ones((1, 64), np.float32)

    monkeypatch.setattr(anatomy, "_BT_V14_EXTRA_BAD_ELECTRODES", {})
    # LA3 bad in trial 0 only
    monkeypatch.setitem(
        anatomy._BT_V14_EXTRA_BAD_ELECTRODES_PER_SESSION, (55, 0), ("LA3",)
    )

    # trial 0: LA3 dropped from both voltage and order
    data0, ch0, _ = bt_load_raw(
        _FakeBT(data=rows.copy(), electrode_labels=list(labels)),
        trial_id=0, subject_id=55,
    )
    assert ch0 == ["LA1", "LA2", "LA4"]
    assert tuple(ch0) == voltage_electrode_order(str(tmp_path), 55, 0)
    np.testing.assert_array_equal(data0[:, 0], np.array([0.0, 1.0, 3.0], np.float32))

    # trial 1: nothing dropped (full montage), both in lockstep
    data1, ch1, _ = bt_load_raw(
        _FakeBT(data=rows.copy(), electrode_labels=list(labels)),
        trial_id=1, subject_id=55,
    )
    assert ch1 == labels
    assert tuple(ch1) == voltage_electrode_order(str(tmp_path), 55, 1)


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
    # Called with no trial_id → the GUARD-1 static drop is PER-SESSION (#213) and
    # the legacy per-subject fallback is retired + empty, so nothing is dropped:
    # all 183 raw voltage contacts survive. (Per-session static-drop realness has
    # its own guard, test_static_excluded_contacts_are_real_and_actually_dropped.)
    # The 2 interior-unmapped Inf-Lat-Vent contacts (LT2bHb3/4) are the only
    # unmapped rows — exactly what this test exercises.
    assert n_voltage == 183
    assert n_mapped == 181  # still exactly 2 unmapped

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


@pytest.mark.parametrize("subject_id", _VENDORED_SUBJECTS)
def test_aligned_voltage_support_lite_equals_all_restricted_to_lite_rows(
    subject_id: int,
) -> None:
    """``aligned_voltage_support(electrode_set="lite")`` aligns support + valid to
    ``lite_voltage_order`` — exactly the "all" support/valid restricted to the
    Lite rows (per-row parcel one-hot is montage-independent). This is the
    load-bearing invariant: support[c] / valid[c] / electrode_tokens[c] all
    describe the same Lite electrode (Lite voltage row c), so the pre-CAR Lite
    loader subset stays row-for-row aligned with the Lite support extractor."""
    _require_vendored(subject_id)

    full_order = voltage_electrode_order(str(_BT_CACHE), subject_id)
    lite_order = lite_voltage_order(str(_BT_CACHE), subject_id)
    mask = lite_voltage_mask(str(_BT_CACHE), subject_id)

    all_sup = aligned_voltage_support(
        str(_BT_CACHE), subject_id, unmapped_policy="zero"
    )
    lite_sup = aligned_voltage_support(
        str(_BT_CACHE), subject_id, unmapped_policy="zero", electrode_set="lite"
    )

    # Row count == Lite montage size; the masked full montage selects the same rows.
    assert lite_sup.support.shape[0] == len(lite_order)
    assert lite_sup.valid.shape[0] == len(lite_order)
    assert mask.sum() == len(lite_order)
    np.testing.assert_array_equal(lite_sup.support, all_sup.support[mask])
    np.testing.assert_array_equal(lite_sup.valid, all_sup.valid[mask])
    # Sanity: the Lite montage is a strict ordered subsequence of the full
    # montage (same order, not just same set — stronger than the support/valid
    # equality, which collapses same-parcel rows).
    assert list(lite_order) == [e for e, keep in zip(full_order, mask) if keep]


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
