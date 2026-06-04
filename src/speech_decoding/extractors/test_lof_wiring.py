"""D2 wiring tests: MNE-LOF per-session bad-channel drop in MultiStftView (T1.7).

The LOF math itself is covered in ``test_quality.py``. These pin the *wiring*:
the per-clip ``raw.info["bads"]`` stamp (intersection / no-op / missing-key), the
``drop_bads`` requirement, the drop-count report (Ben review gate), and the
load-bearing assumption that the ``car=None`` pre-CAR sibling carries the
prepared private state so the fit can read filtered, pre-CAR voltage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from speech_decoding.extractors.view import MultiStftView


class _StubRaw:
    """Minimal stand-in: ``_stamp_lof_bads`` only reads ``ch_names`` and writes
    ``info["bads"]``."""

    def __init__(self, ch_names: list[str]) -> None:
        self.ch_names = list(ch_names)
        self.info = {"bads": []}


class _StubEvent:
    def __init__(self, uid: str, subject_id: int = 3) -> None:
        self._uid = uid
        self._sid = subject_id

    def _splittable_event_uid(self) -> str:
        return self._uid

    def _get_field_or_extra(self, field: str):
        return {"subject_id": self._sid}[field]


def _view(**kw) -> MultiStftView:
    # LOF-on now requires a report path (model_validator); inject one for the
    # wiring tests that aren't specifically exercising that guard.
    if kw.get("lof_bad_channels") and kw.get("drop_bads") and "lof_report_path" not in kw:
        kw["lof_report_path"] = "/tmp/_lof_wiring_test.json"
    v = MultiStftView(event_types="Ieeg", **kw)
    v._session_bads = {}  # don't inherit any mutable-default sharing
    return v


def test_stamp_intersects_with_present_channels() -> None:
    v = _view(lof_bad_channels=True, drop_bads=True)
    v._session_bads = {"s1": ["A2", "ZZZ"]}  # ZZZ not present in this clip
    v._bads_ready = True
    raw = _StubRaw(["A1", "A2", "A3"])
    v._stamp_lof_bads(raw, _StubEvent("s1"))
    assert raw.info["bads"] == ["A2"]


def test_stamp_unions_with_preexisting_bads() -> None:
    # Loader-marked bads must be preserved, LOF bads added (no clobber, no dup).
    v = _view(lof_bad_channels=True, drop_bads=True)
    v._session_bads = {"s1": ["A2", "A1"]}  # A1 already pre-marked below
    v._bads_ready = True
    raw = _StubRaw(["A1", "A2", "A3"])
    raw.info["bads"] = ["A1"]  # pre-marked by the loader
    v._stamp_lof_bads(raw, _StubEvent("s1"))
    assert raw.info["bads"] == ["A1", "A2"]  # A1 kept once, A2 added


def test_stamp_is_noop_before_fit_armed() -> None:
    v = _view(lof_bad_channels=True, drop_bads=True)
    v._session_bads = {"s1": ["A2"]}
    v._bads_ready = False  # prepare's shape-probe runs through here pre-fit
    raw = _StubRaw(["A1", "A2", "A3"])
    v._stamp_lof_bads(raw, _StubEvent("s1"))
    assert raw.info["bads"] == []


def test_stamp_is_noop_when_lof_disabled() -> None:
    v = _view(lof_bad_channels=False)
    v._bads_ready = True
    raw = _StubRaw(["A1", "A2"])
    v._stamp_lof_bads(raw, _StubEvent("s1"))
    assert raw.info["bads"] == []


def test_stamp_missing_session_raises_loud() -> None:
    v = _view(lof_bad_channels=True, drop_bads=True)
    v._session_bads = {"s1": ["A2"]}
    v._bads_ready = True
    raw = _StubRaw(["A1", "A2"])
    with pytest.raises(KeyError, match="no fitted bad-channel set for session"):
        v._stamp_lof_bads(raw, _StubEvent("UNSEEN"))


def test_construction_requires_drop_bads() -> None:
    # lof on but drop_bads off would silently flag-and-never-drop — fail at
    # construction, before any run starts.
    with pytest.raises(ValidationError, match="requires drop_bads=True"):
        MultiStftView(
            event_types="Ieeg", lof_bad_channels=True, drop_bads=False,
            lof_report_path="/tmp/x.json",
        )


def test_construction_requires_report_path() -> None:
    # lof on but no report path would drop-but-never-report — defeats the Ben
    # review gate. Fail at construction.
    with pytest.raises(ValidationError, match="requires lof_report_path"):
        MultiStftView(event_types="Ieeg", lof_bad_channels=True, drop_bads=True)


def test_lof_off_needs_no_report_path() -> None:
    # The default (LOF off) must construct freely with no report path — the
    # maiden run depends on this.
    v = MultiStftView(event_types="Ieeg")
    assert v.lof_bad_channels is False
    assert v.lof_report_path is None


def test_report_path_excluded_from_cache_uid_but_data_knobs_kept() -> None:
    # lof_report_path is output-location only → must NOT perturb the STFT cache
    # uid; the data-determining lof_* knobs MUST stay in it.
    v = _view(lof_bad_channels=True, drop_bads=True)
    excluded = v._exclude_from_cache_uid()
    assert "lof_report_path" in excluded
    for data_knob in ("lof_bad_channels", "lof_threshold", "lof_n_neighbors", "lof_ch_type"):
        assert data_knob not in excluded


def test_fit_bads_armed_before_super_prepare(monkeypatch) -> None:
    # THE GATE-C SHOWSTOPPER PIN: _fit_session_bads + _bads_ready=True must run
    # BEFORE super().prepare(), or super().prepare()'s _get_data materialization
    # memoizes the pre-LOF (no-drop) raw under a key blind to _bads_ready and the
    # drop is silently defeated. Record call order; assert fit-then-super and that
    # bads are armed by the time super().prepare runs.
    from neuralset.extractors import neuro

    v = _view(lof_bad_channels=True, drop_bads=True)
    calls: list[str] = []

    def fake_fit(_obj):
        calls.append("fit")

    def fake_super_prepare(self, _obj):
        # the invariant super().prepare relies on: bads already armed here.
        calls.append(f"super(bads_ready={self._bads_ready})")

    monkeypatch.setattr(v, "_fit_session_bads", fake_fit)
    monkeypatch.setattr(neuro.MneRaw, "prepare", fake_super_prepare)
    v.prepare(obj=None)

    assert calls == ["fit", "super(bads_ready=True)"]


def test_precar_sibling_disables_car_and_carries_private_state() -> None:
    # Load-bearing assumption of the pre-CAR fit: model_copy(update=) flips car
    # off (so _get_data returns filtered PRE-CAR voltage) while carrying the
    # prepared private state (e.g. _channels) so _get_data still works.
    v = _view(car="shaft", lof_bad_channels=True, drop_bads=True)
    v._session_bads = {"keep": ["X1"]}
    sib = v.model_copy(update={"car": None})
    assert sib.car is None
    assert v.car == "shaft"  # original untouched
    assert sib._session_bads == {"keep": ["X1"]}  # private state carried over


def test_report_aggregates_and_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "lof.json"
    v = _view(lof_threshold=1.5, lof_n_neighbors=20, lof_report_path=str(out))
    report = [
        {"session": "s1", "subject_id": 3, "n_channels": 100, "n_bad": 4,
         "bad_channels": ["A1", "A2", "A3", "A4"]},
        {"session": "s2", "subject_id": 3, "n_channels": 120, "n_bad": 2,
         "bad_channels": ["B1", "B2"]},
        {"session": "s3", "subject_id": 7, "n_channels": 90, "n_bad": 0,
         "bad_channels": []},
    ]
    v._log_and_write_lof_report(report)
    assert out.is_file()  # nested parent created
    payload = json.loads(out.read_text())
    assert payload["total_sessions"] == 3
    assert payload["total_bad"] == 6
    assert payload["total_channels"] == 310
    assert payload["by_subject"]["3"] == {"n_bad": 6, "n_channels": 220}
    assert payload["by_subject"]["7"] == {"n_bad": 0, "n_channels": 90}
    assert payload["threshold"] == 1.5
    assert payload["n_neighbors"] == 20


def test_report_without_path_does_not_write(tmp_path: Path) -> None:
    v = _view(lof_report_path=None)
    # No path set -> log only, no file. (Just exercise the no-crash path.)
    v._log_and_write_lof_report(
        [{"session": "s1", "subject_id": 3, "n_channels": 10, "n_bad": 1,
          "bad_channels": ["A1"]}]
    )
    assert list(tmp_path.iterdir()) == []
