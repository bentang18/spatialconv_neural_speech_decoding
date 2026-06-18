"""Tests for `bt_load_raw()` against a `BrainTreebankSubject`-shaped stub.

Real h5 reads only happen on DCC; the laptop test exercises the wrapping
contract — shape, dtype, channel-name ↔ row alignment, sfreq passthrough.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pytest

from speech_decoding.extractors.reference import CARIeegExtractor, parse_shaft
from speech_decoding.studies.braintreebank.anatomy import clean_bt_electrode_label
from speech_decoding.studies.braintreebank.loader import bt_load_raw

# Real BT per-subject label JSONs live in the untracked vendored cache (not in
# CI). The synthetic fixtures below run everywhere; the all-subjects test skips
# when the cache is absent. `parents[4]` == repo root (mirrors test_anatomy.py).
_BT_CACHE = Path(__file__).resolve().parents[4] / ".cache" / "braintreebank"

# Two real mark geometries. sub_2 has 34 TRAILING-marked contacts (22 `*` + 12
# `#`, across 3 shafts) — the only geometry that fragments `parse_shaft` into
# singleton shafts (the silent-zeroing bug). sub_7/9/10 carry a MID-LABEL `*`
# (`RF2I*1`) that keeps its trailing digit and groups fine even raw. The loader
# strips both. `_SUB2_RAW_LABELS` magnifies the bug into one synthetic shaft;
# d1d checks the real per-shaft split across the vendored cohort.
_SUB2_RAW_LABELS = [f"RT1aIa{i}*" for i in range(1, 35)]


@dataclass
class _FakeBT:
    data: np.ndarray
    electrode_labels: list[str]

    def get_all_electrode_data(self, trial_id: int) -> np.ndarray:
        assert trial_id == 1
        return self.data


def test_bt_load_raw_returns_raw_voltage_at_native_rate() -> None:
    n_ch, n_t = 4, 2048
    rng = np.random.default_rng(0)
    fake = _FakeBT(
        data=rng.standard_normal((n_ch, n_t)).astype(np.float64),
        electrode_labels=[f"E{i:03d}" for i in range(n_ch)],
    )

    data, ch_names, sfreq = bt_load_raw(fake, trial_id=1, subject_id=1)

    assert data.shape == (n_ch, n_t)
    assert data.dtype == np.float32  # cast from float64
    assert ch_names == [f"E{i:03d}" for i in range(n_ch)]
    assert sfreq == 2048.0


def test_bt_load_raw_resamples_s9_native_1024_to_2048() -> None:
    """LG1: S9 is distributed at 1024 Hz; the loader polyphase-resamples it UP to
    the pipeline's 2048 Hz. The sample count doubles, wall-clock duration is
    preserved, returned sfreq is 2048, and the result is a faithful interpolation
    (not zero-stuffing/hold) — a low-freq sinusoid below both Nyquists must land on
    the analytic 2048-grid waveform."""
    native_rate, n_native = 1024, 4096  # 4 s at 1024 Hz
    t_native = np.arange(n_native) / native_rate
    sine = np.sin(2 * np.pi * 10.0 * t_native).astype(np.float64)  # 10 Hz, 1 channel
    fake = _FakeBT(data=sine[None, :], electrode_labels=["E000"])

    data, ch_names, sfreq = bt_load_raw(fake, trial_id=1, subject_id=9)

    assert sfreq == 2048.0
    assert data.shape == (1, 2 * n_native)  # 1024 -> 2048 doubles the samples
    assert data.dtype == np.float32
    # Wall-clock duration preserved: n_native/1024 == n_resampled/2048.
    assert n_native / native_rate == data.shape[1] / sfreq
    # Faithful interpolation: interior samples (away from FIR edges) match the
    # analytic 10 Hz sine on the 2048 grid.
    t_2048 = np.arange(data.shape[1]) / sfreq
    analytic = np.sin(2 * np.pi * 10.0 * t_2048)
    interior = slice(256, -256)
    assert np.max(np.abs(data[0, interior] - analytic[interior])) < 1e-2


def test_bt_load_raw_requires_subject_id() -> None:
    """`subject_id` is a required keyword (no silent default) so a non-2048
    subject can never slip through un-resampled — the S9 failure mode."""
    fake = _FakeBT(
        data=np.zeros((2, 100), dtype=np.float32),
        electrode_labels=["a", "b"],
    )
    with pytest.raises(TypeError):
        bt_load_raw(fake, trial_id=1)  # type: ignore[call-arg]


def test_bt_load_raw_rejects_label_count_mismatch() -> None:
    fake = _FakeBT(
        data=np.zeros((4, 100), dtype=np.float32),
        electrode_labels=["only_two", "labels"],
    )
    with pytest.raises(ValueError, match="electrode_labels len"):
        bt_load_raw(fake, trial_id=1, subject_id=1)


def test_bt_load_raw_rejects_non_2d_voltage() -> None:
    fake = _FakeBT(
        data=np.zeros((4,), dtype=np.float32),
        electrode_labels=["x", "y", "z", "w"],
    )
    with pytest.raises(ValueError, match="expected"):
        bt_load_raw(fake, trial_id=1, subject_id=1)


# --- PRE-CAR Lite montage subset (electrode_set="lite") ----------------------


def test_bt_load_raw_lite_subsets_to_lite_montage(monkeypatch) -> None:
    """electrode_set="lite" applies the Neuroprobe-Lite subset at the voltage
    boundary (PRE-CAR), order-preserving, so a Lite result references zero
    out-of-budget contacts and the rows == lite_voltage_order (row-aligned with
    the Lite support / valid_mask)."""
    import speech_decoding.studies.braintreebank.loader as loader_mod

    monkeypatch.setattr(loader_mod, "lite_electrode_set", lambda sid: frozenset({"E1", "E3"}))
    data = np.arange(4 * 8, dtype=np.float64).reshape(4, 8)
    fake = _FakeBT(data=data, electrode_labels=["E0", "E1", "E2", "E3"])

    out, ch_names, sfreq = bt_load_raw(fake, trial_id=1, subject_id=1, electrode_set="lite")

    assert ch_names == ["E1", "E3"]  # voltage order preserved, Lite subset
    assert np.array_equal(out, data[[1, 3]].astype(np.float32))
    assert sfreq == 2048.0


def test_bt_load_raw_lite_composes_after_static_drop(monkeypatch) -> None:
    """STATIC drop runs first, then the Lite subset over the survivors. A Lite
    electrode that is also STATIC-bad is still dropped (STATIC wins), and the
    composed order stays a subsequence of voltage_electrode_order — keeping the
    front-end voltage rows in lockstep with lite_voltage_order."""
    import speech_decoding.studies.braintreebank.loader as loader_mod

    monkeypatch.setattr(loader_mod, "extra_bad_electrodes", lambda sid, tid=None: frozenset({"E1"}))
    monkeypatch.setattr(
        loader_mod, "lite_electrode_set", lambda sid: frozenset({"E1", "E2", "E3"})
    )
    data = np.arange(4 * 8, dtype=np.float64).reshape(4, 8)
    fake = _FakeBT(data=data, electrode_labels=["E0", "E1", "E2", "E3"])

    out, ch_names, _ = bt_load_raw(fake, trial_id=1, subject_id=2, electrode_set="lite")

    # E1 dropped by STATIC even though it is in the Lite set; E2, E3 survive both.
    assert ch_names == ["E2", "E3"]
    assert np.array_equal(out, data[[2, 3]].astype(np.float32))


def test_bt_load_raw_all_is_byte_identical_default(monkeypatch) -> None:
    """electrode_set defaults to "all" and never invokes the Lite subset — the
    BT-FULL pretrain path is unchanged (its existing cache stays valid)."""
    import speech_decoding.studies.braintreebank.loader as loader_mod

    def _boom(sid):  # pragma: no cover - must never be called under "all"
        raise AssertionError("lite_electrode_set called on the 'all' montage")

    monkeypatch.setattr(loader_mod, "lite_electrode_set", _boom)
    data = np.arange(4 * 8, dtype=np.float64).reshape(4, 8)
    fake = _FakeBT(data=data, electrode_labels=["E0", "E1", "E2", "E3"])

    out, ch_names, _ = bt_load_raw(fake, trial_id=1, subject_id=1)

    assert ch_names == ["E0", "E1", "E2", "E3"]
    assert np.array_equal(out, data.astype(np.float32))


# --- drop_static=False: full-montage load for the Guard-1 static re-derivation --


def test_bt_load_raw_drop_static_false_keeps_full_montage(monkeypatch) -> None:
    """drop_static=False bypasses the STATIC exclusion so the Guard-1 precompute
    re-derives the bad set over the FULL montage (incl. the contacts it will later
    drop). The default (True) still drops them, so the cache path is unchanged."""
    import speech_decoding.studies.braintreebank.loader as loader_mod

    monkeypatch.setattr(
        loader_mod, "extra_bad_electrodes", lambda sid, tid=None: frozenset({"E1"})
    )
    data = np.arange(4 * 8, dtype=np.float64).reshape(4, 8)
    fake = _FakeBT(data=data, electrode_labels=["E0", "E1", "E2", "E3"])

    out_d, names_d, _ = bt_load_raw(fake, trial_id=1, subject_id=2)
    assert names_d == ["E0", "E2", "E3"]  # default drops the STATIC contact

    out_f, names_f, _ = bt_load_raw(
        fake, trial_id=1, subject_id=2, drop_static=False
    )
    assert names_f == ["E0", "E1", "E2", "E3"]  # full montage retained
    assert np.array_equal(out_f, data.astype(np.float32))


def test_bt_load_raw_lite_requires_static_drop() -> None:
    """electrode_set="lite" is defined over the post-STATIC survivors, so the
    drop_static=False full-montage mode is incompatible with it — guard against a
    silently non-lite montage."""
    fake = _FakeBT(data=np.zeros((2, 8)), electrode_labels=["E0", "E1"])
    with pytest.raises(ValueError, match="drop_static"):
        bt_load_raw(
            fake, trial_id=1, subject_id=1, electrode_set="lite", drop_static=False
        )


# --- WS-D1: `*`/`#` mark -> singleton shaft -> shaft-CAR zeroing regression ---


def _make_raw(ch_names: list[str], data: np.ndarray, sfreq: float = 200.0) -> mne.io.RawArray:
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")
    return mne.io.RawArray(np.asarray(data, dtype=np.float64), info, verbose=False)


def test_d1a_raw_trailing_mark_fragments_shaft_parser_into_singletons() -> None:
    """(a) On RAW sub_2 labels the trailing `*` defeats `parse_shaft`'s
    numeric-suffix match -> 34 distinct singleton stems; cleaning collapses them
    back to one real `RT1aIa` shaft."""
    raw_stems = {parse_shaft(label)[0] for label in _SUB2_RAW_LABELS}
    assert len(raw_stems) == 34  # every contact its own shaft — the bug
    clean_stems = {parse_shaft(clean_bt_electrode_label(label))[0] for label in _SUB2_RAW_LABELS}
    assert clean_stems == {"RT1aIa"}  # one real shaft — the fix


def test_d1b_loader_chnames_equal_independent_expected_no_mark_survives() -> None:
    """(b) `bt_load_raw` strips marks so its `ch_names` equal a HAND-SPELLED
    expected list (independent of the SUT cleaner — catches a wrong/partial
    strip), no `*`/`#` survives, and the `parse_shaft` grouping actually differs
    from the raw labels (cleaning is not a no-op). Mixes trailing + mid-label +
    plain marks."""
    labels = ["RT1aIa1*", "RT1aIa2*", "RF2I*1", "RF2I*2", "LF1#", "LF2"]
    expected = ["RT1aIa1", "RT1aIa2", "RF2I1", "RF2I2", "LF1", "LF2"]
    fake = _FakeBT(
        data=np.zeros((len(labels), 16), dtype=np.float32),
        electrode_labels=labels,
    )
    _, ch_names, _ = bt_load_raw(fake, trial_id=1, subject_id=1)

    assert ch_names == expected
    assert not any("*" in n or "#" in n for n in ch_names)
    assert [parse_shaft(lab) for lab in labels] != [parse_shaft(n) for n in ch_names]


def test_d1c_loader_clean_prevents_shaft_car_zeroing() -> None:
    """(c) Differential control: shaft-CAR on the RAW sub_2 labels zeros all 34
    contacts (singleton self-subtraction — the data loss); after `bt_load_raw`
    cleans them into one shaft, shaft-CAR leaves nonzero residuals and no channel
    is wiped."""
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((34, 128)) + np.arange(34)[:, None]).astype(np.float32)

    raw_out = CARIeegExtractor(car="shaft")._apply_car(_make_raw(_SUB2_RAW_LABELS, data.copy()))
    assert np.all(np.isclose(raw_out._data, 0.0))  # positive control: the bug

    fake = _FakeBT(data=data, electrode_labels=list(_SUB2_RAW_LABELS))
    _, ch_names, _ = bt_load_raw(fake, trial_id=1, subject_id=1)
    fixed_out = CARIeegExtractor(car="shaft")._apply_car(_make_raw(ch_names, data.copy()))
    all_zero = np.all(np.isclose(fixed_out._data, 0.0), axis=1)
    assert not all_zero.any()  # the fix: the 34 zeroed contacts are gone


def test_d1d_loader_cleans_every_mark_across_all_real_subjects() -> None:
    """(b, spec-literal "all subjects") Over the real per-subject
    `electrode_labels.json` for every vendored BT subject, `bt_load_raw` strips
    every `*`/`#` so its `ch_names` equal an INDEPENDENT in-test strip and no
    mark survives to the shaft parser / DK support table. Exercises the
    mid-label cohort (sub_7/9/10) the synthetic fixtures cannot. Skips when the
    untracked vendored cache is absent (CI without BT fixtures)."""
    labels_root = _BT_CACHE / "electrode_labels"
    if not labels_root.exists():
        pytest.skip(f"vendored BT fixtures absent: {labels_root}")

    marked_present: set[int] = set()
    checked = 0
    for sid in range(1, 11):
        path = labels_root / f"sub_{sid}" / "electrode_labels.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        if any("*" in lab or "#" in lab for lab in raw):
            marked_present.add(sid)
        expected = [lab.replace("*", "").replace("#", "") for lab in raw]  # independent of SUT
        fake = _FakeBT(
            data=np.zeros((len(raw), 4), dtype=np.float32),
            electrode_labels=list(raw),
        )
        _, ch_names, _ = bt_load_raw(fake, trial_id=1, subject_id=1)
        assert ch_names == expected
        assert not any("*" in n or "#" in n for n in ch_names)
        checked += 1

    assert checked >= 1
    # Coverage guard: the known marked cohort that is present MUST have carried
    # marks, else the cleaning path was never exercised (a vacuous pass).
    known_marked_present = {
        s for s in (2, 7, 9, 10)
        if (labels_root / f"sub_{s}" / "electrode_labels.json").exists()
    }
    assert known_marked_present  # at least one marked subject exercised
    assert known_marked_present <= marked_present
