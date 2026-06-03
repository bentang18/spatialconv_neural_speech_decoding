"""Tests for REF-01 / REF-02 per-clip reference-operator augmentation.

Wires :mod:`speech_decoding.extractors.ref_aug` against the lock memo
``memory/project_v14_ref_aug_input_distribution_lock_2026_05_27.md``.
"""

from __future__ import annotations

import numpy as np
import pytest
from neuralset.base import TimedArray

from speech_decoding.extractors.reference import CARIeegExtractor
from speech_decoding.extractors.ref_aug import (
    SUPPORTED_REF_MODES,
    REF_MODE_TO_IDX,
    REF_MODES,
    RefAugMultiStftView,
    RefIdxExtractor,
    draw_ref_idx,
)


class _FakeEvent:
    """Tiny event stand-in matching the patterns in test_view.py.

    Mirrors ``neuralset.events.etypes.Event`` — the natural identity is
    the ``timeline`` string, not an ``event_id`` (which the upstream
    class does not expose).
    """

    def __init__(self, timeline: str = "tl-0") -> None:
        self.timeline = timeline
        self.start = 0.0
        self.duration = 1.0
        self.frequency = 2048.0


def test_ref_modes_enum_unchanged_default_excludes_laplacian() -> None:
    """``REF_MODES`` keeps the 3-cell enum from the lock memo so the
    int → name lookup stays stable, but ``SUPPORTED_REF_MODES`` drops
    ``"laplacian"`` because the laplacian branch raises
    :class:`NotImplementedError` until ``REF-01-LAP`` lands. Drawing
    the 3-cell default would crash ~33% of clips at runtime.
    """
    assert REF_MODES == ("shaft_car", "bipolar", "laplacian")
    assert SUPPORTED_REF_MODES == ("shaft_car", "bipolar")
    assert REF_MODE_TO_IDX == {"shaft_car": 0, "bipolar": 1, "laplacian": 2}


def test_draw_ref_idx_is_deterministic_per_event_key() -> None:
    """Same (seed, event, start, duration) → same ref_idx (reproducibility)."""
    evt = _FakeEvent("tl-0")
    a = draw_ref_idx(evt, 0.0, 1.0, seed=0)
    b = draw_ref_idx(evt, 0.0, 1.0, seed=0)
    assert a == b


def test_draw_ref_idx_depends_on_seed() -> None:
    """Different seeds change the draw distribution per-event."""
    evt = _FakeEvent("tl-0")
    # Sample many seeds; not all should agree (uniform → flat hash).
    draws = {draw_ref_idx(evt, 0.0, 1.0, seed=s) for s in range(100)}
    assert len(draws) > 1  # not pinned to a single mode


def test_draw_ref_idx_distributes_across_default_modes() -> None:
    """Over many event ids, the draw covers both default modes
    (``shaft_car`` and ``bipolar``) roughly uniformly. ``laplacian``
    has been dropped from ``SUPPORTED_REF_MODES`` until ``REF-01-LAP``
    lands, so we no longer assert a third bucket.
    """
    seen = [0, 0, 0]
    for i in range(300):
        idx = draw_ref_idx(_FakeEvent(f"tl-{i}"), 0.0, 1.0, seed=0)
        seen[idx] += 1
    # Both default modes should be hit a meaningful number of times;
    # ``laplacian`` should never appear since it is not in the default
    # draw set.
    assert seen[0] > 100, f"shaft_car under-drawn: {seen}"
    assert seen[1] > 100, f"bipolar under-drawn: {seen}"
    assert seen[2] == 0, f"laplacian drawn from default set: {seen}"


def test_draw_ref_idx_covers_3_modes_when_ref_modes_overridden() -> None:
    """Caller-opt-in to the 3-cell ``REF_MODES`` still draws all three
    indices (this is what the future REF-01-LAP completion unlocks)."""
    seen = [0, 0, 0]
    for i in range(300):
        idx = draw_ref_idx(
            _FakeEvent(f"tl-{i}"), 0.0, 1.0, seed=0, ref_modes=REF_MODES,
        )
        seen[idx] += 1
    assert all(c > 50 for c in seen), f"3-cell draw too skewed: {seen}"


def test_draw_ref_idx_collapses_to_single_mode_when_pinned() -> None:
    """Sister R-no-ref-aug: ref_modes=('shaft_car',) → always idx=0."""
    for i in range(20):
        assert draw_ref_idx(
            _FakeEvent(f"tl-{i}"), 0.0, 1.0,
            ref_modes=("shaft_car",),
        ) == 0


def test_draw_ref_idx_rejects_empty_ref_modes() -> None:
    with pytest.raises(ValueError, match="ref_modes must be non-empty"):
        draw_ref_idx(_FakeEvent(), 0.0, 1.0, ref_modes=())


def test_draw_ref_idx_raises_when_event_has_no_timeline() -> None:
    """``_event_key`` raises if the event exposes no ``timeline`` —
    otherwise every such event would hash to the same key (``None:…``)
    and the per-clip draw would silently collapse, losing all ref-aug
    entropy. ``neuralset.events.etypes.Event`` and all subclasses
    expose ``timeline`` so this branch is purely defensive against
    malformed stand-ins.
    """

    class _NoTimelineEvent:
        start = 0.0
        duration = 1.0
        frequency = 2048.0

    with pytest.raises(ValueError, match="timeline"):
        draw_ref_idx(_NoTimelineEvent(), 0.0, 1.0, seed=0)


def test_ref_idx_extractor_returns_channel_time_timed_array() -> None:
    """RefIdxExtractor emits per-event ref_idx as a (1, 1) TimedArray —
    NeuralSet's BaseExtractor needs at least one non-time axis so the
    missing-default placeholder (``shape = tensor.shape[:-1]``) is a
    non-empty tuple."""
    extractor = RefIdxExtractor(event_types="Ieeg", seed=0)
    out = extractor._get_timed_array(_FakeEvent("tl-0"), start=0.0, duration=1.0)
    assert isinstance(out, TimedArray)
    arr = np.asarray(out.data)
    assert arr.shape == (1, 1)
    assert arr.dtype == np.int64
    assert 0 <= int(arr.item()) < 3


def test_ref_idx_extractor_validates_ref_modes() -> None:
    with pytest.raises(ValueError, match="unknown modes"):
        RefIdxExtractor(
            event_types="Ieeg", ref_modes=("shaft_car", "nope"),
        )


def test_ref_idx_extractor_reproducible_per_seed() -> None:
    a = RefIdxExtractor(event_types="Ieeg", seed=0)
    b = RefIdxExtractor(event_types="Ieeg", seed=0)
    evt = _FakeEvent("tl-7")
    out_a = a._get_timed_array(evt, 0.0, 1.0)
    out_b = b._get_timed_array(evt, 0.0, 1.0)
    assert int(np.asarray(out_a.data).item()) == int(np.asarray(out_b.data).item())


def test_ref_aug_multi_stft_view_inherits_multi_stft_view() -> None:
    from speech_decoding.extractors.view import MultiStftView

    assert issubclass(RefAugMultiStftView, MultiStftView)


def test_ref_aug_multi_stft_view_defaults_to_2_cell_draw() -> None:
    """Default draw set is ``("shaft_car", "bipolar")`` — laplacian is
    held back until ``REF-01-LAP`` lands so the default extractor does
    not crash ~33% of clips at runtime.
    """
    view = RefAugMultiStftView(event_types="Ieeg", car="shaft", notch_filter=60.0)
    assert view.ref_modes == SUPPORTED_REF_MODES
    assert "laplacian" not in view.ref_modes


def test_ref_aug_multi_stft_view_validates_ref_modes() -> None:
    with pytest.raises(ValueError, match="unknown modes"):
        RefAugMultiStftView(
            event_types="Ieeg", car="shaft", notch_filter=60.0,
            ref_modes=("shaft_car", "bogus"),
        )


def test_ref_aug_multi_stft_view_no_ref_aug_sister_passes_shaft_car_only(
    monkeypatch,
) -> None:
    """Sister R-no-ref-aug = pre-5/27 default: only shaft_car is drawn so
    output equals the un-augmented MultiStftView output byte-for-byte."""
    view = RefAugMultiStftView(
        event_types="Ieeg", car="shaft", notch_filter=60.0,
        ref_modes=("shaft_car",), seed=42,
    )

    rng = np.random.default_rng(0)
    waveform_np = rng.standard_normal((4, 2048)).astype(np.float32)
    fake_waveform_ta = TimedArray(
        frequency=2048.0, start=0.0, duration=1.0, data=waveform_np,
    )

    def _fake_super_get_timed_array(self, event, start, duration):
        return fake_waveform_ta

    monkeypatch.setattr(
        CARIeegExtractor, "_get_timed_array", _fake_super_get_timed_array,
        raising=False,
    )

    out = view._get_timed_array(_FakeEvent("tl-0"), start=0.0, duration=1.0)
    assert isinstance(out, TimedArray)
    arr = np.asarray(out.data)
    # 30-bin filterbank @ hop=128 over 2048 samples → T_bin = 17 (time unchecked).
    assert arr.shape[0] == 4
    assert arr.shape[1] == 30


def test_ref_aug_multi_stft_view_laplacian_raises_until_wired(monkeypatch) -> None:
    """REF-01-LAP gap: drawing 'laplacian' must raise NotImplementedError
    with a memo pointer until the upstream CAR/MNE Laplacian path lands."""
    view = RefAugMultiStftView(
        event_types="Ieeg", car="shaft", notch_filter=60.0,
        ref_modes=("laplacian",), seed=0,
    )

    rng = np.random.default_rng(0)
    waveform_np = rng.standard_normal((4, 2048)).astype(np.float32)
    fake_waveform_ta = TimedArray(
        frequency=2048.0, start=0.0, duration=1.0, data=waveform_np,
    )

    def _fake_super_get_timed_array(self, event, start, duration):
        return fake_waveform_ta

    monkeypatch.setattr(
        CARIeegExtractor, "_get_timed_array", _fake_super_get_timed_array,
        raising=False,
    )

    with pytest.raises(NotImplementedError, match="REF-01"):
        view._get_timed_array(_FakeEvent("tl-0"), start=0.0, duration=1.0)


def test_b_m3_bipolar_sign_matches_lock_memo() -> None:
    """The bipolar proxy must compute ``x_e − x_{e+1}`` (the operator
    the lock memo §Operators table specifies), not ``x_{e+1} − x_e``.
    The downstream ``|STFT|`` makes the forward output sign-invariant,
    so the sign error is silent at runtime — guard it with a unit test
    instead.
    """
    import torch as _torch

    view = RefAugMultiStftView(
        event_types="Ieeg", car="shaft", notch_filter=60.0,
        ref_modes=("bipolar",),
    )
    waveform_t = _torch.tensor(
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]],
        dtype=_torch.float32,
    )
    out = view._apply_ref_operator(waveform_t, sample_rate=2048, ref_mode="bipolar")
    # Expected: channel 0 = x_0 − x_1, channel 1 = x_1 − x_2,
    # channel 2 = x_2 (shape preservation placeholder until REF-01-BIP-RAW).
    expected = _torch.tensor(
        [[-9.0, -18.0, -27.0], [-90.0, -180.0, -270.0], [100.0, 200.0, 300.0]],
        dtype=_torch.float32,
    )
    assert _torch.allclose(out, expected), (
        f"bipolar sign mismatch: expected x_e − x_{{e+1}} = {expected}, got {out}"
    )


def test_ref_aug_multi_stft_view_bipolar_proxy_changes_output(monkeypatch) -> None:
    """Bipolar branch produces a different spec than shaft_car for the same
    waveform — confirms the ref-operator switch actually reaches the STFT."""
    rng = np.random.default_rng(0)
    waveform_np = rng.standard_normal((4, 2048)).astype(np.float32)
    fake_waveform_ta = TimedArray(
        frequency=2048.0, start=0.0, duration=1.0, data=waveform_np,
    )

    def _fake_super_get_timed_array(self, event, start, duration):
        return fake_waveform_ta

    monkeypatch.setattr(
        CARIeegExtractor, "_get_timed_array", _fake_super_get_timed_array,
        raising=False,
    )

    shaft_view = RefAugMultiStftView(
        event_types="Ieeg", car="shaft", notch_filter=60.0,
        ref_modes=("shaft_car",),
    )
    bipolar_view = RefAugMultiStftView(
        event_types="Ieeg", car="shaft", notch_filter=60.0,
        ref_modes=("bipolar",),
    )
    out_shaft = shaft_view._get_timed_array(
        _FakeEvent("tl-0"), start=0.0, duration=1.0,
    )
    out_bipolar = bipolar_view._get_timed_array(
        _FakeEvent("tl-0"), start=0.0, duration=1.0,
    )
    assert not np.allclose(np.asarray(out_shaft.data), np.asarray(out_bipolar.data))


def test_ref_aug_multi_stft_view_seed_pins_per_clip_choice(monkeypatch) -> None:
    """Same seed + same event → identical spec across two view instances."""
    rng = np.random.default_rng(0)
    waveform_np = rng.standard_normal((4, 2048)).astype(np.float32)
    fake_waveform_ta = TimedArray(
        frequency=2048.0, start=0.0, duration=1.0, data=waveform_np,
    )

    def _fake_super_get_timed_array(self, event, start, duration):
        return fake_waveform_ta

    monkeypatch.setattr(
        CARIeegExtractor, "_get_timed_array", _fake_super_get_timed_array,
        raising=False,
    )

    a = RefAugMultiStftView(
        event_types="Ieeg", car="shaft", notch_filter=60.0,
        ref_modes=("shaft_car", "bipolar"), seed=11,
    )
    b = RefAugMultiStftView(
        event_types="Ieeg", car="shaft", notch_filter=60.0,
        ref_modes=("shaft_car", "bipolar"), seed=11,
    )
    out_a = a._get_timed_array(_FakeEvent("tl-3"), start=0.0, duration=1.0)
    out_b = b._get_timed_array(_FakeEvent("tl-3"), start=0.0, duration=1.0)
    np.testing.assert_array_equal(np.asarray(out_a.data), np.asarray(out_b.data))


def test_ref_idx_extractor_inherits_car_ieeg_extractor() -> None:
    """RefIdxExtractor shares the same pydantic base as the other v14 extractors."""
    assert issubclass(RefIdxExtractor, CARIeegExtractor)
