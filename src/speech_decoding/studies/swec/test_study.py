"""T3.4: SWECStudy scaffold tests — class is importable, all data-loading
methods raise NotImplementedError citing the gating blocker IDs."""

from __future__ import annotations

import pytest

from speech_decoding.studies.swec.study import SWECStudy


def test_swecstudy_imports_with_audited_classvars() -> None:
    """ClassVars come from the 5/19 SWEC audit memo. These are de-duped
    facts that are safe to encode now (independent of DP03)."""
    assert SWECStudy.N_UNIQUE_PATIENTS == 50
    assert SWECStudy.TOTAL_HOURS == 6672.0
    assert SWECStudy.N_EVENTS == 460
    assert SWECStudy.HARDWARE_BANDPASS_HZ == (0.5, 120.0)
    assert "SWEC" in SWECStudy.aliases


def test_swec_mains_notch_is_50_hz_ch_site() -> None:
    """SCAFFOLD-03: CH-sited (Inselspital Bern); mains = 50 Hz, distinct
    from BT/D/AJILE12 (US 60 Hz)."""
    assert SWECStudy.MAINS_NOTCH_HZ == 50.0


def test_swec_valid_bin_range_is_k0_to_k21() -> None:
    """SCAFFOLD-03 / B1 closure 5/19: hardware band-pass 0.5–120 Hz →
    trainable filterbank bins k0..k21 inclusive → half-open (0, 22)."""
    assert SWECStudy.VALID_BIN_RANGE == (0, 22)


def test_swec_is_phase1_only() -> None:
    """SWEC is anatomy-blind; Phase-2 anatomy-ON pretrain excludes it."""
    assert SWECStudy.PHASE_SCOPE == ("p1",)


@pytest.mark.parametrize(
    "method, args",
    [
        ("_download", ()),
        ("iter_timelines", ()),
        ("_load_timeline_events", ({},)),
        ("_load_raw", ({},)),
    ],
)
def test_data_loading_methods_raise_with_blocker_ids(method: str, args: tuple) -> None:
    s = SWECStudy(path="/tmp/swec_placeholder")
    fn = getattr(s, method)
    with pytest.raises(NotImplementedError) as exc_info:
        result = fn(*args)
        # iter_timelines returns an iterator; consume it to trigger the raise.
        if method == "iter_timelines":
            list(result)
    message = str(exc_info.value)
    for token in ("DP03", "M21", "DP02", "DP06", "M07", "M18"):
        assert token in message, f"{method}: missing blocker id {token}"
