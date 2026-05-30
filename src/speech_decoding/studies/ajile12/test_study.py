"""SCAFFOLD-05 tests: AJILE12Study structural contract."""

from __future__ import annotations

import pytest

from speech_decoding.studies.ajile12 import AJILE12Study


def test_ajile12_mains_notch_is_60_hz_us_site() -> None:
    """Peterson 2022 cohort is US-sited; mains notch = 60 Hz (matches BT
    + D-cohort; SWEC is 50 Hz CH)."""
    assert AJILE12Study.MAINS_NOTCH_HZ == 60.0


def test_ajile12_valid_bin_range_is_k0_to_k20() -> None:
    """500 Hz Nyquist (1000 Hz acquisition) limits the trainable v14
    30-bin log ⅓-octave filterbank to k0–k20 (per
    docs/neuroprobe/v14_implementation_fix_list.md SCAFFOLD-05).
    Pin the half-open ``(start, stop)`` so the dispatch valid-bin mask
    builder reads a single source."""
    assert AJILE12Study.VALID_BIN_RANGE == (0, 21)  # k0..k20 inclusive → stop=21


def test_ajile12_ecog_share_recorded() -> None:
    """Peterson 2022 Table 2: 89.7% ECoG (rest is mixed-electrode)."""
    assert AJILE12Study.ECOG_FRACTION == pytest.approx(0.897)


def test_ajile12_methods_raise_until_dp03_lands() -> None:
    """Data-loading methods raise NotImplementedError citing DP03 until
    the studies/ contract ratifies."""
    s = AJILE12Study(path="/tmp/ajile12_placeholder")
    with pytest.raises(NotImplementedError, match="DP03"):
        s._download()
    with pytest.raises(NotImplementedError, match="DP03"):
        list(s.iter_timelines())
    with pytest.raises(NotImplementedError, match="DP03"):
        s._load_timeline_events({"x": 1})
    with pytest.raises(NotImplementedError, match="DP03"):
        s._load_raw({"x": 1})


def test_ajile12_aliases_include_canonical_names() -> None:
    """Dispatch-side string lookup uses these aliases."""
    assert "AJILE12" in AJILE12Study.aliases
    assert "Peterson2022" in AJILE12Study.aliases
