"""T3.5: DCohortStudy scaffold tests — class is importable, all data-loading
methods raise NotImplementedError citing the gating blocker IDs."""

from __future__ import annotations

import pytest

from speech_decoding.studies.cogan_dcohort.study import DCohortStudy


def test_dcohort_imports_with_sample_rate_classvar() -> None:
    # v14 canonical resample target; native rate is mixed (mostly 2048).
    assert DCohortStudy.SAMPLE_RATE_HZ == 2048.0
    assert "DCohort" in DCohortStudy.aliases


def test_dcohort_mains_notch_is_60_hz_us_site() -> None:
    """SCAFFOLD-04: US-sited (Duke); mains = 60 Hz."""
    assert DCohortStudy.MAINS_NOTCH_HZ == 60.0


def test_dcohort_valid_bin_range_is_full_30_bins() -> None:
    """SCAFFOLD-04: resampled to 2048 Hz → 1024 Hz Nyquist → all 30 v14
    filterbank bins valid → half-open (0, 30)."""
    assert DCohortStudy.VALID_BIN_RANGE == (0, 30)


def test_dcohort_87_subjects_from_inventory() -> None:
    """SCAFFOLD-04 + project_d_cohort_data_inventory_2026_06_03:
    87 / 87 speech-subset D-pts pass FS + RAS + .fif gate (D107/D139 earlier
    mis-flagged — recons live under suffix variants D107A/B, D139A/B)."""
    assert DCohortStudy.N_UNIQUE_PATIENTS == 87


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
    s = DCohortStudy(path="/tmp/dcohort_placeholder")
    fn = getattr(s, method)
    with pytest.raises(NotImplementedError) as exc_info:
        result = fn(*args)
        if method == "iter_timelines":
            list(result)
    message = str(exc_info.value)
    for token in ("DP03", "DP05", "DP06", "M07", "M18"):
        assert token in message, f"{method}: missing blocker id {token}"
