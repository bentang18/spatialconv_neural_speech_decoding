"""T3.5: DCohortStudy scaffold tests — class is importable, all data-loading
methods raise NotImplementedError citing the gating blocker IDs."""

from __future__ import annotations

import pytest

from speech_decoding.studies.cogan_dcohort.study import DCohortStudy


def test_dcohort_imports_with_sample_rate_classvar() -> None:
    assert DCohortStudy.SAMPLE_RATE_HZ == 2000.0
    assert "DCohort" in DCohortStudy.aliases


def test_dcohort_mains_notch_is_60_hz_us_site() -> None:
    """SCAFFOLD-04: US-sited (Duke); mains = 60 Hz."""
    assert DCohortStudy.MAINS_NOTCH_HZ == 60.0


def test_dcohort_valid_bin_range_is_full_30_bins() -> None:
    """SCAFFOLD-04: 2000 Hz acquisition → 1000 Hz Nyquist → all 30 v14
    filterbank bins valid → half-open (0, 30)."""
    assert DCohortStudy.VALID_BIN_RANGE == (0, 30)


def test_dcohort_85_subjects_from_phase2_audit() -> None:
    """SCAFFOLD-04 + project_d_cohort_phase2_cohort_audit_2026_05_23:
    85 / 87 D-pts pass FS + RAS + .fif gate."""
    assert DCohortStudy.N_UNIQUE_PATIENTS == 85


def test_dcohort_phase_scope_is_p1_p2() -> None:
    """D-cohort is anatomy-bearing sEEG; contributes to both P1 and P2."""
    assert DCohortStudy.PHASE_SCOPE == ("p1", "p2")


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
