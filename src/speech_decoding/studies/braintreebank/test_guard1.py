"""TDD for the GUARD-1 static detectors. Each test injects a contact with a
known pathology into an otherwise-clean cohort and asserts the right detector
fires, the wrong ones don't (orthogonality), and the clip-router + per-subject
aggregation behave per the Ben-locked spec."""

from __future__ import annotations

import numpy as np
import pytest

from speech_decoding.studies.braintreebank import guard1


def _clean_cohort(n_chan: int, n_clip: int, fs: float, base: float = 1.0,
                  seed: int = 0) -> np.ndarray:
    """iid Gaussian cohort, ``(n_chan, n_clip·clip_s·fs)``, per-channel std=base."""
    rng = np.random.default_rng(seed)
    t = int(n_clip * guard1.SPIKE_CLIP_S * fs)
    return (rng.standard_normal((n_chan, t)) * base).astype(np.float64)


# ---------------------------------------------------------------- clean baseline
def test_clean_cohort_fires_nothing() -> None:
    v = _clean_cohort(40, 15, fs=512.0)
    m = guard1.static_bad_mask(v, fs=512.0)
    assert not m["spike"].any(), "clean gaussian dV/dt must not breach 100σ"
    assert not m["noisy"].any()
    assert not m["dead"].any()
    assert not m["static"].any()


# ------------------------------------------------------------------------ noisy
def test_detect_noisy_high_amplitude_and_orthogonal() -> None:
    v = _clean_cohort(40, 15, fs=512.0, seed=1)
    v[7] *= 10.0  # chronic 10x amplitude -> wide rmad
    noisy, _ = guard1.detect_noisy(v)
    dead, _ = guard1.detect_dead(v)
    spike, _ = guard1.detect_spike(v, fs=512.0)
    assert noisy[7] and noisy.sum() == 1, "only the 10x contact is noisy"
    assert not dead[7] and not spike[7], "wide-amplitude ≠ dead/spike (orthogonal)"


# ------------------------------------------------------------------------- dead
def test_detect_dead_attenuated_and_orthogonal() -> None:
    v = _clean_cohort(40, 15, fs=512.0, seed=2)
    v[13] *= 0.1  # chronic attenuation -> rmad ≈ 0.1·median < 0.35
    dead, ratio = guard1.detect_dead(v)
    noisy, _ = guard1.detect_noisy(v)
    assert dead[13] and dead.sum() == 1
    assert ratio[13] < guard1.DEAD_RATIO
    assert not noisy[13], "attenuated contact must not also read noisy"


def test_dead_ratio_threshold_is_relative() -> None:
    # A contact at 0.5x median amplitude is quiet-but-healthy (ratio 0.5 > 0.35),
    # NOT dead; at 0.2x it is dead. Verifies the bar is the locked 0.35 ratio.
    v = _clean_cohort(40, 15, fs=512.0, seed=3)
    v[5] *= 0.5
    v[6] *= 0.2
    dead, _ = guard1.detect_dead(v)
    assert not dead[5] and dead[6]


# ------------------------------------------------------------------------ spike
def _inject_spike(v: np.ndarray, ch: int, clip_idxs: list[int], fs: float,
                  amp: float = 500.0) -> None:
    win = int(round(guard1.SPIKE_CLIP_S * fs))
    for ci in clip_idxs:
        v[ch, ci * win + win // 2] += amp  # one transient per chosen clip


def test_detect_spike_transient_clips_static_and_orthogonal() -> None:
    fs = 512.0
    v = _clean_cohort(40, 15, fs=fs, seed=4)
    _inject_spike(v, 9, clip_idxs=[2, 7, 11], fs=fs)  # 3/15 clips = 20% > 1%
    spike, clip_frac = guard1.detect_spike(v, fs=fs)
    noisy, _ = guard1.detect_noisy(v)
    dead, _ = guard1.detect_dead(v)
    assert spike[9] and spike.sum() == 1
    # 3 of ~14 clips corrupted (dV/dt windows the diff array, T−1) ≫ 1% router.
    assert clip_frac[9] > guard1.SPIKE_CLIP_FRAC
    assert 0.15 < clip_frac[9] < 0.25
    assert not noisy[9] and not dead[9], "robust spread ⊥ transients"


def test_spike_sparse_below_1pct_router_not_static() -> None:
    # 1 corrupted clip out of 150 = 0.67% < 1% ⇒ the router KEEPS the contact
    # (its bad clip handled by CLIP), even though a spike event is present.
    fs = 256.0
    v = _clean_cohort(5, 150, fs=fs, seed=5)
    _inject_spike(v, 3, clip_idxs=[40], fs=fs)
    spike, clip_frac = guard1.detect_spike(v, fs=fs)
    # 1 of ~149 clips ≈ 0.67% < 1% router bar.
    assert 0.0 < clip_frac[3] < guard1.SPIKE_CLIP_FRAC
    assert not spike[3], "sub-1% clip-corruption is CLIP's job, not STATIC"


# ------------------------------------------------------- cohort-z lower bound
def test_cohort_z_lower_bounded_by_rel_floor() -> None:
    # The memo's reason `dead` uses a ratio not cohort-z: rel_floor=0.15 bounds
    # cohort-z ≥ −1/0.15 = −6.67, so a symmetric −8 bar could never fire.
    v = np.array([1.0, 1.0, 1.0, 1.0, 1e-6])
    z = guard1.cohort_z(v)
    assert z.min() >= -1.0 / guard1.COHORT_REL_FLOOR - 1e-6
    assert z[-1] == pytest.approx(-1.0 / guard1.COHORT_REL_FLOOR, rel=1e-3)


# ---------------------------------------------------------- static union mask
def test_static_is_union_of_three() -> None:
    fs = 512.0
    v = _clean_cohort(40, 15, fs=fs, seed=6)
    v[7] *= 10.0                      # noisy
    v[13] *= 0.1                      # dead
    _inject_spike(v, 9, [2, 7, 11], fs=fs)  # spike
    m = guard1.static_bad_mask(v, fs=fs)
    idx = set(np.flatnonzero(m["static"]).tolist())
    assert idx == {7, 9, 13}


# -------------------------------------------------- per-subject aggregation
def test_aggregate_subject_spike_majority_noisy_dead_any() -> None:
    per_trial = [
        {"spike": {"A", "B"}, "noisy": {"C"}, "dead": set()},
        {"spike": {"B"}, "noisy": set(), "dead": {"D"}},
        {"spike": set(), "noisy": set(), "dead": set()},
    ]
    agg = guard1.aggregate_subject(per_trial)
    # B fired spike in 2/3 trials (majority) → drop; A only 1/3 → kept.
    assert agg["spike"] == frozenset({"B"})
    # noisy/dead drop on ANY trial.
    assert agg["noisy"] == frozenset({"C"})
    assert agg["dead"] == frozenset({"D"})
    assert agg["static"] == frozenset({"B", "C", "D"})


def test_aggregate_subject_empty() -> None:
    agg = guard1.aggregate_subject([])
    assert agg["static"] == frozenset()


# ----------------------------------------------- signature collector core
def test_classify_from_signature_matches_live_scan() -> None:
    # The JSON-collector path must classify identically to a live scan. Build a
    # cohort with one of each pathology, run static_bad_mask (live), then feed
    # its emitted (clip_frac, rmad) signature to classify_from_signature and
    # assert the fired label sets agree.
    fs = 512.0
    v = _clean_cohort(40, 15, fs=fs, seed=6)
    v[7] *= 10.0                            # noisy
    v[13] *= 0.1                            # dead
    _inject_spike(v, 9, [2, 7, 11], fs=fs)  # spike
    m = guard1.static_bad_mask(v, fs=fs)
    labels = [f"E{i}" for i in range(40)]
    cls = guard1.classify_from_signature(labels, m["clip_frac"], m["rmad"])
    assert cls["spike"] == {"E9"}
    assert cls["noisy"] == {"E7"}
    assert cls["dead"] == {"E13"}


def test_classify_from_signature_thresholds_are_locked() -> None:
    # A signature straddling each bar: contact just over / just under fires / not.
    labels = ["a", "b", "c", "d"]
    # rmad cohort: three ~1.0, one tiny (dead) — median≈1.0.
    rmad = np.array([1.0, 1.0, 1.0, 0.2])          # d: ratio 0.2 < 0.35 ⇒ dead
    clip = np.array([0.02, 0.005, 0.0, 0.0])        # a: 2% > 1% ⇒ spike; b: 0.5% not
    cls = guard1.classify_from_signature(labels, clip, rmad)
    assert cls["spike"] == {"a"}
    assert cls["dead"] == {"d"}
    assert cls["noisy"] == set()
