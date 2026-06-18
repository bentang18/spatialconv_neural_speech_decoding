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


def test_session_signature_roundtrips_through_collector() -> None:
    # The #207 scan side: static_bad_mask -> session_signature -> JSON must be
    # consumable by the #208 collector and reproduce the same drop. Build a cohort
    # with one of each pathology, run the live scan, serialize, then (a) assert the
    # payload is JSON-clean with the collector-required keys, (b) classify_from_
    # signature reproduces the fired sets, (c) collate_sessions recovers the static.
    import json

    fs = 512.0
    v = _clean_cohort(40, 15, fs=fs, seed=6)
    v[7] *= 10.0                            # noisy
    v[13] *= 0.1                            # dead
    _inject_spike(v, 9, [2, 7, 11], fs=fs)  # spike
    sig = guard1.static_bad_mask(v, fs=fs)
    labels = [f"E{i}" for i in range(40)]

    payload = guard1.session_signature(2, 4, labels, sig)
    json.dumps(payload)  # raises if any value is not JSON-serializable
    for key in ("subject", "trial", "labels", "clip_frac", "rmad"):
        assert key in payload
    assert payload["subject"] == 2 and payload["trial"] == 4
    assert payload["spike"] == ["E9"]
    assert payload["noisy"] == ["E7"]
    assert payload["dead"] == ["E13"]

    cls = guard1.classify_from_signature(
        payload["labels"], np.asarray(payload["clip_frac"]),
        np.asarray(payload["rmad"]),
    )
    assert cls["spike"] == set(payload["spike"])
    assert cls["noisy"] == set(payload["noisy"])
    assert cls["dead"] == set(payload["dead"])

    per_subject = guard1.collate_sessions([payload])
    assert per_subject[2]["static"] == frozenset({"E7", "E9", "E13"})


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


# ------------------------------------------- #208 collector: collate + diff
def _session(subject: int, trial: int, labels, clip_frac, rmad) -> dict:
    return {
        "subject": subject, "trial": trial, "labels": list(labels),
        "clip_frac": list(clip_frac), "rmad": list(rmad),
    }


def test_collate_sessions_aggregates_and_classifies_per_subject() -> None:
    # Subject 2, two trials. rmad cohort median≈1.0 in both.
    #   d: dead (ratio 0.2 < 0.35) in BOTH trials → chronic dead.
    #   a: spike (2% > 1%) in BOTH trials → spike majority → static.
    labels = ["a", "b", "c", "d"]
    rmad = [1.0, 1.0, 1.0, 0.2]
    s1 = _session(2, 0, labels, [0.02, 0.0, 0.0, 0.0], rmad)
    s2 = _session(2, 1, labels, [0.02, 0.0, 0.0, 0.0], rmad)
    out = guard1.collate_sessions([s1, s2])
    assert set(out) == {2}
    sub = out[2]
    assert sub["n_trials"] == 2
    assert sub["dead"] == frozenset({"d"})
    assert sub["spike"] == frozenset({"a"})        # 2/2 majority
    assert sub["static"] == frozenset({"a", "d"})
    # both fired in every trial → chronic, none variable
    assert sub["chronic"] == frozenset({"a", "d"})
    assert sub["variable"] == frozenset()


def test_collate_sessions_chronic_vs_variable_split() -> None:
    # 'a' spikes in only 1 of 3 trials → variable AND below spike-majority (kept
    # OUT of static); 'd' dead in all 3 → chronic + static.
    labels = ["a", "b", "c", "d"]
    rmad = [1.0, 1.0, 1.0, 0.2]
    sess = [
        _session(7, 0, labels, [0.02, 0.0, 0.0, 0.0], rmad),  # a spikes
        _session(7, 1, labels, [0.0, 0.0, 0.0, 0.0], rmad),
        _session(7, 2, labels, [0.0, 0.0, 0.0, 0.0], rmad),
    ]
    out = guard1.collate_sessions(sess)[7]
    assert out["n_trials"] == 3
    assert "a" in out["variable"] and "a" not in out["chronic"]
    assert out["spike"] == frozenset()              # 1/3 < majority → not static
    assert out["chronic"] == frozenset({"d"})
    assert out["static"] == frozenset({"d"})


def test_collate_sessions_groups_multiple_subjects() -> None:
    labels = ["a", "b"]
    rmad = [1.0, 0.2]                                # b dead in any trial it appears
    sess = [
        _session(2, 0, labels, [0.0, 0.0], rmad),
        _session(4, 0, labels, [0.0, 0.0], rmad),
    ]
    out = guard1.collate_sessions(sess)
    assert set(out) == {2, 4}
    assert out[2]["dead"] == frozenset({"b"}) and out[4]["dead"] == frozenset({"b"})


def test_diff_static_drops_added_removed_unchanged() -> None:
    would_be = {2: frozenset({"X", "Y"}), 9: frozenset({"P2e6"})}
    locked = {2: frozenset({"Y", "Z"}), 8: frozenset({"Q"})}
    d = guard1.diff_static_drops(would_be, locked)
    # union of subjects {2, 8, 9}
    assert set(d) == {2, 8, 9}
    assert d[2] == {"added": ["X"], "removed": ["Z"], "unchanged": ["Y"]}
    # subject 8 only in locked → all removed
    assert d[8] == {"added": [], "removed": ["Q"], "unchanged": []}
    # subject 9 only in would_be → all added
    assert d[9] == {"added": ["P2e6"], "removed": [], "unchanged": []}


def test_diff_against_locked11_self_consistent() -> None:
    # Reproduce the locked-11 set exactly → zero added/removed everywhere.
    from speech_decoding.studies.braintreebank.anatomy import (
        _BT_V14_EXTRA_BAD_ELECTRODES,
        extra_bad_electrodes,
    )

    locked = {s: extra_bad_electrodes(s) for s in _BT_V14_EXTRA_BAD_ELECTRODES}
    d = guard1.diff_static_drops(locked, locked)
    for sid, rec in d.items():
        assert rec["added"] == [] and rec["removed"] == [], sid
        assert set(rec["unchanged"]) == set(locked[sid])
