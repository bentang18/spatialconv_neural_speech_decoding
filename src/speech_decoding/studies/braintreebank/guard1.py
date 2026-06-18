"""GUARD-1 self-calibrating static bad-electrode detectors.

Spec: ``memory/project_guard1_self_calibrating_rule_set_2026_06_17``. Three
voltage-domain detectors, run per session on **pre-CAR** notch+HPF voltage
``(C, T)``; ``STATIC = spike ∪ noisy ∪ dead``. Thresholds Ben-locked 2026-06-17:

  spike  robust-z(dV/dt) > 100σ inner; a 5 s clip is corrupted if it holds ≥1
         such event; the contact is a STATIC drop only if **>1%** of its clips
         are corrupted (the static-vs-CLIP router — chronic ⇒ ~100% clips,
         sparse ⇒ stays, its bad clips handled by the CLIP layer).
  noisy  cohort-z(rmad of filtered V) > **+8** — chronic high amplitude
         (floating / railing / wide). rmad (robust spread), NOT raw std, so it
         is orthogonal to the intermittent transients ``spike`` already owns.
  dead   ratio (rmad / cohort-median rmad) < **0.35** — chronic attenuation
         (dead / out-of-brain). Ratio, not cohort-z: cohort-z is bounded below
         by −1/rel_floor = −6.67 so a symmetric −8 bar can never fire.

All three are voltage-domain by design (v6: STFT-only catches = 0 — every
spectral-burst contact also fires ``spike``). Spectral artifacts (HFO/ripple,
stft-burst) and freezes route to the **CLIP** layer (``precompute_bad_windows``),
NOT here. This module is pure-numpy and operates on already-filtered voltage;
the load + harmonic-notch + 0.5 Hz HPF lives in the DCC precompute script.

Lifted byte-faithfully from the v6 probe (``guard1_v6_derive.py``): same
``robust_z_np`` / ``cohort_z`` / spike-clip / rmad math that locked the bars.
"""

from __future__ import annotations

import numpy as np

# Robust-spread constant: σ ≈ 1.4826 · MAD recovers the Gaussian std from the
# median-absolute-deviation. Matches SessionRobustZNormalizer + the v6 probe.
SCALE = 1.4826
SIGMA_FLOOR = 1e-6
EPS = 1e-9

# Ben-locked thresholds (2026-06-17). Exposed as module constants so the
# precompute script and the tests reference one source.
SPIKE_SIGMA = 100.0       # robust-z(dV/dt) inner "what is a spike"
SPIKE_CLIP_S = 5.0        # clip granularity (the static-vs-CLIP router unit)
SPIKE_CLIP_FRAC = 0.01    # >1% of clips corrupted ⇒ STATIC (else CLIP)
NOISY_Z = 8.0             # cohort-z(rmad) high tail
DEAD_RATIO = 0.35         # rmad / cohort-median rmad low tail
COHORT_REL_FLOOR = 0.15   # cohort-z denominator floor (bounds z ≥ −6.67)


def _robust_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row robust-z and robust spread (rmad) of ``x`` (C, T).

    Returns ``(z, rmad)`` where ``z = (x − median) / max(σ, floor)`` and
    ``rmad = σ = SCALE · MAD`` per row. float64 internally for stability."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True)
    rmad = SCALE * mad
    z = (x - med) / np.maximum(rmad, SIGMA_FLOOR)
    return z, rmad.ravel()


def cohort_z(v: np.ndarray, rel_floor: float = COHORT_REL_FLOOR) -> np.ndarray:
    """Cohort-relative robust-z of a per-contact statistic ``v`` (C,).

    Median/MAD taken ACROSS contacts (chronic deviation from peers). The
    denominator floor ``rel_floor·|median|`` bounds ``z ≥ −1/rel_floor`` — which
    is exactly why ``dead`` uses a ratio, not this z, on the low tail."""
    v = np.asarray(v, dtype=np.float64)
    m = np.median(v)
    s = max(SCALE * np.median(np.abs(v - m)), rel_floor * abs(m), EPS)
    return (v - m) / s


def detect_spike(
    v_filt: np.ndarray,
    fs: float,
    sigma: float = SPIKE_SIGMA,
    clip_s: float = SPIKE_CLIP_S,
    clip_frac_thresh: float = SPIKE_CLIP_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """Transient-spike STATIC detector on filtered voltage ``v_filt`` (C, T).

    A clip (``clip_s`` seconds) is corrupted if it contains ≥1 sample with
    ``|robust-z(dV/dt)| > sigma``. Returns ``(is_static, clip_frac)`` where
    ``is_static[c] = clip_frac[c] > clip_frac_thresh``. The clip-corruption
    fraction (not raw event rate) is the burst-robust router."""
    v_filt = np.asarray(v_filt, dtype=np.float64)
    C = v_filt.shape[0]
    g = np.diff(v_filt, axis=1)
    gz, _ = _robust_rows(g)
    agz = np.abs(gz)
    win = max(1, int(round(clip_s * fs)))
    nwin = agz.shape[1] // win
    if nwin < 1:
        clip_frac = np.zeros(C)
    else:
        corrupted = (agz[:, : nwin * win].reshape(C, nwin, win) > sigma).any(axis=2)
        clip_frac = corrupted.mean(axis=1)
    return clip_frac > clip_frac_thresh, clip_frac


def detect_noisy(
    v_filt: np.ndarray, z_thresh: float = NOISY_Z
) -> tuple[np.ndarray, np.ndarray]:
    """Chronic-high-amplitude STATIC detector. ``cohort-z(rmad) > z_thresh``.

    Returns ``(is_noisy, rmad)``. rmad = robust spread of the filtered voltage
    per contact (the model's view); cohort-z is across contacts."""
    _, rmad = _robust_rows(v_filt)
    return cohort_z(rmad) > z_thresh, rmad


def detect_dead(
    v_filt: np.ndarray, ratio_thresh: float = DEAD_RATIO
) -> tuple[np.ndarray, np.ndarray]:
    """Chronic-attenuation STATIC detector. ``rmad / median_cohort(rmad) < ratio_thresh``.

    Returns ``(is_dead, ratio)``. Scale-free ratio (not cohort-z, which is
    bounded below); < 0.35 ≈ contact carries < 1/3 the session-typical
    amplitude = dead / out-of-brain / severely attenuated."""
    _, rmad = _robust_rows(v_filt)
    med = np.median(np.asarray(rmad, dtype=np.float64))
    ratio = np.asarray(rmad, dtype=np.float64) / max(med, EPS)
    return ratio < ratio_thresh, ratio


def static_bad_mask(v_filt: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Run all three detectors on one session's filtered voltage ``(C, T)``.

    Returns per-contact boolean masks ``spike`` / ``noisy`` / ``dead`` and their
    union ``static`` (the per-session STATIC drop), plus the raw diagnostics
    (``clip_frac``, ``rmad``, ``dead_ratio``) for provenance dumps."""
    spike, clip_frac = detect_spike(v_filt, fs)
    noisy, rmad = detect_noisy(v_filt)
    dead, ratio = detect_dead(v_filt)
    return {
        "spike": spike,
        "noisy": noisy,
        "dead": dead,
        "static": spike | noisy | dead,
        "clip_frac": clip_frac,
        "rmad": rmad,
        "dead_ratio": ratio,
    }


def classify_from_signature(
    labels: list[str],
    clip_frac: np.ndarray,
    rmad: np.ndarray,
    *,
    spike_thresh: float = SPIKE_CLIP_FRAC,
    noisy_z: float = NOISY_Z,
    dead_ratio: float = DEAD_RATIO,
) -> dict[str, set[str]]:
    """Apply the locked thresholds to a precomputed per-contact signature.

    The collector core: maps one session's ``(clip_frac, rmad)`` per-contact
    arrays (from a fresh scan or a probe-output JSON) to the three fired
    detector label sets. ``rmad`` is the robust spread of filtered V per
    contact; ``clip_frac`` the dV/dt>100σ clip-corruption fraction. Shares the
    cohort-z / ratio math with ``detect_noisy`` / ``detect_dead`` so a JSON
    collector and a live scan classify identically."""
    clip_frac = np.asarray(clip_frac, dtype=np.float64)
    rmad = np.asarray(rmad, dtype=np.float64)
    spike = {labels[i] for i in np.flatnonzero(clip_frac > spike_thresh)}
    noisy = {labels[i] for i in np.flatnonzero(cohort_z(rmad) > noisy_z)}
    med = max(float(np.median(rmad)), EPS)
    dead = {labels[i] for i in np.flatnonzero(rmad / med < dead_ratio)}
    return {"spike": spike, "noisy": noisy, "dead": dead}


def aggregate_subject(
    per_trial: list[dict[str, set[str]]],
) -> dict[str, frozenset[str]]:
    """Aggregate per-session detections of one subject into a per-subject drop.

    DP4 forces a per-SUBJECT static set (``voltage_electrode_order`` is per
    subject, no trial axis), so per-session detections must collapse over the
    subject's loaded trials. Rule (recommended-pending-Ben, the original
    locked-11 convention):
      - ``noisy`` / ``dead`` — chronic ⇒ consistent ⇒ drop if fired in **any**
        loaded trial.
      - ``spike`` — drop if fired in a **majority** of the subject's loaded
        trials (sparse single-trial bursts handled by CLIP, not static).

    ``per_trial`` is a list (one entry per loaded trial) of dicts mapping
    detector name → set of contact labels that fired. Returns the per-subject
    ``spike`` / ``noisy`` / ``dead`` / ``static`` frozensets."""
    n = len(per_trial)
    if n == 0:
        empty = frozenset()
        return {"spike": empty, "noisy": empty, "dead": empty, "static": empty}

    noisy: set[str] = set()
    dead: set[str] = set()
    spike_counts: dict[str, int] = {}
    for trial in per_trial:
        noisy |= set(trial.get("noisy", ()))
        dead |= set(trial.get("dead", ()))
        for lbl in trial.get("spike", ()):
            spike_counts[lbl] = spike_counts.get(lbl, 0) + 1
    spike = {lbl for lbl, c in spike_counts.items() if c > n / 2.0}

    static = frozenset(spike | noisy | dead)
    return {
        "spike": frozenset(spike),
        "noisy": frozenset(noisy),
        "dead": frozenset(dead),
        "static": static,
    }
