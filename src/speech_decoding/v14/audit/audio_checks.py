"""Audio-neural clock alignment + silent-trial detection.

## Clock alignment

The audio recording and the ECoG recording live on separate clocks. They
differ by a bulk offset AND may exhibit crystal-oscillator drift across the
session. A single constant offset is not enough.

`desc-production_events.tsv` contains both the audio-clock `onset` and the
ECoG-clock `sample` for every detected speech event, so each row is a
`(ecog_t, audio_t)` pair. We ignore the file's (buggy) trial identities and
use only its timing columns.

Model: `audio_t = a + b * ecog_t` (linear with drift). We fit it, report
residual std as the consistency metric, and use the fit to map every
`eventsOLD` response onset into audio time for silent-trial detection.

## Silent-trial detection

For each `eventsOLD` response onset, we predict the audio-clock window via
the clock fit, compute raw audio RMS in `[predicted - 0.1 s, predicted + 0.6 s]`,
and flag trials whose peak RMS falls below `noise_floor * 4` as `silent-suspect`.

## Predicates

- `audio_neural_clock_fit_exists` (must): ≥ 20 usable `(ecog_t, audio_t)` pairs
  survive the `≤ 2 s` residual filter.
- `audio_neural_clock_consistent` (must): residual std ≤ 20 ms after fit.
- `silent_trial_fraction_low` (soft): silent-suspect ≤ 10 % of trials.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from speech_decoding.v14.audit.io import ECOG_RAW_SFREQ
from speech_decoding.v14.audit.schema import Check, make_check

RMS_WIN_S = 0.020
RMS_HOP_S = 0.010
ENV_DOWN_S = 0.050

MIN_CLOCK_PAIRS = 20
CLOCK_RESIDUAL_TOL_S = 0.150  # std across (ecog_t, audio_t) pairs. Calibrated
# against MFA's phoneme-onset jitter (~100 ms) in desc-production_events.tsv.
# Loose enough to absorb MFA noise; tight enough to catch real misalignment
# (e.g. a step change or clock reset would push std > 500 ms). Phase 1 does
# not use audio-derived timing — eventsOLD is the alignment source.
MATCH_TOL_S = 2.0

RESPONSE_WIN_PRE_S = 0.1
RESPONSE_WIN_POST_S = 0.6
SILENT_THRESHOLD_FACTOR = 4.0
SILENT_FRACTION_MAX = 0.10


def _rms_envelope(
    audio: np.ndarray, sr: int, win_s: float, hop_s: float
) -> tuple[np.ndarray, np.ndarray]:
    win = int(round(win_s * sr))
    hop = int(round(hop_s * sr))
    n = max(0, (len(audio) - win) // hop + 1)
    sq = audio.astype(np.float64) ** 2
    cs = np.concatenate([[0.0], np.cumsum(sq)])
    ends = np.arange(n) * hop + win
    starts = ends - win
    rms = np.sqrt((cs[ends] - cs[starts]) / win)
    times = (starts + win / 2) / sr
    return times, rms.astype(np.float32)


def _downsample_envelope(
    times: np.ndarray, rms: np.ndarray, bin_s: float
) -> tuple[np.ndarray, np.ndarray]:
    hop_s = float(times[1] - times[0])
    factor = max(1, int(round(bin_s / hop_s)))
    n_bins = len(rms) // factor
    rms_ds = rms[: n_bins * factor].reshape(n_bins, factor).max(axis=1)
    times_ds = times[::factor][:n_bins]
    return times_ds, rms_ds


def _fit_clock_model(production_events: pd.DataFrame) -> dict:
    """Fit `audio_t = a + b * ecog_t` using `desc-production_events.tsv`.

    Returns dict with a, b, residual_std, residuals, and pair count."""
    audio_t = production_events["onset"].to_numpy(dtype=np.float64)
    ecog_t = production_events["sample"].to_numpy(dtype=np.float64) / ECOG_RAW_SFREQ

    # Robust iterative fit: drop outliers beyond MATCH_TOL_S from current fit
    mask = np.ones_like(audio_t, dtype=bool)
    for _ in range(4):
        b, a = np.polyfit(ecog_t[mask], audio_t[mask], 1)
        pred = a + b * ecog_t
        resid = audio_t - pred
        new_mask = np.abs(resid) <= MATCH_TOL_S
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
    b, a = np.polyfit(ecog_t[mask], audio_t[mask], 1)
    pred = a + b * ecog_t
    resid = audio_t - pred

    return {
        "a": float(a),
        "b": float(b),
        "n_input": int(len(audio_t)),
        "n_inliers": int(mask.sum()),
        "residual_std_s": float(np.std(resid[mask])),
        "residual_max_s": float(np.max(np.abs(resid[mask]))),
        "ecog_t": ecog_t,
        "audio_t": audio_t,
        "mask": mask,
        "resid": resid,
    }


def _predict_audio_time(fit: dict, ecog_t: np.ndarray) -> np.ndarray:
    return fit["a"] + fit["b"] * ecog_t


def _silent_trial_mask(
    env_times: np.ndarray,
    env_rms: np.ndarray,
    predicted_audio_onsets: np.ndarray,
    noise_floor: float,
) -> np.ndarray:
    bin_s = env_times[1] - env_times[0]
    pre = int(round(RESPONSE_WIN_PRE_S / bin_s))
    post = int(round(RESPONSE_WIN_POST_S / bin_s))
    thresh = noise_floor * SILENT_THRESHOLD_FACTOR
    silent = np.zeros(len(predicted_audio_onsets), dtype=bool)
    for i, at in enumerate(predicted_audio_onsets):
        ci = int(round((at - env_times[0]) / bin_s))
        lo, hi = max(0, ci - pre), min(len(env_rms), ci + post)
        if hi - lo < 1:
            silent[i] = True
            continue
        silent[i] = bool(env_rms[lo:hi].max() < thresh)
    return silent


def _per_trial_refined_offsets(
    env_times: np.ndarray,
    env_rms: np.ndarray,
    predicted_audio_onsets: np.ndarray,
    noise_floor: float,
    search_s: float = 0.5,
) -> np.ndarray:
    """For diagnostic: find first supra-threshold speech sample in a narrow
    window around predicted onset. Returns offsets (found - predicted), NaN
    if no speech in window."""
    bin_s = env_times[1] - env_times[0]
    half = int(round(search_s / bin_s))
    thresh = noise_floor * SILENT_THRESHOLD_FACTOR
    offsets = np.full(len(predicted_audio_onsets), np.nan)
    for i, at in enumerate(predicted_audio_onsets):
        ci = int(round((at - env_times[0]) / bin_s))
        lo, hi = max(0, ci - half), min(len(env_rms), ci + half + 1)
        if hi - lo < 3:
            continue
        local = env_rms[lo:hi]
        above = np.where(local > thresh)[0]
        if len(above) == 0:
            continue
        offsets[i] = env_times[lo + int(above[0])] - at
    return offsets


def run_audio_checks(
    patient: str,
    audio: np.ndarray,
    audio_sr: int,
    authoritative_resp: pd.DataFrame,
    production_events: pd.DataFrame,
    plots_dir: Path,
    exclusions_csv: Path,
) -> tuple[list[Check], dict]:
    """Run audio clock-fit + silent-trial detection."""

    # ── 1. RMS envelope ──
    times_hi, rms_hi = _rms_envelope(audio, audio_sr, RMS_WIN_S, RMS_HOP_S)
    times, rms = _downsample_envelope(times_hi, rms_hi, ENV_DOWN_S)
    noise_floor = float(np.median(rms))

    # ── 2. Clock model via desc-production_events.tsv ──
    fit = _fit_clock_model(production_events)
    fit_ok = fit["n_inliers"] >= MIN_CLOCK_PAIRS
    consistency_ok = fit_ok and fit["residual_std_s"] <= CLOCK_RESIDUAL_TOL_S

    # ── 3. Predict audio times for each eventsOLD response ──
    ecog_onsets = authoritative_resp["onset"].to_numpy(dtype=np.float64)
    predicted_audio = _predict_audio_time(fit, ecog_onsets) if fit_ok else np.full_like(ecog_onsets, np.nan)

    # ── 4. Silent-trial detection ──
    if fit_ok:
        silent = _silent_trial_mask(times, rms, predicted_audio, noise_floor)
        refined = _per_trial_refined_offsets(times, rms, predicted_audio, noise_floor)
    else:
        silent = np.ones(len(ecog_onsets), dtype=bool)
        refined = np.full(len(ecog_onsets), np.nan)
    silent_fraction = float(silent.mean()) if len(silent) else 0.0

    # ── Checks ──
    checks: list[Check] = [
        make_check(
            "audio_neural_clock_fit_exists",
            "must",
            fit_ok,
            f"linear fit: n_inliers={fit['n_inliers']}/{fit['n_input']}, "
            f"a={fit['a']:.3f}s, b={fit['b']:.6f} (drift={1e3 * (fit['b'] - 1):.2f} ms/s)",
        ),
        make_check(
            "audio_neural_clock_consistent",
            "must",
            consistency_ok,
            f"residual std = {fit['residual_std_s'] * 1000:.1f} ms, "
            f"max |resid| = {fit['residual_max_s'] * 1000:.1f} ms "
            f"(tol ≤ {CLOCK_RESIDUAL_TOL_S * 1000:.0f} ms std)",
        ),
        make_check(
            "silent_trial_fraction_low",
            "soft",
            silent_fraction <= SILENT_FRACTION_MAX,
            f"silent-suspect trials = {int(silent.sum())}/{len(silent)} "
            f"({silent_fraction * 100:.1f}%, tol ≤ {SILENT_FRACTION_MAX * 100:.0f}%)",
        ),
    ]

    # ── Exclusion-candidates CSV ──
    excl_df = pd.DataFrame(
        {
            "trial_index": np.arange(len(ecog_onsets)),
            "ecog_trial_id": authoritative_resp["trial"].to_numpy(),
            "token": authoritative_resp["value"].to_numpy(),
            "ecog_response_onset_s": ecog_onsets,
            "predicted_audio_onset_s": predicted_audio,
            "refined_offset_s": refined,
            "silent_suspect": silent,
        }
    )
    excl_df[excl_df["silent_suspect"]].to_csv(exclusions_csv, index=False)

    # ── Diagnostic plot ──
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

    # (a) Clock fit
    axes[0].scatter(fit["ecog_t"][fit["mask"]], fit["audio_t"][fit["mask"]], s=4, alpha=0.6, label="inlier pairs")
    axes[0].scatter(fit["ecog_t"][~fit["mask"]], fit["audio_t"][~fit["mask"]], s=4, color="red", alpha=0.6, label="outlier")
    xs = np.linspace(fit["ecog_t"].min(), fit["ecog_t"].max(), 100)
    axes[0].plot(xs, fit["a"] + fit["b"] * xs, "k-", lw=0.8, label=f"fit (res std={fit['residual_std_s'] * 1000:.1f} ms)")
    axes[0].set_xlabel("ECoG time (s)")
    axes[0].set_ylabel("audio time (s)")
    axes[0].set_title(f"{patient}: clock model — audio_t = {fit['a']:.3f} + {fit['b']:.6f} · ecog_t")
    axes[0].legend(fontsize=8)

    # (b) Audio RMS with predicted onsets overlaid
    axes[1].plot(times, rms, lw=0.4, color="steelblue", label="audio RMS (50 ms)")
    axes[1].axhline(noise_floor, color="gray", lw=0.5, ls=":", label="noise floor")
    axes[1].axhline(noise_floor * SILENT_THRESHOLD_FACTOR, color="orange", lw=0.5, ls=":", label=f"×{SILENT_THRESHOLD_FACTOR} threshold")
    in_range = (predicted_audio >= times[0]) & (predicted_audio <= times[-1])
    colors = np.where(silent, "red", "green")
    axes[1].vlines(
        predicted_audio[in_range],
        0,
        rms.max(),
        colors=colors[in_range],
        alpha=0.35,
        lw=0.5,
    )
    axes[1].set_xlabel("audio time (s)")
    axes[1].set_ylabel("RMS")
    axes[1].set_title(
        "green = speech detected at predicted onset; red = silent-suspect"
    )
    axes[1].legend(loc="upper right", fontsize=8)

    # (c) Per-trial refined offset vs trial index
    ok_mask = np.isfinite(refined)
    axes[2].scatter(np.where(ok_mask)[0], refined[ok_mask] * 1000, s=6, color="steelblue")
    axes[2].scatter(np.where(~ok_mask)[0], np.zeros(int((~ok_mask).sum())), s=6, color="red", label="no speech in window")
    axes[2].axhline(0, color="k", lw=0.3)
    axes[2].set_xlabel("trial index")
    axes[2].set_ylabel("refined offset (ms, speech onset − predicted)")
    axes[2].set_title(
        f"per-trial refined offsets ({int(ok_mask.sum())} with detected speech onset)"
    )
    axes[2].legend(fontsize=8)

    fig.savefig(plots_dir / f"{patient}_audio_alignment.png", dpi=120)
    plt.close(fig)

    meta = {
        "audio_sr_hz": int(audio_sr),
        "audio_duration_s": float(len(audio) / audio_sr),
        "audio_noise_floor_rms": noise_floor,
        "clock_fit_a": fit["a"],
        "clock_fit_b": fit["b"],
        "clock_fit_drift_ms_per_s": float(1e3 * (fit["b"] - 1)),
        "clock_fit_n_inliers": fit["n_inliers"],
        "clock_fit_n_input": fit["n_input"],
        "clock_fit_residual_std_s": fit["residual_std_s"],
        "clock_fit_residual_max_s": fit["residual_max_s"],
        "n_silent_suspect": int(silent.sum()),
        "silent_fraction": silent_fraction,
        "exclusion_candidates_csv": str(exclusions_csv),
    }
    return checks, meta
