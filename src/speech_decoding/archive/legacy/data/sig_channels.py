"""Significant channel detection and artifact channel exclusion for uECOG HGA.

Sig channel methods:
1. "baseline" — paired t-test: production HGA vs baseline HGA per channel.
   Matches upstream pipeline (permutation cluster test). Identifies channels
   that activate during speech, regardless of phoneme identity.

2. "anova" — one-way ANOVA: does mean HGA differ across phoneme classes?
   Stricter: requires phoneme-specific modulation, not just speech response.

Artifact detection:
    detect_artifact_channels() identifies channels with chronic extreme
    activations (>threshold std in >min_trial_fraction of trials). These are
    electronic artifacts (mic feedback, amplifier saturation), not brain
    signal — they should be excluded entirely, not clipped.

Usage:
    from speech_decoding.data.sig_channels import compute_sig_channels
    sig_mask, stats, pvals = compute_sig_channels(data, method="baseline")

    from speech_decoding.data.sig_channels import detect_artifact_channels
    artifact_mask, spike_rates = detect_artifact_channels(data)
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def compute_sig_channels(
    data: np.ndarray,
    labels: list[int] | None = None,
    alpha: float = 0.05,
    method: str = "baseline",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify significant channels in epoched HGA data.

    Args:
        data: (n_epochs, n_channels, n_times) — epoched HGA data.
            For "baseline" method, epochs should span baseline + production
            (e.g., -1.0s to +1.5s). For "anova", any window works.
        labels: (n_epochs,) integer phoneme labels. Required for "anova",
            ignored for "baseline".
        alpha: FDR-corrected significance threshold.
        method: "baseline" (speech-vs-baseline paired t-test) or
                "anova" (phoneme discrimination F-test).

    Returns:
        Tuple of (sig_mask, stat_values, p_values):
        - sig_mask: (n_channels,) bool — True for significant channels
        - stat_values: (n_channels,) float — t-stat or F-stat per channel
        - p_values: (n_channels,) float — raw p-value per channel
    """
    if method == "baseline":
        return _baseline_test(data, alpha)
    elif method == "anova":
        if labels is None:
            raise ValueError("labels required for anova method")
        return _anova_test(data, labels, alpha)
    else:
        raise ValueError(f"Unknown method: {method}")


def _baseline_test(
    data: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paired t-test: production HGA vs baseline HGA per channel.

    For z-scored productionZscore data (tmin≈-1.0, tmax≈+1.5 at 200Hz):
    - Baseline: first 20% of epoch (pre-stimulus period)
    - Production: middle 40% (covers 0 to ~0.6s post-onset)

    Tests whether mean HGA during production differs from baseline for
    each channel across trials. Equivalent to upstream's speech-vs-baseline
    permutation test but using parametric t-test with FDR correction.
    """
    n_epochs, n_ch, n_times = data.shape

    # Define windows
    t_base_end = n_times // 5           # first 20%
    t_prod_start = n_times * 2 // 5     # 40% mark
    t_prod_end = n_times * 3 // 5       # 60% mark

    # Mean HGA per trial per channel in each window
    baseline_mean = np.mean(data[:, :, :t_base_end], axis=2)        # (epochs, ch)
    production_mean = np.mean(data[:, :, t_prod_start:t_prod_end], axis=2)  # (epochs, ch)

    # Paired t-test per channel
    p_values = np.ones(n_ch)
    t_values = np.zeros(n_ch)

    for ch in range(n_ch):
        diff = production_mean[:, ch] - baseline_mean[:, ch]
        if np.std(diff) < 1e-10:
            continue
        t_val, p_val = stats.ttest_rel(production_mean[:, ch], baseline_mean[:, ch])
        if np.isfinite(t_val) and np.isfinite(p_val):
            t_values[ch] = t_val
            p_values[ch] = p_val

    sig_mask = _fdr_correction(p_values, alpha)
    return sig_mask, t_values, p_values


def _anova_test(
    data: np.ndarray, labels: list[int], alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-way ANOVA: does mean HGA differ across phoneme classes?"""
    n_epochs, n_ch, n_times = data.shape
    labels_arr = np.array(labels)
    unique_labels = np.unique(labels_arr)

    if len(unique_labels) < 2:
        raise ValueError(f"Need ≥2 classes, got {len(unique_labels)}")

    # Mean HGA across production window
    t_prod = n_times // 2
    mean_hga = np.mean(data[:, :, t_prod:], axis=2)

    p_values = np.ones(n_ch)
    f_values = np.zeros(n_ch)

    for ch in range(n_ch):
        groups = [mean_hga[labels_arr == lbl, ch] for lbl in unique_labels]
        if any(np.std(g) < 1e-10 for g in groups):
            continue
        f_val, p_val = stats.f_oneway(*groups)
        if np.isfinite(f_val) and np.isfinite(p_val):
            f_values[ch] = f_val
            p_values[ch] = p_val

    sig_mask = _fdr_correction(p_values, alpha)
    return sig_mask, f_values, p_values


def _fdr_correction(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    thresholds = np.arange(1, n + 1) / n * alpha
    significant = sorted_p <= thresholds
    if not np.any(significant):
        return np.zeros(n, dtype=bool)

    max_sig_idx = np.max(np.where(significant))
    mask = np.zeros(n, dtype=bool)
    mask[sorted_idx[:max_sig_idx + 1]] = True
    return mask


def detect_artifact_channels(
    data: np.ndarray,
    abs_threshold: float = 10.0,
    min_trial_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify channels with chronic electronic artifacts.

    A channel is flagged as artifact if its absolute value exceeds
    ``abs_threshold`` in more than ``min_trial_fraction`` of trials.
    These channels carry electronic noise (mic feedback, amplifier
    saturation), not brain signal, and should be excluded entirely.

    Channels with rare/sporadic spikes (≤min_trial_fraction) are NOT
    flagged — those may be real neural events or one-off glitches that
    can be handled by trial-level rejection if needed.

    Args:
        data: (n_epochs, n_channels, n_times) — epoched HGA data.
        abs_threshold: Maximum absolute z-score considered physiological.
            Default 10.0 — real HGA z-scores rarely exceed 5-8 std.
        min_trial_fraction: Minimum fraction of trials a channel must
            spike in to be considered chronic. Default 0.05 (5%).

    Returns:
        Tuple of (artifact_mask, spike_rates):
        - artifact_mask: (n_channels,) bool — True for artifact channels.
        - spike_rates: (n_channels,) float — fraction of trials each
          channel exceeds the threshold.
    """
    # Per-trial per-channel max absolute value
    ch_trial_max = np.abs(data).max(axis=2)  # (n_epochs, n_channels)

    # Fraction of trials each channel exceeds threshold
    spike_rates = (ch_trial_max > abs_threshold).mean(axis=0)  # (n_channels,)

    artifact_mask = spike_rates > min_trial_fraction
    return artifact_mask, spike_rates
