"""Level 2: Matched-statistics synthetic generator.

Generates synthetic uECOG data that matches the empirical spatiotemporal
statistics of real HGA data (PSD shape, spatial correlation structure,
trial activation envelope, electrode variance distribution).

Unlike smooth_ar (Level 0) and switching_lds (Level 1), this generator
produces data statistically indistinguishable from real data in second-order
statistics, while destroying higher-order structure (phoneme content).

Design from real data analysis (results/real_data_stats.json):
- Temporal: spectral shaping to match empirical PSD (not AR(1))
- Spatial: 3-component decomposition (global + local Gaussian + independent)
  matched to empirical correlation vs Manhattan distance
- Envelope: trial activation profile with per-trial jitter
- Variance: per-electrode scaling from empirical distribution
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from speech_decoding.pretraining.generators.base import Generator

# Empirical PSD from real data (log10 values at key frequencies)
# Measured from 9 source patients, pooled across electrodes and trials
DEFAULT_PSD = {
    1: 0.013045,
    5: 0.001572,
    10: 0.000390,
    20: 3.12e-6,
    50: 2.04e-8,
    100: 4.64e-9,
}

# Spatial correlation decomposition (fit to 8x16 empirical data):
# corr(d) = W_GLOBAL² + W_LOCAL² * exp(-d²/(2*SIGMA_SPATIAL²))
# Verified: d=1→0.650, d=2→0.586, d=5→0.394, d=10→0.350
W_GLOBAL_SQ = 0.35    # global correlation floor
W_LOCAL_SQ = 0.325     # local spatially-smoothed component
SIGMA_SPATIAL = 2.5    # Gaussian smoothing sigma (electrode units)
W_INDEP_SQ = 1.0 - W_GLOBAL_SQ - W_LOCAL_SQ  # independent noise

# Trial envelope (RMS across electrodes vs time, pooled across patients)
# Window: [-0.5, 1.0s] at 200Hz
ENVELOPE_TIMES = [-0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00]
ENVELOPE_VALUES = [1.018, 1.028, 1.173, 1.336, 1.150, 1.083, 1.077]


class MatchedStatsGenerator(Generator):
    """Generate synthetic data matching real uECOG second-order statistics.

    Three-component spatial decomposition:
    - Global: shared signal across all electrodes (r≈0.35 floor)
    - Local: Gaussian-smoothed noise (nearby correlation, σ=2.5)
    - Independent: per-electrode noise

    Temporal structure via spectral shaping (matches empirical PSD),
    not AR(1) — real data decorrelates faster than AR at medium lags.

    Trial envelope with per-trial amplitude jitter.
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        T: int,
        envelope_jitter: float = 0.3,
        dead_frac_range: tuple[float, float] = (0.0, 0.03),
        stats_path: str | None = None,
        acf_speed: float = 1.0,
        baseline_zscore: bool = False,
    ):
        super().__init__(grid_h, grid_w, T)
        self.envelope_jitter = envelope_jitter
        self.dead_frac_range = dead_frac_range
        self.acf_speed = acf_speed
        self.baseline_zscore = baseline_zscore

        # Load custom stats if provided
        if stats_path is not None:
            with open(stats_path) as f:
                stats = json.load(f)
            self._load_psd_from_stats(stats)
            self._load_envelope_from_stats(stats)
        else:
            self.psd_freqs = sorted(DEFAULT_PSD.keys())
            self.psd_vals = [DEFAULT_PSD[f] for f in self.psd_freqs]
            self.env_times = ENVELOPE_TIMES
            self.env_values = ENVELOPE_VALUES

    def _load_psd_from_stats(self, stats: dict) -> None:
        psd = stats["temporal"]["psd"]
        self.psd_freqs = []
        self.psd_vals = []
        for key in sorted(psd.keys(), key=lambda k: float(k.replace("Hz", ""))):
            freq = float(key.replace("Hz", ""))
            self.psd_freqs.append(freq)
            self.psd_vals.append(psd[key]["mean"])

    def _load_envelope_from_stats(self, stats: dict) -> None:
        env = stats["trial_envelope"]["pooled"]
        self.env_times = []
        self.env_values = []
        for key in sorted(env.keys(), key=lambda k: float(k.replace("s", ""))):
            self.env_times.append(float(key.replace("s", "")))
            self.env_values.append(env[key])

    def _build_autocorr_filter(self, T: int) -> np.ndarray:
        """Build frequency-domain filter from target autocorrelation function.

        Uses parametric model: r(h) = exp(-a*h - b*h²)
        Fit to empirical ACF: r(1)=0.96, r(10)=0.276.
        acf_speed multiplies both a and b to accelerate temporal dynamics:
          speed=1.0: r(1)=0.96, r(10)=0.28 (original fit)
          speed=3.0: r(1)=0.88, r(3)=0.28 (3x faster decorrelation)
          speed=5.0: r(1)=0.82, r(2)=0.28 (5x faster decorrelation)
        """
        # Base fit from empirical ACF
        a = 0.03107 * self.acf_speed
        b = 0.009749 * self.acf_speed

        all_lags = np.arange(T)
        acf = np.exp(-a * all_lags - b * all_lags ** 2)

        # Build periodic ACF of length T for correct frequency bins.
        # ACF is even: acf_periodic[k] = r(min(k, T-k))
        acf_periodic = np.zeros(T)
        half = T // 2
        acf_periodic[:half + 1] = acf[:half + 1]
        # Mirror: acf_periodic[k] = acf[T-k] for k = half+1 .. T-1
        tail_indices = np.arange(half + 1, T)
        acf_periodic[half + 1:] = acf[T - tail_indices]

        # PSD = FFT(periodic ACF), gives T//2+1 frequency bins matching rfft(noise)
        psd = np.fft.rfft(acf_periodic).real
        psd = np.maximum(psd, 0)  # ensure non-negative
        return np.sqrt(psd)

    def _spectral_shape(self, white: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Shape white noise to match empirical autocorrelation.

        Uses autocorrelation → PSD → amplitude filter approach.
        More accurate than directly interpolating Welch PSD estimates.

        Args:
            white: (N, T) white noise array
            rng: random state (unused, kept for API consistency)
        Returns:
            (N, T) spectrally shaped noise
        """
        T = white.shape[1]

        cache_key = (T, self.acf_speed)
        if not hasattr(self, '_cached_filter') or self._cache_key != cache_key:
            self._cached_filter = self._build_autocorr_filter(T)
            self._cache_key = cache_key

        amplitude = self._cached_filter

        fft_white = np.fft.rfft(white, axis=-1)
        # Truncate amplitude to match fft output
        n_freq = fft_white.shape[-1]
        amp = amplitude[:n_freq]

        shaped = np.fft.irfft(fft_white * amp[np.newaxis, :], n=T, axis=-1)
        return shaped

    def _build_envelope(self, T: int, rng: np.random.RandomState) -> np.ndarray:
        """Build trial activation envelope with jitter.

        Returns (T,) array of per-timepoint amplitude scaling.
        """
        t_axis = np.linspace(-0.5, 1.0, T)
        base_env = np.interp(t_axis, self.env_times, self.env_values)

        # Per-trial jitter: scale the deviation from baseline
        jitter = 1.0 + rng.randn() * self.envelope_jitter
        envelope = 1.0 + (base_env - 1.0) * max(jitter, 0.1)

        return envelope

    def generate(self, seed: int | None = None) -> np.ndarray:
        """Generate one (H, W, T) trial matching real uECOG statistics.

        Pipeline:
        1. Generate 3 noise components (global, local, independent)
        2. Spectrally shape each to match empirical PSD
        3. Gaussian-smooth the local component for spatial correlation
        4. Mix with empirical weights → correct spatial covariance
        5. Apply trial envelope with jitter
        6. Z-score normalize
        """
        rng = np.random.RandomState(seed)
        H, W, T = self.grid_h, self.grid_w, self.T
        N = H * W

        w_global = np.sqrt(W_GLOBAL_SQ)
        w_local = np.sqrt(W_LOCAL_SQ)
        w_indep = np.sqrt(W_INDEP_SQ)

        # --- Component 1: Global signal (shared across all electrodes) ---
        global_white = rng.randn(1, T)
        global_shaped = self._spectral_shape(global_white, rng)[0]  # (T,)

        # --- Component 2: Local spatially-smoothed noise ---
        local_white = rng.randn(H, W, T)
        # Spectral shape each electrode independently
        local_flat = local_white.reshape(N, T)
        local_shaped = self._spectral_shape(local_flat, rng).reshape(H, W, T)
        # Apply spatial Gaussian smoothing (per time frame)
        for t in range(T):
            local_shaped[:, :, t] = gaussian_filter(
                local_shaped[:, :, t], sigma=SIGMA_SPATIAL, mode="reflect"
            )
        # Re-normalize after smoothing (smoothing reduces variance)
        local_std = local_shaped.std()
        if local_std > 1e-8:
            local_shaped /= local_std

        # --- Component 3: Independent per-electrode noise ---
        indep_white = rng.randn(N, T)
        indep_shaped = self._spectral_shape(indep_white, rng).reshape(H, W, T)

        # --- Mix components ---
        output = (
            w_global * global_shaped[np.newaxis, np.newaxis, :]
            + w_local * local_shaped
            + w_indep * indep_shaped
        )

        # --- Apply trial envelope ---
        envelope = self._build_envelope(T, rng)
        output *= envelope[np.newaxis, np.newaxis, :]

        # --- Add mean activation for baseline-relative mode ---
        # Real HGA power genuinely increases during speech production.
        # The multiplicative envelope above modulates variance but keeps mean≈0.
        # Decompose empirical RMS into variance + mean: RMS² = μ² + σ²
        # baseline σ² ≈ RMS_baseline², so μ(t) = sqrt(RMS(t)² - RMS_baseline²)
        if self.baseline_zscore:
            baseline_end = T // 3
            baseline_rms_sq = np.mean(envelope[:baseline_end]) ** 2
            mean_activation = np.sqrt(
                np.maximum(envelope ** 2 - baseline_rms_sq, 0.0)
            )
            output += mean_activation[np.newaxis, np.newaxis, :]

        # --- Dead electrodes ---
        dead_frac = rng.uniform(*self.dead_frac_range)
        n_dead = int(dead_frac * N)
        if n_dead > 0:
            dead_rows = rng.randint(0, H, size=n_dead)
            dead_cols = rng.randint(0, W, size=n_dead)
            output[dead_rows, dead_cols, :] = 0.0

        # --- Z-score normalize ---
        if self.baseline_zscore:
            # Baseline-relative: z-score using pre-stimulus period only.
            # Real HGA is z-scored relative to pre-stimulus baseline, so
            # during speech production most electrodes are positive.
            # Trial window is [-0.5, 1.0s]; baseline = first 1/3 ([-0.5, 0.0s])
            baseline_end = T // 3
            baseline = output[:, :, :baseline_end]
            mean = baseline.mean()
            std = output.std()  # full-trial std for consistent scaling
        else:
            mean = output.mean()
            std = output.std()
        if std > 1e-8:
            output = (output - mean) / std

        return output.astype(np.float32)
