"""Level 3: Neural Cellular Automata synthetic generator.

Stochastic excitable-medium NCA with strictly local 3x3 update rules.

Two-variable E-I dynamics per electrode:
- u (fast excitatory): threshold-activated, drives output
- v (slow inhibitory): prevents sustained activation -> burst dynamics

Three stochasticity sources:
1. Random initial state
2. Per-cell per-step Gaussian noise (main engine)
3. Envelope jitter (per-trial variation)

Addresses three deficiencies of Gaussian generators:
1. Fast temporal dynamics (burst-suppression, not smooth drift)
2. Positive bias (threshold nonlinearity, not symmetric Gaussian)
3. Fine spatial granularity (local 3x3 rules, not global smoothing)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

from speech_decoding.pretraining.generators.base import Generator

# Trial envelope (from pooled patient RMS profile)
ENVELOPE_TIMES = [-0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00]
ENVELOPE_VALUES = [1.018, 1.028, 1.173, 1.336, 1.150, 1.083, 1.077]


class NCAGenerator(Generator):
    """Stochastic excitable-medium NCA for synthetic uECOG.

    Each electrode has two state variables:
    - u: fast excitatory activation (positive, burst-like)
    - v: slow recovery/inhibition (prevents sustained activation)

    Update rule per step:
        u_local = Conv3x3(u, excite_kernel)
        drive   = u_local - gamma*v + sigma(t)*noise
        u <- decay_fast*u + (1-decay_fast)*ReLU(drive - threshold)
        v <- decay_slow*v + inhibition_rate*u
        output  = u  (naturally non-negative from threshold dynamics)
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        T: int,
        decay_fast: float = 0.85,
        decay_slow: float = 0.97,
        noise_scale: float = 0.5,
        threshold: float = 0.15,
        gamma: float = 0.4,
        inhibition_rate: float = 0.2,
        substeps: int = 3,
        warmup_steps: int = 30,
        envelope_jitter: float = 0.3,
        dead_frac_range: tuple[float, float] = (0.0, 0.03),
    ):
        super().__init__(grid_h, grid_w, T)
        self.decay_fast = decay_fast
        self.decay_slow = decay_slow
        self.noise_scale = noise_scale
        self.threshold = threshold
        self.gamma = gamma
        self.inhibition_rate = inhibition_rate
        self.substeps = substeps
        self.warmup_steps = warmup_steps
        self.envelope_jitter = envelope_jitter
        self.dead_frac_range = dead_frac_range

        # 3x3 excitatory kernel: balanced center + neighbor coupling.
        # Stronger neighbor weights allow activation to spread locally,
        # forming coherent patches of ~10-20 electrodes via multi-step
        # propagation (substeps). Sum = 1.0.
        self.excite_kernel = np.array([
            [0.05, 0.10, 0.05],
            [0.10, 0.40, 0.10],
            [0.05, 0.10, 0.05],
        ])

    def _build_envelope(self, T: int, rng: np.random.RandomState) -> np.ndarray:
        """Build trial activation envelope with jitter.

        Modulates noise_scale over time: lower during baseline,
        higher during production -> speech-like activation profile.
        """
        t_axis = np.linspace(-0.5, 1.0, T)
        base_env = np.interp(t_axis, ENVELOPE_TIMES, ENVELOPE_VALUES)
        jitter = 1.0 + rng.randn() * self.envelope_jitter
        envelope = 1.0 + (base_env - 1.0) * max(jitter, 0.1)
        return envelope

    def _step(
        self,
        u: np.ndarray,
        v: np.ndarray,
        noise_amp: float,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One NCA update step."""
        H, W = u.shape

        # Local excitatory perception (3x3 only)
        u_local = convolve(u, self.excite_kernel, mode="reflect")

        # Stochastic drive
        noise = rng.randn(H, W) * noise_amp

        # E-I drive with threshold
        drive = u_local - self.gamma * v + noise
        activated = np.maximum(drive - self.threshold, 0)

        # Leaky integration
        u_new = self.decay_fast * u + (1 - self.decay_fast) * activated

        # Recovery: slowly tracks excitation
        v_new = self.decay_slow * v + self.inhibition_rate * u_new

        return u_new, v_new

    def generate(self, seed: int | None = None) -> np.ndarray:
        """Generate one (H, W, T) trial via stochastic NCA dynamics.

        Pipeline:
        1. Initialize u, v near zero
        2. Warmup: run NCA steps to establish E-I baseline
        3. Record T steps with envelope-modulated noise
        4. Dead electrodes
        5. Z-score normalize
        """
        rng = np.random.RandomState(seed)
        H, W, T = self.grid_h, self.grid_w, self.T

        # Initialize
        u = rng.randn(H, W) * 0.05
        v = np.zeros((H, W))

        # Build envelope
        envelope = self._build_envelope(T, rng)

        # Warmup with baseline noise level (full substeps)
        baseline_amp = self.noise_scale * envelope[0]
        for _ in range(self.warmup_steps):
            for s in range(self.substeps):
                amp = baseline_amp if s == 0 else baseline_amp * 0.15
                u, v = self._step(u, v, amp, rng)

        # Record T steps with substeps per output frame.
        # Substep 0: noise-driven (new activations appear)
        # Substeps 1+: mostly deterministic spreading (patches form)
        outputs = []
        for t in range(T):
            noise_amp = self.noise_scale * envelope[t]
            for s in range(self.substeps):
                amp = noise_amp if s == 0 else noise_amp * 0.15
                u, v = self._step(u, v, amp, rng)
            outputs.append(u.copy())

        output = np.stack(outputs, axis=-1)  # (H, W, T)

        # Dead electrodes
        N = H * W
        dead_frac = rng.uniform(*self.dead_frac_range)
        n_dead = int(dead_frac * N)
        if n_dead > 0:
            dead_rows = rng.randint(0, H, size=n_dead)
            dead_cols = rng.randint(0, W, size=n_dead)
            output[dead_rows, dead_cols, :] = 0.0

        # Z-score normalize
        mean = output.mean()
        std = output.std()
        if std > 1e-8:
            output = (output - mean) / std

        return output.astype(np.float32)
