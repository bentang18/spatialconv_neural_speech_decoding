"""Signal-level augmentations for the v14 per-phoneme path.

Adapted from ``archive/legacy/data/augmentation.py`` with two v14-specific
fixes:

1. ``time_shift`` uses **zero-pad** instead of circular wrap. The legacy
   code was written for trial-level windows [-0.5, 1.0]s where circular
   wrap on a ±20-frame shift was benign. Our per-phoneme window is
   [-0.15, 0.5]s (130 samples) — circular wrap would leak the response
   onto the stimulus region. Zero-pad preserves causality.
2. ``channel_dropout`` **composes with** (not overwrites) the existing
   ``electrode_active_mask``. A dropped electrode contributes zero to
   the signal AND is marked inactive so downstream masked ops (pool,
   attention) ignore it correctly.

All augmentations are no-ops outside ``model.training``. They operate on
the v14 batch dict produced by ``collate_v14_phoneme_batch`` — same
patient per batch (#31), ``signal[B, N_e, T]``, ``electrode_active_mask
[B, N_e]`` bool.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AugmentationConfig:
    """Augmentation knobs. Any value at the sentinel OFF value disables the op.

    Defaults are the legacy baseline values; call sites override per experiment.
    """

    # Uniform random integer shift in [-max, +max] samples, zero-padded.
    time_shift_max_frames: int = 0
    # Per-electrode log-normal amplitude scaling, scale = exp(N(0, std²)).
    amp_scale_std: float = 0.0
    # Per-electrode Bernoulli dropout, rate ~ U[0, max_p] per step.
    channel_dropout_max_p: float = 0.0
    # Additive Gaussian noise, std = frac · signal.std().
    gaussian_noise_frac: float = 0.0


LEGACY_DEFAULT = AugmentationConfig(
    time_shift_max_frames=20,
    amp_scale_std=0.15,
    channel_dropout_max_p=0.2,
    gaussian_noise_frac=0.02,
)

SHIFT_ONLY = AugmentationConfig(time_shift_max_frames=20)
AMP_ONLY = AugmentationConfig(amp_scale_std=0.15)
DROPOUT_ONLY = AugmentationConfig(channel_dropout_max_p=0.2)
NOISE_ONLY = AugmentationConfig(gaussian_noise_frac=0.02)

PRESETS: dict[str, AugmentationConfig] = {
    "none": AugmentationConfig(),
    "legacy": LEGACY_DEFAULT,
    "shift_only": SHIFT_ONLY,
    "amp_only": AMP_ONLY,
    "dropout_only": DROPOUT_ONLY,
    "noise_only": NOISE_ONLY,
}


def _time_shift_zeropad(signal: torch.Tensor, max_frames: int) -> torch.Tensor:
    """Per-sample integer shift in [-max, +max] samples, zero-padded at both ends.

    `signal`: ``(B, N_e, T)``. Shifts the last dim. Positive shift → delay
    (prepend zeros); negative shift → earlier (append zeros).
    """

    if max_frames <= 0:
        return signal
    B, _, T = signal.shape
    shifts = torch.randint(
        -max_frames, max_frames + 1, (B,), device=signal.device
    )
    out = torch.zeros_like(signal)
    for i in range(B):
        s = int(shifts[i].item())
        if s > 0:
            out[i, :, s:] = signal[i, :, : T - s]
        elif s < 0:
            out[i, :, : T + s] = signal[i, :, -s:]
        else:
            out[i] = signal[i]
    return out


def _amp_scale(signal: torch.Tensor, std: float) -> torch.Tensor:
    """Per-electrode log-normal scale (constant across time per electrode)."""

    if std <= 0.0:
        return signal
    B, N_e, _ = signal.shape
    log_scale = torch.randn(B, N_e, 1, device=signal.device) * std
    return signal * torch.exp(log_scale)


def _channel_dropout(
    signal: torch.Tensor,
    active: torch.Tensor,
    max_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero random electrodes and mark them inactive.

    `max_p` is the MAXIMUM drop rate; each call draws ``p ~ U[0, max_p]``
    and then samples per-electrode with rate `p`. Returns ``(signal, active)``
    — BOTH updated. The active mask is `active AND keep`, never overwritten.
    """

    if max_p <= 0.0:
        return signal, active
    B, N_e, _ = signal.shape
    p = float(torch.rand((), device=signal.device).item()) * max_p
    keep = torch.rand(B, N_e, device=signal.device) > p
    signal = signal * keep.to(signal.dtype).unsqueeze(-1)
    active = active & keep
    return signal, active


def _gaussian_noise(signal: torch.Tensor, frac: float) -> torch.Tensor:
    """Additive Gaussian, std = frac · signal.std() (batch-global scalar)."""

    if frac <= 0.0:
        return signal
    noise_std = frac * signal.std()
    return signal + torch.randn_like(signal) * noise_std


def augment_batch(batch: dict, cfg: AugmentationConfig) -> dict:
    """Return a new batch dict with augmented `signal` + `electrode_active_mask`.

    Caller is responsible for only calling this in training mode.
    """

    signal = batch["signal"]
    active = batch["electrode_active_mask"]
    signal = _time_shift_zeropad(signal, cfg.time_shift_max_frames)
    signal = _amp_scale(signal, cfg.amp_scale_std)
    signal, active = _channel_dropout(signal, active, cfg.channel_dropout_max_p)
    signal = _gaussian_noise(signal, cfg.gaussian_noise_frac)
    out = dict(batch)
    out["signal"] = signal
    out["electrode_active_mask"] = active
    return out
