"""Pre-read-in data augmentations for grid-shaped uECOG HGA.

All augmentations operate on (B, H, W, T) tensors and are applied
before the per-patient read-in layer. Feature dropout and time masking
are handled inside the backbone module.
"""
from __future__ import annotations

import torch


def time_shift(x: torch.Tensor, max_frames: int = 20) -> torch.Tensor:
    """Random per-trial circular time shift ±max_frames.

    Simulates response onset variation and cross-patient neural latency.
    Zero-pads edges after shift.
    """
    if max_frames == 0:
        return x
    B, H, W, T = x.shape
    shifts = torch.randint(-max_frames, max_frames + 1, (B,))
    out = torch.zeros_like(x)
    for i in range(B):
        s = shifts[i].item()
        if s > 0:
            out[i, :, :, s:] = x[i, :, :, :T - s]
        elif s < 0:
            out[i, :, :, :T + s] = x[i, :, :, -s:]
        else:
            out[i] = x[i]
    return out


def amplitude_scale(x: torch.Tensor, std: float = 0.15) -> torch.Tensor:
    """Per-electrode log-normal amplitude scaling.

    scale = exp(N(0, std²)), constant within trial per electrode.
    Simulates electrode impedance variation.
    """
    if std == 0.0:
        return x
    B, H, W, T = x.shape
    # (B, H, W, 1) scale factor per electrode
    log_scale = torch.randn(B, H, W, 1, device=x.device) * std
    return x * torch.exp(log_scale)


def channel_dropout(x: torch.Tensor, max_p: float = 0.2) -> torch.Tensor:
    """Zero entire electrodes with probability p ~ U[0, max_p].

    Simulates bad/missing electrodes and sig channel variation.
    """
    if max_p == 0.0:
        return x
    B, H, W, T = x.shape
    p = torch.rand(1).item() * max_p
    # (B, H, W) mask — same across time
    mask = (torch.rand(B, H, W, device=x.device) > p).float().unsqueeze(-1)
    return x * mask


def gaussian_noise(x: torch.Tensor, frac: float = 0.02) -> torch.Tensor:
    """Additive Gaussian noise: x += N(0, (frac·std(x))²) per frame.

    Simulates recording and biological noise.
    """
    if frac == 0.0:
        return x
    noise_std = frac * x.std()
    return x + torch.randn_like(x) * noise_std


def temporal_stretch(
    x: torch.Tensor, max_rate: float = 0.15,
) -> torch.Tensor:
    """Random per-trial temporal stretch/compress by factor in [1-max_rate, 1+max_rate].

    Simulates speaking rate variation across trials and patients.
    Uses linear interpolation, output length matches input length (padded/cropped).
    """
    if max_rate == 0.0:
        return x
    B, H, W, T = x.shape
    out = torch.zeros_like(x)
    for i in range(B):
        rate = 1.0 + (torch.rand(1).item() * 2 - 1) * max_rate  # U[1-max, 1+max]
        new_T = int(round(T * rate))
        if new_T < 2:
            out[i] = x[i]
            continue
        # (H, W, T) → (H*W, 1, T) for interpolate, then reshape back
        flat = x[i].reshape(-1, 1, T)  # (H*W, 1, T)
        stretched = torch.nn.functional.interpolate(
            flat, size=new_T, mode="linear", align_corners=False,
        )  # (H*W, 1, new_T)
        # Crop or pad to original T
        if new_T >= T:
            out[i] = stretched[:, 0, :T].reshape(H, W, T)
        else:
            out[i, :, :, :new_T] = stretched[:, 0, :].reshape(H, W, new_T)
    return out


def spatial_cutout(
    x: torch.Tensor, max_h: int = 3, max_w: int = 6,
) -> torch.Tensor:
    """Zero a random rectangular patch on the spatial grid per trial.

    Simulates localized signal loss and regularizes Conv2d spatial filters.
    """
    B, H, W, T = x.shape
    cut_h = torch.randint(1, max_h + 1, (1,)).item()
    cut_w = torch.randint(1, max_w + 1, (1,)).item()
    x = x.clone()
    for i in range(B):
        r = torch.randint(0, max(H - cut_h, 1), (1,)).item()
        c = torch.randint(0, max(W - cut_w, 1), (1,)).item()
        x[i, r:r + cut_h, c:c + cut_w, :] = 0.0
    return x


def augment_batch(
    x: torch.Tensor,
    training: bool = True,
    time_shift_frames: int = 20,
    amp_scale_std: float = 0.15,
    channel_dropout_max: float = 0.2,
    noise_frac: float = 0.02,
    do_spatial_cutout: bool = False,
    spatial_cutout_max_h: int = 3,
    spatial_cutout_max_w: int = 6,
    do_temporal_stretch: bool = False,
    temporal_stretch_max_rate: float = 0.15,
) -> torch.Tensor:
    """Apply all pre-read-in augmentations.

    Args:
        x: (B, H, W, T) grid-shaped HGA data.
        training: If False, return x unchanged.
    """
    if not training:
        return x
    x = time_shift(x, max_frames=time_shift_frames)
    if do_temporal_stretch:
        x = temporal_stretch(x, max_rate=temporal_stretch_max_rate)
    x = amplitude_scale(x, std=amp_scale_std)
    x = channel_dropout(x, max_p=channel_dropout_max)
    if do_spatial_cutout:
        x = spatial_cutout(x, max_h=spatial_cutout_max_h, max_w=spatial_cutout_max_w)
    x = gaussian_noise(x, frac=noise_frac)
    return x
