"""P3-02 + P3-03: teacher-side triangular pool 50 → 8 Hz.

Pools the frozen Whisper-L8 hidden state from its 50 Hz native frame
rate (10 ms hop) down to v14's 8 Hz student native rate. Triangular
FWHM = 250 ms (≈ 12.5 Whisper frames per output bucket). Sum-to-1
normalized per bucket. Zero-padded at the start / end.

Locked constants (B05 + B06 lock 2026-05-25 PM)::

    P3_TEACHER_RATE_HZ        = 8.0   # output rate (matches student)
    P3_WHISPER_NATIVE_RATE_HZ = 50.0  # Whisper-large native frame rate

For a 5-s clip this gives:

    (B, 250, 1280)  →  (B, 40, 1280)

40 frames × 5 s = 8 Hz. Student-side stays identity passthrough at
8 Hz — the pool is teacher-side only.
"""

from __future__ import annotations

import torch
from torch import Tensor


# B05 + B06 lock 2026-05-25 PM.
P3_TEACHER_RATE_HZ: float = 8.0
P3_WHISPER_NATIVE_RATE_HZ: float = 50.0

_FWHM_MS: float = 250.0  # 250 ms triangular FWHM
_CLIP_SECONDS: float = 5.0
_EXPECTED_N_IN: int = int(P3_WHISPER_NATIVE_RATE_HZ * _CLIP_SECONDS)   # 250
_EXPECTED_N_OUT: int = int(P3_TEACHER_RATE_HZ * _CLIP_SECONDS)         # 40


def triangular_pool_weight_matrix(n_in: int, n_out: int) -> Tensor:
    """Build the ``(n_out, n_in)`` triangular-pool weight matrix.

    The bucket centred at output index ``i`` sits at input position
    ``i * (n_in / n_out)``; the weight decays linearly to zero at
    ``± FWHM/2`` Whisper frames. Sum-to-1 normalized per row.
    Zero outside the support — no wrap-around, no negative indexing.
    """
    stride = n_in / n_out                   # 50/8 = 6.25 for the 5-s clip
    half_fwhm_frames = (_FWHM_MS / 1000.0) * P3_WHISPER_NATIVE_RATE_HZ / 2.0  # 6.25
    centres = torch.arange(n_out, dtype=torch.float32) * stride              # (n_out,)
    in_positions = torch.arange(n_in, dtype=torch.float32)                   # (n_in,)
    # Distance from each input position to each output bucket centre.
    dist = (in_positions[None, :] - centres[:, None]).abs()                  # (n_out, n_in)
    # Triangle: 1 - dist / half_fwhm; clamp negative to 0.
    W = (1.0 - dist / half_fwhm_frames).clamp(min=0.0)
    # Sum-to-1 per row (with a tiny eps floor for the degenerate
    # edge case where rounding wipes a row, though it shouldn't with
    # FWHM 12.5 ≫ stride 6.25).
    row_sum = W.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    W = W / row_sum
    return W


def triangular_pool_50_to_8_hz(features: Tensor) -> Tensor:
    """Pool ``(B, 250, D) → (B, 40, D)`` for a 5-s Whisper clip.

    Rejects non-250 input lengths — that signals a mis-sized clip
    upstream, which should fail loudly rather than be silently
    re-pooled.
    """
    if features.shape[-2] != _EXPECTED_N_IN:
        raise ValueError(
            f"expected 250 Whisper frames (5 s × 50 Hz); got "
            f"{features.shape[-2]}"
        )
    W = triangular_pool_weight_matrix(_EXPECTED_N_IN, _EXPECTED_N_OUT)  # (40, 250)
    W = W.to(features.device, features.dtype)
    # (B, N_in, D) × (N_out, N_in) → (B, N_out, D)
    return torch.einsum("ot,btd->bod", W, features)
