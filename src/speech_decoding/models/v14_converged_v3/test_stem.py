"""v14_converged_v3 Phase 2 — spectral stem fold (TDD).

Memo (project-v14-converged-v3-sensor-architecture, FRONTEND bullet): the 3 multi-
res |STFT| bands, each already per-(elec,bin) robust-z'd at load, are broadcast
onto the shared 32 Hz token clock (SLOW ×8, MID ×2, HGA ×1 — a hold/repeat, since
a slow frame is constant over its longer window), concatenated to 20 channels, and
folded by ONE weight-shared Linear(20 → d_model) → 1 token per (contact, 31.25 ms).
NO freq embed, NO band embed, NO per-band norm.

Input band layout matches v2's cache convention: (..., F_bins, T_band), freq axis
−2, time axis −1 (v2 forward: lfs (B,C,28,T), hga (B,C,7,T)).
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.stem import SpectralStem

# 4 s clip on the 32 Hz clock = 128 slots; band frame counts at 4/16/32 Hz.
T32 = 128
T_SLOW, T_MID, T_HGA = 16, 64, 128
B, C = 2, 5


def _bands(bB: int = B, cC: int = C):
    slow = torch.randn(bB, cC, 7, T_SLOW)
    mid = torch.randn(bB, cC, 6, T_MID)
    hga = torch.randn(bB, cC, 7, T_HGA)
    return slow, mid, hga


def test_output_shape_is_tokens_on_the_32hz_clock() -> None:
    stem = SpectralStem(d_model=256)
    out = stem(_bands())
    assert out.shape == (B, C, T32, 256)


def test_single_weight_shared_linear_no_embed_no_norm() -> None:
    stem = SpectralStem(d_model=256)
    linears = [m for m in stem.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 1
    assert (linears[0].in_features, linears[0].out_features) == (20, 256)
    # memo: NO freq/band embedding, NO per-band norm.
    assert not any(isinstance(m, torch.nn.Embedding) for m in stem.modules())
    assert not any(
        isinstance(m, (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.GroupNorm))
        for m in stem.modules()
    )


def test_total_channels_is_twenty() -> None:
    stem = SpectralStem(d_model=256)
    folded = stem.broadcast_concat(_bands())
    assert folded.shape == (B, C, 20, T32)


def test_broadcast_is_a_hold_repeat_along_time() -> None:
    # SLOW frame k must occupy the 8 consecutive 32 Hz slots [8k, 8k+8).
    slow = torch.arange(T_SLOW, dtype=torch.float32).reshape(1, 1, 1, T_SLOW).expand(1, 1, 7, T_SLOW)
    mid = torch.zeros(1, 1, 6, T_MID)
    hga = torch.zeros(1, 1, 7, T_HGA)
    stem = SpectralStem(d_model=256)
    folded = stem.broadcast_concat((slow, mid, hga))  # (1,1,20,128)
    slow_rows = folded[0, 0, :7, :]  # (7, 128)
    for k in range(T_SLOW):
        block = slow_rows[:, 8 * k : 8 * k + 8]
        assert torch.allclose(block, torch.full_like(block, float(k)))


def test_concat_order_is_slow_mid_hga() -> None:
    slow = torch.ones(1, 1, 7, T_SLOW)
    mid = torch.full((1, 1, 6, T_MID), 2.0)
    hga = torch.full((1, 1, 7, T_HGA), 3.0)
    stem = SpectralStem(d_model=256)
    folded = stem.broadcast_concat((slow, mid, hga))[0, 0, :, 0]  # (20,)
    assert folded[:7].tolist() == [1.0] * 7
    assert folded[7:13].tolist() == [2.0] * 6
    assert folded[13:].tolist() == [3.0] * 7


def test_wrong_bin_count_raises() -> None:
    stem = SpectralStem(d_model=256)
    bad_slow = torch.randn(B, C, 6, T_SLOW)  # 6 bins, expected 7
    _, mid, hga = _bands()
    with pytest.raises(ValueError):
        stem((bad_slow, mid, hga))


def test_non_multiple_time_length_raises() -> None:
    stem = SpectralStem(d_model=256)
    bad_slow = torch.randn(B, C, 7, 15)  # 15 * 8 = 120 ≠ 128
    _, mid, hga = _bands()
    with pytest.raises(ValueError):
        stem((bad_slow, mid, hga))


def test_arbitrary_leading_dims() -> None:
    # No batch axis: (C, F, T) → (C, T32, d).
    slow = torch.randn(C, 7, T_SLOW)
    mid = torch.randn(C, 6, T_MID)
    hga = torch.randn(C, 7, T_HGA)
    stem = SpectralStem(d_model=256)
    out = stem((slow, mid, hga))
    assert out.shape == (C, T32, 256)
