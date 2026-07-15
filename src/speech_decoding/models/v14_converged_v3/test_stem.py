"""v14_converged_v3 Phase 2 — spectral stem fold (TDD).

Memo (project-v14-converged-v3-sensor-architecture, FRONTEND bullet): the 3 multi-
res |STFT| bands, each already per-(elec,bin) robust-z'd at load, sit on the shared
32 Hz token clock. As of the uniform-hop=64 fix (2026-07-10) every band is extracted
at 32 Hz natively (SLOW/MID/HGA all one frame per 31.25 ms slot — NO hold/repeat),
concatenated to 20 channels, and folded by ONE weight-shared Linear(20 → d_model)
→ 1 token per (contact, 31.25 ms). NO freq embed, NO band embed, NO per-band norm.

Input band layout matches v2's cache convention: (..., F_bins, T_band), freq axis
−2, time axis −1 (v2 forward: lfs (B,C,28,T), hga (B,C,7,T)).
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.stem import PerBandStem, SpectralStem

# 4 s clip on the 32 Hz clock = 128 slots. Uniform hop=64 → every band at 32 Hz,
# so all three arrive with 128 frames (no hold).
T32 = 128
T_SLOW, T_MID, T_HGA = 128, 128, 128
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


def test_no_hold_each_band_frame_maps_one_to_one_to_a_slot() -> None:
    # Uniform hop=64: SLOW frame k occupies EXACTLY slot k (no ×8 hold). Adjacent
    # slots carry independent slow content — the whole point of the hop fix.
    slow = torch.arange(T_SLOW, dtype=torch.float32).reshape(1, 1, 1, T_SLOW).expand(1, 1, 7, T_SLOW)
    mid = torch.zeros(1, 1, 6, T_MID)
    hga = torch.zeros(1, 1, 7, T_HGA)
    stem = SpectralStem(d_model=256)
    folded = stem.broadcast_concat((slow, mid, hga))  # (1,1,20,128)
    slow_rows = folded[0, 0, :7, :]  # (7, 128)
    expected = torch.arange(T32, dtype=torch.float32).expand(7, T32)
    assert torch.allclose(slow_rows, expected)


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


def test_mismatched_band_length_raises() -> None:
    # Uniform hop=64 (factor 1): a band whose frame count != the clock must raise.
    stem = SpectralStem(d_model=256)
    bad_slow = torch.randn(B, C, 7, 127)  # 127 ≠ 128
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


# ── r4 PerBandStem (decimate both; Design B) ───────────────────────────────────
D = 256
# decimated token counts on a 128-slot 32 Hz clip: SLOW /8 = 16, MID /2 = 64, HGA /1 = 128.
T_SLOW_TOK, T_MID_TOK, T_HGA_TOK = 16, 64, 128


def test_perband_shapes_and_position_lengths() -> None:
    stem = PerBandStem(d_model=D)
    toks, pos = stem(_bands())
    assert [t.shape for t in toks] == [
        (B, C, T_SLOW_TOK, D), (B, C, T_MID_TOK, D), (B, C, T_HGA_TOK, D),
    ]
    assert [p.shape[0] for p in pos] == [T_SLOW_TOK, T_MID_TOK, T_HGA_TOK]


def test_perband_lattice_positions_are_strided_on_one_32hz_lattice() -> None:
    stem = PerBandStem(d_model=D)
    _, pos = stem(_bands())
    assert torch.equal(pos[0], torch.arange(T_SLOW_TOK) * 8)  # SLOW stride 8
    assert torch.equal(pos[1], torch.arange(T_MID_TOK) * 2)   # MID stride 2
    assert torch.equal(pos[2], torch.arange(T_HGA_TOK) * 1)   # HGA stride 1
    # band mixing alignment: SLOW token k and HGA token 8k share the same lattice phase.
    for k in range(T_SLOW_TOK):
        assert pos[0][k].item() == pos[2][8 * k].item()


def test_perband_decimation_picks_every_stride_th_frame() -> None:
    # frame content = its 32 Hz index (broadcast over bins) ⇒ decimated frames are exactly
    # the strided indices, checked BEFORE projection via the pure staticmethod.
    x = torch.arange(T32, dtype=torch.float32).reshape(1, 1, 1, T32).expand(1, 1, 6, T32)
    d_mid = PerBandStem.decimate(x, 2)
    assert d_mid.shape[-1] == T_MID_TOK
    assert torch.equal(d_mid[0, 0, 0], torch.arange(0, T32, 2, dtype=torch.float32))
    d_slow = PerBandStem.decimate(x, 8)
    assert torch.equal(d_slow[0, 0, 0], torch.arange(0, T32, 8, dtype=torch.float32))


def test_perband_decimation_rejects_indivisible_stride() -> None:
    x = torch.randn(1, 1, 7, 127)  # 127 not a multiple of 8
    with pytest.raises(ValueError):
        PerBandStem.decimate(x, 8)


def test_perband_separate_projection_per_band() -> None:
    stem = PerBandStem(d_model=D)
    assert len(stem.projs) == 3
    assert [p.in_features for p in stem.projs] == [7, 6, 7]  # SLOW, MID, HGA bins
    assert all(p.out_features == D for p in stem.projs)
    # weights are independent objects (no accidental sharing).
    assert stem.projs[0].weight.data_ptr() != stem.projs[2].weight.data_ptr()


def test_perband_band_emb_is_additive_and_per_band_distinct() -> None:
    stem = PerBandStem(d_model=D)
    # per-band rows differ at init (deliberate structure).
    assert not torch.allclose(stem.band_type_emb[0], stem.band_type_emb[1])
    # additive: zero the band emb ⇒ output == proj(decimate(x).T) exactly.
    slow, mid, hga = _bands(1, 1)
    with torch.no_grad():
        stem.band_type_emb.zero_()
    toks, _ = stem((slow, mid, hga))
    xd = PerBandStem.decimate(mid, 2).transpose(-1, -2)  # (1,1,64,6)
    assert torch.allclose(toks[1], stem.projs[1](xd), atol=1e-6)


def test_perband_wrong_band_count_and_wrong_bins_raise() -> None:
    stem = PerBandStem(d_model=D)
    slow, mid, hga = _bands()
    with pytest.raises(ValueError):
        stem((slow, mid))  # 2 bands, expected 3
    bad_mid = torch.randn(B, C, 7, T_MID)  # 7 bins, MID expects 6
    with pytest.raises(ValueError):
        stem((slow, bad_mid, hga))
