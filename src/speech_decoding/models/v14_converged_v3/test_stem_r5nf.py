"""v3r5nf NoFusionStem — EarlyFusionStem's fully-separated sibling (TDD).

Same two 64 Hz streams as r5-fused (HGA 4-bin |STFT|, LFS 1-ch raw voltage), IDENTICAL
caches — but each conv-pooled by its OWN stem into its OWN 32 Hz token stream, tagged by a
2-way band_type_emb. TWO token streams per (contact, 32 Hz slot), both stride-1 lattice.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.stem import (
    NOFUSION_BINS,
    NOFUSION_DECIMATE,
    NoFusionStem,
    clock_length_32hz,
)

D = 256
T = 128  # 32 Hz tokens on a 4 s clip
L = NOFUSION_DECIMATE * T  # 64 Hz frames = 2T = 256
B, C = 2, 5
NF_HGA_BINS, NF_LFS_CH = NOFUSION_BINS  # (4, 1)


def _streams(bB: int = B, cC: int = C):
    hga = torch.randn(bB, cC, NF_HGA_BINS, L)
    lfs = torch.randn(bB, cC, NF_LFS_CH, L)
    return hga, lfs


def test_two_token_streams_on_the_32hz_clock() -> None:
    stem = NoFusionStem(d_model=D)
    toks, pos = stem(_streams())
    assert len(toks) == 2 and len(pos) == 2  # HGA + LFS, NOT fused into one
    assert toks[0].shape == (B, C, T, D)  # HGA stream
    assert toks[1].shape == (B, C, T, D)  # LFS stream
    assert pos[0].shape[0] == T and pos[1].shape[0] == T


def test_both_streams_share_stride_one_positions() -> None:
    # single-rate: lattice position == token index (stride 1); HGA and LFS at the same
    # (elec, frame) share the SAME RoPE position — band_type_emb is the only distinguisher.
    stem = NoFusionStem(d_model=D)
    _, pos = stem(_streams())
    assert torch.equal(pos[0], torch.arange(T))
    assert torch.equal(pos[1], torch.arange(T))


def test_band_type_emb_shape_and_two_way() -> None:
    stem = NoFusionStem(d_model=D)
    assert stem.band_type_emb.shape == (2, D)
    # per-stream rows differ at init (deliberate structure the model uses).
    assert not torch.allclose(stem.band_type_emb[0], stem.band_type_emb[1])


def test_band_type_emb_is_additive_per_stream() -> None:
    # zeroing the band emb ⇒ each stream's tokens == its conv-pool output exactly.
    stem = NoFusionStem(d_model=D)
    hga, lfs = _streams(1, 1)
    with torch.no_grad():
        stem.band_type_emb.zero_()
    toks, _ = stem((hga, lfs))
    assert torch.allclose(toks[0], stem._pool(hga, stem.hga_stem), atol=1e-6)
    assert torch.allclose(toks[1], stem._pool(lfs, stem.lfs_stem), atol=1e-6)


def test_hga_and_lfs_stems_are_separate_params() -> None:
    stem = NoFusionStem(d_model=D)
    # first conv of each stem: HGA in_channels 4, LFS in_channels 1 (SEPARATE weights).
    assert stem.hga_stem[0].in_channels == NF_HGA_BINS
    assert stem.lfs_stem[0].in_channels == NF_LFS_CH
    assert stem.hga_stem[0].out_channels == D and stem.lfs_stem[0].out_channels == D
    # independent objects — no accidental weight sharing.
    assert stem.hga_stem[0].weight.data_ptr() != stem.lfs_stem[0].weight.data_ptr()


def test_stem_shape_mirrors_early_fusion_two_convs_kernel3() -> None:
    stem = NoFusionStem(d_model=D)
    for name, s, c_in in (("hga", stem.hga_stem, NF_HGA_BINS), ("lfs", stem.lfs_stem, NF_LFS_CH)):
        convs = [m for m in s.modules() if isinstance(m, torch.nn.Conv1d)]
        assert len(convs) == 2, name
        assert (convs[0].in_channels, convs[0].out_channels) == (c_in, D)
        assert convs[0].kernel_size == (3,) and convs[0].stride == (1,) and convs[0].padding == (1,)
        assert (convs[1].in_channels, convs[1].out_channels) == (D, D)
        assert convs[1].kernel_size == (3,) and convs[1].stride == (2,) and convs[1].padding == (1,)
        assert any(isinstance(m, torch.nn.GELU) for m in s.modules())
    # no Linear proj, no per-band norm.
    assert not any(isinstance(m, torch.nn.Linear) for m in stem.modules())
    assert not any(
        isinstance(m, (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.GroupNorm))
        for m in stem.modules()
    )


def test_arbitrary_leading_dims() -> None:
    stem = NoFusionStem(d_model=D)
    hga = torch.randn(C, NF_HGA_BINS, L)  # no batch axis
    lfs = torch.randn(C, NF_LFS_CH, L)
    toks, pos = stem((hga, lfs))
    assert toks[0].shape == (C, T, D) and toks[1].shape == (C, T, D)
    assert pos[0].shape[0] == T


def test_wrong_stream_count_raises() -> None:
    stem = NoFusionStem(d_model=D)
    hga, lfs = _streams()
    with pytest.raises(ValueError, match="2 streams"):
        stem((hga,))
    with pytest.raises(ValueError, match="2 streams"):
        stem((hga, lfs, hga))


def test_wrong_bins_raise() -> None:
    stem = NoFusionStem(d_model=D)
    _, lfs = _streams()
    bad_hga = torch.randn(B, C, 7, L)  # 7 bins, expected 4
    with pytest.raises(ValueError, match="HGA has"):
        stem((bad_hga, lfs))
    hga, _ = _streams()
    bad_lfs = torch.randn(B, C, 2, L)  # 2 channels, expected 1
    with pytest.raises(ValueError, match="LFS has"):
        stem((hga, bad_lfs))


def test_mismatched_or_odd_frame_count_raises() -> None:
    stem = NoFusionStem(d_model=D)
    hga = torch.randn(B, C, NF_HGA_BINS, L)
    lfs_short = torch.randn(B, C, NF_LFS_CH, L - 2)  # disagree with HGA
    with pytest.raises(ValueError, match="frame counts disagree"):
        stem((hga, lfs_short))
    hga_odd = torch.randn(B, C, NF_HGA_BINS, L - 1)  # 255, odd
    lfs_odd = torch.randn(B, C, NF_LFS_CH, L - 1)
    with pytest.raises(ValueError, match="must be even"):
        stem((hga_odd, lfs_odd))


# ── clock_length_32hz for no_fusion (identical derivation to early_fusion) ────────
def test_clock_no_fusion_halves_64hz_to_32hz() -> None:
    hga, lfs = _streams()
    assert clock_length_32hz((hga, lfs), no_fusion=True) == T


def test_clock_no_fusion_stream_disagreement_raises() -> None:
    hga = torch.randn(B, C, NF_HGA_BINS, L)
    lfs = torch.randn(B, C, NF_LFS_CH, L - 2)
    with pytest.raises(ValueError, match="disagree on the 32 Hz clock"):
        clock_length_32hz((hga, lfs), no_fusion=True)
    odd = torch.randn(B, C, NF_HGA_BINS, L - 1)
    with pytest.raises(ValueError, match="must be even"):
        clock_length_32hz((odd, odd), no_fusion=True)


def test_clock_no_fusion_mutually_exclusive_with_others() -> None:
    hga, lfs = _streams()
    with pytest.raises(ValueError, match="mutually exclusive"):
        clock_length_32hz((hga, lfs), no_fusion=True, early_fusion=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        clock_length_32hz((hga, lfs), no_fusion=True, native_fine_hga=True)
