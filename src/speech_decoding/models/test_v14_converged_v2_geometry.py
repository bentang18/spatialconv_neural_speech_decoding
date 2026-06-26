"""P2.1 geometry tests for the converged-v2 2-band magnitude frontend.

Locks the token layout BEFORE any stem/pool code lands: token totals
22 (1 s) / 110 (5 s), the 4 freq-patches (LFS 4/8/16 + HGA 7), the RoPE
8 : 1 clock, and the metadata-tensor shapes/ranges.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models import v14_converged_v2 as v2


def test_band_constants_bins_and_patches():
    assert v2.LFS.freq_patch_bins == (4, 8, 16)
    assert v2.LFS.n_freq_bins == 28
    assert v2.LFS.n_freq_patches == 3
    assert v2.HGA.freq_patch_bins == (7,)
    assert v2.HGA.n_freq_bins == 7
    assert v2.HGA.n_freq_patches == 1
    assert v2.BANDS_V2 == (v2.LFS, v2.HGA)  # low → high
    assert v2.N_FREQ_PATCHES_V2 == 4


@pytest.mark.parametrize(
    "clip_s, lfs_frames, hga_frames, lfs_tok, hga_tok, total",
    [
        (1.0, 5, 33, 6, 16, 22),
        (5.0, 21, 161, 30, 80, 110),
    ],
)
def test_token_geometry(clip_s, lfs_frames, hga_frames, lfs_tok, hga_tok, total):
    lfs, hga = v2.bands_for_clip_len(clip_s)
    assert (lfs.name, hga.name) == ("lfs", "hga")
    assert lfs.n_time_frames == lfs_frames
    assert hga.n_time_frames == hga_frames
    # tk=2 non-overlapping time-patches, × freq-patches
    assert lfs.n_tokens == lfs_tok
    assert hga.n_tokens == hga_tok
    assert sum(b.n_tokens for b in (lfs, hga)) == total


def test_one_second_constants_match_module_total():
    assert v2.N_TOKENS_V2 == 22
    assert sum(b.n_tokens for b in v2.bands_for_clip_len(1.0)) == v2.N_TOKENS_V2


def test_rope_clock_8_to_1():
    assert v2.LFS.time_patch_stride_samples == 1024   # tk2 × hop512
    assert v2.HGA.time_patch_stride_samples == 128    # tk2 × hop64
    assert v2.band_slot_mults() == [8, 1]
    # clock survives the 5 s retiming (stride is clip-independent)
    assert v2.band_slot_mults(v2.bands_for_clip_len(5.0)) == [8, 1]


def test_token_metadata_shapes_and_ranges():
    for clip_s, total in ((1.0, 22), (5.0, 110)):
        bands = v2.bands_for_clip_len(clip_s)
        band_id, freq_patch_id, time_slot = v2.token_metadata(bands)
        assert band_id.shape == freq_patch_id.shape == time_slot.shape == (total,)
        # LFS (band 0) tokens come first, then HGA (band 1)
        n_lfs = bands[0].n_tokens
        assert torch.equal(band_id[:n_lfs], torch.zeros(n_lfs, dtype=torch.long))
        assert torch.equal(band_id[n_lfs:], torch.ones(total - n_lfs, dtype=torch.long))
        # 4 distinct freq-patches: LFS 0/1/2, HGA 3
        assert set(freq_patch_id.tolist()) == {0, 1, 2, 3}
        assert set(freq_patch_id[:n_lfs].tolist()) == {0, 1, 2}
        assert set(freq_patch_id[n_lfs:].tolist()) == {3}
        # time_slot on the shared HGA-stride clock: LFS steps by 8, HGA by 1
        lfs_slots = sorted(set(time_slot[:n_lfs].tolist()))
        assert lfs_slots == [8 * i for i in range(bands[0].n_time_patches)]
        hga_slots = sorted(set(time_slot[n_lfs:].tolist()))
        assert hga_slots == list(range(bands[1].n_time_patches))


def test_clip_too_short_raises():
    with pytest.raises(ValueError, match="clip too short"):
        v2.bands_for_clip_len(0.05)  # ~102 samples → LFS 1 frame < tk2
