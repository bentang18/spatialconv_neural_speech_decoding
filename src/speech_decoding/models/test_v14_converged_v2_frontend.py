"""P2.2 tests for the converged-v2 per-electrode frontend (stems + encoder)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models import v14_converged_v2 as v2
from speech_decoding.models.v14_converged_v2 import (
    FrontendEncoderV2,
    TwoBandTokenizerV2,
    _freq_patch_slices,
)


def _inputs(clip_s: float, b: int = 2, c: int = 3):
    lfs_band, hga_band = v2.bands_for_clip_len(clip_s)
    torch.manual_seed(0)
    lfs = torch.randn(b, c, lfs_band.n_freq_bins, lfs_band.n_time_frames)
    hga = torch.randn(b, c, hga_band.n_freq_bins, hga_band.n_time_frames)
    return lfs, hga, lfs_band, hga_band


def test_freq_patch_slices():
    assert _freq_patch_slices(v2.LFS) == [(0, 4), (4, 12), (12, 28)]
    assert _freq_patch_slices(v2.HGA) == [(0, 7)]


def test_tokenizer_stem_geometry():
    tok = TwoBandTokenizerV2(d_model=32)
    # 4 stems: LFS 3 (fk 4/8/16) + HGA 1 (fk 7)
    assert len(tok.stems) == 4
    assert [s.kernel_freq for s in tok.stems] == [4, 8, 16, 7]
    assert all(s.kernel_time == 2 for s in tok.stems)
    assert tok._stem_slice == [(0, 4), (4, 12), (12, 28), (0, 7)]
    assert tok._stem_band == [0, 0, 0, 1]
    assert tok._stem_name == ["lfs_fp0", "lfs_fp1", "lfs_fp2", "hga_fp0"]


@pytest.mark.parametrize("clip_s, total", [(1.0, 22), (5.0, 110)])
def test_tokenizer_output_shape(clip_s, total):
    lfs, hga, lfs_band, hga_band = _inputs(clip_s)
    bands = (lfs_band, hga_band)
    tok = TwoBandTokenizerV2(d_model=32, bands=bands)
    out = tok(lfs, hga)
    assert out.shape == (2, 3, total, 32)
    # per-stem monitor populated for all 4 stems
    assert set(tok.last_band_token_norm) == {"lfs_fp0", "lfs_fp1", "lfs_fp2", "hga_fp0"}


def test_tokenizer_token_order_lfs_then_hga():
    """LFS tokens (band 0) precede HGA (band 1); buffers agree with metadata."""
    lfs, hga, lfs_band, hga_band = _inputs(1.0)
    bands = (lfs_band, hga_band)
    tok = TwoBandTokenizerV2(d_model=16, bands=bands)
    bid, fpid, slot = v2.token_metadata(bands)
    assert torch.equal(tok.band_id, bid)
    assert torch.equal(tok.freq_patch_id, fpid)
    assert torch.equal(tok.time_slot, slot)
    out = tok(lfs, hga)
    n_lfs = lfs_band.n_tokens
    assert n_lfs == 6 and out.shape[2] == 22


def test_tokenizer_rejects_wrong_freq_bins():
    tok = TwoBandTokenizerV2(d_model=16)
    bad_lfs = torch.randn(2, 3, 27, 5)   # 27 ≠ 28
    hga = torch.randn(2, 3, 7, 33)
    with pytest.raises(ValueError, match="expected 28"):
        tok(bad_lfs, hga)


@pytest.mark.parametrize("clip_s, total", [(1.0, 22), (5.0, 110)])
def test_frontend_encoder_output_shape(clip_s, total):
    lfs, hga, lfs_band, hga_band = _inputs(clip_s)
    bands = (lfs_band, hga_band)
    enc = FrontendEncoderV2(d_model=32, n_heads=4, n_layers=2, bands=bands)
    out = enc(lfs, hga)
    assert out.shape == (2, 3, total, 32)


def test_frontend_key_mask_leak_free():
    """Masked cells never serve as keys ⇒ visible-cell outputs are invariant to
    the masked cells' VALUES (the batchable equivalent of physically dropping)."""
    lfs, hga, lfs_band, hga_band = _inputs(1.0)
    bands = (lfs_band, hga_band)
    enc = FrontendEncoderV2(d_model=32, n_heads=4, n_layers=2, bands=bands)
    enc.eval()
    n_lfs = lfs_band.n_tokens                       # first 6 = LFS (visible)
    S = lfs_band.n_tokens + hga_band.n_tokens
    # Hide ALL HGA tokens; LFS stays visible.
    key_mask = torch.zeros(2, 3, S, dtype=torch.bool)
    key_mask[:, :, :n_lfs] = True
    with torch.no_grad():
        out1 = enc(lfs, hga, key_mask=key_mask)
        out2 = enc(lfs, hga + 9.0, key_mask=key_mask)  # corrupt every HGA cell
    # LFS (visible) outputs identical; HGA (masked queries) free to differ.
    assert torch.allclose(out1[:, :, :n_lfs], out2[:, :, :n_lfs], atol=1e-6)
    assert not torch.allclose(out1[:, :, n_lfs:], out2[:, :, n_lfs:])


def test_frontend_rope_has_enough_slots():
    enc = FrontendEncoderV2(d_model=32, n_heads=4, n_layers=1)
    # 1 s: max time_slot = 15 (HGA) ⇒ ≥16 slots
    assert enc.key_rope.shape[1] >= int(enc.tokenizer.time_slot.max()) + 1
