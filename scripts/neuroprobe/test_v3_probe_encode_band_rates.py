"""--band-rates: an enc0 whose HGA is baked FINER than the 32 Hz clip clock.

The 64 Hz HGA question ("does a finer high-gamma stream beat the current 32 Hz one at enc0")
needs slow/mid read at 32 Hz beside an HGA read at 64 Hz. Two things can silently produce a
complete, plausible, WRONG answer here, and both are what these tests pin:

  1. A declared rate that does not match the bake turns the per-band window from a decimation
     into a compressed, time-shifted slice AT THE RIGHT SHAPE — the bug that invalidated four
     r6 runs (2026-07-23). ``assert_band_rates_match_cache`` is the guard; these tests pin that
     the parse feeds it the right thing and that a mixed declaration survives round-trip.
  2. ``band_lengths`` in the record is grid-derived (``clip_frames // BAND_STRIDES`` on the
     32 Hz clock), which stops describing enc0 the moment a band is baked off-clock. A record
     saying HGA=32 over a payload carrying 64 would hand every consumer a wrong slicing of
     enc0 that still passes a width check on the WRONG width.
"""
from __future__ import annotations

import pytest
import torch

from scripts.neuroprobe.v3_probe_encode_r4 import _enc0_band_lengths, _parse_band_rates
from speech_decoding.models.v14_converged_v3.dataset import assert_band_rates_match_cache
from speech_decoding.models.v14_converged_v3.pack_r4 import BAND_STRIDES, band_token_counts

CLIP_FRAMES = 32  # 1 s on the 32 Hz clip clock
UNIFORM = ((1, 1), (1, 1), (1, 1))
FINE_HGA = ((1, 1), (1, 1), (2, 1))
FDIMS = (7, 6, 7)          # published SLOW/MID/HGA bins
FDIMS_V3HGA = (7, 6, 4)    # band_v3hga is 4 bins over 64-160 Hz


def _bands(frames):
    """(n, N, F_b, T_b) per band — only the last two axes matter to the functions under test."""
    return [torch.zeros(2, 3, f, t) for f, t in zip(FDIMS, frames)]


def test_parse_band_rates_reads_num_den_and_bare_ints() -> None:
    assert _parse_band_rates("1/1,1/1,2/1", 3) == FINE_HGA
    assert _parse_band_rates("1, 1, 2", 3) == FINE_HGA, "a bare int must mean num/1"


def test_parse_band_rates_rejects_a_count_that_misaligns_with_the_caches() -> None:
    # Silent misalignment would zip a rate onto the WRONG band, which is exactly failure mode 1.
    with pytest.raises(SystemExit):
        _parse_band_rates("1/1,2/1", 3)


def test_declared_fine_hga_matches_a_64hz_bake_and_rejects_a_32hz_one() -> None:
    """The parse is only useful if it feeds the real guard correctly."""
    assert_band_rates_match_cache([32, 32, 64], _parse_band_rates("1/1,1/1,2/1", 3), where="fine")
    with pytest.raises(ValueError, match="but its cache is 32 Hz"):
        # Declaring 64 Hz HGA against the CURRENT 32 Hz bake: the shapes still work out, so
        # nothing but this guard catches it.
        assert_band_rates_match_cache([32, 32, 32], FINE_HGA, where="fine")


def test_enc0_band_lengths_reproduces_the_published_layout_on_uniform_caches() -> None:
    """The default path must be unchanged: (4,16,32) = 348 columns per unit."""
    got = _enc0_band_lengths(_bands((CLIP_FRAMES,) * 3), None)
    assert got == band_token_counts(CLIP_FRAMES), "diverged from pack_r4's own grid lengths"
    assert got == (4, 16, 32)
    assert sum(t * f for t, f in zip(got, FDIMS)) == 348


def test_enc0_band_lengths_tracks_a_64hz_hga_instead_of_the_32hz_grid() -> None:
    """HGA read at 64 Hz keeps stride 1, so enc0 carries 64 frames — not the grid's 32."""
    bands = [torch.zeros(2, 3, f, t) for f, t in zip(FDIMS_V3HGA, (32, 32, 64))]
    got = _enc0_band_lengths(bands, None)
    assert got == (4, 16, 64)
    assert got[2] != band_token_counts(CLIP_FRAMES)[2], (
        "setup failed: the grid and the payload agree, so this proves nothing")
    assert sum(t * f for t, f in zip(got, FDIMS_V3HGA)) == 380


def test_enc0_band_lengths_matches_the_decimation_it_describes() -> None:
    """Not a formula check: the length must equal what x[..., ::stride] actually yields."""
    frames = (32, 32, 64)
    bands = [torch.zeros(2, 3, f, t) for f, t in zip(FDIMS_V3HGA, frames)]
    got = _enc0_band_lengths(bands, None)
    for b, st, want in zip(bands, BAND_STRIDES, got):
        assert b[..., ::st].shape[-1] == want
