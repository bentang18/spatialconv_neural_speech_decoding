"""v14_converged_v3 — spectral stem fold (Phase 2).

Fold the 3 multi-resolution |STFT| bands into one content token per (contact,
31.25 ms slot). Memo: project-v14-converged-v3-sensor-architecture (FRONTEND).

Each band arrives as ``(..., F_bins, T_band)`` (v2 cache convention: freq axis −2,
time axis −1), already per-(elec,bin) robust-z'd at load. The stem:

  1. broadcasts each band's time axis onto the shared 32 Hz clock by a per-band
     integer HOLD factor (SLOW ×8 @4Hz, MID ×2 @16Hz, HGA ×1 @32Hz) —
     repeat_interleave, because a slow frame is constant across its longer window;
  2. concatenates the bands along freq → 20 channels;
  3. applies ONE weight-shared ``Linear(20 → d_model)`` per (contact, slot).

Deliberately bare: NO freq embedding, NO band embedding, NO per-band norm (the
projection's weight columns self-identify each band; per-band norm would
reintroduce delta's within-band 1/f dominance — memo). Band time-lengths must be
exact integer sub-multiples of the 32 Hz clock; cropping the stft's center-pad
off-by-one to clean multiples is the assembly's job, upstream of the stem.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

# (n_bins, hold_factor) per band, in concat order. 7·1 + 6·... note factors are
# T32/T_band = 32Hz / band-frame-rate: SLOW 4Hz→8, MID 16Hz→2, HGA 32Hz→1.
V3_BANDS: tuple[tuple[int, int], ...] = ((7, 8), (6, 2), (7, 1))


class SpectralStem(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        *,
        bands: Sequence[tuple[int, int]] = V3_BANDS,
    ) -> None:
        super().__init__()
        self.bands = tuple((int(nb), int(f)) for nb, f in bands)
        self.total_bins = sum(nb for nb, _ in self.bands)
        self.proj = nn.Linear(self.total_bins, d_model)

    def broadcast_concat(self, band_inputs: Sequence[Tensor]) -> Tensor:
        """Bands ``[(...,F_b,T_b)]`` → ``(..., total_bins, T32)`` on the 32 Hz clock."""
        if len(band_inputs) != len(self.bands):
            raise ValueError(
                f"expected {len(self.bands)} bands, got {len(band_inputs)}"
            )
        t32: int | None = None
        held: list[Tensor] = []
        for x, (n_bins, factor) in zip(band_inputs, self.bands):
            if x.shape[-2] != n_bins:
                raise ValueError(
                    f"band input has {x.shape[-2]} freq bins, expected {n_bins}"
                )
            up = x.repeat_interleave(factor, dim=-1)  # hold along time
            if t32 is None:
                t32 = up.shape[-1]
            elif up.shape[-1] != t32:
                raise ValueError(
                    f"band time {x.shape[-1]}×{factor}={up.shape[-1]} != clock {t32}; "
                    "band lengths must be exact sub-multiples of the 32 Hz clock"
                )
            held.append(up)
        return torch.cat(held, dim=-2)  # (..., total_bins, T32)

    def forward(self, band_inputs: Sequence[Tensor]) -> Tensor:
        """Bands ``[(...,F_b,T_b)]`` → tokens ``(..., T32, d_model)``."""
        folded = self.broadcast_concat(band_inputs)  # (..., total_bins, T32)
        folded = folded.transpose(-1, -2)  # (..., T32, total_bins)
        return self.proj(folded)  # (..., T32, d_model)
