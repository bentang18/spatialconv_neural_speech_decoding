"""v14_converged_v3 — spectral stem fold (Phase 2).

Fold the 3 multi-resolution |STFT| bands into one content token per (contact,
31.25 ms slot). Memo: project-v14-converged-v3-sensor-architecture (FRONTEND).

Each band arrives as ``(..., F_bins, T_band)`` (v2 cache convention: freq axis −2,
time axis −1), already per-(elec,bin) robust-z'd at load. The stem:

  1. aligns each band's time axis onto the shared 32 Hz clock. As of the
     uniform-hop fix (2026-07-10) every band is EXTRACTED at hop=64 → 32 Hz
     natively, so the per-band factors are all 1 (no hold). The repeat_interleave
     is retained as a no-op that also GUARDS every band's frame count == the clock;
  2. concatenates the bands along freq → 20 channels;
  3. applies ONE weight-shared ``Linear(20 → d_model)`` per (contact, slot).

  Historical note: before the fix, slow/mid were extracted at hop=512/128 (4/16 Hz)
  and HELD up (×8/×2) to the 32 Hz clock — held stairsteps with no real sub-250 ms
  / 62.5 ms timing. Uniform hop=64 gives all bands genuine 32 Hz temporal detail.

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

from speech_decoding.models.v14_converged_v3.pe import init_transformer_weights

# (n_bins, hold_factor) per band, in concat order: SLOW 7, MID 6, HGA 7 = 20 ch.
# hold_factor = T32 / T_band = 32Hz / band-frame-rate. With the uniform-hop=64 fix
# (2026-07-10) every band is extracted at 32 Hz ⇒ all factors are 1 (no hold). The
# factor is retained (=1) so broadcast_concat still asserts each band == the clock.
V3_BANDS: tuple[tuple[int, int], ...] = ((7, 1), (6, 1), (7, 1))


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
        self.apply(init_transformer_weights)  # V-JEPA 2 trunc_normal(0.02)+zero-bias

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
