"""v14_converged_v3 — spectral stem fold (Phase 2).

Fold the 3 multi-resolution |STFT| bands into one content token per (contact,
31.25 ms slot). Memo: project-v14-converged-v3-sensor-architecture (FRONTEND).

Each band arrives as ``(..., F_bins, T_band)`` (v2 cache convention: freq axis −2,
time axis −1), already per-(elec,bin) robust-z'd at load. The stem:

  1. aligns each band's time axis onto the shared 32 Hz clock. As of the
     uniform-hop fix (2026-07-10) every band is EXTRACTED at hop=64 → 32 Hz
     natively, so the per-band factors are all 1 (no hold). The repeat_interleave
     is SKIPPED when factor==1 (it is NOT free — ``repeat_interleave(1)`` still
     allocates a fresh copy of every band each step); the frame-count == clock
     guard is the separate ``t32`` check below, so nothing is lost;
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
# (2026-07-10) every band is extracted at 32 Hz ⇒ all factors are 1 (no hold), so the
# hold is skipped; broadcast_concat still asserts each band == the clock via t32.
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
            # hold along time; skip the alloc when factor==1 (native 32 Hz, the v3
            # uniform-hop case) — repeat_interleave(1) still copies every band/step.
            up = x if factor == 1 else x.repeat_interleave(factor, dim=-1)
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


# ── r4 / Design B per-band decimated stem ──────────────────────────────────────
# (n_bins, lattice_stride) per band in SLOW, MID, HGA order. lattice_stride = the
# 32 Hz-frame decimation step = the shared-lattice position increment per token:
# HGA 1 (32 Hz), MID 2 (16 Hz), SLOW 8 (4 Hz). Bins 7/6/7 = the v2 cache band widths.
PER_BAND_SPECS: tuple[tuple[int, int], ...] = ((7, 8), (6, 2), (7, 1))


class PerBandStem(nn.Module):
    """Per-band DECIMATED token stem (r4 / Design B, contract project-r4-contract-2026-07-15).

    Each band arrives on the shared 32 Hz clock as ``(..., F_b, T32)`` (v2 cache
    convention: freq axis −2, time −1, per-(elec,bin) robust-z'd at load). This stem:

      1. DECIMATES each band to its own token rate by a strided time slice — HGA stride 1
         (32 Hz), MID stride 2 (16 Hz), SLOW stride 8 (4 Hz). The dropped frames are the
         ~75%/94% window-overlap-redundant frames that ARE the M14 leak; the surviving
         token hop = nperseg/2 makes the overlap factor EXACTLY 2 in every band, so a
         width-4 mask block buries the deepest cell at margin 2 = zero raw-sample overlap.
      2. projects each band with its OWN ``Linear(F_b → d)`` (separate weights self-identify
         the band) and ADDS a learnable per-band ``band_type_emb`` (Ben: "100% add a band
         embed" — REVERSES the old single-concat-token / no-band-embed frontend).
      3. emits, per band, each token's SHARED-32Hz-LATTICE position ``token_index · stride``
         — the L1 time-RoPE coordinate. HGA stride 1 keeps the tuned base regime; MID/SLOW
         inherit the same per-unit frequency at wider strides, so a SLOW token at lattice
         8k and an HGA token at 8k share phase ⇒ band mixing aligns them in physical time.

    Ragged BY DESIGN: bands carry different token counts (T32 / stride). Deliberately NO
    freq embed and NO per-band norm (per-band norm reintroduces within-band 1/f dominance);
    band identity rides the separate projections + the additive band embed.
    """

    def __init__(
        self,
        d_model: int = 256,
        *,
        bands: Sequence[tuple[int, int]] = PER_BAND_SPECS,
        band_emb_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.specs = tuple((int(nb), int(st)) for nb, st in bands)
        self.projs = nn.ModuleList(nn.Linear(nb, d_model) for nb, _ in self.specs)
        # additive per-band identity, one d-vector per band; standard 0.02 init (the band
        # is deliberate structure we WANT the model to use, unlike the near-zero parcel
        # nuisance embed). band_emb_std is the A/B knob.
        self.band_type_emb = nn.Parameter(torch.empty(len(self.specs), d_model))
        self.projs.apply(init_transformer_weights)  # V-JEPA trunc_normal(0.02)+zero-bias
        nn.init.trunc_normal_(self.band_type_emb, std=band_emb_std)

    @staticmethod
    def decimate(x: Tensor, stride: int) -> Tensor:
        """Strided time slice ``x[..., ::stride]`` — the band's own-rate frames. Requires
        the 32 Hz length to be an exact multiple of ``stride`` (128 % {1,2,8} == 0)."""
        t32 = x.shape[-1]
        if t32 % stride != 0:
            raise ValueError(f"32 Hz length {t32} not a multiple of stride {stride}")
        return x[..., ::stride]

    def forward(
        self, band_inputs: Sequence[Tensor]
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Bands ``[(...,F_b,T32)]`` → (per-band tokens ``(..., T_b, d)``, per-band lattice
        positions ``(T_b,)`` long). Order is SLOW, MID, HGA (the spec order)."""
        if len(band_inputs) != len(self.specs):
            raise ValueError(f"expected {len(self.specs)} bands, got {len(band_inputs)}")
        tokens: list[Tensor] = []
        positions: list[Tensor] = []
        for b, (x, (n_bins, stride), proj) in enumerate(
            zip(band_inputs, self.specs, self.projs)
        ):
            if x.shape[-2] != n_bins:
                raise ValueError(
                    f"band {b} has {x.shape[-2]} freq bins, expected {n_bins}"
                )
            xd = self.decimate(x, stride)  # (..., F_b, T_b)
            xd = xd.transpose(-1, -2)  # (..., T_b, F_b)
            tok = proj(xd) + self.band_type_emb[b]  # (..., T_b, d)
            t_b = xd.shape[-2]
            pos = torch.arange(t_b, device=x.device, dtype=torch.long) * stride  # (T_b,)
            tokens.append(tok)
            positions.append(pos)
        return tuple(tokens), tuple(positions)


# ── fine-HGA native-rate stem (2026-07-21, native-rate rebake) ──────────────────
# (n_bins,) per band in SLOW, MID, HGA order. HGA is the FINE band (4 bins 64-160 Hz,
# k2..k5) — down from the coarse 7 (STFT_2BAND_HGA). SLOW/MID keep the v2 widths.
FINE_HGA_BINS: tuple[int, int, int] = (7, 6, 4)
# lattice stride per band (32 Hz-frame position increment) — IDENTICAL to PER_BAND_SPECS
# so the emitted RoPE positions match PerBandStem exactly and pack/mask/pe are unchanged.
FINE_LATTICE_STRIDES: tuple[int, int, int] = (8, 2, 1)
HGA_POOL_FACTOR: int = 4  # 128 Hz → 32 Hz via 2× stride-2 convs (2² = 4)


class FineHgaStem(nn.Module):
    """Native-rate per-band token stem — fine-HGA OFAT (2026-07-21).

    The native-rate rebake extracts each band at ITS OWN rate (view.py: SLOW hop 512
    → 4 Hz, MID hop 128 → 16 Hz, HGA hop 16 → 128 Hz), so this stem receives bands at
    DIFFERENT frame counts on a clip: SLOW ``T/8``, MID ``T/2``, HGA ``4T`` (T = the
    32 Hz clip length). It emits the SAME per-band token grid + lattice positions as
    ``PerBandStem`` (T/8, T/2, T at strides 8, 2, 1) so packing/masking/pe/attention
    are byte-identical downstream — the OFAT changes ONLY the HGA time front-end:

      • SLOW/MID: already at their token rate ⇒ NO decimate, a plain per-band
        ``Linear(F_b → d)`` + additive ``band_type_emb`` (matches PerBandStem's
        SLOW/MID path; those tokens are bit-identical given the same weights + native
        inputs, since native extraction == PerBandStem's ::8/::2 decimate — proven in
        test_native_rate_*_equals_decimated_32hz).
      • HGA: arrives at 128 Hz and is LEARN-pooled 128 → 32 Hz by 2×``Conv1d(k3, s2,
        pad1)`` + GELU (4 → d → d), replacing the fixed strided decimate. RF = 7 frames
        @128 Hz ≈ 54.7 ms reaches ±1 output token, so a width-4 mask block buries the
        deepest cell at margin 2 (leak-clean; test_fine_hga_conv_pool_is_local).

    Deliberately NO freq embed / NO per-band norm — band identity rides the separate
    projections + the additive band embed, exactly as PerBandStem.
    """

    def __init__(self, d_model: int = 256, *, band_emb_std: float = 0.02) -> None:
        super().__init__()
        n_slow, n_mid, n_hga = FINE_HGA_BINS
        self.slow_proj = nn.Linear(n_slow, d_model)
        self.mid_proj = nn.Linear(n_mid, d_model)
        # HGA conv-pool 128 → 32 Hz: 4 bins → d → d, two stride-2 k3 convs, GELU between.
        self.hga_pool = nn.Sequential(
            nn.Conv1d(n_hga, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
        )
        self.band_type_emb = nn.Parameter(torch.empty(3, d_model))
        self.apply(init_transformer_weights)  # Linear trunc_normal(0.02)+zero-bias; convs default
        nn.init.trunc_normal_(self.band_type_emb, std=band_emb_std)

    def _pool_hga(self, hga: Tensor) -> Tensor:
        """(..., F_hga, 4T) → (..., T, d) via the 2× stride-2 conv-pool."""
        *lead, f, length = hga.shape
        if length % HGA_POOL_FACTOR != 0:
            raise ValueError(
                f"HGA length {length} not a multiple of pool factor {HGA_POOL_FACTOR}"
            )
        x = hga.reshape(-1, f, length)  # (prod_lead, F_hga, 4T)
        x = self.hga_pool(x)  # (prod_lead, d, T)
        x = x.transpose(-1, -2)  # (prod_lead, T, d)
        return x.reshape(*lead, x.shape[-2], x.shape[-1])  # (..., T, d)

    def forward(
        self, band_inputs: Sequence[Tensor]
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Bands ``[SLOW(...,7,T/8), MID(...,6,T/2), HGA(...,4,4T)]`` → (per-band tokens
        ``(..., T_b, d)``, per-band lattice positions ``(T_b,)`` long), SLOW/MID/HGA order."""
        if len(band_inputs) != 3:
            raise ValueError(f"expected 3 bands, got {len(band_inputs)}")
        slow, mid, hga = band_inputs
        for name, x, nb in (
            ("slow", slow, FINE_HGA_BINS[0]),
            ("mid", mid, FINE_HGA_BINS[1]),
            ("hga", hga, FINE_HGA_BINS[2]),
        ):
            if x.shape[-2] != nb:
                raise ValueError(f"band {name} has {x.shape[-2]} freq bins, expected {nb}")

        slow_tok = self.slow_proj(slow.transpose(-1, -2)) + self.band_type_emb[0]
        mid_tok = self.mid_proj(mid.transpose(-1, -2)) + self.band_type_emb[1]
        hga_tok = self._pool_hga(hga) + self.band_type_emb[2]
        tokens = (slow_tok, mid_tok, hga_tok)

        s_slow, s_mid, s_hga = FINE_LATTICE_STRIDES
        positions = (
            torch.arange(slow_tok.shape[-2], device=slow.device, dtype=torch.long) * s_slow,
            torch.arange(mid_tok.shape[-2], device=mid.device, dtype=torch.long) * s_mid,
            torch.arange(hga_tok.shape[-2], device=hga.device, dtype=torch.long) * s_hga,
        )
        return tokens, positions
