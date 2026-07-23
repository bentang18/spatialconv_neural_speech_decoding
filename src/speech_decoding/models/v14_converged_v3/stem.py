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


class NativePerBandStem(PerBandStem):
    """r6 stem: PerBandStem on NATIVE-RATE bands — plain per-band Linear, ZERO decimate/conv.

    r6 bakes each band at its own native rate (SLOW @4 Hz, MID @16 Hz, HGA @32 Hz coarse
    7-bin ``STFT_2BAND_HGA``), so the bands arrive ALREADY at token rate — the r4/PerBandStem
    strided decimate has nothing to drop. This is the ONLY difference from PerBandStem: same
    ``specs``/``projs``/``band_type_emb`` (⇒ ``load_state_dict`` across the two is direct), same
    additive band-embed, same emitted lattice positions ``token_index · stride`` (8/2/1 on the
    shared 32 Hz reference lattice). Feeding this stem ``band[..., ::stride]`` reproduces
    PerBandStem's tokens byte-for-byte — the native bake IS the r4 decimate, precomputed
    (test_native_perband_equals_perband_decimated). Conv stem OUT (OFAT #1); no per-band norm.
    """

    def forward(
        self, band_inputs: Sequence[Tensor]
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Native bands ``[(...,F_b,T_b)]`` (T_b already the token count) → (per-band tokens
        ``(...,T_b,d)``, per-band lattice positions ``token_index·stride``). SLOW, MID, HGA order."""
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
            xt = x.transpose(-1, -2)  # (..., T_b, F_b) — NO decimate, native rate in
            tok = proj(xt) + self.band_type_emb[b]  # (..., T_b, d)
            t_b = xt.shape[-2]
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


def clock_length_32hz(
    band_inputs: Sequence[Tensor],
    *,
    native_fine_hga: bool = False,
    early_fusion: bool = False,
    no_fusion: bool = False,
    native_perband: bool = False,
) -> int:
    """The 32 Hz clip-clock length T from the input bands.

    Uniform arm0: SLOW is already at 32 Hz ⇒ T = SLOW.shape[-1]. Native fine-HGA: bands
    arrive at SLOW T/8, MID T/2, HGA 4T ⇒ derive from HGA (128 = 4·32, integer-exact) and
    assert all three agree, so a mis-shaped native cache fails loud instead of silently
    mis-gridding. r6 native-per-band (``native_perband``): SLOW T/8, MID T/2, HGA T (HGA at 32 Hz,
    7-bin coarse — NO conv-pool, factor 1, unlike fine-HGA's 128 Hz/4T) ⇒ T = HGA.shape[-1], same
    SLOW·8 / MID·2 agreement asserts. r5 early-fusion (EarlyFusionStem) AND v3r5nf no-fusion
    (NoFusionStem): 2 streams HGA/LFS both at 64 Hz = 2T frames ⇒ T = HGA/2 (stem stride-2 decimate
    64→32), assert LFS agrees (the derivation is IDENTICAL — both consume the same 2×64 Hz caches).
    Both the masking (model.py) and the grid (objective.py) call this so the two never
    disagree on T."""
    if (
        int(early_fusion) + int(native_fine_hga) + int(no_fusion) + int(native_perband) > 1
    ):
        raise ValueError(
            "early_fusion, native_fine_hga, no_fusion and native_perband are mutually exclusive"
        )
    if native_perband:
        s_slow, s_mid, _ = FINE_LATTICE_STRIDES  # (8, 2, 1) — SLOW/MID native = T/stride
        T = int(band_inputs[2].shape[-1])  # HGA native = T (32 Hz, stride 1, NO pool)
        if not (
            int(band_inputs[0].shape[-1]) * s_slow == T
            and int(band_inputs[1].shape[-1]) * s_mid == T
        ):
            raise ValueError(
                f"r6 native band frame counts disagree on the 32 Hz clock: "
                f"slow={int(band_inputs[0].shape[-1])} mid={int(band_inputs[1].shape[-1])} "
                f"hga={int(band_inputs[2].shape[-1])} → T={T}"
            )
        return T
    if early_fusion or no_fusion:
        L = int(band_inputs[0].shape[-1])  # HGA @ 64 Hz = 2T frames
        if L % 2 != 0:
            raise ValueError(f"early-fusion 64 Hz frame count {L} must be even (= 2T)")
        T = L // 2
        if int(band_inputs[1].shape[-1]) // 2 != T:  # LFS must ride the same 64 Hz clock
            raise ValueError(
                f"early-fusion streams disagree on the 32 Hz clock: "
                f"hga={L} lfs={int(band_inputs[1].shape[-1])} → T={T}"
            )
        return T
    if not native_fine_hga:
        return int(band_inputs[0].shape[-1])
    s_slow, s_mid, _ = FINE_LATTICE_STRIDES  # (8, 2, 1) — SLOW/MID native = T/stride
    T = int(band_inputs[2].shape[-1]) // HGA_POOL_FACTOR
    if not (
        int(band_inputs[0].shape[-1]) * s_slow == T
        and int(band_inputs[1].shape[-1]) * s_mid == T
    ):
        raise ValueError(
            f"native band frame counts disagree on the 32 Hz clock: "
            f"slow={int(band_inputs[0].shape[-1])} mid={int(band_inputs[1].shape[-1])} "
            f"hga={int(band_inputs[2].shape[-1])} → T={T}"
        )
    return T


def nf_token_geometry(t_32hz: int, *, decimate: int) -> tuple[int, int]:
    """(n_tokens_per_stream, time_stride) for a ``NoFusionStem`` of net decimation ``decimate``.

    The stem consumes 2×64 Hz caches (L = 2·T_32 frames) and emits L//decimate tokens/stream:
      • ``decimate=2`` (v3r5nf): n_tok = T_32, stride 1 — 32 Hz tokens, byte-identical to before.
      • ``decimate=4`` (v3r5nffast OFAT): n_tok = T_32//2, stride 2 — 16 Hz tokens. The stride
        places each 16 Hz token at its TRUE 32 Hz-lattice position (token p → lattice 2p), so the
        L1 RoPE geometry is physically identical to the 32 Hz arm (``base_time=64`` stays matched)
        — only the temporal sampling coarsens. Masks/grid index tokens by ``bandpos`` (0..n_tok−1);
        RoPE uses ``time_pos = bandpos·stride`` (build_r5nf_grid). Both the model's mask sampling
        and the objective's grid derive n_tok from HERE so they never disagree on the token count
        (the 07-22 plan-cache clock bug invariant)."""
    L = 2 * int(t_32hz)  # 64 Hz frame count
    if decimate not in (2, 4):
        raise ValueError(f"nf decimate must be 2 or 4, got {decimate}")
    if L % decimate != 0:
        raise ValueError(f"64 Hz frame count {L} (2·T_32) not divisible by decimate {decimate}")
    return L // decimate, decimate // 2


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


# ── r5 EarlyFusionStem (Chang 2-stream single-rate; 2026-07-21) ──────────────────
# TWO streams early-fused to one token per (contact, 32 Hz slot): HGA 4-bin |STFT|
# magnitude (64–160 Hz) ⊕ LFS 1-ch raw 1–30 Hz voltage, BOTH at 64 Hz frames (hop=32,
# frame-for-frame aligned). Concat → 5 channels → Conv1d(5→d, k3, s1) → GELU →
# Conv1d(d→d, k3, s2) → GELU decimates 64 → 32 Hz. Single-rate (stride-1 lattice), NO
# band embed, NO SLOW/MID, NO multi-rate. Memo: project-r5-config-canonical-2026-07-21.
EARLY_FUSION_BINS: tuple[int, int] = (4, 1)  # (HGA bins, LFS channels), concat HGA⊕LFS = 5
EARLY_FUSION_CHANNELS: int = sum(EARLY_FUSION_BINS)  # 5 fused channels per 64 Hz frame
EARLY_FUSION_DECIMATE: int = 2  # stem stride-2 (64→32 Hz) ⇒ 2 raw 64 Hz frames per token

# ── v3r5nf NoFusionStem (r5-fused's fully-separated sibling; 2026-07-22) ──────────
# The r5 contract WITHOUT the fusion: the SAME two 64 Hz streams (HGA 4-bin |STFT|, LFS
# 1-ch raw voltage), IDENTICAL cache content — but each conv-pooled by its OWN stem into
# its OWN 32 Hz token stream (no 5-channel concat), tagged by a 2-way band embed. Two token
# streams per (contact, 32 Hz slot). Memo: project-r5-config-canonical-2026-07-21.
NOFUSION_BINS: tuple[int, int] = (4, 1)  # (HGA bins, LFS channels), SEPARATE streams
NOFUSION_DECIMATE: int = 2  # each stem stride-2 (64→32 Hz) ⇒ 2 raw 64 Hz frames per token


class EarlyFusionStem(nn.Module):
    """Chang 2-stream early-fusion stem (r5) — HGA ⊕ LFS → 32 Hz tokens.

    Receives TWO streams, both at 64 Hz frames (hop=32) on the same clip clock:
      • HGA: ``(..., 4, L)`` — |STFT| magnitude, 4 bins 64–160 Hz, per-(elec,bin) robust-z.
      • LFS: ``(..., 1, L)`` — raw 1–30 Hz voltage, per-elec robust-z.
    with ``L = 2·T`` (T = the 32 Hz token count). Stacks them to 5 channels per (elec,
    64 Hz frame) and conv-pools 64 → 32 Hz:

      ``Conv1d(5 → d, k3, s1, pad1) → GELU → Conv1d(d → d, k3, s2, pad1) → GELU``

    one stride-2 decimate (64 → 32). Emits a SINGLE-rate token stream + stride-1 lattice
    positions ``arange(T)``, returned as 1-element tuples so the r5 path reuses the same
    pack/mask/pe machinery as the multi-rate stems (a degenerate one-band case).

    k=3 BOTH layers (Ben 2026-07-21, LOCKED — NOT k=5, NOT blurpool; Whisper ``Conv1d(k3,
    s2)`` precedent, accepts the small 1/f residual β alias). ACCEPT THE BLEED: each 32 Hz
    token's receptive field is 5 input frames (~78 ms @64 Hz), so adjacent tokens share 3
    of 5 frames — that overlap is scored anyway (``in_loss = masked``, no margin gate).
    Deliberately NO band embed, NO per-band norm: single early-fused token, single axial
    time-RoPE.
    """

    def __init__(
        self, d_model: int = 256, *, bins: tuple[int, int] = EARLY_FUSION_BINS
    ) -> None:
        super().__init__()
        self.n_hga, self.n_lfs = int(bins[0]), int(bins[1])
        c_in = self.n_hga + self.n_lfs
        # 5→d band-mix + short temporal filter (k3,s1); d→d decimate 64→32 (k3,s2). GELU
        # after each. No Linear ⇒ convs keep PyTorch default init (matches FineHgaStem pool).
        self.fuse = nn.Sequential(
            nn.Conv1d(c_in, d_model, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.apply(init_transformer_weights)  # no Linear here ⇒ no-op on convs (default init)

    def forward(
        self, band_inputs: Sequence[Tensor]
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Streams ``[HGA(...,4,L), LFS(...,1,L)]`` (L = 2T, 64 Hz) → ((tokens ``(...,T,d)``,),
        (positions ``(T,)`` long,)). Single-rate: one token stream, lattice stride 1."""
        if len(band_inputs) != 2:
            raise ValueError(
                f"EarlyFusionStem expects 2 streams (HGA, LFS), got {len(band_inputs)}"
            )
        hga, lfs = band_inputs
        if hga.shape[-2] != self.n_hga:
            raise ValueError(f"HGA has {hga.shape[-2]} bins, expected {self.n_hga}")
        if lfs.shape[-2] != self.n_lfs:
            raise ValueError(f"LFS has {lfs.shape[-2]} channels, expected {self.n_lfs}")
        if hga.shape[-1] != lfs.shape[-1]:
            raise ValueError(
                f"HGA/LFS 64 Hz frame counts disagree: {hga.shape[-1]} vs {lfs.shape[-1]}"
            )
        length = hga.shape[-1]
        if length % 2 != 0:
            raise ValueError(
                f"64 Hz frame count {length} must be even (= 2× the 32 Hz token count)"
            )
        x = torch.cat((hga, lfs), dim=-2)  # (..., 5, L) — HGA bins 0..3, LFS ch 4
        *lead, c, L = x.shape
        x = x.reshape(-1, c, L)  # (prod_lead, 5, L)
        x = self.fuse(x)  # (prod_lead, d, T)
        x = x.transpose(-1, -2)  # (prod_lead, T, d)
        tok = x.reshape(*lead, x.shape[-2], x.shape[-1])  # (..., T, d)
        t = tok.shape[-2]
        pos = torch.arange(t, device=hga.device, dtype=torch.long)  # single-rate, stride 1
        return (tok,), (pos,)


class NoFusionStem(nn.Module):
    """v3r5nf no-fusion stem — EarlyFusionStem's fully-separated sibling (2026-07-22).

    Consumes the SAME two 64 Hz streams as ``EarlyFusionStem`` (byte-identical caches, zero
    re-bake) but does NOT concat them: each stream gets its OWN conv-pool stem into its OWN
    32 Hz token stream, and a 2-way ``band_type_emb`` (like ``PerBandStem``/``FineHgaStem``)
    tags which stream a token came from. The ONLY change vs r5-fused is fusion → full
    separation:

      • HGA: ``(..., 4, L)`` — |STFT| magnitude, 4 bins 64–160 Hz, per-(elec,bin) robust-z.
      • LFS: ``(..., 1, L)`` — raw 1–30 Hz voltage, per-elec robust-z.
    with ``L = 2·T``. Each stem is ``Conv1d(c_in→d, k3, s1, pad1) → GELU → Conv1d(d→d, k3, s2,
    pad1) → GELU`` (mirrors ``EarlyFusionStem.fuse`` exactly, per-stream ``c_in`` = 4 / 1); the
    single stride-2 decimates 64 → 32 Hz. Emits TWO token streams (HGA d, LFS d) + stride-1
    lattice positions ``arange(T)`` shared by both (the two streams sit at the same (elec,
    frame) RoPE position; ``band_type_emb`` is the only distinguisher, exactly like the r4
    multi-band path with both bands at stride 1).

    Deliberately NO per-band norm — stream identity rides the separate stems + the additive
    band embed. band_type_emb init order matches FineHgaStem: ``apply(init_transformer_weights)``
    first (no-op on the convs), then ``trunc_normal_`` the band embed.
    """

    def __init__(
        self,
        d_model: int = 256,
        *,
        bins: tuple[int, int] = NOFUSION_BINS,
        decimate: int = NOFUSION_DECIMATE,
        band_emb_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.n_hga, self.n_lfs = int(bins[0]), int(bins[1])
        if decimate not in (2, 4):
            raise ValueError(f"NoFusionStem decimate must be 2 or 4, got {decimate}")
        self.decimate = int(decimate)
        s1 = self.decimate // 2  # net stem stride = s1·2 = decimate (64 Hz → 64/decimate Hz)

        def _stem(c_in: int, _s1: int = s1) -> nn.Sequential:
            # identical shape to EarlyFusionStem.fuse, per-stream in_channels. The FIRST conv's
            # stride carries the fast-frontend decimation (v3r5nffast OFAT): s1=1 → net 2 = 32 Hz
            # tokens (v3r5nf, byte-identical); s1=2 → net 4 = 16 Hz tokens. Second conv stays s2.
            return nn.Sequential(
                nn.Conv1d(c_in, d_model, kernel_size=3, stride=_s1, padding=1),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            )

        self.hga_stem = _stem(self.n_hga)
        self.lfs_stem = _stem(self.n_lfs)
        self.band_type_emb = nn.Parameter(torch.empty(2, d_model))
        self.apply(init_transformer_weights)  # no Linear here ⇒ no-op on convs (default init)
        nn.init.trunc_normal_(self.band_type_emb, std=band_emb_std)

    def _pool(self, x: Tensor, stem: nn.Sequential) -> Tensor:
        """(..., C, L) → (..., T, d) via a stream's stride-2 conv-pool (64→32 Hz)."""
        *lead, c, L = x.shape
        x = x.reshape(-1, c, L)  # (prod_lead, C, L)
        x = stem(x)  # (prod_lead, d, T)
        x = x.transpose(-1, -2)  # (prod_lead, T, d)
        return x.reshape(*lead, x.shape[-2], x.shape[-1])  # (..., T, d)

    def forward(
        self, band_inputs: Sequence[Tensor]
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Streams ``[HGA(...,4,L), LFS(...,1,L)]`` (L = 2T, 64 Hz) → ((hga_tok ``(...,T,d)``,
        lfs_tok ``(...,T,d)``), (pos ``(T,)``, pos ``(T,)``)). Both streams stride-1, SAME
        lattice positions; band_type_emb[0]/[1] the only stream distinguisher."""
        if len(band_inputs) != 2:
            raise ValueError(
                f"NoFusionStem expects 2 streams (HGA, LFS), got {len(band_inputs)}"
            )
        hga, lfs = band_inputs
        if hga.shape[-2] != self.n_hga:
            raise ValueError(f"HGA has {hga.shape[-2]} bins, expected {self.n_hga}")
        if lfs.shape[-2] != self.n_lfs:
            raise ValueError(f"LFS has {lfs.shape[-2]} channels, expected {self.n_lfs}")
        if hga.shape[-1] != lfs.shape[-1]:
            raise ValueError(
                f"HGA/LFS 64 Hz frame counts disagree: {hga.shape[-1]} vs {lfs.shape[-1]}"
            )
        length = hga.shape[-1]
        if length % self.decimate != 0:
            raise ValueError(
                f"64 Hz frame count {length} must be divisible by decimate {self.decimate} "
                f"(= the tokens/stream count)"
            )
        hga_tok = self._pool(hga, self.hga_stem) + self.band_type_emb[0]  # (..., T, d)
        lfs_tok = self._pool(lfs, self.lfs_stem) + self.band_type_emb[1]  # (..., T, d)
        t = hga_tok.shape[-2]
        pos = torch.arange(t, device=hga.device, dtype=torch.long)  # stride 1, shared
        return (hga_tok, lfs_tok), (pos, pos)
