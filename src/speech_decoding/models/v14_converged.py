"""v14 converged architecture (``memory/project_v14_converged_arch_2026_06_17``).

A clean build of the 3-stage converged v14 architecture, composing the reusable
primitives from :mod:`v14_encoder` (``_PatchStem``, ``JepaPredictor``, RoPE
rotation) rather than bending the live B37 mean-pool path. The live encoder is
left untouched; this module is the new flag-gated forward path, built
component-by-component with colocated TDD.

Stage 1 here: the **3STFT per-electrode frontend tokenizer** (FE spec
``reports/fe_3stft_2of2of2_spec_2026_06_17.md`` §2/§5). Per electrode, per 1 s
clip, three STFTs at different resolutions are patch-embedded into **38 tokens**
(slow 6 + beta 16 + HG 16). Electrodes ride in the batch dim — the tokenizer has
**no cross-electrode pathway** (Stage-1 isolation, load-bearing: ~10 co-shaft
near-duplicate electrodes per parcel would give M2 a trivial copy shortcut).

The 2/2/2 ladder geometry is LOCKED (FE spec §2). All hops are ``N/2`` (50%
overlap); all ``tk=2`` (FE §6 window-tiling: ``tk·hop == N``). The three time-
patch strides (``tk·hop``) are slow 1024 / beta 256 / HG 128 samples = **8:2:1**,
so they share one RoPE clock in units of the HG stride (128 samples = 62.5 ms,
the FE §1 reference grid). NB the freq-pos memo's "8/4/1" multipliers are stale
(the 2/*4*/2 predecessor); the locked 2/2/2 is 8/2/1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from speech_decoding.models.v14_encoder import (
    _JointTokenBlock,
    _PatchStem,
    _rope_freqs,
    _sincos_1d,
)

# Reference sample rate (FE spec §1: 1 s clip @ 2048 Hz).
FS_HZ: int = 2048


@dataclass(frozen=True)
class BandSpec:
    """One band of the locked 3STFT 2/2/2 ladder (FE spec §2).

    ``n_freq_bins`` / ``n_time_frames`` are the in-band ``torch.stft`` output
    dims (center=True) the cache stores per band; ``kernel_freq`` = ``fk`` and
    ``kernel_time`` = ``tk`` set the patch grid; ``in_channels`` = 2 for the
    phase-bearing slow band (Cartesian Re/Im), 1 for the magnitude bands.
    """

    name: str
    nperseg: int          # N
    hop: int              # N // 2
    n_freq_bins: int      # in-band bins (slow 6, beta 6, HG 9)
    n_time_frames: int    # stft frames over a 2048-sample clip (5 / 17 / 33)
    kernel_freq: int      # fk
    kernel_time: int      # tk (== 2 for all locked bands)
    in_channels: int      # 2 (slow Re,Im) | 1 (mag)

    @property
    def n_freq_patches(self) -> int:
        return (self.n_freq_bins - self.kernel_freq) // self.kernel_freq + 1

    @property
    def n_time_patches(self) -> int:
        return (self.n_time_frames - self.kernel_time) // self.kernel_time + 1

    @property
    def n_tokens(self) -> int:
        return self.n_freq_patches * self.n_time_patches

    @property
    def time_patch_stride_samples(self) -> int:
        # Non-overlapping patches advance by tk·hop samples (= N at hop=N/2).
        return self.kernel_time * self.hop

    @property
    def patch_input_dim(self) -> int:
        # The lossless linear lift's fan-in (FE §5): fk·tk·channels.
        return self.kernel_freq * self.kernel_time * self.in_channels


# The locked 2/2/2 ladder (FE spec §2, every number verified by verify_3stft.py).
SLOW = BandSpec("slow", 1024, 512, 6, 5, kernel_freq=2, kernel_time=2, in_channels=2)
BETA = BandSpec("beta", 256, 128, 6, 17, kernel_freq=3, kernel_time=2, in_channels=1)
HG = BandSpec("hg", 128, 64, 9, 33, kernel_freq=9, kernel_time=2, in_channels=1)
BANDS: tuple[BandSpec, ...] = (SLOW, BETA, HG)

N_TOKENS: int = sum(b.n_tokens for b in BANDS)            # 6 + 16 + 16 = 38
N_FREQ_PATCHES: int = sum(b.n_freq_patches for b in BANDS)  # 3 + 2 + 1 = 6


@dataclass(frozen=True)
class M2MaskConfig:
    """M2 within-electrode masking config (FE spec §8, locked starting point).

    slow is EXEMPT (never masked — always-visible forward-PAC context). beta is
    one freq-tubed time-mask (both freq-patches co-masked across ``beta_span``
    time columns = 50% of beta). HG is a wav2vec2 span mask: ``round(hg_start_rate
    · T_p)`` distinct span starts × ``hg_span`` forward, overlaps allowed. The
    §8.7 sisters tune ``hg_start_rate`` (coverage dial) and ``beta_span``
    (redundancy knob, never < 3 per the §8.3 bleed floor).
    """

    hg_start_rate: float = 0.15   # §8.5: 15–20% of HG time positions are starts
    hg_span: int = 3              # §8.4: HG span width 3 (event-scale, bleed-floor)
    beta_span: int = 4            # §8.4: beta tube width 4 (sustained-burst scale)


def sample_m2_mask(
    generator: torch.Generator,
    bands: tuple[BandSpec, ...] = BANDS,
    cfg: M2MaskConfig = M2MaskConfig(),
) -> Tensor:
    """Sample ONE un-tubed electrode's M2 token mask (FE §8). Returns a bool
    ``(N_TOKENS,)`` in tokenizer order (band slow→beta→HG, freq-major time-minor);
    ``True`` = held out (an M2 target). slow stays all-``False`` (exempt)."""
    parts: list[Tensor] = []
    for b in bands:
        m = torch.zeros(b.n_freq_patches, b.n_time_patches, dtype=torch.bool)
        if b.name == "slow":
            pass  # EXEMPT — never an M2 target
        elif b.name == "beta":
            # One freq-tube: 1 random start, width beta_span, BOTH freq-patches.
            hi = b.n_time_patches - cfg.beta_span + 1
            start = int(torch.randint(0, max(hi, 1), (1,), generator=generator).item())
            m[:, start : start + cfg.beta_span] = True
        elif b.name == "hg":
            # wav2vec2 span mask: distinct starts in [0, T_p−span], span forward,
            # overlaps allowed (merge into longer spans).
            allowed = b.n_time_patches - cfg.hg_span + 1
            n_starts = min(round(cfg.hg_start_rate * b.n_time_patches), allowed)
            if n_starts > 0:
                starts = torch.randperm(allowed, generator=generator)[:n_starts]
                for s in starts.tolist():
                    m[:, s : s + cfg.hg_span] = True
        else:
            raise ValueError(f"unknown band {b.name!r} for M2 masking")
        parts.append(m.reshape(-1))
    return torch.cat(parts)


@dataclass(frozen=True)
class M4MaskConfig:
    """M4 whole-parcel tube config (converged memo). ``parcel_mask_ratio`` carries
    B36's 0.20 anchor (memo-open: tune after the first probe)."""

    parcel_mask_ratio: float = 0.20


def sample_parcel_tube(
    present_parcels: Tensor,
    generator: torch.Generator,
    cfg: M4MaskConfig = M4MaskConfig(),
) -> Tensor:
    """Sample the tubed (whole-parcel-masked) parcels for one clip.

    ``present_parcels`` = the parcel ids present in this (ragged) montage. Returns
    a LongTensor of the tubed parcel ids (a subset). Whole-parcel masking removes
    ALL electrodes of each tubed parcel together (shortcut-resistant: co-shaft
    near-duplicates go together). Guards: ≥1 parcel tubed (M4 needs a target) and
    ≥1 parcel un-tubed (M4 needs visible context) — so a 1-parcel clip tubes
    nothing (M4 inert, no cross-region context to infer from)."""
    present = torch.as_tensor(present_parcels).reshape(-1).unique()
    n = int(present.numel())
    if n < 2:
        return present[:0]
    n_tube = round(cfg.parcel_mask_ratio * n)
    n_tube = max(1, min(n_tube, n - 1))
    idx = torch.randperm(n, generator=generator)[:n_tube]
    return present[idx]


def electrode_tube_mask(parcel_per_electrode: Tensor, tubed_parcels: Tensor) -> Tensor:
    """Whole-parcel tube application: ``True`` for every electrode whose parcel is
    tubed (all electrodes of a tubed parcel go together). ``(C,)`` bool over the
    ragged electrode axis; these electrodes are DROPPED from the student encoder."""
    pe = torch.as_tensor(parcel_per_electrode).reshape(-1)
    tubed = torch.as_tensor(tubed_parcels).reshape(-1)
    if tubed.numel() == 0:
        return torch.zeros(pe.numel(), dtype=torch.bool)
    return torch.isin(pe, tubed)


class ThreeBandTokenizer(nn.Module):
    """3STFT per-electrode frontend tokenizer (Stage 1, FE spec §5).

    Runs one ``_PatchStem`` per band over the per-electrode ``(F, T)`` plane and
    concatenates the three bands' tokens into the 38-token sequence. Electrodes
    ride in the batch dim — **isolated**, no cross-electrode mixing.

    Forward inputs (normalization applied upstream at load, FE §4):
      - ``slow``: ``(B, C, 2, 6, 5)`` Cartesian (Re, Im)
      - ``beta``: ``(B, C, 6, 17)`` ``|STFT|``
      - ``hg``:   ``(B, C, 9, 33)`` ``|STFT|``

    Forward output: ``tokens`` ``(B, C, 38, d_model)`` in band-then-(freq-major,
    time) order. Per-token metadata is precomputed (geometry-fixed) and exposed
    as registered buffers:
      - ``band_id`` ``(38,)`` ∈ {0=slow, 1=beta, 2=hg}
      - ``freq_global_id`` ``(38,)`` ∈ [0, 6) — index into a shared freq-pos table
      - ``time_slot`` ``(38,)`` — RoPE clock in HG-stride units (×8 slow / ×2 beta
        / ×1 HG); ``time_slot · 62.5 ms`` = physical time-in-clip.
    """

    def __init__(self, d_model: int, bands: tuple[BandSpec, ...] = BANDS) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.d_model = d_model
        self.bands = bands

        # Shared RoPE clock unit = the finest band's time-patch stride. All
        # strides must be integer multiples of it, else the bands cannot share
        # one clock (FE §7 — load-bearing; assert at construction).
        min_stride = min(b.time_patch_stride_samples for b in bands)
        self._slot_mult: list[int] = []
        for b in bands:
            mult, rem = divmod(b.time_patch_stride_samples, min_stride)
            if rem != 0:
                raise ValueError(
                    f"band {b.name!r} time-patch stride {b.time_patch_stride_samples} "
                    f"is not an integer multiple of the finest stride {min_stride}; "
                    f"the 3 bands cannot share one RoPE clock"
                )
            self._slot_mult.append(mult)

        self.stems = nn.ModuleList(
            [
                _PatchStem(
                    d_model,
                    kernel_freq=b.kernel_freq,
                    kernel_time=b.kernel_time,
                    in_channels=b.in_channels,
                )
                for b in bands
            ]
        )

        band_id, freq_global_id, time_slot = self._build_token_metadata()
        self.register_buffer("band_id", band_id, persistent=False)
        self.register_buffer("freq_global_id", freq_global_id, persistent=False)
        self.register_buffer("time_slot", time_slot, persistent=False)

    def _build_token_metadata(self) -> tuple[Tensor, Tensor, Tensor]:
        """Geometry-fixed per-token (band, global-freq-patch, RoPE-slot) ids.

        Token order within a band is the ``_PatchStem`` output flattened
        ``(F_p, T_p)`` row-major → freq-patch-major, time-minor; bands concatenate
        slow → beta → HG."""
        band_id: list[int] = []
        freq_global_id: list[int] = []
        time_slot: list[int] = []
        freq_base = 0
        for bi, (b, mult) in enumerate(zip(self.bands, self._slot_mult)):
            for fp in range(b.n_freq_patches):
                for tp in range(b.n_time_patches):
                    band_id.append(bi)
                    freq_global_id.append(freq_base + fp)
                    time_slot.append(mult * tp)
            freq_base += b.n_freq_patches
        return (
            torch.tensor(band_id, dtype=torch.long),
            torch.tensor(freq_global_id, dtype=torch.long),
            torch.tensor(time_slot, dtype=torch.long),
        )

    def forward(self, slow: Tensor, beta: Tensor, hg: Tensor) -> Tensor:
        """``(B,C,2,6,5) / (B,C,6,17) / (B,C,9,33)`` → tokens ``(B,C,38,d)``."""
        per_band = []
        for b, stem, x in zip(self.bands, self.stems, (slow, beta, hg)):
            # _PatchStem: (B,C,Cin,F,T) [2ch] or (B,C,F,T) [1ch] → (B,C,F_p,T_p,d).
            out = stem(x)                                   # (B, C, F_p, T_p, d)
            B, C, F_p, T_p, d = out.shape
            if F_p != b.n_freq_patches or T_p != b.n_time_patches:
                raise ValueError(
                    f"band {b.name!r} stem produced ({F_p},{T_p}) freq/time patches, "
                    f"expected ({b.n_freq_patches},{b.n_time_patches}); check the input "
                    f"({b.n_freq_bins},{b.n_time_frames}) bins/frames"
                )
            per_band.append(out.reshape(B, C, F_p * T_p, d))  # (B, C, n_tok_b, d)
        return torch.cat(per_band, dim=2)                     # (B, C, 38, d)


class FrontendEncoder(nn.Module):
    """Stage 1 (converged arch): per-electrode ISOLATED 3STFT frontend transformer.

    Tokenize → add the freq-tag → ``n_layers`` joint freq×time self-attention
    blocks (RoPE on the shared physical-time clock) → LayerNorm. Electrodes ride
    in the batch dim throughout: **no cross-electrode pathway** (Stage-1 isolation).

    The freq-tag defaults to a **1-D LEARNABLE** table (Ben 2026-06-18,
    [[project_v14_freq_pos_learned_decision_2026_06_18]]) indexed by the token's
    global freq-patch id; the ``"sinusoidal"`` sister is retained.

    **Multirate RoPE (the rigorous reuse of the 2STFT dual-rate handling):** every
    token's rotation is gathered from one ``key_rope`` table at the token's
    ``time_slot`` (8/2/1 × time-patch-idx) — identical to ``_forward_dual_band``'s
    ``key_rope[:, slot, :]`` gather. Because the slots are physical-time indices on
    the shared HG-stride (62.5 ms) grid, tokens at the same real time across the 3
    bands receive the SAME rotation, and the RoPE attention between any two tokens
    depends only on their physical-time difference — band-agnostic (proven in the
    tests). Works precisely because the hops 512:128:64 are exact integer multiples
    (asserted in :class:`ThreeBandTokenizer`).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        *,
        freq_pos: str = "learned",
        bands: tuple[BandSpec, ...] = BANDS,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        head_dim = d_model // n_heads
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        self.tokenizer = ThreeBandTokenizer(d_model, bands)
        n_fp = sum(b.n_freq_patches for b in bands)

        # Freq-tag (additive, indexed by global freq-patch id). Learned by default.
        if freq_pos == "learned":
            self.freq_embed = nn.Parameter(torch.empty(n_fp, d_model))
            nn.init.trunc_normal_(self.freq_embed, std=0.02)
        elif freq_pos == "sinusoidal":
            self.register_buffer("freq_embed", _sincos_1d(n_fp, d_model), persistent=False)
        else:
            raise ValueError(f"freq_pos must be 'learned' or 'sinusoidal', got {freq_pos!r}")

        # One RoPE table over the shared clock; gathered per token at time_slot.
        n_slots = int(self.tokenizer.time_slot.max().item()) + 1
        self.register_buffer("key_rope", _rope_freqs(head_dim, n_slots), persistent=False)

        self.blocks = nn.ModuleList(
            [_JointTokenBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, slow: Tensor, beta: Tensor, hg: Tensor) -> Tensor:
        """``(B,C,2,6,5)/(B,C,6,17)/(B,C,9,33)`` → per-electrode frontend features
        ``(B, C, 38, d)`` (the teacher's full-input output; M2/M4 targets read it)."""
        tok = self.tokenizer(slow, beta, hg)                     # (B, C, 38, d)
        B, C, S, d = tok.shape
        tok = tok + self.freq_embed[self.tokenizer.freq_global_id]  # + (38, d)
        rope = self.key_rope[:, self.tokenizer.time_slot, :]     # (2, 38, head_dim)
        x = tok.reshape(B * C, S, d)
        for blk in self.blocks:
            x = blk(x, rope)
        return self.ln_out(x).reshape(B, C, S, d)
