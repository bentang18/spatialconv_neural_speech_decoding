"""v14 converged architecture (``memory/project_v14_converged_arch_2026_06_17``).

A clean build of the 3-stage converged v14 architecture, composing the reusable
primitives from :mod:`v14_encoder` (``_PatchStem``, ``JepaPredictor``, RoPE
rotation) rather than bending the live B37 mean-pool path. The live encoder is
left untouched; this module is the new flag-gated forward path, built
component-by-component with colocated TDD.

Stage 1 here: the **3STFT per-electrode frontend tokenizer** (FE spec
``reports/fe_3stft_2of2of2_spec_2026_06_17.md`` §2/§5). Per electrode, per 1 s
clip, three STFTs at different resolutions are patch-embedded into **30 tokens**
(slow 6 + beta 8 + HG 16). Electrodes ride in the batch dim — the tokenizer has
**no cross-electrode pathway** (Stage-1 isolation, load-bearing: ~10 co-shaft
near-duplicate electrodes per parcel would give M2 a trivial copy shortcut).

The ladder geometry is LOCKED. All hops are ``N/2`` (50% overlap). slow and HG
keep ``tk=2`` (FE §6 window-tiling: ``tk·hop == N``); **beta is ``tk=4``**
(2026-06-19, Ben): a deliberate 2× coarsening to 250 ms beta patches — halves
beta to 8 tokens AND lets a fixed-width span-mask hit ~50% of beta on the 5 s SSL
clip (the old tk=2 width-4 tube was sized for 1 s and masked only ~10% at 5 s).
HG is UNCHANGED (tk=2, 16 tokens). The three time-patch strides (``tk·hop``) are
slow 1024 / beta **512** / HG 128 samples = **8:4:1**, so they share one RoPE
clock in units of the HG stride (128 samples = 62.5 ms, the FE §1 reference grid).
"""

from __future__ import annotations

import contextlib
import copy
import os
import typing as tp
from collections import defaultdict
from dataclasses import dataclass, replace
from math import comb

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from speech_decoding.models.v14_encoder import (
    _JointTokenBlock,
    _MultiHeadCrossAttention,
    _PatchStem,
    _ragged_gather_idx,
    _rope_freqs,
    _sincos_1d,
)

# Reference sample rate (FE spec §1: 1 s clip @ 2048 Hz).
FS_HZ: int = 2048

# Canonical SDPA kernel priority for the masked cross-electrode latent (the
# large-L O(Nv²) all-pairs elephant). On sm90 (Hopper / GH200, aarch64,
# torch-2.10) flash declines ANY mask, so the auto-dispatcher falls to the sm80
# mem-efficient (CUTLASS) kernel for the masked+grad latent call even with cuDNN
# globally enabled. Forcing this priority with ``set_priority=True`` runs the
# Hopper-native sm90 cuDNN kernel instead (~2.7× the mem-efficient one on the
# latent's shapes; probe 2026-06-21). EFFICIENT is the graceful fallback for any
# call cuDNN declines — NOT math, which materialises the full (B,H,L,L) score
# matrix and OOMs at the latent's L. Kernel selection only: identical attention
# math, output-identical. The experiments module imports this for the global
# ``cudnn`` path; the latent encoder uses it for the scoped ``cudnn_latent``
# path (see ``LatentEncoder._force_cudnn``).
_CUDNN_SDPA_PRIORITY = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def cudnn_sdpa_context(enabled: bool) -> tp.ContextManager[None]:
    """Force the cuDNN-first SDPA priority when ``enabled`` and CUDA is live.

    Used to scope the cuDNN force to exactly the call(s) where its host-side
    ~19 ms/call plan-building is repaid by the GPU win (the large-L masked
    latent); small-L frontend / predictor / time-SA calls keep the cheaper
    default dispatcher. Backward inherits the forward's chosen kernel, so
    wrapping only the forward suffices. ``contextlib.nullcontext`` otherwise."""
    if enabled and torch.cuda.is_available():
        return sdpa_kernel(_CUDNN_SDPA_PRIORITY, set_priority=True)
    return contextlib.nullcontext()


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


# The locked ladder (FE spec §2, every number verified by verify_3stft.py).
# beta kernel_time=4 (2026-06-19): 2× coarser time patch (250 ms) halves beta's
# token count at every clip length and re-grounds its 50% mask to the real grid.
# HG stays kernel_time=2 (62.5 ms — phonetic resolution preserved, Ben's call).
SLOW = BandSpec("slow", 1024, 512, 6, 5, kernel_freq=2, kernel_time=2, in_channels=2)
BETA = BandSpec("beta", 256, 128, 6, 17, kernel_freq=3, kernel_time=4, in_channels=1)
HG = BandSpec("hg", 128, 64, 9, 33, kernel_freq=9, kernel_time=2, in_channels=1)
BANDS: tuple[BandSpec, ...] = (SLOW, BETA, HG)

N_TOKENS: int = sum(b.n_tokens for b in BANDS)            # 6 + 8 + 16 = 30
N_FREQ_PATCHES: int = sum(b.n_freq_patches for b in BANDS)  # 3 + 2 + 1 = 6


def bands_for_clip_len(
    clip_len_s: float, fs: int = FS_HZ, base: tuple[BandSpec, ...] = BANDS,
) -> tuple[BandSpec, ...]:
    """The locked 2/2/2 ladder retimed for a ``clip_len_s``-second clip.

    SSL pretraining runs on **5 s** clips; only Phase-4 eval uses the 1 s clip the
    `BANDS` constants encode (``n_time_frames`` 5/17/33). Every BandSpec field is
    clip-length-INVARIANT except ``n_time_frames`` — the per-band ``torch.stft``
    frame count, which for ``center=True`` is exactly ``1 + n_samples // hop``
    (verified against ``torch.stft`` at 1 s and 5 s). Only the TIME axis grows;
    the freq grid, kernels, and channels are untouched, so token order and the
    RoPE clock multipliers are preserved — just more time-patches per band.

    5 s ⇒ slow 21 / beta 81 / HG 161 frames → 10 / 20 / 80 time-patches (beta tk=4)
    → 30 + 40 + 80 = **150 tokens** (vs 30 at 1 s)."""
    n_samples = int(round(clip_len_s * fs))
    out: list[BandSpec] = []
    for b in base:
        n_frames = 1 + n_samples // b.hop
        if n_frames < b.kernel_time:
            raise ValueError(
                f"clip_len_s={clip_len_s}s gives band {b.name!r} only {n_frames} "
                f"stft frames (< kernel_time {b.kernel_time}); clip too short"
            )
        out.append(replace(b, n_time_frames=n_frames))
    return tuple(out)


def band_slot_mults(bands: tuple[BandSpec, ...] = BANDS) -> list[int]:
    """Per-band RoPE-clock multiplier = time-patch stride ÷ finest stride.

    The shared clock unit is the finest band's time-patch stride (HG, 128
    samples = 62.5 ms). Every band's stride must be an integer multiple of it,
    else the 3 bands cannot share one clock (FE §7, load-bearing). Raises if not.
    Locked 2/2/2 ⇒ slow 1024 / beta 256 / HG 128 = **8 : 2 : 1**."""
    min_stride = min(b.time_patch_stride_samples for b in bands)
    mults: list[int] = []
    for b in bands:
        mult, rem = divmod(b.time_patch_stride_samples, min_stride)
        if rem != 0:
            raise ValueError(
                f"band {b.name!r} time-patch stride {b.time_patch_stride_samples} "
                f"is not an integer multiple of the finest stride {min_stride}; "
                f"the 3 bands cannot share one RoPE clock"
            )
        mults.append(mult)
    return mults


def token_metadata(bands: tuple[BandSpec, ...] = BANDS) -> tuple[Tensor, Tensor, Tensor]:
    """Geometry-fixed per-token ``(band_id, freq_global_id, time_slot)`` longs.

    Single source for the tokenizer AND the latent. Token order: bands concat
    slow→beta→HG, each flattened ``(F_p, T_p)`` row-major (freq-patch-major,
    time-minor). ``time_slot = mult · time_patch_idx`` puts all bands on the
    shared HG-stride clock; ``freq_global_id ∈ [0, ΣF_p)`` indexes a shared
    freq-pos table."""
    mults = band_slot_mults(bands)
    band_id: list[int] = []
    freq_global_id: list[int] = []
    time_slot: list[int] = []
    freq_base = 0
    for bi, (b, mult) in enumerate(zip(bands, mults)):
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


@dataclass(frozen=True)
class M2MaskConfig:
    """M2 within-electrode masking config (FE spec §8, locked starting point).

    slow is EXEMPT (never masked — always-visible forward-PAC context). beta and
    HG both use the wav2vec2 span mask: ``round(start_rate · T_p)`` distinct span
    starts × ``span`` forward, overlaps allowed (merge into longer runs), beta's
    span freq-tubed (both freq-patches co-masked). beta defaults (rate 0.30,
    span 2) realize ~50% of beta on the coarse 250 ms grid (2026-06-19 fix: the
    old single width-4 tube was sized for the 1 s clip and only masked ~10% at
    5 s). The §8.7 sisters tune the per-band start_rate (coverage dial).
    """

    hg_start_rate: float = 0.20   # §8.5: 20% of HG time positions are starts (Ben 2026-06-18)
    hg_span: int = 3              # §8.4: HG span width 3 (event-scale, bleed-floor)
    beta_start_rate: float = 0.30  # 2026-06-19: 30% of beta time positions are starts (~50%)
    beta_span: int = 2            # 2026-06-19: beta span width 2 on the coarse 250 ms grid


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
            # wav2vec2 freq-tubed span mask: round(beta_start_rate·T_p) distinct
            # starts × beta_span forward, overlaps allowed, BOTH freq-patches.
            allowed = b.n_time_patches - cfg.beta_span + 1
            n_starts = min(round(cfg.beta_start_rate * b.n_time_patches), allowed)
            if n_starts > 0:
                starts = torch.randperm(allowed, generator=generator)[:n_starts]
                for s in starts.tolist():
                    m[:, s : s + cfg.beta_span] = True
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


def parcel_electrode_mean(
    feats: Tensor, parcel_per_electrode: Tensor, target_parcels: Tensor
) -> Tensor:
    """M4 teacher TARGET: per-parcel electrode-MEAN of the teacher frontend grid.

    For one clip, given the EMA-teacher's **frontend** features ``feats``
    ``(C, 38, d)`` (post-frontend, NOT post-latent — the teacher halts after the
    frontend, converged memo Q1) and each electrode's parcel id, return the
    electrode-MEAN over each target (tubed) parcel's electrodes, **keeping the
    full 38-token freq-time grid** → ``(P, 38, d)``. **std dropped** — the mean
    is the DeepSets-canonical permutation-invariant aggregation ``ρ(mean φ(eᵢ))``
    and the only subject-invariant (montage-noise ~1/√n) region descriptor.

    Caller passes the teacher features already **detached** (stop-grad target).
    Mean over the electrode axis makes the ragged within-parcel electrode count
    irrelevant; the output is rectangular ``(P, 38, d)``."""
    if feats.dim() != 3:
        raise ValueError(f"feats must be (C, 38, d); got {tuple(feats.shape)}")
    C, S, d = feats.shape
    pe = parcel_per_electrode.reshape(-1)
    tp = target_parcels.reshape(-1)
    membership = (pe[None, :] == tp[:, None]).to(feats.dtype)     # (P, C) one-hot
    counts = membership.sum(dim=1, keepdim=True).clamp_min(1.0)   # (P, 1)
    summed = membership @ feats.reshape(C, S * d)                 # (P, S·d)
    return (summed / counts).reshape(-1, S, d)                    # (P, 38, d)


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
        # one clock (FE §7 — load-bearing; band_slot_mults asserts this).
        band_slot_mults(bands)

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

        band_id, freq_global_id, time_slot = token_metadata(bands)
        self.register_buffer("band_id", band_id, persistent=False)
        self.register_buffer("freq_global_id", freq_global_id, persistent=False)
        self.register_buffer("time_slot", time_slot, persistent=False)

        # Per-band stem-output magnitude, stashed each forward for the monitor
        # (Ben 2026-06-18): with three SEPARATE stems, one running 10× hotter would
        # silently dominate the additive latent. Mean per-token L2 norm of each
        # band's raw stem output (pre freq-tag, pre blocks); a plain detached dict,
        # not state. {band_name: scalar}; populated by ``forward``.
        self.last_band_token_norm: dict[str, Tensor] = {}

    def forward(self, slow: Tensor, beta: Tensor, hg: Tensor) -> Tensor:
        """``(B,C,2,6,5) / (B,C,6,17) / (B,C,9,33)`` → tokens ``(B,C,38,d)``."""
        per_band = []
        band_norms: dict[str, Tensor] = {}
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
            # Mean per-token L2 norm of this band's stem output — the magnitude that
            # would dominate the latent if one stem runs hot. Detached (monitor only).
            band_norms[b.name] = out.detach().norm(dim=-1).mean()
            per_band.append(out.reshape(B, C, F_p * T_p, d))  # (B, C, n_tok_b, d)
        self.last_band_token_norm = band_norms
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

    def forward(
        self,
        slow: Tensor,
        beta: Tensor,
        hg: Tensor,
        *,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        """``(B,C,2,6,5)/(B,C,6,17)/(B,C,9,33)`` → per-electrode frontend features
        ``(B, C, 38, d)``.

        ``key_mask``: optional ``(B, C, 38)`` bool, ``True`` = a token that may be
        attended (the paradigm-B M2 visibility mask for the STUDENT; the teacher
        passes ``None`` = full input). Masked tokens never serve as keys, so the
        visible tokens' features are leakage-free — identical to physically
        dropping the masked tokens, but batchable."""
        tok = self.tokenizer(slow, beta, hg)                     # (B, C, 38, d)
        B, C, S, d = tok.shape
        tok = tok + self.freq_embed[self.tokenizer.freq_global_id]  # + (38, d)
        rope = self.key_rope[:, self.tokenizer.time_slot, :]     # (2, 38, head_dim)
        x = tok.reshape(B * C, S, d)
        km = None if key_mask is None else key_mask.reshape(B * C, S)
        for blk in self.blocks:
            x = blk(x, rope, km)
        return self.ln_out(x).reshape(B, C, S, d)

    def forward_ragged(
        self,
        slow: Tensor,
        beta: Tensor,
        hg: Tensor,
        keep_elec: Tensor,
        visible: Tensor,
        *,
        fixed_r: int | None = None,
        fixed_lk: int | None = None,
    ) -> Tensor:
        """Ragged frontend — **physically gather**, never process a dense set.

        ``fixed_r`` / ``fixed_lk`` (V-JEPA-2 static path, step B): CPU-known kept-
        electrode count (``B·c_real`` teacher / ``B·n_vis`` student) and kept-token
        count (``S`` teacher / ``s_vis`` student). When given, BOTH the electrode-
        axis selection and the token-axis gather slice to fixed lengths — no
        ``nonzero``/``.item()`` GPU sync, static shapes — and since every selected
        row is then genuinely real (``fixed_r == keep.sum()``, ``fixed_lk ==``
        per-row visible count), the blocks run **maskless** and the scatter-back is
        a plain ``scatter_``. ``None`` = legacy ``nonzero`` + key-mask (bit-identical).


        Ben's directive (2026-06-18): "Always gather only visible/un-tubed
        tokens. NEVER OVER DENSE NEVER." This is the production path; the dense
        ``forward(key_mask=...)`` above survives only as the equivalence oracle
        (a masked key contributes 0 to the softmax and masked queries are never
        read, so the two are output-identical on the kept tokens — that identity
        is the TDD veracity test).

        ``keep_elec``: ``(B, C)`` bool, ``True`` = an electrode to encode (real
        AND, for the student, un-tubed). Dropped electrodes (padded or tubed) are
        never tokenized through the blocks. ``visible``: ``(B, C, 38)`` bool,
        ``True`` = a token to keep (the student's un-M2-masked tokens; the
        teacher passes all-``True``). Returns ``(B, C, 38, d)`` with kept-
        electrode visible positions filled and everything else exactly zero.

        Two-level ragged gather: electrodes are isolated (no cross-electrode
        pathway in the frontend), so kept electrodes flatten into one batch with
        **no electrode-axis padding**; within each kept electrode the visible
        tokens pad to the per-batch-max count ``Lk`` (the ``_ragged_gather_idx``
        2STFT primitive). Every kept electrode keeps ≥1 token (slow is M2-exempt
        for the student; the teacher keeps all 38), so no row is all-pad → no
        empty-softmax NaN.
        """
        tok = self.tokenizer(slow, beta, hg)                     # (B, C, 38, d)
        B, C, S, d = tok.shape
        tok = tok + self.freq_embed[self.tokenizer.freq_global_id]
        elec = B * C
        out = tok.new_zeros(elec, S, d)
        keep_flat = keep_elec.reshape(elec)
        static = fixed_r is not None
        if not static and not bool(keep_flat.any()):
            return out.reshape(B, C, S, d)

        if static:                                               # fixed kept count
            e_sort, _, R = _ragged_gather_idx(keep_flat.unsqueeze(0), fixed_k=fixed_r)
            e_idx = e_sort[0]                                     # (R,) kept electrodes
        else:
            e_idx = keep_flat.nonzero(as_tuple=False).squeeze(1)  # (R,) kept electrodes
            R = int(e_idx.numel())
        toks = tok.reshape(elec, S, d)[e_idx]                    # (R, 38, d)
        vis = visible.reshape(elec, S)[e_idx]                    # (R, 38) bool

        t_idx, t_real, Lk = _ragged_gather_idx(vis, fixed_k=fixed_lk)  # (R, Lk)
        xv = torch.gather(toks, 1, t_idx.unsqueeze(-1).expand(R, Lk, d))  # (R, Lk, d)
        slot_v = self.tokenizer.time_slot[t_idx]                 # (R, Lk)
        rope = self.key_rope[:, slot_v, :]                       # (2, R, Lk, head_dim)
        block_mask = None if static else t_real                  # static → maskless
        for blk in self.blocks:
            xv = blk(xv, rope, block_mask)
        xv = self.ln_out(xv)                                     # (R, Lk, d)

        if static:                       # both axes all-real ⇒ sync-free scatter_
            e_full = e_idx.unsqueeze(1).expand(R, Lk)            # (R, Lk)
            flat_dst = (e_full * S + t_idx).reshape(R * Lk)      # into (elec·S,)
            out.reshape(elec * S, d).scatter_(
                0, flat_dst.unsqueeze(-1).expand(R * Lk, d), xv.reshape(R * Lk, d))
        else:
            # Scatter the genuinely-kept (real) outputs back to their (electrode,
            # token) home; padded slots are dropped.
            rows = torch.arange(R, device=tok.device).unsqueeze(1).expand(R, Lk)
            e_sel = e_idx[rows][t_real]                          # (M,)
            t_sel = t_idx[t_real]                                # (M,)
            out[e_sel, t_sel] = xv[t_real]
        return out.reshape(B, C, S, d)


class LatentEncoder(nn.Module):
    """Stage 2 (converged arch): FLAT token-level cross-electrode latent.

    The redesign's load-bearing piece — the stage that must finally earn its
    keep (the prior B37 run showed no post-latent uplift over post-frontend).

    Bridge (frontend→latent): add a learned **parcel-tag** per electrode,
    broadcast across its 38 tokens. This is the **only additive/spatial PE**
    (DK parcel ids, shared cross-subject vocab; **MNI BANNED**; **no distance
    bias, no same-parcel boost** — either would re-open the co-shaft copy
    shortcut and let M4 cheat). It is the PopT zero-per-subject bridge: subjects
    align through parcels, never coordinates.

    Then **global ALL-PAIRS self-attention** over the flattened ``C·38``
    electrode-token set — electrodes now ride the **sequence** dim (cross-
    electrode mixing, the opposite of the isolated frontend), **electrode-token
    granularity preserved** (no pooling bottleneck in the SSL gradient; inter-
    areal CFC gets a direct edge). RoPE keys off the **shared physical-time
    clock** (``time_slot``, the same 8:4:1 grid the frontend uses), so the three
    band grids align in the cross-electrode attention.

    Ragged: variable electrode count per subject. ``electrode_mask`` (``True`` =
    real electrode) becomes the per-token key mask so padded electrodes are
    never attended to — the valid electrodes' outputs are independent of any
    padding (no pad-to-max contamination).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_parcels: int,
        *,
        bands: tuple[BandSpec, ...] = BANDS,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        head_dim = d_model // n_heads
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")

        # The only additive spatial PE — learned DK parcel-tag (MNI banned).
        self.parcel_embed = nn.Embedding(n_parcels, d_model)
        nn.init.trunc_normal_(self.parcel_embed.weight, std=0.02)

        _, _, time_slot = token_metadata(bands)
        self.register_buffer("time_slot", time_slot, persistent=False)  # (38,)
        n_slots = int(time_slot.max().item()) + 1
        self.register_buffer("key_rope", _rope_freqs(head_dim, n_slots), persistent=False)

        self.blocks = nn.ModuleList(
            [_JointTokenBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_out = nn.LayerNorm(d_model)
        self.tokens_per_electrode = int(time_slot.numel())  # 38

        # Kernel selection only (output-identical). This cross-electrode latent
        # is the large-L all-pairs elephant where cuDNN's masked sm90 kernel is
        # ~2.7× the sm80 mem-efficient fallback. Scope the cuDNN force to THIS
        # encoder's block loops so the small-L frontend / predictor / time-SA
        # calls keep the cheaper default dispatcher (cuDNN adds ~19 ms/call host
        # plan-building not repaid at small L). Read once at construction from
        # the run's V14_SDPA_BACKEND: ``cudnn`` (global force, also wraps here —
        # nested same-priority context is a no-op) and ``cudnn_latent`` (scoped
        # to here only) both opt in; any other value leaves the default kernel.
        self._force_cudnn = (os.environ.get("V14_SDPA_BACKEND") or "").strip().lower() in (
            "cudnn",
            "cudnn_latent",
        )

    def forward(
        self,
        feats: Tensor,
        parcel_per_electrode: Tensor,
        *,
        electrode_mask: Tensor | None = None,
        token_mask: Tensor | None = None,
    ) -> Tensor:
        """Cross-electrode latent over the frontend features.

        ``feats``: ``(B, C, 38, d)`` per-electrode frontend features.
        ``parcel_per_electrode``: ``(B, C)`` long — each electrode's DK parcel id.
        ``electrode_mask``: ``(B, C)`` bool, ``True`` = real electrode (ragged
        padding); broadcast to all 38 tokens.
        ``token_mask``: ``(B, C, 38)`` bool, ``True`` = visible key — the finer
        per-token mask the STUDENT uses (drop tubed electrodes AND the un-tubed
        electrodes' M2-masked tokens, converged Q2). Takes precedence over
        ``electrode_mask``; ``None`` for both ⇒ full attention. Output:
        ``(B, C, 38, d)`` (electrode-token granularity preserved)."""
        B, C, S, d = feats.shape
        # Frontend→latent bridge: + parcel-tag (broadcast across the 38 tokens).
        x = feats + self.parcel_embed(parcel_per_electrode)[:, :, None, :]
        x = x.reshape(B, C * S, d)                                 # all-pairs token set
        rope = self.key_rope[:, self.time_slot.repeat(C), :]       # (2, C·38, head_dim)
        key_mask: Tensor | None = None
        if token_mask is not None:
            key_mask = token_mask.reshape(B, C * S)
        elif electrode_mask is not None:
            key_mask = electrode_mask[:, :, None].expand(B, C, S).reshape(B, C * S)
        with cudnn_sdpa_context(self._force_cudnn):
            for blk in self.blocks:
                x = blk(x, rope, key_mask)
        return self.ln_out(x).reshape(B, C, S, d)

    def forward_ragged(
        self,
        feats: Tensor,
        parcel_per_electrode: Tensor,
        token_mask: Tensor,
        *,
        fixed_nv: int | None = None,
    ) -> Tensor:
        """Ragged latent — **physically gather** the visible-un-tubed cross-
        electrode token set, never run all-pairs over a dense padded/masked set.

        ``fixed_nv`` (V-JEPA-2 static path, step B): the CPU-known visible SEE-set
        size. When given, the gather slices to exactly ``fixed_nv`` (no ``.item()``
        sync, static shape) and — since every row is then genuinely visible
        (``real`` all-True) — the all-pairs SA runs **maskless** (cuDNN-nomask) and
        the scatter-back is a plain ``scatter_`` (no boolean-index sync). ``None`` =
        legacy pad-to-batch-max + key-mask (bit-identical to before).

        ``token_mask``: ``(B, C, 38)`` bool, ``True`` = a token in the student's
        SEE-set (``latent_vis = real & un-tubed & un-M2-masked``). Per sample the
        True tokens are gathered across ALL electrodes into one set (pad to the
        per-batch-max visible count), the all-pairs SA runs over only that set,
        and the outputs scatter back to ``(B, C, 38, d)`` (zeros elsewhere).

        Output-identical to the dense ``forward(token_mask=...)`` on the visible
        tokens — a masked key contributes 0 to the cross-electrode softmax and
        masked tokens are never read — but no padded/masked token is ever built
        into the sequence. Each sample needs ≥1 visible token (guaranteed: a clip
        keeps ≥1 un-tubed parcel and slow tokens are M2-exempt)."""
        B, C, S, d = feats.shape
        x = feats + self.parcel_embed(parcel_per_electrode)[:, :, None, :]
        x = x.reshape(B, C * S, d)
        vis = token_mask.reshape(B, C * S)                        # (B, C·38)
        out = feats.new_zeros(B, C * S, d)

        idx, real, Nv = _ragged_gather_idx(vis, fixed_k=fixed_nv)  # (B, Nv)
        xv = torch.gather(x, 1, idx.unsqueeze(-1).expand(B, Nv, d))  # (B, Nv, d)
        slot_full = self.time_slot.repeat(C)                      # (C·38,)
        slot_v = slot_full[idx]                                   # (B, Nv)
        rope = self.key_rope[:, slot_v, :]                        # (2, B, Nv, head_dim)
        block_mask = None if fixed_nv is not None else real       # static → maskless
        with cudnn_sdpa_context(self._force_cudnn):
            for blk in self.blocks:
                xv = blk(xv, rope, block_mask)
        xv = self.ln_out(xv)                                      # (B, Nv, d)

        if fixed_nv is not None:                                  # real all-True
            out.scatter_(1, idx.unsqueeze(-1).expand(B, Nv, d), xv)
        else:
            rows = torch.arange(B, device=feats.device).unsqueeze(1).expand(B, Nv)
            out[rows[real], idx[real]] = xv[real]
        return out.reshape(B, C, S, d)


class M4Predictor(nn.Module):
    """Paradigm-B own-predictor for M4 (the latent parcel JEPA).

    I-JEPA-style joint predictor (per [[project_v14_predictor_design_rope_lock_2026_06_04]]):
    a shared learnable **mask token** + additive **parcel-tag** (which tubed
    parcel) + additive **freq-tag** (which of the 38 freq-time cells), positioned
    in time by **RoPE on the shared clock**. The query tokens are concatenated
    with the visible latent **context** and run through ``n_layers`` joint self-
    attention blocks, so each query attends to ALL visible un-tubed context (the
    M4 SEE-set) and the queries co-resolve; the query positions are read out and
    projected (raw, **no LN before L1**) to the teacher feature dim.

    SEE = all visible un-tubed latent tokens (``ctx``). PREDICT = each tubed
    parcel's 38-token electrode-mean teacher grid (``parcel_electrode_mean``).
    Independent params from the M2 predictor. freq-tag LEARNED by default
    (converged arch, 2026-06-18); ``"sinusoidal"`` sister retained.

    Ragged: ``key_mask`` (``True`` = real context token) drops padded context;
    only the requested ``query_parcels`` × 38 query tokens are ever built, so no
    out-of-scope compute (the locked ragged-predictor contract)."""

    def __init__(
        self,
        d_model: int,
        pred_dim: int,
        n_heads: int,
        n_layers: int,
        n_parcels: int,
        *,
        freq_pos: str = "learned",
        bands: tuple[BandSpec, ...] = BANDS,
    ) -> None:
        super().__init__()
        if pred_dim % n_heads != 0:
            raise ValueError(f"pred_dim={pred_dim} not divisible by n_heads={n_heads}")
        head_dim = pred_dim // n_heads
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")

        self.ctx_proj = nn.Linear(d_model, pred_dim, bias=False)  # context → pred space
        self.mask_token = nn.Parameter(torch.zeros(pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.parcel_embed = nn.Embedding(n_parcels, pred_dim)     # OWN parcel tag
        nn.init.trunc_normal_(self.parcel_embed.weight, std=0.02)

        _, freq_global_id, time_slot = token_metadata(bands)
        self.register_buffer("freq_global_id", freq_global_id, persistent=False)  # (38,)
        self.register_buffer("time_slot", time_slot, persistent=False)            # (38,)
        n_fp = sum(b.n_freq_patches for b in bands)
        if freq_pos == "learned":
            self.freq_embed = nn.Parameter(torch.empty(n_fp, pred_dim))
            nn.init.trunc_normal_(self.freq_embed, std=0.02)
        elif freq_pos == "sinusoidal":
            self.register_buffer("freq_embed", _sincos_1d(n_fp, pred_dim), persistent=False)
        else:
            raise ValueError(f"freq_pos must be 'learned' or 'sinusoidal', got {freq_pos!r}")

        n_slots = int(time_slot.max().item()) + 1
        self.register_buffer("key_rope", _rope_freqs(head_dim, n_slots), persistent=False)
        self.blocks = nn.ModuleList(
            [_JointTokenBlock(pred_dim, n_heads) for _ in range(n_layers)]
        )
        self.head = nn.Linear(pred_dim, d_model, bias=False)      # raw pred, NO LN
        self.tokens_per_electrode = int(time_slot.numel())        # 38

    def _build_queries(self, query_parcels: Tensor) -> Tensor:
        """``(B, P)`` tubed parcel ids → query tokens ``(B, P, 38, pred_dim)`` =
        mask-token + parcel-tag + freq-tag (RoPE-time added in attention)."""
        parcel = self.parcel_embed(query_parcels)[:, :, None, :]  # (B, P, 1, d)
        freq = self.freq_embed[self.freq_global_id][None, None]   # (1, 1, 38, d)
        return self.mask_token + parcel + freq                    # (B, P, 38, d)

    def forward(
        self,
        ctx: Tensor,
        ctx_slot: Tensor,
        query_parcels: Tensor,
        *,
        key_mask: Tensor | None = None,
        query_valid: Tensor | None = None,
    ) -> Tensor:
        """Predict each tubed parcel's teacher grid from the visible context.

        ``ctx``: ``(B, N, d_model)`` visible un-tubed latent tokens (M4 SEE-set).
        ``ctx_slot``: ``(N,)`` long — each context token's RoPE time-slot.
        ``query_parcels``: ``(B, P)`` long — the tubed parcel ids to predict.
        ``ctx_slot``: ``(N,)`` shared OR ``(B, N)`` per-row RoPE time-slots — the
        ragged gather hands a different visible-token subset per sample, so each
        row carries its own slots (the per-row RoPE path).
        ``key_mask``: ``(B, N)`` bool, ``True`` = real context token (ragged
        padding); ``None`` ⇒ all real.
        ``query_valid``: ``(B, P)`` bool, ``True`` = a REAL tubed parcel, ``False``
        = a ``P_FIXED`` pad slot. Pad-parcel query tokens are masked OUT of every
        block's keys (Ben 2026-06-22) so a real parcel's prediction is INVARIANT
        to how many pad slots were appended — otherwise the joint self-attention
        lets pad "parcel-0" queries leak into real predictions, making the answer
        depend on ``P_FIXED``. ``None`` ⇒ all queries real (legacy, pre-static).
        Output: ``(B, P, 38, d_model)`` predicted feature per (tubed parcel,
        freq-time cell), to be L1'd vs the stop-grad teacher target."""
        B, N, _ = ctx.shape
        P = query_parcels.shape[1]
        S = self.tokens_per_electrode
        q = self._build_queries(query_parcels).reshape(B, P * S, -1)  # (B, P·38, d)
        c = self.ctx_proj(ctx)                                        # (B, N, d)
        tokens = torch.cat([c, q], dim=1)                             # (B, N+P·38, d)

        q_slot = self.time_slot.repeat(P)                             # (P·38,)
        if ctx_slot.dim() == 1:                                       # shared (dense)
            slots = torch.cat([ctx_slot.reshape(-1), q_slot])         # (N+P·38,)
            rope = self.key_rope[:, slots, :]                         # (2, N+P·38, hd)
        else:                                                         # per-row (ragged)
            qs = q_slot[None, :].expand(B, P * S)                     # (B, P·38)
            slots = torch.cat([ctx_slot, qs], dim=1)                  # (B, N+P·38)
            rope = self.key_rope[:, slots, :]                         # (2, B, N+P·38, hd)

        # Context tokens carry the ragged mask; pad-parcel query tokens are masked
        # out of the keys (a real parcel must not attend a P_FIXED pad query).
        if key_mask is None and query_valid is None:
            full_mask: Tensor | None = None
        else:
            ctx_keep = (
                key_mask if key_mask is not None
                else torch.ones(B, N, dtype=torch.bool, device=ctx.device)
            )
            if query_valid is None:
                q_keep = torch.ones(B, P * S, dtype=torch.bool, device=ctx.device)
            else:                                                     # (B,P) → (B,P·S)
                q_keep = query_valid[:, :, None].expand(B, P, S).reshape(B, P * S)
            full_mask = torch.cat([ctx_keep, q_keep], dim=1)         # (B, N+P·38)

        # S3 (#248, bit-exact): the readout uses ONLY the query rows
        # (``tokens[:, N:]``), so the last block's context-row outputs are
        # discarded. Run all but the last block over the full sequence (context
        # must update as keys/values), then compute the last block for the query
        # rows only — SDPA over a query subset is bit-identical to the full pass
        # sliced to those rows, but skips the N context rows' SDPA-query,
        # out-projection, and FFN (the M4 predictor's largest single-layer cost).
        for blk in self.blocks[:-1]:
            tokens = blk(tokens, rope, full_mask)
        last = self.blocks[-1]
        n_q = P * S
        attn_q = last.attn(
            last.ln_attn(tokens), rope, key_mask=full_mask, n_query=n_q
        )                                                            # (B, P·38, d)
        xq = tokens[:, N:] + attn_q
        xq = xq + last.ffn(last.ln_ffn(xq))
        pred = self.head(xq)                                         # (B, P·38, d_model)
        return pred.reshape(B, P, S, -1)                            # (B, P, 38, d_model)


class M2Predictor(nn.Module):
    """Paradigm-B own-predictor for M2 (the frontend per-electrode JEPA).

    ISOLATED per electrode (electrodes ride the batch dim): each electrode's
    predictor SEES **only that electrode's own visible frontend tokens** and
    PREDICTS **only that electrode's own masked frontend tokens** (no cross-
    electrode pathway — the frontend isolation is what makes M2 shortcut-proof).

    Query = a shared learnable **mask token** + additive **freq-tag** ONLY (per
    [[project_v14_m2_m4_predictor_scopes_2026_06_18]]: the parcel tag does not
    exist yet at the frontend stage), positioned by **RoPE on the shared clock**.
    Because each electrode masks a different set of positions, both the context
    and the query slots are **per-row** (the ragged-drop RoPE path). Independent
    params from M4; freq-tag LEARNED by default, ``"sinusoidal"`` sister kept.

    Ragged: ``ctx_mask`` / ``query_mask`` (``True`` = real) pad electrodes with
    differing visible/masked counts to a rectangular batch without leakage."""

    def __init__(
        self,
        d_model: int,
        pred_dim: int,
        n_heads: int,
        n_layers: int,
        *,
        freq_pos: str = "learned",
        bands: tuple[BandSpec, ...] = BANDS,
    ) -> None:
        super().__init__()
        if pred_dim % n_heads != 0:
            raise ValueError(f"pred_dim={pred_dim} not divisible by n_heads={n_heads}")
        head_dim = pred_dim // n_heads
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")

        self.ctx_proj = nn.Linear(d_model, pred_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        _, _, time_slot = token_metadata(bands)
        n_fp = sum(b.n_freq_patches for b in bands)
        if freq_pos == "learned":
            self.freq_embed = nn.Parameter(torch.empty(n_fp, pred_dim))
            nn.init.trunc_normal_(self.freq_embed, std=0.02)
        elif freq_pos == "sinusoidal":
            self.register_buffer("freq_embed", _sincos_1d(n_fp, pred_dim), persistent=False)
        else:
            raise ValueError(f"freq_pos must be 'learned' or 'sinusoidal', got {freq_pos!r}")

        n_slots = int(time_slot.max().item()) + 1
        self.register_buffer("key_rope", _rope_freqs(head_dim, n_slots), persistent=False)
        self.blocks = nn.ModuleList(
            [_JointTokenBlock(pred_dim, n_heads) for _ in range(n_layers)]
        )
        self.head = nn.Linear(pred_dim, d_model, bias=False)      # raw pred, NO LN

    def forward(
        self,
        ctx: Tensor,
        ctx_slot: Tensor,
        query_freq: Tensor,
        query_slot: Tensor,
        *,
        ctx_mask: Tensor | None = None,
        query_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict one electrode-batch's masked frontend tokens from its visible.

        ``ctx``: ``(B', N, d_model)`` per-electrode visible frontend tokens.
        ``ctx_slot``: ``(B', N)`` long — visible tokens' RoPE time-slots (per row).
        ``query_freq``: ``(B', M)`` long — global freq-patch id of each masked token.
        ``query_slot``: ``(B', M)`` long — RoPE time-slot of each masked token.
        ``ctx_mask`` / ``query_mask``: ``(B', N)`` / ``(B', M)`` bool, ``True`` =
        real (ragged padding). Output: ``(B', M, d_model)`` predicted feature per
        masked position, L1'd vs the stop-grad teacher frontend target there."""
        B, N, _ = ctx.shape
        q = self.mask_token + self.freq_embed[query_freq]            # (B', M, d)
        c = self.ctx_proj(ctx)                                       # (B', N, d)
        tokens = torch.cat([c, q], dim=1)                            # (B', N+M, d)

        slots = torch.cat([ctx_slot, query_slot], dim=1)            # (B', N+M)
        rope = self.key_rope[:, slots, :]                           # (2, B', N+M, hd)

        if ctx_mask is None and query_mask is None:
            full_mask: Tensor | None = None
        else:
            M = query_freq.shape[1]
            cm = ctx_mask if ctx_mask is not None else torch.ones(
                B, N, dtype=torch.bool, device=ctx.device)
            qm = query_mask if query_mask is not None else torch.ones(
                B, M, dtype=torch.bool, device=ctx.device)
            full_mask = torch.cat([cm, qm], dim=1)                  # (B', N+M)

        for blk in self.blocks:
            tokens = blk(tokens, rope, full_mask)
        return self.head(tokens[:, N:])                            # (B', M, d_model)


class ParcelReadout(nn.Module):
    """Stage 3: hard-PMA k=1/parcel readout (downstream; encoder FROZEN).

    One learned **seed query per parcel** (PMA, Set Transformer). Parcel ``p``'s
    query attends ONLY to the tokens of electrodes assigned to parcel ``p`` — a
    **hard one-hot DK grouping** (NO support bias: under hard membership
    ``log(support)`` ≡ a hard mask, so the support table's job is *grouping*, not
    biasing; bias would only matter under the gated soft-BNA path). Empty parcels
    pool to a **zero** slot (missing token). The fixed ``K``-parcel rep is
    flattened and linearly mapped to the task logits — **montage-invariant**: any
    electrode montage maps to the same ``K`` slots, so no per-subject params.

    Reuses the B36 ``_MultiHeadCrossAttention`` hard block-diagonal pool (its
    no-coverage row → exactly-zero context is the empty-parcel handling)."""

    def __init__(
        self,
        d_model: int,
        n_parcels: int,
        n_classes: int,
        n_heads: int,
        *,
        bands: tuple[BandSpec, ...] = BANDS,
    ) -> None:
        super().__init__()
        self.n_parcels = n_parcels
        self.tokens_per_electrode = sum(b.n_tokens for b in bands)  # 38
        # k=1 learned seed query per parcel group.
        self.seed = nn.Parameter(torch.empty(n_parcels, d_model))
        nn.init.trunc_normal_(self.seed, std=0.02)
        self.pool = _MultiHeadCrossAttention(d_model, n_heads)
        self.head = nn.Linear(n_parcels * d_model, n_classes)
        self.register_buffer(
            "_parcel_ids", torch.arange(n_parcels), persistent=False
        )

    def forward(
        self,
        feats: Tensor,
        parcel_per_electrode: Tensor,
        *,
        electrode_mask: Tensor | None = None,
    ) -> Tensor:
        """``feats`` ``(B, C, 38, d)`` encoder output → task logits ``(B, n_classes)``.

        ``parcel_per_electrode`` ``(B, C)`` long; ``electrode_mask`` ``(B, C)``
        bool (``True`` = real). Pools each parcel's tokens with its seed query,
        flattens the K-parcel rep, and linearly reads out."""
        B = feats.shape[0]
        pooled = self.pool_parcels(feats, parcel_per_electrode, electrode_mask=electrode_mask)
        return self.head(pooled.reshape(B, -1))                    # (B, n_classes)

    def pool_parcels(
        self,
        feats: Tensor,
        parcel_per_electrode: Tensor,
        *,
        electrode_mask: Tensor | None = None,
    ) -> Tensor:
        """The fixed ``(B, K, d)`` K-parcel representation (pre-readout, reusable
        for analysis). Each slot is the hard-PMA pool of its parcel's tokens;
        absent parcels are exactly zero."""
        B, C, S, d = feats.shape
        tokens = feats.reshape(B, C * S, d)
        token_parcel = (
            parcel_per_electrode[:, :, None].expand(B, C, S).reshape(B, C * S)
        )
        membership = token_parcel[:, None, :] == self._parcel_ids[None, :, None]
        if electrode_mask is not None:
            token_valid = electrode_mask[:, :, None].expand(B, C, S).reshape(B, C * S)
            membership = membership & token_valid[:, None, :]      # (B, K, C·38)
        seed = self.seed[None].expand(B, self.n_parcels, d)        # (B, K, d)
        return self.pool(seed, tokens, membership)                 # (B, K, d)


def _padded_true_indices(mask: Tensor, max_m: int) -> tuple[Tensor, Tensor]:
    """Per-row True-position gather for ragged masks.

    ``mask`` ``(B', S)`` bool → ``(idx (B', max_m), valid (B', max_m))`` where
    ``idx[r]`` lists the column indices where ``mask[r]`` is True (in original
    order), right-padded to ``max_m`` (pad value 0), and ``valid`` flags the real
    entries. Vectorized: a stable descending argsort floats the True columns to
    the front; ``valid`` is ``rank < row_count``."""
    order = torch.argsort(mask.to(torch.int8), dim=1, descending=True, stable=True)
    idx = order[:, :max_m]
    counts = mask.sum(dim=1)                                       # (B',)
    ranks = torch.arange(max_m, device=mask.device)
    valid = ranks[None, :] < counts[:, None]
    return idx, valid & mask.any(dim=1, keepdim=True)


class V14ConvergedSSL(nn.Module):
    """The converged v14 SSL forward — ONE student pass, TWO heads, EMA teacher.

    Wires [[project_v14_converged_arch_2026_06_17]] +
    [[project_v14_m2_m4_predictor_scopes_2026_06_18]] into a single loss module:

    * **Teacher = frontend-ONLY** (EMA of the student frontend; halts after the
      frontend — never builds the parcel-embed or runs the latent, the single
      biggest compute saving since BOTH targets are post-frontend). No grad.
    * **Student = one pass.** frontend(visible, M2 key-masked) → +parcel-tag →
      latent(visible un-tubed, M2-masked-also-hidden). slow is M2-EXEMPT, so
      every electrode keeps ≥6 visible frontend tokens (no all-masked NaN).
    * **M2 head** (frontend JEPA): per un-tubed electrode, predict its own masked
      frontend tokens from its own visible (isolated). Target = stop-grad teacher
      frontend features there.
    * **M4 head** (latent parcel JEPA): predict each tubed parcel's electrode-MEAN
      teacher-frontend grid from ALL visible un-tubed latent tokens. Target =
      ``parcel_electrode_mean`` of the stop-grad teacher.

    Both paradigm-B own-predictor, pure L1. Leakage-free by construction (visible
    tokens never attend masked/tubed tokens). The masks are sampled outside (see
    ``sample_ssl_masks``) and passed in, so the forward is deterministic/testable.

    Compute structure (Ben 2026-06-18, HARD requirement — NOT a deferred perf
    opt): :meth:`forward` PHYSICALLY GATHERS only the visible/un-tubed tokens at
    every stage — the teacher (full real set, ragged, post-frontend only), the
    student frontend (un-tubed real electrodes, visible tokens, 38-at-a-time per
    electrode), the latent (visible cross-electrode set), and both predictors
    (M2 one electrode at a time → its masked; M4 all-visible → tubed parcels).
    Never a dense padded/masked set. The dense attention-key-masked path survives
    only as :meth:`_forward_dense`, the output-identical equivalence oracle."""

    def __init__(
        self,
        d_model: int,
        n_parcels: int,
        *,
        n_heads: int,
        frontend_layers: int,
        latent_layers: int,
        m2_pred_dim: int,
        m2_pred_layers: int,
        m4_pred_dim: int,
        m4_pred_layers: int,
        lambda_m2: float = 1.0,
        lambda_m4: float = 1.0,
        freq_pos: str = "learned",
        m4_precision_weight: bool = True,
        m4_precision_alpha: float = 1.0,
        m4_precision_n_ref: float = 11.0,
        bands: tuple[BandSpec, ...] = BANDS,
        clip_len_s: float | None = None,
    ) -> None:
        super().__init__()
        # SSL pretraining runs on 5 s clips; Phase-4 eval on 1 s. ``clip_len_s``
        # (a serializable float threaded through the config) retimes the locked
        # ladder's TIME axis; the default ``bands`` (BANDS) is the 1 s geometry.
        # Passing both is contradictory, so clip_len_s wins and rebuilds bands.
        if clip_len_s is not None:
            bands = bands_for_clip_len(clip_len_s)
        self.bands = bands
        self.clip_len_s = clip_len_s
        self.lambda_m2 = float(lambda_m2)
        self.lambda_m4 = float(lambda_m4)
        # M4 heteroscedastic down-weight (Ben 2026-06-18): a parcel's electrode-MEAN
        # target has sampling variance ∝ 1/n (n = real electrodes in the parcel), so
        # low-n parcels carry a noisier target → down-weight. n-only (σ-free), so the
        # B37 downweight_dof form (masked_jepa._precision_weight_downweight) ports
        # unchanged. w(n=1)=0, w(n=n_ref)=1, capped at 1 above n_ref.
        self.m4_precision_weight = bool(m4_precision_weight)
        self.m4_precision_alpha = float(m4_precision_alpha)
        self.m4_precision_n_ref = float(m4_precision_n_ref)
        self.tokens_per_electrode = sum(b.n_tokens for b in bands)  # 38

        self.student_frontend = FrontendEncoder(
            d_model, n_heads, frontend_layers, freq_pos=freq_pos, bands=bands
        )
        # Teacher = EMA shadow of the student frontend ONLY (no latent). Cloned so
        # init matches exactly; never optimizer-trained.
        self.teacher_frontend = copy.deepcopy(self.student_frontend)
        for p in self.teacher_frontend.parameters():
            p.requires_grad_(False)

        self.latent = LatentEncoder(d_model, n_heads, latent_layers, n_parcels, bands=bands)
        self.m2_predictor = M2Predictor(
            d_model, m2_pred_dim, n_heads, m2_pred_layers, freq_pos=freq_pos, bands=bands
        )
        self.m4_predictor = M4Predictor(
            d_model, m4_pred_dim, n_heads, m4_pred_layers, n_parcels,
            freq_pos=freq_pos, bands=bands,
        )
        band_id, freq_global_id, time_slot = token_metadata(bands)
        # band_id (38,) ∈ {0=slow,1=beta,2=hg} — the per-band loss-diagnostic axis.
        self.register_buffer("band_id", band_id, persistent=False)
        self.register_buffer("freq_global_id", freq_global_id, persistent=False)
        self.register_buffer("time_slot", time_slot, persistent=False)

        # Forward-tap stash for the RankMe monitor (Ben 2026-06-19): the training
        # forward already computes the teacher-frontend target and the student
        # latent, so the monitor reads them HERE instead of re-running a dense
        # full-input forward every step (the #245 double-forward). Detached, one
        # step's worth, overwritten each forward — mirrors `last_band_token_norm`.
        #   teacher_frontend: (B,C,38,d) full real set — byte-identical to a fresh
        #     `teacher_frontend(slow,beta,hg)` (electrode-isolated, teacher_vis all-True).
        #   student_latent: the latent representation as actually shaped this step
        #     (ragged `forward` → (B,C,38,d) with zeros off the visible set; dense
        #     oracle → (B,C,38,d) full). The monitor gates the zero/pad rows out.
        self.last_rank_taps: dict[str, Tensor] = {}

    @torch.no_grad()
    def update_teacher(self, tau: float) -> None:
        """EMA: ``teacher ← τ·teacher + (1−τ)·student`` (frontend only)."""
        for t, s in zip(self.teacher_frontend.parameters(),
                        self.student_frontend.parameters()):
            t.mul_(tau).add_(s.detach(), alpha=1.0 - tau)
        for t, s in zip(self.teacher_frontend.buffers(),
                        self.student_frontend.buffers()):
            t.copy_(s)

    # ----------------------------------------------------------- monitor diagnostics
    def _stem_norm_diag(self) -> dict[str, Tensor]:
        """Per-band STUDENT stem-output norm (Ben 2026-06-18): one of the three
        separate stems running hot would silently dominate the additive latent.
        Read from the student tokenizer's stash, populated by the just-completed
        student frontend pass. ``{stem_norm_slow, stem_norm_beta, stem_norm_hg}``."""
        return {
            f"stem_norm_{k}": v.detach()
            for k, v in self.student_frontend.tokenizer.last_band_token_norm.items()
        }

    # ----------------------------------------------------------- eval feature taps
    @torch.no_grad()
    def encode_frontend(self, slow: Tensor, beta: Tensor, hg: Tensor) -> Tensor:
        """Eval tap: student FRONTEND features on the FULL input, no mask. ``(B,C,38,d)``.
        Electrode-isolated (no cross-electrode mixing) — the post-frontend probe tap."""
        return self.student_frontend(slow, beta, hg)

    @torch.no_grad()
    def encode_latent(
        self, slow: Tensor, beta: Tensor, hg: Tensor,
        parcel_per_electrode: Tensor, *, electrode_mask: Tensor | None = None,
    ) -> Tensor:
        """Eval tap: frontend→LATENT on the FULL input, no mask. ``(B,C,38,d)``.
        Cross-electrode mixed (parcel-tagged global SA) — the post-latent probe tap.
        The converged success criterion = a linear probe on this must beat one on
        :meth:`encode_frontend`."""
        feats = self.student_frontend(slow, beta, hg)
        # The probe taps store per-electrode anatomy as (C,) (constant across a
        # subject's windows); :meth:`latent` wants (B, C). Expand a 1-D map/mask to
        # the batch — a no-op on the training path, which calls model.forward with
        # (B, C) already, not this eval tap.
        if parcel_per_electrode.ndim == 1:
            parcel_per_electrode = parcel_per_electrode.unsqueeze(0).expand(feats.shape[0], -1)
        if electrode_mask is not None and electrode_mask.ndim == 1:
            electrode_mask = electrode_mask.unsqueeze(0).expand(feats.shape[0], -1)
        return self.latent(feats, parcel_per_electrode, electrode_mask=electrode_mask)

    def forward(
        self,
        slow: Tensor,
        beta: Tensor,
        hg: Tensor,
        parcel_per_electrode: Tensor,
        electrode_mask: Tensor,
        m2_mask: Tensor,
        tube_mask: Tensor,
        tubed_parcels: Tensor,
        tubed_parcel_mask: Tensor,
        static: "StaticShapes | None" = None,
    ) -> dict[str, Tensor]:
        """Run the one-pass two-head forward; return ``{loss, l_m2, l_m4}``.

        ``static`` (step B): when given, every ragged gather slices to a CPU-known
        fixed length (no GPU sync, one compiled graph/session) and the maskless
        sites drop their all-True key-mask → cuDNN-nomask. ``None`` = legacy
        pad-to-batch-max ragged (bit-identical). Converted incrementally; sites not
        yet wired ignore ``static`` and stay on the legacy path.

        Shapes — ``slow (B,C,2,6,5)`` / ``beta (B,C,6,17)`` / ``hg (B,C,9,33)``;
        ``parcel_per_electrode (B,C)`` long; ``electrode_mask (B,C)`` bool real;
        ``m2_mask (B,C,38)`` bool (True = M2 target, only on un-tubed electrodes);
        ``tube_mask (B,C)`` bool (electrode in a tubed parcel); ``tubed_parcels
        (B,P)`` long + ``tubed_parcel_mask (B,P)`` bool (real tubed parcels).

        RAGGED production path (Ben 2026-06-18): every stage **physically gathers**
        only the visible/un-tubed tokens — never a dense padded/masked set.
        - TEACHER: full REAL electrode set (incl tubed — M4 targets need them),
          ragged (drops padding), STOPS post-frontend (the big compute saving).
        - STUDENT frontend: only un-tubed real electrodes, only un-M2-masked
          tokens (per-electrode, 38 tokens at a time, never dense over electrodes).
        - LATENT: gathers the visible un-tubed cross-electrode token set.
        - M2: one electrode at a time, its visible→its masked.
        - M4: all visible tokens from all electrodes → only the tubed parcels.

        Output-identical to :meth:`_forward_dense` (the equivalence oracle): a
        masked key adds 0 to the softmax and masked tokens are never read."""
        # ---- teacher: full REAL electrode set, RAGGED (drop padding), post-FE ---
        teacher_vis = electrode_mask.new_ones(m2_mask.shape)       # (B,C,38) all True
        with torch.no_grad():
            t_f = self.teacher_frontend.forward_ragged(
                slow, beta, hg, electrode_mask, teacher_vis,       # (B,C,38,d)
                fixed_r=None if static is None else static.teacher_elec_k,
                fixed_lk=None if static is None else static.s)
        t_f = t_f.detach()

        # ---- student: only un-tubed real electrodes, only visible tokens --------
        student_vis = ~m2_mask                                     # (B,C,38) visible
        student_keep = electrode_mask & ~tube_mask                 # drop tubed+padded
        s_f = self.student_frontend.forward_ragged(
            slow, beta, hg, student_keep, student_vis,
            fixed_r=None if static is None else static.student_elec_k,
            fixed_lk=None if static is None else static.s_vis)
        latent_vis = (
            electrode_mask[:, :, None] & (~tube_mask)[:, :, None] & student_vis
        )                                                          # (B,C,38)
        s_l = self.latent.forward_ragged(
            s_f, parcel_per_electrode, latent_vis,
            fixed_nv=None if static is None else static.latent_nv,
        )
        # Stash the taps the RankMe monitor needs — t_f is already detached; s_l
        # carries grad so detach. `latent_vis` (B,C,38) marks the rows the latent
        # actually wrote (zeros elsewhere) so the monitor ranks only real tokens.
        self.last_rank_taps = {
            "teacher_frontend": t_f,
            "student_latent": s_l.detach(),
            "student_latent_valid": latent_vis,
        }

        l_m2, m2_bands = self._m2_loss_ragged(
            s_f, t_f, m2_mask, student_vis, electrode_mask, tube_mask,
            static=static)
        l_m4, m4_bands = self._m4_loss_ragged(
            s_l, t_f, parcel_per_electrode, electrode_mask, latent_vis,
            tubed_parcels, tubed_parcel_mask, static=static,
        )
        loss = self.lambda_m2 * l_m2 + self.lambda_m4 * l_m4
        out = _assemble_losses(loss, l_m2, l_m4, m2_bands, m4_bands)
        out.update(self._stem_norm_diag())
        return out

    def _forward_dense(
        self,
        slow: Tensor,
        beta: Tensor,
        hg: Tensor,
        parcel_per_electrode: Tensor,
        electrode_mask: Tensor,
        m2_mask: Tensor,
        tube_mask: Tensor,
        tubed_parcels: Tensor,
        tubed_parcel_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Dense attention-key-masked path — retained ONLY as the equivalence
        oracle for the ragged :meth:`forward`. Mathematically output-identical
        (a masked key contributes 0; masked outputs are never read), but it builds
        the full padded/masked token set, which the production path must not do."""
        with torch.no_grad():
            t_f = self.teacher_frontend(slow, beta, hg)            # (B,C,38,d)
        t_f = t_f.detach()

        student_vis = ~m2_mask                                     # (B,C,38) visible
        s_f = self.student_frontend(slow, beta, hg, key_mask=student_vis)
        latent_vis = (
            electrode_mask[:, :, None] & (~tube_mask)[:, :, None] & student_vis
        )                                                          # (B,C,38)
        s_l = self.latent(s_f, parcel_per_electrode, token_mask=latent_vis)
        self.last_rank_taps = {
            "teacher_frontend": t_f,
            "student_latent": s_l.detach(),
            "student_latent_valid": latent_vis,
        }

        l_m2, m2_bands = self._m2_loss(
            s_f, t_f, m2_mask, student_vis, electrode_mask, tube_mask)
        l_m4, m4_bands = self._m4_loss(
            s_l, t_f, parcel_per_electrode, electrode_mask, latent_vis,
            tubed_parcels, tubed_parcel_mask,
        )
        loss = self.lambda_m2 * l_m2 + self.lambda_m4 * l_m4
        out = _assemble_losses(loss, l_m2, l_m4, m2_bands, m4_bands)
        out.update(self._stem_norm_diag())
        return out

    # --------------------------------------------------------------- M2 head
    def _m2_loss(
        self, s_f: Tensor, t_f: Tensor, m2_mask: Tensor, student_vis: Tensor,
        electrode_mask: Tensor, tube_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        B, C, S, d = s_f.shape
        elec = B * C
        m2_flat = m2_mask.reshape(elec, S)
        max_m = int(m2_flat.sum(dim=1).max().item())
        # un-tubed real electrodes that actually carry an M2 target
        elec_ok = (electrode_mask & ~tube_mask).reshape(elec) & (m2_flat.any(dim=1))
        if max_m == 0 or not bool(elec_ok.any()):
            return s_f.new_zeros(()), _zero_bands(s_f, _M2_BAND_IDS, "m2")

        qidx, qvalid = _padded_true_indices(m2_flat, max_m)        # (elec, max_m)
        qvalid = qvalid & elec_ok[:, None]
        ctx = s_f.reshape(elec, S, d)
        ctx_slot = self.time_slot[None, :].expand(elec, S)
        q_freq = self.freq_global_id[qidx]                         # (elec, max_m)
        q_slot = self.time_slot[qidx]
        pred = self.m2_predictor(
            ctx, ctx_slot, q_freq, q_slot,
            ctx_mask=student_vis.reshape(elec, S), query_mask=qvalid,
        )                                                          # (elec, max_m, d)
        tgt = torch.gather(
            t_f.reshape(elec, S, d), 1, qidx[:, :, None].expand(elec, max_m, d)
        )
        bands = _band_diagnostics(
            pred, tgt, qvalid, self.band_id[qidx], _M2_BAND_IDS, "m2")
        return _masked_l1(pred, tgt, qvalid), bands

    def _m2_loss_ragged(
        self, s_f: Tensor, t_f: Tensor, m2_mask: Tensor, student_vis: Tensor,
        electrode_mask: Tensor, tube_mask: Tensor,
        static: "StaticShapes | None" = None,
    ) -> Tensor:
        """Ragged M2 (production) — Ben 2026-06-18: "the M2 PREDICTOR only
        GATHERING ITS INDIVIDUAL ELECTRODE, PROCESSING EACH ELECTRODE ONE AT A
        TIME ... ONLY PREDICTING THAT ONE ELECTRODE's masked tokens NEVER DENSE."

        Two physical gathers, never a dense set: (1) ELECTRODE axis — only the
        un-tubed real electrodes that carry an M2 target are batched (tubed/padded
        electrodes are never run through the predictor); (2) TOKEN axis — each
        electrode's visible context and its masked queries are gathered to their
        own ragged lengths (the dense path passed all 38 context tokens key-masked
        to visible). Output-identical to the dense ``_m2_loss`` oracle on the
        per-electrode masked targets (the predictor is per-electrode in the batch
        dim either way; masked context keys add 0 and no-target rows contribute 0
        to the L1 mean), with no dense electrode/token set ever built."""
        B, C, S, d = s_f.shape
        elec = B * C
        m2_flat = m2_mask.reshape(elec, S)
        vis_flat = student_vis.reshape(elec, S)
        elec_ok = (electrode_mask & ~tube_mask).reshape(elec) & m2_flat.any(dim=1)
        if static is None and not bool(elec_ok.any()):
            return s_f.new_zeros(()), _zero_bands(s_f, _M2_BAND_IDS, "m2")

        if static is not None:    # fixed target-electrode count, no nonzero sync
            e_sort, _, R = _ragged_gather_idx(
                elec_ok.unsqueeze(0), fixed_k=static.student_elec_k)
            e_idx = e_sort[0]                                       # (R,) target elecs
        else:
            e_idx = elec_ok.nonzero(as_tuple=False).squeeze(1)     # (R,) target elecs
            R = int(e_idx.numel())
        sf = s_f.reshape(elec, S, d)[e_idx]                        # (R, 38, d)
        tf = t_f.reshape(elec, S, d)[e_idx]                        # (R, 38, d)

        f_ctx = None if static is None else static.s_vis
        f_q = None if static is None else static.n_mask
        c_idx, c_real, Nv = _ragged_gather_idx(vis_flat[e_idx], fixed_k=f_ctx)
        ctx = torch.gather(sf, 1, c_idx.unsqueeze(-1).expand(R, Nv, d))
        ctx_slot = self.time_slot[c_idx]                           # (R, Nv)

        q_idx, q_real, Mq = _ragged_gather_idx(m2_flat[e_idx], fixed_k=f_q)
        q_freq = self.freq_global_id[q_idx]                        # (R, Mq)
        q_slot = self.time_slot[q_idx]
        pred = self.m2_predictor(
            ctx, ctx_slot, q_freq, q_slot,
            ctx_mask=None if static is not None else c_real,       # static → maskless
            query_mask=None if static is not None else q_real,
        )                                                          # (R, Mq, d)
        tgt = torch.gather(tf, 1, q_idx.unsqueeze(-1).expand(R, Mq, d))
        bands = _band_diagnostics(
            pred, tgt, q_real, self.band_id[q_idx], _M2_BAND_IDS, "m2")
        return _masked_l1(pred, tgt, q_real), bands

    # --------------------------------------------------------------- M4 head
    def _m4_loss(
        self, s_l: Tensor, t_f: Tensor, parcel_per_electrode: Tensor,
        electrode_mask: Tensor, latent_vis: Tensor,
        tubed_parcels: Tensor, tubed_parcel_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        B, C, S, d = s_l.shape
        if not bool(tubed_parcel_mask.any()):
            return s_l.new_zeros(()), _zero_bands(s_l, _M4_BAND_IDS, "m4")

        ctx = s_l.reshape(B, C * S, d)
        ctx_slot = self.time_slot.repeat(C)                        # (C·38,)
        pred = self.m4_predictor(
            ctx, ctx_slot, tubed_parcels.clamp_min(0),
            key_mask=latent_vis.reshape(B, C * S),
            query_valid=tubed_parcel_mask,
        )                                                          # (B, P, 38, d)

        # teacher electrode-MEAN target over each tubed parcel (padded elec → -1)
        pe = torch.where(electrode_mask, parcel_per_electrode,
                         torch.full_like(parcel_per_electrode, -1))
        member = (tubed_parcels[:, :, None] == pe[:, None, :]).to(t_f.dtype)  # (B,P,C)
        n_p = member.sum(dim=-1)                                    # (B,P) real elec/parcel
        tgt = (member @ t_f.reshape(B, C, S * d)) / n_p.clamp_min(1.0)[:, :, None]
        tgt = tgt.reshape(B, -1, S, d).detach()                     # (B,P,38,d)
        cell_valid = tubed_parcel_mask[:, :, None, None].expand_as(pred)
        weight = self._m4_weight(n_p, tubed_parcel_mask)           # (B,P) or None
        bands = _band_diagnostics(
            pred, tgt, cell_valid[..., 0], self.band_id.view(1, 1, S),
            _M4_BAND_IDS, "m4")
        return _masked_l1(pred, tgt, cell_valid, weight=weight), bands

    def _m4_loss_ragged(
        self, s_l: Tensor, t_f: Tensor, parcel_per_electrode: Tensor,
        electrode_mask: Tensor, latent_vis: Tensor,
        tubed_parcels: Tensor, tubed_parcel_mask: Tensor,
        static: "StaticShapes | None" = None,
    ) -> Tensor:
        """Ragged M4 (production) — Ben 2026-06-18: "the M4 PREDICTOR MUST GATHER
        ALL THE VISIBLE TOKENS FROM ALL ELECTRODES — NEVER DENSE — AND ONLY
        PREDICT THE TUBED PARCELS — NEVER DENSE AGAIN."

        The visible un-tubed latent SEE-set (``latent_vis``) is **physically
        gathered** across all electrodes (pad-to-batch-max), never the dense C·38
        token set key-masked to visible. The queries are already only the tubed
        parcels. Output-identical to the dense ``_m4_loss`` oracle (the visible
        context attention set is the same; masked context keys add 0), with no
        dense context ever built. Teacher target is unchanged (electrode-MEAN of
        the tubed parcels' teacher frontend grid)."""
        B, C, S, d = s_l.shape
        if static is None and not bool(tubed_parcel_mask.any()):
            return s_l.new_zeros(()), _zero_bands(s_l, _M4_BAND_IDS, "m4")

        vis = latent_vis.reshape(B, C * S)
        f_nv = None if static is None else static.latent_nv
        c_idx, c_real, Nv = _ragged_gather_idx(vis, fixed_k=f_nv)  # (B, Nv)
        ctx = torch.gather(
            s_l.reshape(B, C * S, d), 1, c_idx.unsqueeze(-1).expand(B, Nv, d)
        )                                                          # (B, Nv, d)
        ctx_slot = self.time_slot.repeat(C)[c_idx]                 # (B, Nv) per-row
        pred = self.m4_predictor(
            ctx, ctx_slot, tubed_parcels.clamp_min(0),
            key_mask=None if static is not None else c_real,       # static → maskless ctx
            query_valid=tubed_parcel_mask,
        )                                                          # (B, P, 38, d)

        # teacher electrode-MEAN target over each tubed parcel (padded elec → -1)
        pe = torch.where(electrode_mask, parcel_per_electrode,
                         torch.full_like(parcel_per_electrode, -1))
        member = (tubed_parcels[:, :, None] == pe[:, None, :]).to(t_f.dtype)  # (B,P,C)
        n_p = member.sum(dim=-1)                                    # (B,P) real elec/parcel
        tgt = (member @ t_f.reshape(B, C, S * d)) / n_p.clamp_min(1.0)[:, :, None]
        tgt = tgt.reshape(B, -1, S, d).detach()                     # (B,P,38,d)
        cell_valid = tubed_parcel_mask[:, :, None, None].expand_as(pred)
        weight = self._m4_weight(n_p, tubed_parcel_mask)           # (B,P) or None
        bands = _band_diagnostics(
            pred, tgt, cell_valid[..., 0], self.band_id.view(1, 1, S),
            _M4_BAND_IDS, "m4")
        return _masked_l1(pred, tgt, cell_valid, weight=weight), bands

    def _m4_weight(self, n_p: Tensor, tubed_parcel_mask: Tensor) -> Tensor | None:
        """Per-parcel M4 down-weight ``(B,P)`` (real-parcel cells only), or ``None``
        when the precision weight is off. ``n_p`` = real electrode count per tubed
        parcel; pad parcels are zeroed (cell_valid already drops them, this just
        keeps the weight clean)."""
        if not self.m4_precision_weight:
            return None
        w = _downweight_dof(
            n_p, n_ref=self.m4_precision_n_ref, alpha=self.m4_precision_alpha)
        return w * tubed_parcel_mask.to(w.dtype)


# Per-band diagnostic axes (band_id buffer ∈ {0=slow,1=beta,2=hg}). slow is M2-
# EXEMPT so it never carries an M2 target; M4 predicts the whole 38-token grid so
# all three bands appear there.
_M2_BAND_IDS: tuple[tuple[str, int], ...] = (("beta", 1), ("hg", 2))
_M4_BAND_IDS: tuple[tuple[str, int], ...] = (("slow", 0), ("beta", 1), ("hg", 2))


def _downweight_dof(n: Tensor, *, n_ref: float, alpha: float) -> Tensor:
    """M4 per-parcel down-weight ``w = min(1, ((n-1)/(n_ref-1))^alpha)``, DETACHED.

    ``n`` = the real electrode count of each (tubed) parcel. Ports B37's
    ``masked_jepa._precision_weight_downweight`` (the live default) to the
    converged electrode-MEAN target: n-only (σ-free), so the form is unchanged.
    Passes through 0 at n=1 (a single-electrode parcel's mean is just that
    electrode — no aggregation, no cross-electrode structure to trust), reaches 1
    at ``n_ref`` and saturates above it."""
    ref = max(float(n_ref) - 1.0, 1e-6)
    dof = (n - 1.0).clamp(min=0.0)
    return (dof / ref).pow(alpha).clamp(max=1.0).detach()


# Monitor-stat eps — matches masked_jepa._JEPA_STATS_EPS (detached scalars only).
_STATS_EPS = 1e-8


def _jepa_stats_on_cells(
    pred: Tensor, target: Tensor, cell_valid: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """``(loss, explained_var, target_var)`` over the ``cell_valid`` scored cells.

    ``cell_valid`` is the PER-CELL mask (``pred.shape[:-1]``-broadcastable, no
    feature axis); the scored ``(n, d)`` rows are gathered and reduced. Matches
    ``masked_jepa._jepa_pred_target_stats`` L1 semantics:
      * ``loss`` = mean ``|pred − target|`` over scored cells×d (== the masked-L1).
      * ``target_var`` = mean over feature dims of the target's per-dim variance
        across scored cells (→ 0 on EMA-teacher collapse).
      * ``explained_var`` = ``1 − err / Var(target)`` (scale-free structure captured).
    ``ev``/``tv`` are ``NaN`` with < 2 scored cells (variance undefined → logger
    skips them); ``loss`` is 0 with no scored cells. All detached — pure monitor.

    MASKED-REDUCTION form (no boolean indexing): masked-out cells contribute 0 to
    every sum, so the result is identical to gather-then-reduce on the valid cells,
    but the shapes are static (no ``nonzero``) → torch.compile-traceable. NaN/zero
    branches are tensor ``torch.where`` (the count drives them, not a python ``if``
    on a data-dependent shape)."""
    d = pred.shape[-1]
    m = cell_valid.detach().reshape(-1).to(pred.dtype)         # (N,) 0/1
    mc = m[:, None]                                            # (N,1)
    p = pred.detach().reshape(-1, d)
    t = target.detach().reshape(-1, d)
    n = m.sum()                                                # scored-cell count
    nz = n.clamp_min(1.0)
    nd = (n * d).clamp_min(1.0)
    loss = ((p - t).abs() * mc).sum() / nd                     # masked-L1 over cells×d
    mean_pd = (t * mc).sum(0) / nz                             # (d,) per-dim mean
    target_var = ((t.pow(2) * mc).sum(0) / nz - mean_pd.pow(2)).mean()
    mean_all = (t * mc).sum() / nd                             # flat mean over cells×d
    var_all = (t.pow(2) * mc).sum() / nd - mean_all.pow(2)
    explained_var = 1.0 - loss / (var_all + _STATS_EPS)
    nan = pred.new_full((), float("nan"))
    two = n >= 2
    return (
        torch.where(n >= 1, loss, pred.new_zeros(())),
        torch.where(two, explained_var, nan),
        torch.where(two, target_var, nan),
    )


def _collapse_stats(
    pred: Tensor, target: Tensor, cell_valid: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """``(pred_var, target_norm, pred_target_var_ratio)`` over scored cells — the
    joint module's ``_log_term_stats`` collapse triad. ``pred_var`` → 0 when the
    predictor hedges to a constant; ``target_norm`` is the mean target-row L2;
    ``pred_target_var_ratio = pred_var/target_var`` (≈1 healthy, ≪1 = the
    predictor collapsing to the batch-mean target). All NaN with < 2 scored
    cells (variance undefined → logger skips). Detached — pure monitor.

    MASKED-REDUCTION form (no boolean indexing) — see :func:`_jepa_stats_on_cells`:
    identical to gather-then-reduce on the valid cells, but static-shaped so it
    traces under torch.compile."""
    d = pred.shape[-1]
    m = cell_valid.detach().reshape(-1).to(pred.dtype)         # (N,) 0/1
    mc = m[:, None]
    p = pred.detach().reshape(-1, d)
    t = target.detach().reshape(-1, d)
    n = m.sum()
    nz = n.clamp_min(1.0)
    mean_p = (p * mc).sum(0) / nz
    pred_var = ((p.pow(2) * mc).sum(0) / nz - mean_p.pow(2)).mean()
    mean_t = (t * mc).sum(0) / nz
    target_var = ((t.pow(2) * mc).sum(0) / nz - mean_t.pow(2)).mean()
    target_norm = (t.norm(dim=-1) * m).sum() / nz
    ratio = pred_var / (target_var + _STATS_EPS)
    nan = pred.new_full((), float("nan"))
    two = n >= 2
    return (
        torch.where(two, pred_var, nan),
        torch.where(n >= 1, target_norm, nan),
        torch.where(two, ratio, nan),
    )


def _band_diagnostics(
    pred: Tensor, target: Tensor, cell_valid: Tensor, band_of_cell: Tensor,
    pairs: tuple[tuple[str, int], ...], head: str,
) -> dict[str, Tensor]:
    """Aggregate + per-band ``loss``/``explained_var``/``target_var`` for one head.

    Ben 2026-06-18: split the term's loss/explained_var/target_var by band {slow,
    beta, hg} — the generalized slow-easiness guard (catches a trivial/low-grad
    band, a dead band, or one band dominating). ``head`` ∈ {'m2','m4'} sets the key
    prefix; ``cell_valid``/``band_of_cell`` are per-cell (``pred.shape[:-1]``).
    Detached monitors — never folded into the loss. Keys: ``ev_{head}``/``tv_{head}``
    (aggregate over all scored cells) and per band ``l_{head}_{b}``/``ev_{head}_{b}``
    /``tv_{head}_{b}``. The per-band ``l_{head}_{b}`` is the unweighted masked-L1
    (unchanged from the prior per-band loss diagnostic)."""
    _, ev, tv = _jepa_stats_on_cells(pred, target, cell_valid)
    pv, tn, ptvr = _collapse_stats(pred, target, cell_valid)
    out: dict[str, Tensor] = {
        f"ev_{head}": ev, f"tv_{head}": tv,
        f"pv_{head}": pv, f"tn_{head}": tn, f"ptvr_{head}": ptvr,
    }
    for name, bid in pairs:
        bl, bev, btv = _jepa_stats_on_cells(
            pred, target, cell_valid & (band_of_cell == bid)
        )
        out[f"l_{head}_{name}"] = bl
        out[f"ev_{head}_{name}"] = bev
        out[f"tv_{head}_{name}"] = btv
    return out


def _zero_bands(
    ref: Tensor, pairs: tuple[tuple[str, int], ...], head: str,
) -> dict[str, Tensor]:
    """Graph-free zero/NaN diagnostics (the no-target early-return case) — SAME key
    set as :func:`_band_diagnostics` so the forward dict keys are batch-stable."""
    nan = ref.new_full((), float("nan"))
    out: dict[str, Tensor] = {
        f"ev_{head}": nan, f"tv_{head}": nan,
        f"pv_{head}": nan, f"tn_{head}": nan, f"ptvr_{head}": nan,
    }
    for name, _ in pairs:
        out[f"l_{head}_{name}"] = ref.new_zeros(())
        out[f"ev_{head}_{name}"] = nan
        out[f"tv_{head}_{name}"] = nan
    return out


def _assemble_losses(
    loss: Tensor, l_m2: Tensor, l_m4: Tensor,
    m2_diag: dict[str, Tensor], m4_diag: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Pack the scalar losses + the full M2/M4 monitor diagnostics (per-band loss/
    explained_var/target_var + aggregate ev/tv) into the forward return dict. The
    diag dicts are already fully keyed (``l_m2_beta``, ``ev_m2``, ``tv_m4_slow`` …)."""
    out = {"loss": loss, "l_m2": l_m2, "l_m4": l_m4}
    out.update(m2_diag)
    out.update(m4_diag)
    return out


def _masked_l1(
    pred: Tensor, target: Tensor, valid: Tensor, weight: Tensor | None = None,
) -> Tensor:
    """Mean L1 over the ``valid`` entries; 0 if none. ``valid`` is broadcast over
    any trailing feature dims, so the mean is per-element over (valid cells × d).

    ``weight`` (optional, detached) is a per-cell multiplier broadcast the same
    way; the DENOMINATOR stays the **unweighted** valid-cell count, so a mean
    weight < 1 genuinely shrinks the loss (the B37 ``downweight_dof`` semantics —
    masked_jepa._weighted_l1_or_zero's ``(per_cell·w).mean()`` = sum / count).
    ``weight=None`` ⇒ byte-identical to the plain masked mean (the M2 path)."""
    diff = (pred - target).abs()
    v = valid.to(diff.dtype)
    while v.dim() < diff.dim():
        v = v.unsqueeze(-1)
    v = v.expand_as(diff)
    denom = v.sum().clamp_min(1.0)
    if weight is None:
        return (diff * v).sum() / denom
    w = weight.to(diff.dtype)
    while w.dim() < diff.dim():
        w = w.unsqueeze(-1)
    return (diff * v * w.expand_as(diff)).sum() / denom


def sample_ssl_masks(
    parcel_per_electrode: Tensor,
    electrode_mask: Tensor,
    generator: torch.Generator,
    *,
    m2_cfg: M2MaskConfig = M2MaskConfig(),
    m4_cfg: M4MaskConfig = M4MaskConfig(),
    bands: tuple[BandSpec, ...] = BANDS,
) -> dict[str, Tensor]:
    """Compose the per-clip mask draws for a batch (the data-path bridge).

    Returns ``m2_mask (B,C,S)`` / ``tube_mask (B,C)`` / ``tubed_parcels (B,Pmax)``
    / ``tubed_parcel_mask (B,Pmax)`` where ``S = Σ band.n_tokens`` (38 at the 1 s
    eval clip, 190 at the 5 s SSL clip — set by ``bands``). Per sample: tube whole
    parcels (M4), then draw an M2 within-electrode mask for every un-tubed real
    electrode. Tubed and padded electrodes get no M2 target. ``Pmax`` is the
    batch-max tubed count."""
    B, C = parcel_per_electrode.shape
    S = sum(b.n_tokens for b in bands)
    m2 = torch.zeros(B, C, S, dtype=torch.bool)
    tube = torch.zeros(B, C, dtype=torch.bool)
    tubed_lists: list[Tensor] = []
    for b in range(B):
        real = electrode_mask[b]
        present = parcel_per_electrode[b][real].unique()
        tubed = sample_parcel_tube(present, generator, m4_cfg)
        tubed_lists.append(tubed)
        tb = electrode_tube_mask(parcel_per_electrode[b], tubed) & real
        tube[b] = tb
        for e in range(C):
            if real[e] and not tb[e]:
                m2[b, e] = sample_m2_mask(generator, bands=bands, cfg=m2_cfg)
    pmax = max(1, max(t.numel() for t in tubed_lists))
    tubed_parcels = torch.zeros(B, pmax, dtype=torch.long)
    tubed_parcel_mask = torch.zeros(B, pmax, dtype=torch.bool)
    for b, t in enumerate(tubed_lists):
        tubed_parcels[b, : t.numel()] = t
        tubed_parcel_mask[b, : t.numel()] = True
    return {
        "m2_mask": m2,
        "tube_mask": tube,
        "tubed_parcels": tubed_parcels,
        "tubed_parcel_mask": tubed_parcel_mask,
    }


# ============================ static-shape SSL masks ==========================
# V-JEPA-2 static regime (Ben 2026-06-22): a tight-pack tube + rand_unmask M2 make
# the visible-electrode count (n_vis) AND the per-electrode visible-token count
# (S − N_mask) CONSTANT per session, so every forward gather is a static shape →
# one compiled graph per distinct n_vis (~19 across the BT corpus), no within-
# session jitter, no batch-min, no bucketing. Drop-not-pad (the masked tokens are
# physically removed, never padded+masked), so the latent runs the fast cuDNN
# nomask kernel (step B). The old `sample_ssl_masks` variable-count path stays the
# default until this is proven.


def _m2_expected_cols(allowed: int, span: int, n_starts: int, n_time: int) -> float:
    """E[|covered time columns|] when ``n_starts`` distinct uniform span starts in
    ``[0, allowed)`` each cover ``span`` columns forward. Exact via per-column
    coverage probability (deterministic — no Monte-Carlo)."""
    if n_starts <= 0:
        return 0.0
    total = comb(allowed, n_starts)
    e = 0.0
    for p in range(n_time):
        lo = max(0, p - span + 1)
        hi = min(allowed - 1, p)
        w = hi - lo + 1 if hi >= lo else 0           # starts that cover column p
        if w <= 0:
            continue
        uncov = comb(allowed - w, n_starts) / total if allowed - w >= n_starts else 0.0
        e += 1.0 - uncov
    return e


def m2_fixed_col_targets(
    bands: tuple[BandSpec, ...] = BANDS, cfg: M2MaskConfig = M2MaskConfig(),
) -> dict[str, int]:
    """Per-band rand_unmask target = ``round(E[covered columns])`` under the
    current span sampler, so the fixed-count mask keeps the realized ~40% ratio
    (1 s: beta 2 / HG 8 cols → 12/30 tok; 5 s: beta 10 / HG 39 → 59/150). slow
    is exempt (0)."""
    targets: dict[str, int] = {}
    for b in bands:
        if b.name == "slow":
            targets[b.name] = 0
            continue
        rate = cfg.beta_start_rate if b.name == "beta" else cfg.hg_start_rate
        span = cfg.beta_span if b.name == "beta" else cfg.hg_span
        allowed = b.n_time_patches - span + 1
        n_starts = min(round(rate * b.n_time_patches), allowed)
        targets[b.name] = round(
            _m2_expected_cols(allowed, span, n_starts, b.n_time_patches))
    return targets


def _rand_unmask_columns(
    allowed: int, span: int, n_starts: int, n_time: int, target: int,
    generator: torch.Generator,
) -> Tensor:
    """Sample the span mask, then randomly unmask (overflow) or add (underflow) to
    EXACTLY ``target`` covered columns. Exact count, no positional bias (the
    overflow/deficit columns are chosen with uniform random identity)."""
    cols = torch.zeros(n_time, dtype=torch.bool)
    if n_starts > 0:
        starts = torch.randperm(allowed, generator=generator)[:n_starts]
        for s in starts.tolist():
            cols[s : s + span] = True
    cur = int(cols.sum())
    if cur > target:
        on = cols.nonzero(as_tuple=False).flatten()
        drop = on[torch.randperm(on.numel(), generator=generator)[: cur - target]]
        cols[drop] = False
    elif cur < target:
        off = (~cols).nonzero(as_tuple=False).flatten()
        add = off[torch.randperm(off.numel(), generator=generator)[: target - cur]]
        cols[add] = True
    return cols


def sample_m2_mask_fixed(
    generator: torch.Generator,
    bands: tuple[BandSpec, ...] = BANDS,
    cfg: M2MaskConfig = M2MaskConfig(),
    targets: dict[str, int] | None = None,
) -> Tensor:
    """rand_unmask M2: like :func:`sample_m2_mask` but every band hits an EXACT
    column count (``targets``) → constant ``N_mask`` per electrode (static maskless
    frontend shape). beta/HG mask whole time-columns (freq-tube preserved — both
    freq patches co-masked); slow exempt. Returns ``(N_TOKENS,)`` bool in tokenizer
    order; ``True`` = M2 target."""
    if targets is None:
        targets = m2_fixed_col_targets(bands, cfg)
    parts: list[Tensor] = []
    for b in bands:
        m = torch.zeros(b.n_freq_patches, b.n_time_patches, dtype=torch.bool)
        if b.name != "slow":
            span = cfg.beta_span if b.name == "beta" else cfg.hg_span
            rate = cfg.beta_start_rate if b.name == "beta" else cfg.hg_start_rate
            allowed = b.n_time_patches - span + 1
            n_starts = min(round(rate * b.n_time_patches), allowed)
            cols = _rand_unmask_columns(
                allowed, span, n_starts, b.n_time_patches, targets[b.name], generator)
            m[:, cols] = True                        # freq-tube: all freq patches
        parts.append(m.reshape(-1))
    return torch.cat(parts)


@dataclass(frozen=True)
class TightPackConfig:
    """Static-shape M4 tube (Ben 2026-06-22). ``ratio`` of the real electrodes are
    dropped, hit EXACTLY: first-fit tight-pack whole parcels to ``≤`` budget then
    water-fill the residual from the largest visible (non-target) parcels → n_vis =
    C − round(ratio·C) constant per session. At ratio 0.25 every BT parcel fits
    (no coverage gap). ``floor`` = min electrodes a visible parcel keeps (inert on
    BT). ``p_fixed`` pads the M4 query parcel axis (measured BT max 17 at ratio
    0.25 → 18, +1 headroom; the small wasted query rows are noise vs the
    latent — see [[project_v14_static_shape_ssl_2026_06_22]])."""

    ratio: float = 0.25
    floor: int = 1
    p_fixed: int = 18


def tight_pack_tube(
    parcel_per_electrode: Tensor, real: Tensor, generator: torch.Generator,
    cfg: TightPackConfig = TightPackConfig(),
) -> tuple[Tensor, Tensor]:
    """One clip's tube. Returns ``(tube_mask (C,), tubed_parcels (P,))``:
    ``tube_mask`` = electrodes DROPPED from the student (whole tubed parcels ∪
    water-fill residual); ``tubed_parcels`` = the WHOLE tubed parcel ids (M4
    targets only — residual-trimmed electrodes are NOT here).

    INVARIANT (shortcut-resistance): every tubed parcel is whole, and the residual
    drops come only from VISIBLE (non-target) parcels — so no parcel ever has both
    a target and a visible electrode (no co-shaft leak). Guards: ≥1 parcel tubed
    (M4 needs a target) and ≥1 parcel visible (M4 needs context); a <2-parcel clip
    tubes nothing (M4 inert)."""
    C = parcel_per_electrode.shape[0]
    tube = torch.zeros(C, dtype=torch.bool)
    ridx = real.nonzero(as_tuple=False).flatten()
    n_real = int(ridx.numel())
    if n_real == 0:
        return tube, parcel_per_electrode.new_zeros(0)
    pe = parcel_per_electrode[ridx]
    parcels, sizes = pe.unique(return_counts=True)
    nP = int(parcels.numel())
    if nP < 2:
        return tube, parcel_per_electrode.new_zeros(0)        # M4 inert
    D_target = round(cfg.ratio * n_real)
    size_of = {int(parcels[i]): int(sizes[i]) for i in range(nP)}
    order = torch.randperm(nP, generator=generator).tolist()
    tubed: list[int] = []
    cum = 0
    for oi in order:                                          # first-fit, skip non-fitters
        p = int(parcels[oi])
        if cum + size_of[p] <= D_target:
            tubed.append(p)
            cum += size_of[p]
    if not tubed:                                             # budget < smallest parcel
        sp = int(parcels[int(torch.argmin(sizes))])
        tubed = [sp]
        cum = size_of[sp]
    if len(tubed) == nP:                                      # never empty-visible
        cum -= size_of[tubed.pop()]
    tubed_t = torch.tensor(sorted(tubed), dtype=parcel_per_electrode.dtype)
    in_tube = torch.isin(pe, tubed_t)                         # over real electrodes
    tube[ridx[in_tube]] = True
    # water-fill the residual to EXACTLY D_target dropped, from the largest visible
    # parcels, random electrode identity, never below floor.
    residual = D_target - cum
    if residual > 0:
        buckets: dict[int, list[int]] = defaultdict(list)
        for j in (~in_tube).nonzero(as_tuple=False).flatten().tolist():
            buckets[int(pe[j])].append(int(ridx[j]))
        for p in buckets:
            perm = torch.randperm(len(buckets[p]), generator=generator).tolist()
            buckets[p] = [buckets[p][k] for k in perm]        # random identity
        for _ in range(residual):
            maxn = max((len(lst) for lst in buckets.values()), default=0)
            if maxn <= cfg.floor:
                break                                         # floor blocks (inert on BT)
            cand = [p for p, lst in buckets.items() if len(lst) == maxn]
            p = cand[int(torch.randint(len(cand), (1,), generator=generator))]
            tube[buckets[p].pop()] = True
    return tube, tubed_t


def sample_ssl_masks_static(
    parcel_per_electrode: Tensor,
    electrode_mask: Tensor,
    generator: torch.Generator,
    *,
    m2_cfg: M2MaskConfig = M2MaskConfig(),
    tube_cfg: TightPackConfig = TightPackConfig(),
    bands: tuple[BandSpec, ...] = BANDS,
) -> dict[str, Tensor]:
    """Static-shape variant of :func:`sample_ssl_masks` (V-JEPA-2 regime). Same
    return keys, but (1) the tube is tight-pack → ``n_vis = C − round(ratio·C)``
    constant per session; (2) M2 is rand_unmask → constant ``N_mask`` per kept
    electrode; (3) ``tubed_parcels`` padded to a FIXED ``p_fixed`` (not batch-max).
    Those constants make every forward gather a static shape (step B). ``tube_mask``
    = dropped-from-student (whole tubed ∪ residual); ``tubed_parcels`` = whole M4
    targets only."""
    B, C = parcel_per_electrode.shape
    S = sum(b.n_tokens for b in bands)
    targets = m2_fixed_col_targets(bands, m2_cfg)
    P = tube_cfg.p_fixed
    m2 = torch.zeros(B, C, S, dtype=torch.bool)
    tube = torch.zeros(B, C, dtype=torch.bool)
    tubed_parcels = torch.zeros(B, P, dtype=torch.long)
    tubed_parcel_mask = torch.zeros(B, P, dtype=torch.bool)
    for b in range(B):
        real = electrode_mask[b]
        tb, tparc = tight_pack_tube(parcel_per_electrode[b], real, generator, tube_cfg)
        n_t = int(tparc.numel())
        if n_t > P:
            raise ValueError(
                f"tubed parcels {n_t} exceed p_fixed {P}; raise TightPackConfig."
                f"p_fixed (BT max measured 17 at ratio {tube_cfg.ratio})")
        tube[b] = tb
        tubed_parcels[b, :n_t] = tparc
        tubed_parcel_mask[b, :n_t] = True
        keep = real & ~tb
        for e in range(C):
            if bool(keep[e]):
                m2[b, e] = sample_m2_mask_fixed(generator, bands, m2_cfg, targets)
    return {
        "m2_mask": m2,
        "tube_mask": tube,
        "tubed_parcels": tubed_parcels,
        "tubed_parcel_mask": tubed_parcel_mask,
    }


@dataclass(frozen=True)
class StaticShapes:
    """CPU-known gather lengths for the V-JEPA-2 static forward (step B).

    Under the tight-pack tube + rand_unmask M2 + the session-homogeneous batch
    sampler, every per-forward gather length is constant across the batch and the
    session. Passing these as Python ints lets each ragged gather slice to a fixed
    column count — no ``.item()``/``nonzero`` GPU sync, one compiled graph per
    session — and, where the gather has no padding (``real`` all-True), the caller
    drops the key-mask → cuDNN-nomask kernel. Built by :func:`compute_static_shapes`."""

    b: int          # batch size
    c: int          # padded electrode count (token-axis width)
    c_real: int     # real electrodes (teacher encode set)
    n_vis: int      # un-tubed real electrodes (student encode set)
    s: int          # tokens / electrode
    s_vis: int      # visible tokens / kept electrode (S − n_mask)
    n_mask: int     # masked tokens / kept electrode (M2 query count)
    p_fixed: int    # M4 query parcel pad width

    @property
    def teacher_elec_k(self) -> int:
        return self.b * self.c_real          # frontend flattens elec = B·C

    @property
    def student_elec_k(self) -> int:
        return self.b * self.n_vis

    @property
    def latent_nv(self) -> int:
        return self.n_vis * self.s_vis       # visible cross-electrode SEE-set


def compute_static_shapes(
    electrode_mask: Tensor, m2_mask: Tensor, tube_mask: Tensor, p_fixed: int,
) -> StaticShapes:
    """Derive :class:`StaticShapes` from the CPU masks. Fail loud if the batch is
    not session-homogeneous (the counts MUST be uniform across ``B`` for a static
    shape) or if a kept electrode's ``N_mask`` varies (rand_unmask guarantees it
    does not). Cheap CPU reductions — run on the CPU masks before the device move."""
    B, C, S = m2_mask.shape
    c_real_v = electrode_mask.sum(dim=1)                  # (B,)
    n_tube_v = tube_mask.sum(dim=1)                       # (B,)
    keep = electrode_mask & ~tube_mask                    # (B, C) student-kept
    nmask_v = m2_mask.sum(dim=2)                          # (B, C) per electrode
    c_real, n_tube = int(c_real_v[0]), int(n_tube_v[0])
    if not (torch.all(c_real_v == c_real) and torch.all(n_tube_v == n_tube)):
        raise ValueError(
            "static forward needs a session-homogeneous batch (uniform real / "
            f"tubed electrode counts); got real={c_real_v.tolist()} "
            f"tubed={n_tube_v.tolist()}")
    if not bool(keep.any()):
        raise ValueError("static forward: no un-tubed real electrodes in the batch")
    kept_nmask = nmask_v[keep]
    n_mask = int(kept_nmask[0])
    if not torch.all(kept_nmask == n_mask):
        raise ValueError(
            "static forward needs a constant M2 N_mask per kept electrode "
            "(rand_unmask should guarantee this)")
    return StaticShapes(
        b=B, c=C, c_real=c_real, n_vis=c_real - n_tube,
        s=S, s_vis=S - n_mask, n_mask=n_mask, p_fixed=p_fixed,
    )


def probe_pool(feats: Tensor, electrode_mask: Tensor | None = None) -> Tensor:
    """Parameter-free probe feature: masked mean of ``feats (B,C,38,d)`` over the
    real electrodes and their 38 tokens → ``(B, d)``.

    Used for the converged success criterion (post-latent linear probe MUST beat
    post-frontend): applied identically to :meth:`V14ConvergedSSL.encode_frontend`
    and :meth:`V14ConvergedSSL.encode_latent` outputs so the only thing that
    differs between the two probe inputs is the encoder depth, NOT a learned pool.
    """
    if electrode_mask is None:
        return feats.mean(dim=(1, 2))
    S = feats.shape[2]
    w = electrode_mask.to(feats.dtype)                       # (B,C)
    summed = (feats * w[:, :, None, None]).sum(dim=(1, 2))   # (B,d)
    denom = (w.sum(dim=1) * S).clamp_min(1.0)[:, None]       # (B,1)
    return summed / denom
