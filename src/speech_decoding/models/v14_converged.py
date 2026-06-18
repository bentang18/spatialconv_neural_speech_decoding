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

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from speech_decoding.models.v14_encoder import (
    _JointTokenBlock,
    _MultiHeadCrossAttention,
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
    clock** (``time_slot``, the same 8:2:1 grid the frontend uses), so the three
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
        for blk in self.blocks:
            x = blk(x, rope, key_mask)
        return self.ln_out(x).reshape(B, C, S, d)


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
    ) -> Tensor:
        """Predict each tubed parcel's teacher grid from the visible context.

        ``ctx``: ``(B, N, d_model)`` visible un-tubed latent tokens (M4 SEE-set).
        ``ctx_slot``: ``(N,)`` long — each context token's RoPE time-slot.
        ``query_parcels``: ``(B, P)`` long — the tubed parcel ids to predict.
        ``key_mask``: ``(B, N)`` bool, ``True`` = real context token (ragged
        padding); ``None`` ⇒ all real. Output: ``(B, P, 38, d_model)`` predicted
        feature per (tubed parcel, freq-time cell), to be L1'd vs the stop-grad
        teacher ``parcel_electrode_mean`` target."""
        B, N, _ = ctx.shape
        P = query_parcels.shape[1]
        S = self.tokens_per_electrode
        q = self._build_queries(query_parcels).reshape(B, P * S, -1)  # (B, P·38, d)
        c = self.ctx_proj(ctx)                                        # (B, N, d)
        tokens = torch.cat([c, q], dim=1)                             # (B, N+P·38, d)

        q_slot = self.time_slot.repeat(P)                             # (P·38,)
        slots = torch.cat([ctx_slot.reshape(-1), q_slot])             # (N+P·38,)
        rope = self.key_rope[:, slots, :]                             # (2, N+P·38, hd)

        # Context tokens carry the ragged mask; query tokens are always real keys.
        if key_mask is None:
            full_mask: Tensor | None = None
        else:
            q_ok = torch.ones(B, P * S, dtype=torch.bool, device=key_mask.device)
            full_mask = torch.cat([key_mask, q_ok], dim=1)           # (B, N+P·38)

        for blk in self.blocks:
            tokens = blk(tokens, rope, full_mask)
        pred = self.head(tokens[:, N:])                              # (B, P·38, d_model)
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

    Compute note: this batched form key-MASKS the masked/tubed tokens rather than
    physically dropping them (correctness-identical paradigm-B; the used visible
    features match a physical drop). Reclaiming the ~2× by ragged token-drop is a
    perf optimization over this correct baseline, to land in DCC run-prep."""

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
        bands: tuple[BandSpec, ...] = BANDS,
    ) -> None:
        super().__init__()
        self.lambda_m2 = float(lambda_m2)
        self.lambda_m4 = float(lambda_m4)
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
        _, freq_global_id, time_slot = token_metadata(bands)
        self.register_buffer("freq_global_id", freq_global_id, persistent=False)
        self.register_buffer("time_slot", time_slot, persistent=False)

    @torch.no_grad()
    def update_teacher(self, tau: float) -> None:
        """EMA: ``teacher ← τ·teacher + (1−τ)·student`` (frontend only)."""
        for t, s in zip(self.teacher_frontend.parameters(),
                        self.student_frontend.parameters()):
            t.mul_(tau).add_(s.detach(), alpha=1.0 - tau)
        for t, s in zip(self.teacher_frontend.buffers(),
                        self.student_frontend.buffers()):
            t.copy_(s)

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
    ) -> dict[str, Tensor]:
        """Run the one-pass two-head forward; return ``{loss, l_m2, l_m4}``.

        Shapes — ``slow (B,C,2,6,5)`` / ``beta (B,C,6,17)`` / ``hg (B,C,9,33)``;
        ``parcel_per_electrode (B,C)`` long; ``electrode_mask (B,C)`` bool real;
        ``m2_mask (B,C,38)`` bool (True = M2 target, only on un-tubed electrodes);
        ``tube_mask (B,C)`` bool (electrode in a tubed parcel); ``tubed_parcels
        (B,P)`` long + ``tubed_parcel_mask (B,P)`` bool (real tubed parcels)."""
        # ---- teacher: frontend-only, full input, stop-grad ------------------
        with torch.no_grad():
            t_f = self.teacher_frontend(slow, beta, hg)            # (B,C,38,d)
        t_f = t_f.detach()

        # ---- student: one pass, M2-key-masked frontend then latent ----------
        student_vis = ~m2_mask                                     # (B,C,38) visible
        s_f = self.student_frontend(slow, beta, hg, key_mask=student_vis)
        latent_vis = (
            electrode_mask[:, :, None] & (~tube_mask)[:, :, None] & student_vis
        )                                                          # (B,C,38)
        s_l = self.latent(s_f, parcel_per_electrode, token_mask=latent_vis)

        l_m2 = self._m2_loss(s_f, t_f, m2_mask, student_vis, electrode_mask, tube_mask)
        l_m4 = self._m4_loss(
            s_l, t_f, parcel_per_electrode, electrode_mask, latent_vis,
            tubed_parcels, tubed_parcel_mask,
        )
        loss = self.lambda_m2 * l_m2 + self.lambda_m4 * l_m4
        return {"loss": loss, "l_m2": l_m2, "l_m4": l_m4}

    # --------------------------------------------------------------- M2 head
    def _m2_loss(
        self, s_f: Tensor, t_f: Tensor, m2_mask: Tensor, student_vis: Tensor,
        electrode_mask: Tensor, tube_mask: Tensor,
    ) -> Tensor:
        B, C, S, d = s_f.shape
        elec = B * C
        m2_flat = m2_mask.reshape(elec, S)
        max_m = int(m2_flat.sum(dim=1).max().item())
        # un-tubed real electrodes that actually carry an M2 target
        elec_ok = (electrode_mask & ~tube_mask).reshape(elec) & (m2_flat.any(dim=1))
        if max_m == 0 or not bool(elec_ok.any()):
            return s_f.new_zeros(())

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
        return _masked_l1(pred, tgt, qvalid)

    # --------------------------------------------------------------- M4 head
    def _m4_loss(
        self, s_l: Tensor, t_f: Tensor, parcel_per_electrode: Tensor,
        electrode_mask: Tensor, latent_vis: Tensor,
        tubed_parcels: Tensor, tubed_parcel_mask: Tensor,
    ) -> Tensor:
        B, C, S, d = s_l.shape
        if not bool(tubed_parcel_mask.any()):
            return s_l.new_zeros(())

        ctx = s_l.reshape(B, C * S, d)
        ctx_slot = self.time_slot.repeat(C)                        # (C·38,)
        pred = self.m4_predictor(
            ctx, ctx_slot, tubed_parcels.clamp_min(0),
            key_mask=latent_vis.reshape(B, C * S),
        )                                                          # (B, P, 38, d)

        # teacher electrode-MEAN target over each tubed parcel (padded elec → -1)
        pe = torch.where(electrode_mask, parcel_per_electrode,
                         torch.full_like(parcel_per_electrode, -1))
        member = (tubed_parcels[:, :, None] == pe[:, None, :]).to(t_f.dtype)  # (B,P,C)
        counts = member.sum(dim=-1, keepdim=True).clamp_min(1.0)
        tgt = (member @ t_f.reshape(B, C, S * d)) / counts          # (B,P,S·d)
        tgt = tgt.reshape(B, -1, S, d).detach()
        cell_valid = tubed_parcel_mask[:, :, None, None].expand_as(pred)
        return _masked_l1(pred, tgt, cell_valid)


def _masked_l1(pred: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Mean L1 over the ``valid`` entries; 0 if none. ``valid`` is broadcast over
    any trailing feature dims, so the mean is per-element over (valid cells × d)."""
    diff = (pred - target).abs()
    v = valid.to(diff.dtype)
    while v.dim() < diff.dim():
        v = v.unsqueeze(-1)
    v = v.expand_as(diff)
    denom = v.sum().clamp_min(1.0)
    return (diff * v).sum() / denom


def sample_ssl_masks(
    parcel_per_electrode: Tensor,
    electrode_mask: Tensor,
    generator: torch.Generator,
    *,
    m2_cfg: M2MaskConfig = M2MaskConfig(),
    m4_cfg: M4MaskConfig = M4MaskConfig(),
) -> dict[str, Tensor]:
    """Compose the per-clip mask draws for a batch (the data-path bridge).

    Returns ``m2_mask (B,C,38)`` / ``tube_mask (B,C)`` / ``tubed_parcels (B,Pmax)``
    / ``tubed_parcel_mask (B,Pmax)``. Per sample: tube whole parcels (M4), then
    draw an M2 within-electrode mask for every un-tubed real electrode. Tubed and
    padded electrodes get no M2 target. ``Pmax`` is the batch-max tubed count."""
    B, C = parcel_per_electrode.shape
    S = N_TOKENS
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
                m2[b, e] = sample_m2_mask(generator, cfg=m2_cfg)
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
