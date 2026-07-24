"""v14_converged_v3 r4 — plain-JEPA objective on the FLAT per-band path (Design B).

Rewritten 2026-07-15 for r4 (contract project-r4-contract-2026-07-15). The dual-axis
PACKED path (``packing.gather_cells`` / ``build_pack_plan``) is RETIRED: r4 tokens are
RAGGED per (contact, band) and laid out FLAT (``pack_r4.build_r4_grid``). The JEPA
mechanics are unchanged in spirit, only the layout differs:

  online   encoder over the VISIBLE tokens only — masked tokens are physically DROPPED
           from the packed sequence (``pack_r4.build_visible_pack``). The flat
           block-diagonal kernel has NO key-mask, so dropping is the ONLY exclusion and a
           target can never leak into a visible latent (leak-safety proven in test_pack_r4).
  teacher  EMA of the encoder over the FULL grid → deep-sup {3,6,9,12} concat ``[·,1024]``
           targets, each level affine-LN'd (encoder ``norms_block``) then per-level
           affine-free ``_ln_target`` (the DOUBLE norm) + stop-grad.
  predictor  ``enc_to_pred`` (1024→256→128) at visible tokens + a learnable mask-query at
           masked tokens (``scatter_visible``), 6× L1 d128 within-shaft, then
           ``pred_to_target`` (128→1024) predicting ALL 4 teacher levels.
  loss     L1 over the 1024-dim stack at the MARGIN-GATED masked tokens (``token_flags``
           ``in_loss`` — #26: every scored query sits ≥2 tokens from a visible same-band
           frame, so no raw-sample overlap can be photocopied; M14). All 3 bands contribute.

The EMA teacher wraps the WHOLE target path — ``PerBandStem`` AND encoder — matching
V-JEPA 2 (its target encoder EMA-blends the patch-embed too).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from speech_decoding.models.v14_converged_v3.geometry import L1Geometry
from speech_decoding.models.v14_converged_v3.masking import V3Masks
from speech_decoding.models.v14_converged_v3.monitor_taps import (
    early_fusion_recon_stats,
    nofusion_recon_stats,
    per_band_jepa_stats,
)
from speech_decoding.models.v14_converged_v3.pack_r4 import (
    R4Grid,
    VisiblePack,
    build_r4_grid,
    build_r5_grid,
    build_r5nf_grid,
    build_visible_pack,
    pack_band_tokens,
    scatter_visible,
    token_flags,
    token_flags_r5,
    token_flags_r5nf,
    token_flags_r6,
)
from speech_decoding.models.v14_converged_v3.pe import init_transformer_weights
from speech_decoding.models.v14_converged_v3.stem import (
    EARLY_FUSION_BINS,
    EARLY_FUSION_CHANNELS,
    EARLY_FUSION_DECIMATE,
    NOFUSION_BINS,
    NOFUSION_DECIMATE,
    PER_BAND_SPECS,
    EarlyFusionStem,
    FineHgaStem,
    NoFusionStem,
    PerBandStem,
    clock_length_32hz,
    nf_token_geometry,
)
from speech_decoding.models.v14_converged_v3.towers import (
    N_LEVELS,
    PRED_D_MODEL,
    build_encoder,
    build_predictor,
)
from speech_decoding.ssl.ema import EmaTeacher, fixed_ema_schedule, stop_grad

D_MODEL = 256
EMA_TAU = 0.99925  # V-JEPA 2 / 2.1 default (B26 lock)


def _masked_mean_l1(pred: Tensor, tgt: Tensor, weight: Tensor) -> Tensor:
    """Weighted-mean per-token L1 over the (B, total) grid: the |pred-tgt| feature-mean at
    each token, averaged over the tokens ``weight`` selects. ``weight`` is a (B, total) 0/1
    float mask; the denominator is its sum (clamped ≥1 so an all-zero mask is a safe 0). Both
    the primary (masked-token) JEPA term and the V-JEPA-2.1 context term (visible tokens)
    share this exact arithmetic — only the token set (the ``weight``) differs."""
    ae = (pred - tgt).abs().mean(-1)  # (B, total) per-token mean over the feature stack
    return (ae * weight).sum() / weight.sum().clamp(min=1.0)


def _ln_target(t: Tensor, n_levels: int = 1) -> Tensor:
    """Affine-free V-JEPA target norm (matches v14_converged_v2._ln_target).

    Deep-sup (``n_levels>1``): the teacher emits ``n_levels`` concatenated per-level
    chunks (each already through the encoder's affine ``norms_block``), and upstream
    applies a SECOND parameter-free ``F.layer_norm`` to EACH chunk independently
    (`train.py:591-611`) — the DOUBLE norm. Split, LN each, re-cat. ``n_levels==1`` is
    exactly the single-tap affine-free LN over the whole vector."""
    t = t.detach()
    if n_levels == 1:
        return F.layer_norm(t, (t.shape[-1],))
    d = t.shape[-1] // n_levels
    return torch.cat([F.layer_norm(c, (d,)) for c in t.split(d, dim=-1)], dim=-1)


@dataclass
class JepaOutput:
    loss: Tensor  # the masked L1 (JEPA) or MSE (MAE) loss the trainer backprops
    n_masked: Tensor  # 0-dim: margin-gated scored-token count (realization-dependent; sync deferred)
    taps: dict[str, Tensor] | None = None


class _TargetTower(nn.Module):
    """The full target-producing path: ``PerBandStem`` (patch-embed) + encoder.

    This is the unit the EMA teacher mirrors — V-JEPA's target encoder is an EMA of the
    ENTIRE context encoder INCLUDING the patch-embed, so the stem lives INSIDE the EMA'd
    tower, not shared. The predictor is NOT part of this tower (it is the online-only map
    from context latents into target space, with no teacher counterpart in JEPA).

    ``forward`` runs the FULL grid (teacher / ``pack=None``) OR the per-clip VISIBLE subset
    (online / ``pack`` given) through the SAME stem + encoder; the only difference is which
    flat entry point the encoder takes (``forward_flat`` vs ``forward_flat_pack``).
    """

    def __init__(
        self,
        *,
        n_parcels: int,
        deep_sup: bool = True,
        native_fine_hga: bool = False,
        early_fusion: bool = False,
        no_fusion: bool = False,
        nf_decimate: int = NOFUSION_DECIMATE,
    ) -> None:
        super().__init__()
        # native_fine_hga: consume native-rate bands (SLOW 4Hz / MID 16Hz / HGA 128Hz,
        # HGA 4 bins conv-pooled 128→32Hz) instead of arm0's uniform-32Hz PerBandStem.
        # Same (tokens, positions) contract + same 32Hz output lattice, so masking/pack/
        # pe/encoder are byte-identical (memo project-fine-hga-bt-rebake-tasklist-2026-07-21).
        # early_fusion (r5): the Chang 2-stream EarlyFusionStem — HGA⊕LFS → ONE 32Hz token stream
        # (a degenerate single-band grid). no_fusion (v3r5nf): the same 2 streams but NOT fused —
        # NoFusionStem emits TWO 32Hz token streams. At most one is set (mutually exclusive).
        # r6 is NOT a frontend switch: it rides arm0's PerBandStem on the 32 Hz caches verbatim
        # (its deltas are the data regime + the loss, not the stem — see V3JepaObjective).
        if int(early_fusion) + int(native_fine_hga) + int(no_fusion) > 1:
            raise ValueError(
                "early_fusion, native_fine_hga and no_fusion are mutually exclusive"
            )
        if early_fusion:
            self.stem = EarlyFusionStem(D_MODEL)
        elif no_fusion:
            self.stem = NoFusionStem(D_MODEL, decimate=nf_decimate)
        elif native_fine_hga:
            self.stem = FineHgaStem(D_MODEL)
        else:
            self.stem = PerBandStem(D_MODEL)
        self.encoder = build_encoder(n_parcels=n_parcels, deep_sup=deep_sup)

    def forward(
        self,
        bands: Sequence[Tensor],
        grid: R4Grid,
        parcel_packed: Tensor,
        *,
        pack: VisiblePack | None = None,
        tap_blocks: tuple[int, ...] = (),
    ) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
        tokens, _ = self.stem(bands)  # per-band tuple (B, N, T_b, d) on the shared 32 Hz clock
        x_full = pack_band_tokens(tokens, grid)  # (B, total, d) flat grid order
        if pack is None:
            return self.encoder.forward_flat(
                x_full, grid, parcel_packed, tap_blocks=tap_blocks
            )
        d = x_full.shape[-1]
        x_vis = x_full.gather(1, pack.idx[:, :, None].expand(-1, -1, d))  # (B, M_vis, d)
        return self.encoder.forward_flat_pack(x_vis, pack, tap_blocks=tap_blocks)


class V3JepaObjective(nn.Module):
    def __init__(
        self,
        *,
        n_parcels: int,
        target_ln: bool = True,
        ema_tau: float = EMA_TAU,
        deep_sup: bool = True,
        mae: bool = False,
        native_fine_hga: bool = False,
        early_fusion: bool = False,
        no_fusion: bool = False,
        r6: bool = False,
        nf_decimate: int = NOFUSION_DECIMATE,
        mae_stream_weight: str = "equal",
    ) -> None:
        super().__init__()
        self.native_fine_hga = bool(native_fine_hga)
        # r6 (Ben 2026-07-23) = r4 with FOUR deltas and nothing else. The frontend is r4's,
        # VERBATIM: the same 3-band 32 Hz |STFT| caches, the same PerBandStem (which decimates
        # 8/2/1 internally), the same grid, the same per-band recon heads. The deltas:
        #   1. DATA REGIME — shaft-level batching (dispatch_v3, not this class).
        #   2. NO norm_pix on the MAE target (Ben 07-22: "that was dumb"; the r5 arms already
        #      reconstruct a raw target, and per-token renormalisation destroys the amplitude
        #      ratio that makes the band weighting count-proportional).
        #   3. NO margin gate — every masked token is scored (token_flags_r6; Ben 07-23: "no ML
        #      SSL does a margin gate for masked tokens").
        #   4. PER-SENSOR temporal masks + a predictor band embed (below).
        # The name is deliberately just the arm id: the previous flag was called native_perband
        # and asserted a native-rate bake that was never made — a false name that cost four runs
        # (memo project-r6-band-rates-cache-rate-bug-2026-07-23). MAE-only.
        self.r6 = bool(r6)
        # early_fusion (r5, Chang 2-stream): EarlyFusionStem + single-band stride-1 grid
        # (build_r5_grid) + single temporal mask + accept-the-bleed scoring (token_flags_r5,
        # in_loss == masked). Objective stays swappable (JEPA-latent OR MAE-recon).
        self.early_fusion = bool(early_fusion)
        # no_fusion (v3r5nf): r5-fused's fully-separated sibling — NoFusionStem (2 stems) +
        # two-band stride-1 grid (build_r5nf_grid) + INDEPENDENT per-stream masks
        # (sample_masks_r5nf) + accept-the-bleed scoring (token_flags_r5nf). MAE-ONLY (raw-frame
        # per-stream recon, pooled ⇒ 4:1 HGA:LFS); JEPA-nf is deferred.
        self.no_fusion = bool(no_fusion)
        # nf_decimate (v3r5nffast OFAT): NoFusionStem net decimation. 2 = v3r5nf (32 Hz tokens,
        # byte-identical); 4 = fast (16 Hz tokens). Sets the stem stride, the per-stream MAE head
        # widths (DEC·bins) and the target-gather frame count — all keyed off self.nf_decimate.
        self.nf_decimate = int(nf_decimate)
        if (
            int(self.early_fusion) + int(self.native_fine_hga)
            + int(self.no_fusion) + int(self.r6) > 1
        ):
            raise ValueError(
                "early_fusion, native_fine_hga, no_fusion and r6 are mutually exclusive"
            )
        # v3r5nf loss weighting (Ben 07-22): "equal" (1:1, NEW DEFAULT) averages the two
        # per-stream MEAN-MSEs ⇒ each stream 50% of the gradient regardless of feat count,
        # invariant to HGA's arbitrary bin granularity; "pooled" (4:1) sums per-channel SE over
        # both streams ⇒ HGA:LFS = 4:1 (matches r5-fused). Only bites the no_fusion MAE path.
        if mae_stream_weight not in ("equal", "pooled"):
            raise ValueError(f"mae_stream_weight must be equal|pooled, got {mae_stream_weight!r}")
        self.mae_stream_weight = mae_stream_weight
        # MAE arm (Masked Autoencoder target, He 2021 / AudioMAE): the ONLY change vs the
        # JEPA arm is the prediction TARGET — reconstruct this token's OWN norm_pix'd input
        # |STFT| bins instead of the EMA-teacher latent. No EMA teacher. Everything upstream
        # (visible-only encoder, predictor, mask query, margin-gated in_loss, all locked HPs)
        # is byte-identical to the JEPA arm.
        self.mae = bool(mae)
        # v3r5nf is MAE-only (per-stream raw-frame reconstruction); JEPA-nf is deferred.
        if self.no_fusion and not self.mae:
            raise ValueError("v3r5nf is MAE-only")
        # r6 is MAE-only (spectral |STFT|-bin reconstruction, no norm_pix); JEPA-r6 not in scope.
        if self.r6 and not self.mae:
            raise ValueError("r6 is MAE-only")
        self.deep_sup = bool(deep_sup)
        self.n_levels = N_LEVELS if self.deep_sup else 1
        # online target path (stem + encoder) — every param here is EMA-mirrored (JEPA arm).
        self.online = _TargetTower(
            n_parcels=n_parcels, deep_sup=self.deep_sup,
            native_fine_hga=self.native_fine_hga,
            early_fusion=self.early_fusion,
            no_fusion=self.no_fusion,
            nf_decimate=self.nf_decimate,
        )
        # EMA teacher exists ONLY on the JEPA arm; MAE reconstructs the raw input, no teacher.
        self.teacher = (
            None
            if self.mae
            else EmaTeacher(self.online, coeff_schedule=fixed_ema_schedule(tau=ema_tau))
        )
        self.predictor = build_predictor(n_parcels=n_parcels)
        # Encoder→predictor input map. Deep-sup: the encoder emits n_levels concatenated
        # levels, so this is upstream's ``predictor_embed`` 2-layer fusion MLP
        # (Linear(n_levels·d → d)·GELU·Linear(d → d_pred), NO LayerNorm). Single-tap: a
        # plain Linear(d → d_pred). ``pred_to_target`` (upstream ``predictor_proj``) is ONE
        # wide Linear emitting all n_levels·d target dims from the predictor's final block.
        target_dim = N_LEVELS * D_MODEL if self.deep_sup else D_MODEL
        if self.deep_sup:
            self.enc_to_pred = nn.Sequential(
                nn.Linear(N_LEVELS * D_MODEL, D_MODEL),
                nn.GELU(),
                nn.Linear(D_MODEL, PRED_D_MODEL),
            )
        else:
            self.enc_to_pred = nn.Linear(D_MODEL, PRED_D_MODEL)
        self.enc_to_pred.apply(init_transformer_weights)  # V-JEPA 2 trunc_normal(0.02)
        if self.mae:
            self.pred_to_target = None
            if self.no_fusion:
                # v3r5nf MAE decoder = TWO independent Linear heads, one per SEPARATED stream:
                # HGA reconstructs its own DEC(2)·4 raw 64 Hz values, LFS its own DEC(2)·1. No
                # per-token norm_pix; the loss POOLS per-channel MSE over all masked tokens both
                # streams ⇒ HGA:LFS = 4:1 in expectation (equal masked counts × 8 vs 2 feats).
                self.mae_head_hga = nn.Linear(PRED_D_MODEL, self.nf_decimate * NOFUSION_BINS[0])
                self.mae_head_lfs = nn.Linear(PRED_D_MODEL, self.nf_decimate * NOFUSION_BINS[1])
                init_transformer_weights(self.mae_head_hga)
                init_transformer_weights(self.mae_head_lfs)
                self.mae_head_r5 = None
                self.mae_heads = None
                # per-stream band identity ADDED to the mask QUERY (predictor space, d_pred =
                # PRED_D_MODEL) — a SEPARATE param from the stem's 256-dim band_type_emb. Lets
                # the predictor tell apart two co-located masked queries (HGA vs LFS at the same
                # RoPE position) that must predict different targets.
                self.mask_band_emb = nn.Parameter(torch.empty(2, PRED_D_MODEL))
                nn.init.trunc_normal_(self.mask_band_emb, std=0.02)
            elif self.early_fusion:
                # r5 MAE decoder_pred = ONE Linear(d_pred → DEC·C) reconstructing this token's
                # OWN 2 raw 64 Hz frames × 5 fused channels (10 values). Plain per-channel MSE
                # on the load-time z-scored channels ⇒ 4:1 HGA:LFS by construction, NO per-token
                # norm_pix (norm_pix would re-normalise per token and destroy the count=ratio).
                # Linear (He-2022: the predictor transformer IS the nonlinear decoder, the head a
                # linear readout); an MLP head is a flagged OFAT if linear underperforms.
                self.mae_head_r5 = nn.Linear(
                    PRED_D_MODEL, EARLY_FUSION_DECIMATE * EARLY_FUSION_CHANNELS
                )
                self.mae_heads = None
                init_transformer_weights(self.mae_head_r5)
            else:
                # MAE decoder_pred = per-band reconstruction heads, one Linear(d_pred → F_b) per
                # band (SLOW 7 / MID 6 / HGA 7) — the transpose-twin of PerBandStem's per-band
                # INPUT projections. Fed by the predictor's terminal norm_out (= MAE
                # decoder_norm), each emits its band's own |STFT| bins; no wide teacher target.
                self.mae_head_r5 = None
                self.mae_heads = nn.ModuleList(
                    nn.Linear(PRED_D_MODEL, nb) for nb, _ in PER_BAND_SPECS
                )
                self.mae_heads.apply(init_transformer_weights)
        else:
            self.pred_to_target = nn.Linear(PRED_D_MODEL, target_dim)
            init_transformer_weights(self.pred_to_target)
        # r6: per-BAND identity added to EVERY predictor token, visible and masked alike (Ben
        # 2026-07-23: "all tokens in predictor get +parcel embed and +freq embed — same as before
        # the first L1 block"). The parcel half already holds for every arm (towers._run_flat adds
        # parcel_embed to the predictor's input exactly as it does the encoder's); the band half
        # did not, and r4's grid makes that a real degeneracy: SLOW token j sits at lattice 8j,
        # MID at 2j, HGA at j, so at every lattice position divisible by 8 the three co-located
        # tokens of one contact share depth, time_pos AND parcel. Three masked queries there were
        # byte-identical predictor inputs required to reconstruct three DIFFERENT bands' bins —
        # only the per-band output head could tell them apart, and the predictor stack itself
        # could not. This is the r4/r6 analogue of no_fusion's mask_band_emb: predictor space
        # (d_pred), separate from the stem's 256-dim band_type_emb, standard 0.02 init.
        self.pred_band_emb = (
            nn.Parameter(torch.empty(len(PER_BAND_SPECS), PRED_D_MODEL)) if self.r6 else None
        )
        if self.pred_band_emb is not None:
            nn.init.trunc_normal_(self.pred_band_emb, std=0.02)
        # Learnable mask query, zero-init (V-JEPA-2.1 audit). Stored 3-D (1, 1, D) to match
        # upstream: the shared ndim<=1 no-decay rule then DECAYS it (a 1-D store would
        # silently exempt it). Broadcasts identically in scatter_visible (a no-op there).
        self.mask_token = nn.Parameter(torch.zeros(1, 1, PRED_D_MODEL))
        self.target_ln = target_ln

    def forward(
        self,
        bands: Sequence[Tensor],
        geom: L1Geometry,
        parcel_id: Tensor,
        masks: V3Masks,
        *,
        collect_taps: bool = False,
        grid_max_seqlen: int | None = None,
        m_vis: int | None = None,
        pack_max_seqlen: int | None = None,
    ) -> JepaOutput:
        """``bands``: 3-band |STFT| inputs, ``bands[b]`` (B, N, F_b, T) on the shared 32 Hz
        clock (SLOW, MID, HGA). ``masks`` (``V3Masks``): per-band temporal masks
        (``slow_mask``/``mid_mask``/``hga_mask``) + spatial ``contact_mask`` (B, N).

        FLAT r4 path: the online encoder runs the VISIBLE tokens (masked dropped), the EMA
        teacher + predictor the FULL flat grid. ``token_flags`` gives the predictor-query
        set (``masked``) and the margin-gated scored set (``in_loss`` ⊆ ``masked``); both
        counts are per-session CONSTANTS (masking.py balances them), so every packed shape
        is static and the graph never recompiles.

        ``collect_taps`` (monitor cadence): also return the detached encoder block-12
        VISIBLE-cell tap (rankme / feat_std).
        """
        # r6 rides arm0's 32 Hz caches + PerBandStem, so it takes the DEFAULT clock branch —
        # no frontend flag of its own (the r6 deltas are the regime and the loss, not the stem).
        T = clock_length_32hz(
            bands, native_fine_hga=self.native_fine_hga, early_fusion=self.early_fusion,
            no_fusion=self.no_fusion,
        )  # 32 Hz clock
        # grid_max_seqlen / m_vis / pack_max_seqlen are per-session Python-int shape constants
        # (the module caches them via ``V3ConvergedModel.session_plan`` and passes them in);
        # supplying them skips the per-step ``.item()`` host syncs that otherwise break the
        # compiled graph. None ⇒ derive them here (one sync each — the eager/standalone path).
        # r5 (early_fusion): single-band stride-1 grid + accept-the-bleed flags (in_loss==masked).
        if self.early_fusion:
            grid = build_r5_grid(geom, n_time=T, max_seqlen=grid_max_seqlen)
            parcel_packed = parcel_id[grid.contact]
            masked, in_loss = token_flags_r5(grid, masks)  # (B, total) bool; in_loss == masked
        elif self.no_fusion:
            # nf token count = T (decimate 2) or T/2 (decimate 4, fast); time_stride places 16 Hz
            # tokens on the 32 Hz lattice (RoPE physical). Masks were sampled with the SAME n_tok
            # (model.forward derives it identically), so the grid/mask never disagree.
            n_tok, time_stride = nf_token_geometry(T, decimate=self.nf_decimate)
            grid = build_r5nf_grid(
                geom, n_time=n_tok, time_stride=time_stride, max_seqlen=grid_max_seqlen
            )
            parcel_packed = parcel_id[grid.contact]
            masked, in_loss = token_flags_r5nf(grid, masks)  # (B, total); in_loss == masked
        elif self.r6:
            # r6: r4's grid VERBATIM (build_r4_grid) + PER-SENSOR band masks, NO margin gate
            # (token_flags_r6 returns in_loss == masked — every masked token is scored).
            grid = build_r4_grid(geom, n_time=T, max_seqlen=grid_max_seqlen)
            parcel_packed = parcel_id[grid.contact]
            masked, in_loss = token_flags_r6(grid, masks)  # (B, total); in_loss == masked
        else:
            grid = build_r4_grid(geom, n_time=T, max_seqlen=grid_max_seqlen)
            parcel_packed = parcel_id[grid.contact]  # (total,) long
            masked, in_loss = token_flags(grid, masks)  # (B, total) bool each
        pack = build_visible_pack(
            grid, masked, parcel_packed, m_vis=m_vis, max_seqlen=pack_max_seqlen
        )

        # online encoder over the VISIBLE tokens (masked physically dropped) → (B, M_vis, 1024)
        if collect_taps:
            z, enc_taps = self.online(
                bands, grid, parcel_packed, pack=pack, tap_blocks=(12,)
            )
        else:
            z = self.online(bands, grid, parcel_packed, pack=pack)

        # EMA teacher over the FULL grid → deep-sup targets, per-level double-norm + stop-grad.
        # MAE has no teacher (target = the token's OWN norm_pix'd input, built in _mae_output);
        # skip the teacher forward entirely so ``tgt`` never touches the None teacher.
        tgt = None
        if not self.mae:
            with torch.no_grad():
                tgt = self.teacher(bands, grid, parcel_packed)  # (B, total, n_levels·256)
            tgt = _ln_target(tgt, self.n_levels) if self.target_ln else stop_grad(tgt)

        # predictor input: enc_to_pred(z) at visible tokens, mask-query at masked tokens.
        pred_in = scatter_visible(
            self.enc_to_pred(z), pack.idx, grid.total, self.mask_token
        )  # (B, total, 128)
        # v3r5nf: HGA and LFS tokens at the SAME (contact, 32Hz-pos) share IDENTICAL RoPE
        # (same time_pos + depth), so two co-located masked queries would be byte-identical
        # predictor inputs forced to predict DIFFERENT targets. A per-stream band identity
        # (predictor space, d_pred) breaks that. Added to ALL tokens (visible + masked),
        # symmetric with the parcel embed (towers.py:279) and matching the MAE-decoder /
        # I-JEPA-predictor convention of adding position/identity embeds to context AND mask
        # tokens alike. r4/r5 arms never enter this branch (no mask_band_emb).
        if self.no_fusion:
            band_q = self.mask_band_emb[grid.band]  # (total, PRED_D_MODEL)
            pred_in = pred_in + band_q[None]  # all tokens (visible + masked)
        elif self.pred_band_emb is not None:  # r6: same fix for the 3-band grid (see __init__)
            pred_in = pred_in + self.pred_band_emb[grid.band][None]  # (B, total, d_pred)
        # forward_flat adds the parcel embed to every token itself (towers._run_flat), so with the
        # band embed above the predictor now sees parcel + band on ALL tokens — the same identity
        # pair the encoder gets before its first L1 block.
        h = self.predictor.forward_flat(pred_in, grid, parcel_packed)  # (B, total, 128)
        if self.mae:
            if self.no_fusion:
                return self._mae_output_r5nf(
                    bands, grid, h, in_loss,
                    enc_taps=enc_taps if collect_taps else None,
                    collect_taps=collect_taps,
                    # per-VISIBLE-token stream label (0=HGA 1=LFS), aligned to enc_taps[12]'s
                    # (B, M_vis) axis, so the monitor can split enc12 rankme/feat_std by stream.
                    band_vis=grid.band[pack.idx] if collect_taps else None,
                )
            if self.early_fusion:
                return self._mae_output_r5(
                    bands, grid, h, in_loss,
                    enc_taps=enc_taps if collect_taps else None,
                    collect_taps=collect_taps,
                )
            return self._mae_output(
                bands, grid, h, in_loss,
                enc_taps=enc_taps if collect_taps else None,
                collect_taps=collect_taps,
                norm_pix=not self.r6,  # r6's ONLY loss-side delta from r4
            )
        assert self.pred_to_target is not None  # JEPA arm always builds it (mae returned above)
        pred = self.pred_to_target(h)  # (B, total, n_levels·256)

        # L1 at the MARGIN-GATED masked tokens (in_loss). STATIC weighted mean over the
        # fixed (B, total) grid — every scored token shares the same feature width, so this
        # equals the boolean-indexed mean with a shape-static graph (in_loss count is a
        # per-session constant ⇒ the denominator needs no host sync). §5: reuse the tested
        # deep-sup L1 unchanged (pure |·|, NOT smooth-L1 — the DATA-FLOW one-liner's wording).
        w = in_loss.to(pred.dtype)  # (B, total)
        jepa_loss = _masked_mean_l1(pred, tgt, w)

        taps = None
        if collect_taps:
            d_enc = enc_taps[12].shape[-1]
            taps = {"enc12": enc_taps[12].detach().reshape(-1, d_enc)}  # (B·M_vis, 256)
            # per-band JEPA health (#40) — reduce the GB-scale pred/tgt to scalars HERE
            # (they never leave the objective as raw taps). no_grad: monitor-only, must not
            # extend the backward graph the loss above already built.
            with torch.no_grad():
                taps.update(per_band_jepa_stats(pred, tgt, w, grid.band))
        return JepaOutput(
            loss=jepa_loss,
            # 0-dim tensor, NOT an int: the scored-token count is realization-dependent
            # (the margin gate varies per clip — NOT a session constant, so it cannot be
            # cached/passed like m_vis). Returning the tensor defers the host sync to the
            # logger's own cadence instead of forcing one inside the compiled forward each
            # step; consumers ``.item()``/log it lazily.
            n_masked=w.sum().detach(),
            taps=taps,
        )

    # ── MAE arm (Masked Autoencoder target) ──────────────────────────────────────
    def _mae_output(
        self,
        bands: Sequence[Tensor],
        grid: R4Grid,
        h: Tensor,
        in_loss: Tensor,
        *,
        enc_taps: dict[int, Tensor] | None,
        collect_taps: bool,
        norm_pix: bool = True,
    ) -> JepaOutput:
        """MAE loss: reconstruct each masked token's OWN input |STFT| bins.

        Exact He-2021 / AudioMAE recipe — per-token target normalization
        ``(x-mean)/sqrt(var+1e-6)`` (unbiased var) over that token's F_b bins, MSE
        ``(pred-target)^2`` meaned over the bins, then masked-mean over the SAME
        ``in_loss`` tokens the JEPA arm scores. ``h`` (B, total, d_pred) is the
        predictor output AFTER its terminal ``norm_out`` (= MAE ``decoder_norm``).

        ``norm_pix`` (r6 = False, Ben 07-22 "that was dumb"): skip the per-token target
        renormalization and reconstruct the RAW load-time robust-z'd bins, as every r5 arm
        already does. Renormalizing per token rescales each band to unit variance, which
        destroys the amplitude ratio the bands actually carry and leaves the loss weighted
        purely by token count; without it the count-proportional weighting (HGA ~8:1 over
        SLOW, by token count) rides on the true amplitudes."""
        target, feat_valid, feat_count = self._mae_gather_target(bands, grid)
        if norm_pix:
            target = self._norm_pix(target.float(), feat_valid, feat_count)  # fp32 norm_pix
        pred = self._mae_pred(h, grid)  # (B, total, F_MAX); pad slot = 0
        target = target.to(pred.dtype)
        w = in_loss.to(pred.dtype)  # (B, total) masked weight
        fv = feat_valid[None].to(pred.dtype)  # (1, total, F_MAX)
        # per-token MSE over that token's own valid bins (pad excluded), then masked-mean.
        se_tok = (((pred - target) ** 2) * fv).sum(-1) / feat_count[None].to(pred.dtype)
        mae_loss = (se_tok * w).sum() / w.sum().clamp(min=1.0)

        taps = None
        if collect_taps:
            assert enc_taps is not None
            d_enc = enc_taps[12].shape[-1]
            taps = {"enc12": enc_taps[12].detach().reshape(-1, d_enc)}  # (B·M_vis, 256)
            with torch.no_grad():  # per-band recon health; pad zeroed so it can't pollute
                taps.update(per_band_jepa_stats(pred, target * fv, w, grid.band))
        return JepaOutput(
            loss=mae_loss,
            n_masked=w.sum().detach(),
            taps=taps,
        )

    # ── r5 MAE arm (Chang 2-stream raw-frame reconstruction) ─────────────────────
    def _mae_output_r5(
        self,
        bands: Sequence[Tensor],
        grid: R4Grid,
        h: Tensor,
        in_loss: Tensor,
        *,
        enc_taps: dict[int, Tensor] | None,
        collect_taps: bool,
    ) -> JepaOutput:
        """r5 MAE loss: reconstruct each masked token's OWN 2 raw 64 Hz frames × 5 channels.

        Target = the DEC(2)·C(5)=10 fused input values that stride-2 the stem pooled into this
        32 Hz token (already load-time per-(elec,bin)/(elec) robust-z'd). Plain per-channel MSE
        over the 10 values — NO per-token norm_pix, so the 4 HGA bins vs 1 LFS channel keep
        their amplitude ratio ⇒ 4:1 HGA:LFS by construction (memo). Masked-mean over the
        accept-the-bleed ``in_loss`` set (== masked). ``h`` (B, total, d_pred) is the predictor
        output AFTER its terminal ``norm_out`` (= MAE decoder_norm)."""
        assert self.mae_head_r5 is not None  # r5 MAE always builds it
        target = self._mae_gather_target_r5(bands, grid)  # (B, total, DEC·C)
        pred = self.mae_head_r5(h)  # (B, total, DEC·C)
        target = target.to(pred.dtype)
        w = in_loss.to(pred.dtype)  # (B, total) accept-the-bleed masked weight
        se = (pred - target) ** 2  # (B, total, DEC·C)
        se_tok = se.mean(-1)  # (B, total) per-token MSE over the 10 values
        mae_loss = (se_tok * w).sum() / w.sum().clamp(min=1.0)

        taps = None
        if collect_taps:
            assert enc_taps is not None
            d_enc = enc_taps[12].shape[-1]
            taps = {"enc12": enc_taps[12].detach().reshape(-1, d_enc)}  # (B·M_vis, 256)
            with torch.no_grad():
                # r5 recon health split by the early-fused CHANNEL groups: explained-var +
                # pred-target var-ratio for HGA (4 |STFT| bins combined) vs LFS (1 raw) — the r5
                # replacement for r4's per-band {slow,mid,hga} (single band ⇒ that split is here).
                taps.update(early_fusion_recon_stats(
                    pred, target, w, n_hga=EARLY_FUSION_BINS[0], decimate=EARLY_FUSION_DECIMATE
                ))
                # HGA-vs-LFS recon split (LOG-ONLY monitor). The DEC·C target lays out each of
                # the DEC frames as [HGA₀..₃, LFS] (EARLY_FUSION_BINS = (4, 1)), so channels
                # [:4] are the 4 HGA |STFT| bins (combined) and [4:] the LFS raw. Each stream's
                # tap is its CONTRIBUTION to the training loss — the group's squared errors summed
                # and divided by the FULL DEC·C (not the group count), so the displayed split adds
                # up to the optimized loss. Invariant: mae_hga + mae_lfs == mae_loss (the 4:1
                # HGA:LFS dominance is visible directly in the two summands, not a hidden reweight).
                n_hga = EARLY_FUSION_BINS[0]
                se_f = se.reshape(
                    se.shape[0], se.shape[1], EARLY_FUSION_DECIMATE, EARLY_FUSION_CHANNELS
                )
                denom = w.sum().clamp(min=1.0)
                dc = EARLY_FUSION_DECIMATE * EARLY_FUSION_CHANNELS  # 10 = training-loss denominator
                hga_c = se_f[..., :n_hga].sum((-2, -1)) / dc  # HGA share of the 10-value token mean
                lfs_c = se_f[..., n_hga:].sum((-2, -1)) / dc  # LFS share
                taps["mae_hga"] = (hga_c * w).sum() / denom
                taps["mae_lfs"] = (lfs_c * w).sum() / denom
        return JepaOutput(
            loss=mae_loss,
            n_masked=w.sum().detach(),
            taps=taps,
        )

    def _mae_gather_target_r5(
        self, bands: Sequence[Tensor], grid: R4Grid
    ) -> Tensor:
        """Gather each flat token's OWN 2 raw 64 Hz frames × 5 fused channels → (B, total, DEC·C).

        ``bands`` = (HGA (B,N,4,L), LFS (B,N,1,L)) at 64 Hz frames (L = 2T), already robust-z'd
        at load. A token of (contact c, time_pos p) pooled 64→32 Hz from frames {DEC·p, DEC·p+1}
        (stem stride-2, ``time_pos == 32 Hz index`` on the r5 grid). Stack the 2 streams to 5
        channels, flatten (contact, frame) into one axis, and index with ONE static linear index
        per frame offset — no boolean/data-dependent gather (compile-safe). Value order per token:
        [frame0 ch0..4, frame1 ch0..4] (== the head's DEC·C output layout)."""
        fused = torch.cat([bands[0], bands[1]], dim=2)  # (B, N, C=5, L)
        B, N, C, L = fused.shape
        ft = fused.permute(0, 1, 3, 2).reshape(B, N * L, C)  # (B, N·L, 5)
        lin0 = grid.contact * L + EARLY_FUSION_DECIMATE * grid.time_pos  # (total,) frame DEC·p
        frames = [ft[:, lin0 + f, :] for f in range(EARLY_FUSION_DECIMATE)]  # DEC × (B, total, 5)
        return torch.stack(frames, dim=2).reshape(B, grid.total, EARLY_FUSION_DECIMATE * C)

    # ── v3r5nf MAE arm (no-fusion per-stream raw-frame reconstruction) ────────────
    def _mae_gather_target_r5nf(
        self, bands: Sequence[Tensor], grid: R4Grid
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Gather each token's OWN DEC(2) raw 64 Hz frames of its OWN stream → padded
        (B, total, F_MAX=8) + (feat_valid, feat_count).

        ``bands`` = (HGA (B,N,4,L), LFS (B,N,1,L)) at 64 Hz frames (L = 2T), robust-z'd at
        load. Both streams computed over ALL tokens then selected by band (compile-safe, no
        data-dependent gather), mirroring ``_mae_gather_target_r5``'s linear-index gather. A
        token of (contact c, bandpos p) pooled 64→32 Hz from frames {DEC·p, DEC·p+1}. Value
        layout per token = [frame0 ch0..C-1, frame1 ch0..C-1] (matches the head output). LFS is
        zero-padded from DEC·1=2 to F_MAX=8; ``feat_valid`` marks each token's real feats,
        ``feat_count`` = 8 (HGA) or 2 (LFS)."""
        DEC = self.nf_decimate  # 2 (v3r5nf) or 4 (fast); L = DEC·n_tok, so DEC·bandpos covers all L
        hga = bands[0]
        B, N, Ch_h, L = hga.shape  # Ch_h = 4
        hga_ft = hga.permute(0, 1, 3, 2).reshape(B, N * L, Ch_h)  # (B, N·L, 4)
        lin_h = grid.contact * L + DEC * grid.bandpos  # (total,) frame DEC·p
        hga_tgt = torch.stack(
            [hga_ft[:, lin_h + f, :] for f in range(DEC)], dim=2
        ).reshape(B, grid.total, DEC * Ch_h)  # (B, total, 8)

        lfs = bands[1]
        Ch_l = lfs.shape[2]  # 1
        lfs_ft = lfs.permute(0, 1, 3, 2).reshape(B, N * L, Ch_l)  # (B, N·L, 1)
        lfs_tgt = torch.stack(
            [lfs_ft[:, lin_h + f, :] for f in range(DEC)], dim=2
        ).reshape(B, grid.total, DEC * Ch_l)  # (B, total, 2)
        lfs_tgt = F.pad(lfs_tgt, (0, DEC * Ch_h - DEC * Ch_l))  # → (B, total, 8)

        is_hga = (grid.band == 0)[None, :, None]  # (1, total, 1)
        target = torch.where(is_hga, hga_tgt, lfs_tgt)  # (B, total, 8)
        f_max = DEC * Ch_h  # 8
        feat_count = torch.where(
            grid.band == 0,
            torch.full_like(grid.band, DEC * Ch_h),
            torch.full_like(grid.band, DEC * Ch_l),
        )  # (total,) 8 or 2
        feat_valid = (
            torch.arange(f_max, device=grid.band.device)[None, :] < feat_count[:, None]
        )  # (total, 8)
        return target, feat_valid, feat_count

    def _mae_pred_r5nf(self, h: Tensor, grid: R4Grid) -> Tensor:
        """Per-stream reconstruction heads → (B, total, 8), selected per token by band.

        Runs BOTH heads over every token (static shape) and selects by ``grid.band`` — the
        compile-safe analogue of a per-stream boolean gather. LFS output (DEC·1=2) is zero-
        padded to F_MAX=8 (the pad slots masked out of the loss by ``feat_valid``)."""
        o_hga = self.mae_head_hga(h)  # (B, total, DEC·4)
        o_lfs = F.pad(self.mae_head_lfs(h), (0, self.nf_decimate * NOFUSION_BINS[0]
                                             - self.nf_decimate * NOFUSION_BINS[1]))  # → DEC·4
        return torch.where((grid.band == 0)[None, :, None], o_hga, o_lfs)  # (B, total, DEC·4)

    def _mae_output_r5nf(
        self,
        bands: Sequence[Tensor],
        grid: R4Grid,
        h: Tensor,
        in_loss: Tensor,
        *,
        enc_taps: dict[int, Tensor] | None,
        collect_taps: bool,
        band_vis: Tensor | None = None,
    ) -> JepaOutput:
        """v3r5nf MAE loss: reconstruct each masked token's OWN DEC(2) raw 64 Hz frames of its
        OWN stream (HGA 8 feats, LFS 2). Accept-the-bleed ``in_loss`` (== masked).

        ``mae_stream_weight`` (Ben 07-22):
        - ``equal`` (1:1, DEFAULT): average the two per-stream MEAN-MSEs (HGA = mean over its 8
          feats then mean over masked HGA tokens; LFS likewise over its 2) ⇒ each stream is 50%
          of the loss, invariant to HGA's arbitrary bin granularity, lifting LFS 20%→50%.
        - ``pooled`` (4:1): SUM per-channel SE over all masked tokens both streams / total valid
          channel count ⇒ HGA:LFS = 4:1 (matches r5-fused, 8 vs 2 valid feats at equal counts)."""
        target, feat_valid, feat_count = self._mae_gather_target_r5nf(bands, grid)
        pred = self._mae_pred_r5nf(h, grid)  # (B, total, 8); LFS pad slots = 0
        target = target.to(pred.dtype)
        w = in_loss.to(pred.dtype)  # (B, total) accept-the-bleed
        fv = feat_valid[None].to(pred.dtype)  # (1, total, 8)
        se_valid = (((pred - target) ** 2) * fv).sum(-1)  # (B, total) SUM over each token's feats
        is_hga = (grid.band == 0).to(pred.dtype)[None]  # (1, total)
        is_lfs = (grid.band == 1).to(pred.dtype)[None]  # (1, total)
        w_hga, w_lfs = w * is_hga, w * is_lfs  # (B, total) per-stream masked weights
        # per-stream mean-MSE: each token's SE ÷ its own feat_count (mean over its valid feats),
        # then weighted-mean over that stream's masked tokens. Static shapes (no boolean gather).
        tok_mean = se_valid / feat_count[None].to(pred.dtype)  # (B, total) mean over valid feats
        mae_hga = (tok_mean * w_hga).sum() / w_hga.sum().clamp(min=1.0)
        mae_lfs = (tok_mean * w_lfs).sum() / w_lfs.sum().clamp(min=1.0)
        if self.mae_stream_weight == "equal":
            mae_loss = 0.5 * (mae_hga + mae_lfs)  # 1:1, each stream 50%
        else:  # pooled ⇒ 4:1 HGA:LFS by construction
            denom = (feat_count[None].to(pred.dtype) * w).sum().clamp(min=1.0)
            mae_loss = (se_valid * w).sum() / denom

        taps = None
        if collect_taps:
            assert enc_taps is not None
            d_enc = enc_taps[12].shape[-1]
            taps = {"enc12": enc_taps[12].detach().reshape(-1, d_enc)}  # (B·M_vis, 256)
            with torch.no_grad():
                # per-VISIBLE-token stream label aligned to the flattened enc12 rows (static
                # shape — NO boolean gather here, so the compiled collect-taps graph stays
                # shape-stable). The eager health monitor does the boolean split + per-stream
                # SVD off-graph: is r5nf's low pooled rankme the between-stream ID axis, or does
                # each stream's OWN spectrum also sit low? (0=HGA 1=LFS, matches grid.band.)
                if band_vis is not None:
                    taps["enc12_band"] = band_vis.detach().reshape(-1)  # (B·M_vis,)
                # log-only per-stream mean-MSE (the SAME per-token means, independent of the loss
                # mode): reports each stream's raw reconstruction quality for the health monitor.
                taps["mae_hga"] = mae_hga.detach()
                taps["mae_lfs"] = mae_lfs.detach()
                # per-stream explained-var + pred-target var-ratio (collapse detector) — the nf
                # analogue of r5-fused's early_fusion_recon_stats, split by TOKEN-band and sliced
                # to each stream's own valid feats (LFS 2, HGA 8) so pad zeros don't deflate it.
                taps.update(nofusion_recon_stats(
                    pred, target, w_hga, w_lfs,
                    n_hga=self.nf_decimate * NOFUSION_BINS[0],
                    n_lfs=self.nf_decimate * NOFUSION_BINS[1],
                ))
        return JepaOutput(
            loss=mae_loss,
            n_masked=w.sum().detach(),
            taps=taps,
        )

    def _mae_gather_target(
        self, bands: Sequence[Tensor], grid: R4Grid
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Gather each flat token's own band |STFT| bins into a padded (B, total, F_MAX)
        tensor via ONE static index — no boolean/data-dependent gather (compile-safe).

        ``bands[b]`` (B, N, F_b, T32), already per-(elec,bin) robust-z'd at load (the stem's
        input). A token of (band, contact, time_pos) wants ``bands[band][:, contact, :,
        time_pos]`` — ``time_pos`` IS the T32 clock index of that decimated frame
        (bandpos·stride). Pad each band's freq to F_MAX, stack band-major, index with
        ``lin = (band·N + contact)·T32 + time_pos``. Returns (target, feat_valid, feat_count):
        ``feat_valid`` (total, F_MAX) bool marks each token's real bins (MID pad = the F_MAX-th
        slot), ``feat_count`` (total,) = F_b per token."""
        B, N = bands[0].shape[0], bands[0].shape[1]
        T = bands[0].shape[-1]
        f_max = max(nb for nb, _ in PER_BAND_SPECS)
        padded: list[Tensor] = []
        for b, (nb, _) in enumerate(PER_BAND_SPECS):
            xb = bands[b].transpose(-1, -2)  # (B, N, T, F_b)
            if nb < f_max:
                xb = F.pad(xb, (0, f_max - nb))  # zero-pad freq to F_MAX
            padded.append(xb)
        stack = torch.stack(padded, dim=1)  # (B, N_BANDS, N, T, F_MAX)
        flat = stack.reshape(B, len(PER_BAND_SPECS) * N * T, f_max)
        lin = (grid.band * N + grid.contact) * T + grid.time_pos  # (total,)
        target = flat[:, lin, :]  # (B, total, F_MAX)
        device = grid.band.device
        f_by_band = torch.tensor([nb for nb, _ in PER_BAND_SPECS], device=device)
        feat_count = f_by_band[grid.band]  # (total,)
        feat_valid = torch.arange(f_max, device=device)[None, :] < feat_count[:, None]
        return target, feat_valid, feat_count

    def _mae_pred(self, h: Tensor, grid: R4Grid) -> Tensor:
        """Per-band reconstruction heads → (B, total, F_MAX), selected per token by band.

        Runs all 3 heads over every token (static shape) and one-hot-selects by ``grid.band``
        — the compile-safe analogue of a per-band boolean gather. MID's F_MAX-th slot is
        zero-pad (masked out of the loss by ``feat_valid``)."""
        f_max = max(nb for nb, _ in PER_BAND_SPECS)
        outs: list[Tensor] = []
        for head, (nb, _) in zip(self.mae_heads, PER_BAND_SPECS):
            o = head(h)  # (B, total, F_b)
            if nb < f_max:
                o = F.pad(o, (0, f_max - nb))
            outs.append(o)  # (B, total, F_MAX)
        stack = torch.stack(outs, dim=-1)  # (B, total, F_MAX, N_BANDS)
        onehot = F.one_hot(grid.band, len(PER_BAND_SPECS)).to(h.dtype)  # (total, N_BANDS)
        return (stack * onehot[None, :, None, :]).sum(-1)  # (B, total, F_MAX)

    @staticmethod
    def _norm_pix(x: Tensor, feat_valid: Tensor, feat_count: Tensor) -> Tensor:
        """He-2021 norm_pix over each token's VALID bins: ``(x-mean)/sqrt(var+1e-6)`` with
        UNBIASED var (÷(n-1), matching ``torch.var``) and eps 1e-6. Pad bins are excluded
        from mean/var; their post-norm value is masked out of the loss by ``feat_valid``."""
        fv = feat_valid[None].to(x.dtype)  # (1, total, F_MAX)
        fc = feat_count[None, :, None].to(x.dtype)  # (1, total, 1)
        mean = (x * fv).sum(-1, keepdim=True) / fc
        var = ((x - mean) ** 2 * fv).sum(-1, keepdim=True) / (fc - 1.0)
        return (x - mean) / (var + 1e-6).sqrt()

    @torch.no_grad()
    def update_teacher(self, step: int | None = None) -> float:
        if self.teacher is None:  # MAE arm has no EMA teacher — no-op every opt-step.
            return 0.0
        return self.teacher.update_from(self.online, step=step)
