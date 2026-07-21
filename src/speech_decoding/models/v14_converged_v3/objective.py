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
    per_band_jepa_stats,
)
from speech_decoding.models.v14_converged_v3.pack_r4 import (
    R4Grid,
    VisiblePack,
    build_r4_grid,
    build_visible_pack,
    pack_band_tokens,
    scatter_visible,
    token_flags,
)
from speech_decoding.models.v14_converged_v3.pe import init_transformer_weights
from speech_decoding.models.v14_converged_v3.stem import (
    PER_BAND_SPECS,
    FineHgaStem,
    PerBandStem,
    clock_length_32hz,
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
        self, *, n_parcels: int, deep_sup: bool = True, native_fine_hga: bool = False
    ) -> None:
        super().__init__()
        # native_fine_hga: consume native-rate bands (SLOW 4Hz / MID 16Hz / HGA 128Hz,
        # HGA 4 bins conv-pooled 128→32Hz) instead of arm0's uniform-32Hz PerBandStem.
        # Same (tokens, positions) contract + same 32Hz output lattice, so masking/pack/
        # pe/encoder are byte-identical (memo project-fine-hga-bt-rebake-tasklist-2026-07-21).
        self.stem = FineHgaStem(D_MODEL) if native_fine_hga else PerBandStem(D_MODEL)
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
    ) -> None:
        super().__init__()
        self.native_fine_hga = bool(native_fine_hga)
        # MAE arm (Masked Autoencoder target, He 2021 / AudioMAE): the ONLY change vs the
        # JEPA arm is the prediction TARGET — reconstruct this token's OWN norm_pix'd input
        # |STFT| bins instead of the EMA-teacher latent. No EMA teacher. Everything upstream
        # (visible-only encoder, predictor, mask query, margin-gated in_loss, all locked HPs)
        # is byte-identical to the JEPA arm.
        self.mae = bool(mae)
        self.deep_sup = bool(deep_sup)
        self.n_levels = N_LEVELS if self.deep_sup else 1
        # online target path (stem + encoder) — every param here is EMA-mirrored (JEPA arm).
        self.online = _TargetTower(
            n_parcels=n_parcels, deep_sup=self.deep_sup,
            native_fine_hga=self.native_fine_hga,
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
            # MAE decoder_pred = per-band reconstruction heads, one Linear(d_pred → F_b) per
            # band (SLOW 7 / MID 6 / HGA 7) — the transpose-twin of PerBandStem's per-band
            # INPUT projections. Fed by the predictor's terminal norm_out (= MAE decoder_norm),
            # each emits its band's own |STFT| bins; no wide teacher-target projection.
            self.pred_to_target = None
            self.mae_heads = nn.ModuleList(
                nn.Linear(PRED_D_MODEL, nb) for nb, _ in PER_BAND_SPECS
            )
            self.mae_heads.apply(init_transformer_weights)
        else:
            self.pred_to_target = nn.Linear(PRED_D_MODEL, target_dim)
            init_transformer_weights(self.pred_to_target)
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
        T = clock_length_32hz(bands, native_fine_hga=self.native_fine_hga)  # 32 Hz clock
        # grid_max_seqlen / m_vis / pack_max_seqlen are per-session Python-int shape constants
        # (the module caches them via ``V3ConvergedModel.session_plan`` and passes them in);
        # supplying them skips the per-step ``.item()`` host syncs that otherwise break the
        # compiled graph. None ⇒ derive them here (one sync each — the eager/standalone path).
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
        h = self.predictor.forward_flat(pred_in, grid, parcel_packed)  # (B, total, 128)
        if self.mae:
            return self._mae_output(
                bands, grid, h, in_loss,
                enc_taps=enc_taps if collect_taps else None,
                collect_taps=collect_taps,
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
    ) -> JepaOutput:
        """MAE loss: reconstruct each masked token's OWN norm_pix'd input |STFT| bins.

        Exact He-2021 / AudioMAE recipe — per-token target normalization
        ``(x-mean)/sqrt(var+1e-6)`` (unbiased var) over that token's F_b bins, MSE
        ``(pred-target)^2`` meaned over the bins, then masked-mean over the SAME
        margin-gated ``in_loss`` tokens the JEPA arm scores. ``h`` (B, total, d_pred) is the
        predictor output AFTER its terminal ``norm_out`` (= MAE ``decoder_norm``)."""
        target, feat_valid, feat_count = self._mae_gather_target(bands, grid)
        target = self._norm_pix(target.float(), feat_valid, feat_count)  # fp32 norm_pix
        pred = self._mae_pred(h, grid)  # (B, total, F_MAX); pad slot = 0
        target = target.to(pred.dtype)
        w = in_loss.to(pred.dtype)  # (B, total) margin-gated masked weight
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
