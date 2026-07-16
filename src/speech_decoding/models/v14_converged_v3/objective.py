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
V-JEPA 2 (its target encoder EMA-blends the patch-embed too). The secondary perceiver /
Gaussian-NLL head (contract §6–7) is a SEPARATE, unshared fusion wired in a following step.
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
    cov_entropy_vs_floor,
    per_band_jepa_stats,
    per_band_nll,
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
from speech_decoding.models.v14_converged_v3.perceiver import PerceiverHead
from speech_decoding.models.v14_converged_v3.secondary_head import (
    count_dependent_noise_var,
    present_masked_nll,
)
from speech_decoding.models.v14_converged_v3.state_target import (
    SLOT_STRIDE,
    build_state_target,
)
from speech_decoding.models.v14_converged_v3.stem import PerBandStem
from speech_decoding.models.v14_converged_v3.towers import (
    N_LEVELS,
    PRED_D_MODEL,
    build_encoder,
    build_predictor,
)
from speech_decoding.ssl.ema import EmaTeacher, fixed_ema_schedule, stop_grad

LAMBDA_NLL: float = 0.2  # secondary Gaussian-NLL weight (contract §5, ⚙️ open knob λ≈0.2)

D_MODEL = 256
EMA_TAU = 0.99925  # V-JEPA 2 / 2.1 default (B26 lock)


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
    loss: Tensor  # the TOTAL the trainer backprops: jepa_loss + λ·nll_loss (jepa only if secondary off)
    n_masked: Tensor  # 0-dim: margin-gated scored-token count (realization-dependent; sync deferred)
    taps: dict[str, Tensor] | None = None
    jepa_loss: Tensor | None = None  # primary L1 term (set only when the secondary is active)
    nll_loss: Tensor | None = None  # secondary Gaussian-NLL term (None when secondary off)


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

    def __init__(self, *, n_parcels: int, deep_sup: bool = True) -> None:
        super().__init__()
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
        lambda_nll: float = LAMBDA_NLL,
    ) -> None:
        super().__init__()
        self.deep_sup = bool(deep_sup)
        self.n_levels = N_LEVELS if self.deep_sup else 1
        self.lambda_nll = float(lambda_nll)
        # online target path (stem + encoder) — every param here is EMA-mirrored.
        self.online = _TargetTower(n_parcels=n_parcels, deep_sup=self.deep_sup)
        self.teacher = EmaTeacher(
            self.online, coeff_schedule=fixed_ema_schedule(tau=ema_tau)
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
        self.pred_to_target = nn.Linear(PRED_D_MODEL, target_dim)
        self.enc_to_pred.apply(init_transformer_weights)  # V-JEPA 2 trunc_normal(0.02)
        init_transformer_weights(self.pred_to_target)
        # Learnable mask query, zero-init (V-JEPA-2.1 audit). Stored 3-D (1, 1, D) to match
        # upstream: the shared ndim<=1 no-decay rule then DECAYS it (a 1-D store would
        # silently exempt it). Broadcasts identically in scatter_visible (a no-op there).
        self.mask_token = nn.Parameter(torch.zeros(1, 1, PRED_D_MODEL))
        self.target_ln = target_ln
        # Secondary write-only Perceiver head (contract §6–7). It reads the deep-sup {3,6,9,12}
        # concat (1024) the CONTEXT encoder emits, so it exists ONLY under deep_sup; the
        # single-tap ablation arm has no secondary. Its own fusion (1024→d_perc) is SEPARATE
        # from enc_to_pred — the shared trunk is the ENCODER, not the fusion.
        self.perceiver = PerceiverHead(n_parcels=n_parcels) if self.deep_sup else None

    def forward(
        self,
        bands: Sequence[Tensor],
        geom: L1Geometry,
        parcel_id: Tensor,
        masks: V3Masks,
        *,
        collect_taps: bool = False,
        stat_mean: Tensor | None = None,
        stat_std: Tensor | None = None,
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

        ``stat_mean``/``stat_std`` (P, 6) FROZEN per-(subject,parcel,dim) train stats: when
        BOTH are given the secondary write-only Perceiver Gaussian-NLL is added
        (``total = JEPA_L1 + λ·NLL``, contract §5–7). When absent the forward is JEPA-only
        (the primary path is unchanged) — the two are wired together only once the per-session
        stats are plumbed through the batch (the launch config always supplies them).

        ``collect_taps`` (monitor cadence): also return the detached encoder block-12
        VISIBLE-cell tap (rankme / feat_std).
        """
        T = bands[0].shape[-1]  # 32 Hz clock length
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
        with torch.no_grad():
            tgt = self.teacher(bands, grid, parcel_packed)  # (B, total, n_levels·256)
        tgt = _ln_target(tgt, self.n_levels) if self.target_ln else stop_grad(tgt)

        # predictor input: enc_to_pred(z) at visible tokens, mask-query at masked tokens.
        pred_in = scatter_visible(
            self.enc_to_pred(z), pack.idx, grid.total, self.mask_token
        )  # (B, total, 128)
        h = self.predictor.forward_flat(pred_in, grid, parcel_packed)  # (B, total, 128)
        pred = self.pred_to_target(h)  # (B, total, n_levels·256)

        # L1 at the MARGIN-GATED masked tokens (in_loss). STATIC weighted mean over the
        # fixed (B, total) grid — every scored token shares the same feature width, so this
        # equals the boolean-indexed mean with a shape-static graph (in_loss count is a
        # per-session constant ⇒ the denominator needs no host sync). §5: reuse the tested
        # deep-sup L1 unchanged (pure |·|, NOT smooth-L1 — the DATA-FLOW one-liner's wording).
        w = in_loss.to(pred.dtype)  # (B, total)
        ae = (pred - tgt).abs().mean(-1)  # (B, total) per-token mean over the 1024-d stack
        jepa_loss = (ae * w).sum() / w.sum().clamp(min=1.0)

        # ── secondary write-only Perceiver Gaussian-NLL (opt-in on the frozen stats) ──
        nll_loss = None
        sec_taps = None
        total = jepa_loss
        if stat_mean is not None or stat_std is not None:
            if self.perceiver is None:
                raise ValueError(
                    "secondary NLL needs deep_sup=True (the Perceiver reads the deep-sup taps)"
                )
            if stat_mean is None or stat_std is None:
                raise ValueError("stat_mean and stat_std must BOTH be given to enable the NLL")
            nll_loss, sec_taps = self._secondary_nll(
                bands, parcel_id, z, pack, stat_mean, stat_std, collect_taps=collect_taps
            )
            total = jepa_loss + self.lambda_nll * nll_loss

        taps = None
        if collect_taps:
            d_enc = enc_taps[12].shape[-1]
            taps = {"enc12": enc_taps[12].detach().reshape(-1, d_enc)}  # (B·M_vis, 256)
            # per-band JEPA health (#40) — reduce the GB-scale pred/tgt to scalars HERE
            # (they never leave the objective as raw taps). no_grad: monitor-only, must not
            # extend the backward graph the loss above already built.
            with torch.no_grad():
                taps.update(per_band_jepa_stats(pred, tgt, w, grid.band))
            if sec_taps is not None:  # per-band NLL (#41) + cov-entropy vs floor (#42)
                taps.update(sec_taps)
        return JepaOutput(
            loss=total,
            # 0-dim tensor, NOT an int: the scored-token count is realization-dependent
            # (the margin gate varies per clip — NOT a session constant, so it cannot be
            # cached/passed like m_vis). Returning the tensor defers the host sync to the
            # logger's own cadence instead of forcing one inside the compiled forward each
            # step; consumers ``.item()``/log it lazily.
            n_masked=w.sum().detach(),
            taps=taps,
            jepa_loss=jepa_loss if nll_loss is not None else None,
            nll_loss=nll_loss,
        )

    def _secondary_nll(
        self,
        bands: Sequence[Tensor],
        parcel_id: Tensor,
        z: Tensor,
        pack: VisiblePack,
        stat_mean: Tensor,
        stat_std: Tensor,
        *,
        collect_taps: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        """Write-only Perceiver → per-(parcel, 4 Hz slot) Gaussian, scored by the present-
        masked marginal NLL against the model-free 6-D state target.

        ``z`` (B, M_vis, 1024) = the CONTEXT (visible) encoder deep-sup output (WITH grad —
        write-only means no read-BACK into the primary stream, not stop-grad; the NLL DOES
        shape the encoder). Queries = every present (parcel, slot); the count-dependent floor
        is keyed on each query's parcel electrode count (Q1-A: objective owns n_elec, head
        adds the floor). Exact masking ⇒ every packed visible token is real (no key mask)."""
        assert self.perceiver is not None  # guarded by the caller
        target, present, parcels = build_state_target(
            bands, parcel_id, stat_mean, stat_std, slot_stride=SLOT_STRIDE
        )  # target (B, P, S, 6), present (P, 6), parcels (P,)
        B, P, S, _ = target.shape
        dev = target.device
        # per-parcel electrode count → per-query count-dependent 6-D noise floor.
        # ``parcels = unique(parcel_id)`` ⇒ ``parcels.max() == parcel_id.max()``, so a bare
        # ``bincount`` already returns length ``parcel_id.max()+1`` — the old ``minlength=
        # int(parcels.max().item())+1`` was a no-op that cost a per-step host sync (the
        # secondary-path twin of the syncs e3baef6 killed on the primary path).
        counts = torch.bincount(parcel_id)
        n_elec = counts[parcels]  # (P,)
        noise = count_dependent_noise_var(n_elec.repeat_interleave(S))  # (Q, 6)
        # queries parcel-major to match target.reshape(B, P·S, 6): [p0s0,p0s1,…,p1s0,…].
        Q = P * S
        q_parcel = parcels.repeat_interleave(S)  # (Q,) actual parcel ids (index ParcelIdentityEmbed)
        q_slot = torch.arange(S, device=dev).repeat(P)  # (Q,) 0..S-1
        # exact masking ⇒ every packed visible token is real, so the encode key-mask would be
        # all-True and its additive bias identically zero. Pass None to skip the bool alloc +
        # the zero bias-add in the perceiver's encode cross-attention (agent finding #5).
        lat = None
        if collect_taps:  # also pull the processed latent bank for the health monitor
            mu, cov, lat = self.perceiver(
                z,
                pack.time_pos,
                None,
                q_parcel[None].expand(B, Q),
                q_slot[None].expand(B, Q),
                n_slots=S,
                noise=noise[None].expand(B, Q, noise.shape[-1]),
                return_latents=True,
            )
        else:
            mu, cov = self.perceiver(
                z,
                pack.time_pos,
                None,
                q_parcel[None].expand(B, Q),
                q_slot[None].expand(B, Q),
                n_slots=S,
                noise=noise[None].expand(B, Q, noise.shape[-1]),
            )
        present_q = present.repeat_interleave(S, dim=0)  # (Q, 6)
        target_q = target.reshape(B, Q, target.shape[-1])
        present_q_full = present_q[None].expand(B, Q, present_q.shape[-1])  # (B, Q, 6)
        nll = present_masked_nll(mu, cov, target_q, present_q_full)
        aux = None
        if collect_taps:  # per-band NLL (#41) + predicted-cov entropy vs floor (#42)
            with torch.no_grad():
                aux = {
                    **per_band_nll(mu, cov, target_q, present_q_full),
                    **cov_entropy_vs_floor(cov, noise[None].expand(B, Q, noise.shape[-1])),
                }
            # processed-latent bank (B, S·M, d_perc), raw — reduced by the callback's
            # perceiver-health monitor (RankMe/feat_std via the shared _rank_and_std path,
            # dead-frac + latent-latent cosine). Detached: monitor-only, never in backward.
            if lat is not None:
                aux["perc_lat"] = lat.detach()
        return nll, aux

    @torch.no_grad()
    def update_teacher(self, step: int | None = None) -> float:
        return self.teacher.update_from(self.online, step=step)
