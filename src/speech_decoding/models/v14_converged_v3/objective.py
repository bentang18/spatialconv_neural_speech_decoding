"""v14_converged_v3 — plain-JEPA objective (Phase 6).

v1 = PLAIN JEPA, KISS (memo project-v14-converged-v3-sensor-architecture):
EMA teacher + 1 predictor + masked-position L1 loss ONLY. NO dense/context loss,
NO deep supervision. KEEP ``target_ln`` (affine-free ``F.layer_norm`` on the
teacher targets — canonical V-JEPA + our pool-γ/scale-runaway fix). Fundamental
collapse guard = EMA-teacher asymmetry + predictor bottleneck (BYOL→I-JEPA).

Forward (I-JEPA mechanics):
  online   encoder over the VISIBLE electrodes (masked excluded as attention keys,
           so a target can never leak into a visible latent) → z.
  teacher  EMA of the encoder, over the FULL grid → targets at masked positions;
           passed through ``target_ln`` and stop-grad.
  predictor  project z (256→128), overwrite masked (electrode,slot) slots with a
           learnable mask-query token (position supplied by the predictor's own
           L1 index-RoPE + L2 parcel identity), run the narrow tower with FULL
           attention (queries gather visible context), project up (128→256).
  loss     L1 over the masked positions only.

The EMA teacher wraps the WHOLE target path — stem (linear patch-embed) AND encoder
— matching V-JEPA 2, whose target encoder is a deepcopy that EMA-blends the
patch-embed too (`app/vjepa_2_1/train.py:361`). Whole-sensor masks reappear in the
predictor as all-query shafts, reconstructed cross-shaft via L2 (the offloaded task).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from collections.abc import Sequence

from speech_decoding.models.v14_converged_v3.geometry import L1Geometry
from speech_decoding.models.v14_converged_v3.packing import (
    PackPlan,
    build_pack_plan,
    gather_cells,
    gather_tokens,
    scatter_cells,
)
from speech_decoding.models.v14_converged_v3.pe import init_transformer_weights
from speech_decoding.models.v14_converged_v3.stem import SpectralStem
from speech_decoding.models.v14_converged_v3.towers import (
    N_LEVELS,
    PRED_D_MODEL,
    build_encoder,
    build_predictor,
)
from speech_decoding.ssl.ema import EmaTeacher, fixed_ema_schedule, stop_grad
from speech_decoding.ssl.masked_jepa import _l1_or_zero

D_MODEL = 256
EMA_TAU = 0.99925  # V-JEPA 2 / 2.1 default (B26 lock)


def _ln_target(t: Tensor, n_levels: int = 1) -> Tensor:
    """Affine-free V-JEPA target norm (matches v14_converged_v2._ln_target).

    Deep-sup (#61, ``n_levels>1``): the teacher emits ``n_levels`` concatenated
    per-level chunks (each already through the encoder's affine ``norms_block``), and
    upstream applies a SECOND parameter-free ``F.layer_norm`` to EACH chunk
    independently (`train.py:591-611`) — the DOUBLE norm. Split, LN each, re-cat.
    ``n_levels==1`` is exactly the single-tap affine-free LN over the whole vector."""
    t = t.detach()
    if n_levels == 1:
        return F.layer_norm(t, (t.shape[-1],))
    d = t.shape[-1] // n_levels
    return torch.cat(
        [F.layer_norm(c, (d,)) for c in t.split(d, dim=-1)], dim=-1
    )


def _static_off(lambda_context: float | Tensor) -> bool:
    """True ⇒ context loss is statically off (the pure plain-JEPA arm). Only a python
    ``0.0`` counts: it is known at ``torch.compile`` trace time so the branch is
    constant-folded and the context head is never touched. A 0-d tensor (the module's
    on-schedule λ, even value 0 during the 0–15k hold) is NOT static-off — the head
    always runs, keeping the compiled graph invariant across the λ ramp."""
    return isinstance(lambda_context, (int, float)) and lambda_context == 0.0


@dataclass
class JepaOutput:
    loss: Tensor
    n_masked: int
    taps: dict[str, Tensor] | None = None
    loss_context: Tensor | None = None  # #66 monitor: the raw (unweighted) context L1
    loss_intra: Tensor | None = None  # monitor: masked L1 over partial-shaft (intra) cells
    loss_inter: Tensor | None = None  # monitor: masked L1 over whole-shaft (inter) cells


class _TargetTower(nn.Module):
    """The full target-producing path: stem (patch-embed) + encoder.

    This is the unit the EMA teacher mirrors. V-JEPA's target encoder is an EMA of
    the ENTIRE context encoder INCLUDING the patch-embed — so the stem must live
    inside the EMA'd tower, not be shared. The predictor is NOT part of this tower:
    it is the online-only module that maps context latents into target space (it has
    no teacher counterpart in JEPA).
    """

    def __init__(self, *, n_parcels: int, deep_sup: bool = True) -> None:
        super().__init__()
        self.stem = SpectralStem(D_MODEL)
        self.encoder = build_encoder(n_parcels=n_parcels, deep_sup=deep_sup)

    def forward(
        self,
        bands: Sequence[Tensor],
        plan: PackPlan,
        parcel_packed: Tensor,
        *,
        backend: str = "auto",
        tap_blocks: tuple[int, ...] = (),
    ) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
        # PACKED production path (#24, dual-axis): stem over all N×T, gather the plan's
        # selected VISIBLE CELLS (M_vis×T_kept online / N×T teacher — full-grid time_idx
        # is arange(T) ⇒ a no-op on the time axis), run the encoder packed. This is the
        # forward the EMA teacher (a deepcopy) also routes through.
        tokens = self.stem(bands)  # (B, N, T, 256)
        x = gather_cells(tokens, plan.order, plan.time_idx)  # (B, P, T_kept, 256)
        return self.encoder.forward_packed(
            x, plan, parcel_packed, backend=backend, tap_blocks=tap_blocks
        )

    def forward_padded(
        self,
        bands: Sequence[Tensor],
        geom: L1Geometry,
        parcel_id: Tensor,
        visible: Tensor | None,
        *,
        tap_blocks: tuple[int, ...] = (),
    ) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
        # Padded ORACLE (test only): the pre-#24 dense path, kept to pin the packed
        # forward numerically on CPU.
        tokens = self.stem(bands)  # (B, N, T, 256)
        return self.encoder(
            tokens, geom, parcel_id, visible=visible, tap_blocks=tap_blocks
        )


class V3JepaObjective(nn.Module):
    def __init__(
        self,
        *,
        n_parcels: int,
        target_ln: bool = True,
        ema_tau: float = EMA_TAU,
        deep_sup: bool = True,
    ) -> None:
        super().__init__()
        self.deep_sup = bool(deep_sup)
        self.n_levels = N_LEVELS if self.deep_sup else 1
        # online target path (stem + encoder) — every param here is EMA-mirrored.
        self.online = _TargetTower(n_parcels=n_parcels, deep_sup=self.deep_sup)
        self.teacher = EmaTeacher(
            self.online, coeff_schedule=fixed_ema_schedule(tau=ema_tau)
        )
        self.predictor = build_predictor(n_parcels=n_parcels)
        # Encoder→predictor input map. Deep-sup (#61): the encoder emits n_levels
        # concatenated levels, so this is upstream's ``predictor_embed`` 2-layer fusion
        # MLP (`predictor.py:84-89`) — Linear(n_levels·d → d) · GELU · Linear(d →
        # d_pred), NO LayerNorm before/after. Single-tap: a plain Linear(d → d_pred).
        # ``pred_to_target`` is upstream ``predictor_proj`` — ONE wide Linear emitting
        # all n_levels·d target dims from the predictor's final (norm_out'd) block.
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
        # CONTEXT-loss head (#66, upstream ``predictor_proj_context``, predictor.py:183):
        # a SEPARATE wide Linear that predicts the teacher target at the VISIBLE/context
        # positions (not the masked ones). Same target space as ``pred_to_target``. Only
        # trained when the λ schedule is active (off before 15k). Always constructed so
        # the module + optimizer see a static param set regardless of the λ ramp.
        self.pred_to_target_context = nn.Linear(PRED_D_MODEL, target_dim)
        self.enc_to_pred.apply(init_transformer_weights)  # V-JEPA 2 trunc_normal(0.02)
        init_transformer_weights(self.pred_to_target)
        init_transformer_weights(self.pred_to_target_context)
        # Learnable mask query, zero-init (V-JEPA-2.1 audit: mask_token zero-init).
        # Stored 3-D (1, 1, D) to match upstream predictor.py:64-65: the shared
        # ndim<=1 no-decay rule (optim_param_groups.is_no_decay) then DECAYS it, as
        # upstream does — a 1-D (D,) store would silently exempt it (audit L36). It
        # broadcasts identically in the torch.where below (numerically a no-op).
        self.mask_token = nn.Parameter(torch.zeros(1, 1, PRED_D_MODEL))
        self.target_ln = target_ln

    def forward(
        self,
        bands: Sequence[Tensor],
        geom: L1Geometry,
        parcel_id: Tensor,
        contact_mask: Tensor,
        frame_mask: Tensor,
        *,
        m_vis: int,
        t_kept: int,
        backend: str = "auto",
        collect_taps: bool = False,
        whole_contact: Tensor | None = None,
        lambda_context: float | Tensor = 0.0,
    ) -> JepaOutput:
        """bands: 3-band |STFT| inputs. Dual-axis masks (Ben 2026-07-12):
        ``contact_mask`` (B, N) bool = spatially masked contacts; ``frame_mask``
        (B, S, T) bool = temporally masked frames per shaft. A CELL (contact c, frame t)
        is a target iff c is spatially masked OR t is masked for c's shaft.

        PACKED production path (#24 FULL varlen, dual-axis): the online encoder runs over
        the ``m_vis·t_kept`` VISIBLE CELLS (space-kept ∧ frame-kept), compacted and ragged
        on BOTH axes; the EMA teacher + predictor over the FULL grid (all N contacts × all
        T frames, same-t exact). ``m_vis = N − D`` and ``t_kept = T − T_mask`` are
        per-session CONSTANTS the model supplies — they fix the compacted shapes without a
        host-sync and give ``n_masked = B·(N·T − m_vis·t_kept)`` for free (B6).

        ``collect_taps`` (monitor cadence only): also return detached taps — encoder
        block-12 visible cells (rankme/feat_std) + predictor/target cells split by
        whole-sensor vs intra tier. ``whole_contact`` (B, N) bool required then.
        """
        B, N = contact_mask.shape
        T = bands[0].shape[-1]  # 32 Hz clock length (all bands native 32 Hz, factor 1)
        shaft_of = geom.shaft_of_contact  # (N,) long — shaft id per contact
        # dual-axis CELL mask: masked iff contact spatially masked OR frame masked for its
        # shaft (per-sensor outer product; homogeneous within shaft). (B, N, T) True=target.
        cell_masked = contact_mask[:, :, None] | frame_mask[:, shaft_of, :]  # (B, N, T)
        cell_visible = ~cell_masked

        # per-clip visible-CELL pack plan (online, compacted on both axes + L2 real-frame
        # regroup) + static full-grid plan (teacher/predictor, no time compaction/regroup).
        online_plan = build_pack_plan(
            geom, n_time=T, batch=B, n_selected=m_vis,
            visible=~contact_mask, frame_keep=~frame_mask, t_kept_hint=t_kept,
        )
        full_plan = build_pack_plan(geom, n_time=T, batch=B, n_selected=N, visible=None)
        online_parcel = parcel_id[online_plan.order]  # (B, m_vis)
        full_parcel = parcel_id[full_plan.order]  # (B, N)

        # online encoder over the visible cells (packed) → (B, m_vis, T_kept, d)
        if collect_taps:
            z, enc_taps = self.online(
                bands, online_plan, online_parcel, backend=backend, tap_blocks=(12,)
            )
        else:
            z = self.online(bands, online_plan, online_parcel, backend=backend)

        # EMA teacher over the full grid (packed, full_plan order) → targets
        with torch.no_grad():
            tgt = self.teacher(bands, full_plan, full_parcel, backend=backend)
        # (B,N,T,n_levels·256) full order; deep-sup per-level double-norm (see _ln_target)
        tgt = _ln_target(tgt, self.n_levels) if self.target_ln else stop_grad(tgt)

        # predictor input assembly: enc_to_pred(z) at visible CELLS, mask-query at masked
        # cells (built in CONTACT order, then gathered into the predictor's all-N order).
        # scatter_cells places the compacted encoder outputs back at their (order, real
        # frame) cells; masked cells stay 0 then get the mask-query below.
        zp_full = scatter_cells(
            self.enc_to_pred(z), online_plan.order, online_plan.time_idx, N, T
        )  # (B, N, T, 128) contact order
        pred_in_full = torch.where(
            cell_masked[:, :, :, None], self.mask_token, zp_full
        )  # masked cell → query, contact order
        pred_in = gather_tokens(pred_in_full, full_plan.order)  # (B, N, T, 128) full order
        h = self.predictor.forward_packed(pred_in, full_plan, full_parcel, backend=backend)
        pred = self.pred_to_target(h)  # (B, N, T, n_levels·256) full order

        # L1 at masked CELLS; pred & tgt BOTH in full_plan order → reorder the cell mask.
        # STATIC masked-MEAN, not boolean-index: ``x[bool_mask]`` has a data-dependent
        # output shape, which graph-breaks (and specializes) under
        # ``torch.compile(dynamic=False)``. The weighted mean over the fixed (B,N,T) grid
        # is the IDENTICAL value — every masked cell shares the same feature width ``d`` —
        # with a shape-static graph. The masked-cell count is a per-session constant so the
        # denominator needs no host-sync.
        cm_packed = cell_masked.gather(
            1, full_plan.order[:, :, None].expand(B, N, T)
        )  # (B, N, T) bool, full order
        w = cm_packed.to(pred.dtype)  # (B, N, T)
        ae = (pred - tgt).abs().mean(-1)  # (B, N, T) per-cell mean over d
        loss = (ae * w).sum() / w.sum().clamp(min=1.0)

        # intra/inter monitor: split the same masked L1 by whole-shaft (INTER — predict a
        # fully-hidden sensor from OTHER shafts, the cross-sensor task) vs partial-shaft
        # (INTRA — same-shaft context available). Reuses ``ae``; static masked-means, no
        # boolean index. whole_contact is (B, N) contact order ⇒ gather into full order to
        # align with cm_packed. None off-monitor (model passes whole_contact only then).
        loss_intra = loss_inter = None
        if whole_contact is not None:
            whole_packed = whole_contact.gather(1, full_plan.order)[:, :, None]  # (B,N,1)
            wi = (cm_packed & whole_packed).to(pred.dtype)
            wa = (cm_packed & ~whole_packed).to(pred.dtype)
            loss_inter = (ae * wi).sum() / wi.sum().clamp(min=1.0)
            loss_intra = (ae * wa).sum() / wa.sum().clamp(min=1.0)

        # CONTEXT loss (#66, upstream predict_all): a SECOND L1 predicting the teacher
        # target at the VISIBLE/context cells via ``pred_to_target_context``, added as
        # ``λ(step)·loss_context``. ``_static_off`` (a python 0.0 known at trace time ⇒
        # compile constant-folds the branch) skips the head entirely for the pure plain-
        # JEPA arm; when the schedule is active the module ALWAYS passes a 0-d tensor
        # (even value 0 pre-15k) so the graph is static across the λ ramp.
        loss_context = None
        if not _static_off(lambda_context):
            pred_ctx = self.pred_to_target_context(h)  # (B, N, T, target_dim) full order
            ctx_tube = ~cm_packed  # (B, N, T) visible cells
            loss_context = _l1_or_zero(pred_ctx[ctx_tube], tgt[ctx_tube], "l1")
            loss = loss + lambda_context * loss_context

        taps = None
        if collect_taps:
            taps = self._build_taps_packed(
                enc_taps, pred, tgt, cell_masked, whole_contact, full_plan.order
            )
        return JepaOutput(
            loss=loss, n_masked=B * (N * T - m_vis * t_kept), taps=taps,
            loss_context=loss_context, loss_intra=loss_intra, loss_inter=loss_inter,
        )

    @staticmethod
    @torch.compiler.disable
    def _build_taps_packed(
        enc_taps: dict[int, Tensor],
        pred: Tensor,
        tgt: Tensor,
        cell_masked: Tensor,
        whole_contact: Tensor | None,
        full_order: Tensor,
    ) -> dict[str, Tensor]:
        """Detached monitor taps (packed path). Encoder taps are already the packed
        VISIBLE CELLS (B, m_vis, T_kept, d) ⇒ every row is a visible cell, so flatten
        directly. pred/tgt are in ``full_order``; the whole/intra tiers are reordered
        to match before extracting. Targets are already ``target_ln``'d (post-norm).

        The tiers are CELL-level: ``whole`` = every cell of a wholly-dropped shaft (all T
        frames of its contacts); ``intra`` = the remaining masked cells (space depth-blocks
        + per-shaft time-blocks). ``whole_cell ⊆ cell_masked`` since a wholly-masked
        contact masks all T of its frames.

        ``@torch.compiler.disable``: the tiers are boolean-indexed (``p[whole_t]``) with a
        per-step-VARIABLE cell count (which shafts get whole-masked is random and shafts
        differ in size). Under ``torch.compile(dynamic=False)`` that resume frame
        respecialises on every new count and blows past ``recompile_limit`` (64) → the G4
        probe's storm. This path is a DETACHED, monitor-cadence read that never touches the
        loss, so running it in eager (one clean graph break, no guards) is correct."""
        d_enc = enc_taps[12].shape[-1]
        d_out = pred.shape[-1]
        out: dict[str, Tensor] = {
            "enc12": enc_taps[12].detach().reshape(-1, d_enc),  # (n_vis_cells, 256)
        }
        if whole_contact is None:
            return out
        B, N, T = cell_masked.shape
        whole_cell = whole_contact[:, :, None].expand(B, N, T)  # (B, N, T) contact order
        intra_cell = cell_masked & ~whole_cell  # (B, N, T) contact order
        exp = full_order[:, :, None].expand(B, N, T)
        whole_packed = whole_cell.gather(1, exp)  # (B, N, T) full order
        intra_packed = intra_cell.gather(1, exp)
        p, t = pred.detach(), tgt.detach()
        out["pred_whole"] = p[whole_packed].reshape(-1, d_out)
        out["tgt_whole"] = t[whole_packed].reshape(-1, d_out)
        out["pred_intra"] = p[intra_packed].reshape(-1, d_out)
        out["tgt_intra"] = t[intra_packed].reshape(-1, d_out)
        return out

    def _forward_padded(
        self,
        bands: Sequence[Tensor],
        geom: L1Geometry,
        parcel_id: Tensor,
        contact_mask: Tensor,
        frame_mask: Tensor,
        *,
        collect_taps: bool = False,
        whole_contact: Tensor | None = None,
        lambda_context: float | Tensor = 0.0,
    ) -> JepaOutput:
        """Padded ORACLE (test only): the pre-#24 dense JEPA assembly, kept to pin the
        packed ``forward`` numerically on CPU. The teacher is routed through the
        deepcopy's ``forward_padded``. Dual-axis: the encoder gets the CELL visibility
        mask ``cell_visible`` (B, N, T); masked cells are excluded as attention keys (so
        they never leak into a visible latent) and overwritten by the mask-query."""
        B, N = contact_mask.shape
        T = bands[0].shape[-1]
        shaft_of = geom.shaft_of_contact
        cell_masked = contact_mask[:, :, None] | frame_mask[:, shaft_of, :]  # (B, N, T)
        cell_visible = ~cell_masked
        if collect_taps:
            z, enc_taps = self.online.forward_padded(
                bands, geom, parcel_id, cell_visible, tap_blocks=(12,)
            )
        else:
            z = self.online.forward_padded(bands, geom, parcel_id, cell_visible)
        with torch.no_grad():
            tgt = self.teacher.model.forward_padded(bands, geom, parcel_id, None)
        tgt = _ln_target(tgt, self.n_levels) if self.target_ln else stop_grad(tgt)
        zp = self.enc_to_pred(z)  # (B, N, T, 128); masked-cell rows discarded below
        pred_in = torch.where(cell_masked[:, :, :, None], self.mask_token, zp)
        h = self.predictor(pred_in, geom, parcel_id, visible=None)
        pred = self.pred_to_target(h)
        # Same static masked-mean as the packed ``forward`` (identical value) so the
        # oracle stays a faithful numeric pin of the production loss.
        w = cell_masked.to(pred.dtype)  # (B, N, T)
        ae = (pred - tgt).abs().mean(-1)  # (B, N, T) per-cell mean over d
        loss = (ae * w).sum() / w.sum().clamp(min=1.0)
        # intra/inter monitor (oracle, contact order ⇒ no gather). See packed forward.
        loss_intra = loss_inter = None
        if whole_contact is not None:
            whole_c = whole_contact[:, :, None]  # (B, N, 1)
            wi = (cell_masked & whole_c).to(pred.dtype)
            wa = (cell_masked & ~whole_c).to(pred.dtype)
            loss_inter = (ae * wi).sum() / wi.sum().clamp(min=1.0)
            loss_intra = (ae * wa).sum() / wa.sum().clamp(min=1.0)
        loss_context = None
        if not _static_off(lambda_context):
            pred_ctx = self.pred_to_target_context(h)
            ctx_tube = cell_visible  # (B, N, T)
            loss_context = _l1_or_zero(pred_ctx[ctx_tube], tgt[ctx_tube], "l1")
            loss = loss + lambda_context * loss_context
        taps = None
        if collect_taps:
            taps = {
                "enc12": enc_taps[12].detach()[cell_visible],
            }
            if whole_contact is not None:
                whole_t = whole_contact[:, :, None].expand(B, N, T)
                intra_t = cell_masked & ~whole_t
                p, t = pred.detach(), tgt.detach()
                taps.update(
                    pred_whole=p[whole_t], tgt_whole=t[whole_t],
                    pred_intra=p[intra_t], tgt_intra=t[intra_t],
                )
        return JepaOutput(
            loss=loss, n_masked=int(w.sum().item()), taps=taps,
            loss_context=loss_context, loss_intra=loss_intra, loss_inter=loss_inter,
        )

    @torch.no_grad()
    def update_teacher(self, step: int | None = None) -> float:
        return self.teacher.update_from(self.online, step=step)
