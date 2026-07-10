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
    gather_tokens,
    scatter_tokens,
)
from speech_decoding.models.v14_converged_v3.pe import init_transformer_weights
from speech_decoding.models.v14_converged_v3.stem import SpectralStem
from speech_decoding.models.v14_converged_v3.towers import (
    PRED_D_MODEL,
    build_encoder,
    build_predictor,
)
from speech_decoding.ssl.ema import EmaTeacher, fixed_ema_schedule, stop_grad
from speech_decoding.ssl.masked_jepa import _l1_or_zero

D_MODEL = 256
EMA_TAU = 0.99925  # V-JEPA 2 / 2.1 default (B26 lock)


def _ln_target(t: Tensor) -> Tensor:
    """Affine-free V-JEPA target norm (matches v14_converged_v2._ln_target)."""
    return F.layer_norm(t.detach(), (t.shape[-1],))


@dataclass
class JepaOutput:
    loss: Tensor
    n_masked: int
    taps: dict[str, Tensor] | None = None


class _TargetTower(nn.Module):
    """The full target-producing path: stem (patch-embed) + encoder.

    This is the unit the EMA teacher mirrors. V-JEPA's target encoder is an EMA of
    the ENTIRE context encoder INCLUDING the patch-embed — so the stem must live
    inside the EMA'd tower, not be shared. The predictor is NOT part of this tower:
    it is the online-only module that maps context latents into target space (it has
    no teacher counterpart in JEPA).
    """

    def __init__(self, *, n_parcels: int) -> None:
        super().__init__()
        self.stem = SpectralStem(D_MODEL)
        self.encoder = build_encoder(n_parcels=n_parcels)

    def forward(
        self,
        bands: Sequence[Tensor],
        plan: PackPlan,
        parcel_packed: Tensor,
        *,
        backend: str = "auto",
        tap_blocks: tuple[int, ...] = (),
    ) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
        # PACKED production path (#24): stem over all N, gather the plan's selected
        # contacts (M_vis online / N teacher), run the encoder packed. This is the
        # forward the EMA teacher (a deepcopy) also routes through.
        tokens = self.stem(bands)  # (B, N, T, 256)
        x = gather_tokens(tokens, plan.order)  # (B, P, T, 256)
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
    ) -> None:
        super().__init__()
        # online target path (stem + encoder) — every param here is EMA-mirrored.
        self.online = _TargetTower(n_parcels=n_parcels)
        self.teacher = EmaTeacher(
            self.online, coeff_schedule=fixed_ema_schedule(tau=ema_tau)
        )
        self.predictor = build_predictor(n_parcels=n_parcels)
        self.enc_to_pred = nn.Linear(D_MODEL, PRED_D_MODEL)
        self.pred_to_target = nn.Linear(PRED_D_MODEL, D_MODEL)
        init_transformer_weights(self.enc_to_pred)  # V-JEPA 2 trunc_normal(0.02)
        init_transformer_weights(self.pred_to_target)
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
        mask: Tensor,
        *,
        m_masked: int,
        backend: str = "auto",
        collect_taps: bool = False,
        whole_contact: Tensor | None = None,
    ) -> JepaOutput:
        """bands: 3-band |STFT| inputs; mask: (B, N) bool (True = target).

        PACKED production path (#24 FULL varlen): the online encoder runs over the
        ``M_vis = N − m_masked`` visible contacts, the EMA teacher + predictor over
        all N, all shaft-grouped and ragged (no ``max_c`` pad). ``m_masked`` = the
        per-row held-out count ``M = round(mask_frac·N)`` (a per-session CONSTANT the
        model supplies) — it fixes ``M_vis`` without a host-sync on ``mask.sum()`` and
        gives ``n_masked = B·M·T`` for free (B6).

        ``collect_taps`` (monitor cadence only): also return detached taps — encoder
        block-3/12 visible rows (rankme/feat_std) + predictor/target rows split by
        whole-sensor vs intra-sensor tier. ``whole_contact`` (B, N) bool required then.
        """
        visible = ~mask  # (B, N)
        B, N = mask.shape
        T = bands[0].shape[-1]  # 32 Hz clock length (all bands native 32 Hz, factor 1)
        m_vis = N - m_masked

        # per-clip visible pack plan (online) + static all-N plan (teacher/predictor)
        online_plan = build_pack_plan(
            geom, n_time=T, batch=B, n_selected=m_vis, visible=visible
        )
        full_plan = build_pack_plan(geom, n_time=T, batch=B, n_selected=N, visible=None)
        online_parcel = parcel_id[online_plan.order]  # (B, M_vis)
        full_parcel = parcel_id[full_plan.order]  # (B, N)

        # online encoder over the visible contacts (packed)
        if collect_taps:
            z, enc_taps = self.online(
                bands, online_plan, online_parcel, backend=backend, tap_blocks=(3, 12)
            )  # (B, M_vis, T, 256)
        else:
            z = self.online(bands, online_plan, online_parcel, backend=backend)

        # EMA teacher over all N (packed, full_plan order) → targets
        with torch.no_grad():
            tgt = self.teacher(bands, full_plan, full_parcel, backend=backend)
        tgt = _ln_target(tgt) if self.target_ln else stop_grad(tgt)  # (B,N,T,256) full order

        # predictor input assembly: enc_to_pred(z) at visible, mask-query at masked
        # (built in CONTACT order, then gathered into the predictor's all-N order).
        zp_full = scatter_tokens(self.enc_to_pred(z), online_plan.order, N)  # (B,N,T,128)
        m = mask[:, :, None, None]  # (B, N, 1, 1)
        pred_in_full = torch.where(m, self.mask_token, zp_full)  # masked → query, contact order
        pred_in = gather_tokens(pred_in_full, full_plan.order)  # (B, N, T, 128) full order
        h = self.predictor.forward_packed(pred_in, full_plan, full_parcel, backend=backend)
        pred = self.pred_to_target(h)  # (B, N, T, 256) full order

        # L1 at masked positions; pred & tgt BOTH in full_plan order → reorder mask
        mask_packed = mask.gather(1, full_plan.order)  # (B, N) bool, full order
        tube = mask_packed[:, :, None].expand(B, N, T)  # (B, N, T)
        loss = _l1_or_zero(pred[tube], tgt[tube], "l1")

        taps = None
        if collect_taps:
            taps = self._build_taps_packed(
                enc_taps, pred, tgt, mask, whole_contact, full_plan.order
            )
        return JepaOutput(loss=loss, n_masked=B * m_masked * T, taps=taps)

    @staticmethod
    def _build_taps_packed(
        enc_taps: dict[int, Tensor],
        pred: Tensor,
        tgt: Tensor,
        mask: Tensor,
        whole_contact: Tensor | None,
        full_order: Tensor,
    ) -> dict[str, Tensor]:
        """Detached monitor taps (packed path). Encoder taps are already the packed
        VISIBLE tokens (B, M_vis, T, d) ⇒ every row is a visible token, so flatten
        directly. pred/tgt are in ``full_order``; the whole/intra tiers are reordered
        to match before extracting. Targets are already ``target_ln``'d (post-norm)."""
        d_enc = enc_taps[3].shape[-1]
        d_out = pred.shape[-1]
        out: dict[str, Tensor] = {
            "enc3": enc_taps[3].detach().reshape(-1, d_enc),  # (n_vis·T, 256); pre-first-L2
            "enc12": enc_taps[12].detach().reshape(-1, d_enc),
        }
        if whole_contact is None:
            return out
        B, N, T = mask.shape[0], mask.shape[1], pred.shape[2]
        whole_packed = whole_contact.gather(1, full_order)  # (B, N) full order
        intra_packed = (mask & ~whole_contact).gather(1, full_order)
        whole_t = whole_packed[:, :, None].expand(B, N, T)
        intra_t = intra_packed[:, :, None].expand(B, N, T)
        p, t = pred.detach(), tgt.detach()
        out.update(
            pred_whole=p[whole_t].reshape(-1, d_out), tgt_whole=t[whole_t].reshape(-1, d_out),
            pred_intra=p[intra_t].reshape(-1, d_out), tgt_intra=t[intra_t].reshape(-1, d_out),
        )
        return out

    def _forward_padded(
        self,
        bands: Sequence[Tensor],
        geom: L1Geometry,
        parcel_id: Tensor,
        mask: Tensor,
        *,
        collect_taps: bool = False,
        whole_contact: Tensor | None = None,
    ) -> JepaOutput:
        """Padded ORACLE (test only): the pre-#24 dense JEPA assembly, kept to pin the
        packed ``forward`` numerically on CPU. The teacher is routed through the
        deepcopy's ``forward_padded``."""
        visible = ~mask
        if collect_taps:
            z, enc_taps = self.online.forward_padded(
                bands, geom, parcel_id, visible, tap_blocks=(3, 12)
            )
        else:
            z = self.online.forward_padded(bands, geom, parcel_id, visible)
        with torch.no_grad():
            tgt = self.teacher.model.forward_padded(bands, geom, parcel_id, None)
        tgt = _ln_target(tgt) if self.target_ln else stop_grad(tgt)
        zp = self.enc_to_pred(z)
        m = mask[:, :, None, None]
        pred_in = torch.where(m, self.mask_token, zp)
        h = self.predictor(pred_in, geom, parcel_id, visible=None)
        pred = self.pred_to_target(h)
        tube = mask[:, :, None].expand(pred.shape[:-1])
        loss = _l1_or_zero(pred[tube], tgt[tube], "l1")
        taps = None
        if collect_taps:
            B, N, T = mask.shape[0], mask.shape[1], pred.shape[2]
            vis_t = visible[:, :, None].expand(B, N, T)
            taps = {
                "enc3": enc_taps[3].detach()[vis_t],
                "enc12": enc_taps[12].detach()[vis_t],
            }
            if whole_contact is not None:
                whole_t = whole_contact[:, :, None].expand(B, N, T)
                intra_t = (mask & ~whole_contact)[:, :, None].expand(B, N, T)
                p, t = pred.detach(), tgt.detach()
                taps.update(
                    pred_whole=p[whole_t], tgt_whole=t[whole_t],
                    pred_intra=p[intra_t], tgt_intra=t[intra_t],
                )
        return JepaOutput(loss=loss, n_masked=int(tube.sum()), taps=taps)

    @torch.no_grad()
    def update_teacher(self, step: int | None = None) -> float:
        return self.teacher.update_from(self.online, step=step)
