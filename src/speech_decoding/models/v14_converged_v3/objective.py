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
        geom: L1Geometry,
        parcel_id: Tensor,
        visible: Tensor | None,
        *,
        tap_blocks: tuple[int, ...] = (),
    ) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
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
        collect_taps: bool = False,
        whole_contact: Tensor | None = None,
    ) -> JepaOutput:
        """bands: 3-band |STFT| inputs; mask: (B, N) bool (True = target).

        ``collect_taps`` (monitor cadence only): also return detached taps —
        encoder block-6/12 visible-token rows (rankme/feat_std depth probe) and the
        predictor/target rows split by whole-sensor vs intra-sensor tier (EV +
        var-ratio + L1). ``whole_contact`` (B, N) bool marks the whole-sensor tier;
        required when ``collect_taps`` for the tier split.
        """
        visible = ~mask  # (B, N)

        # online tower (stem + encoder) over visible electrodes only
        if collect_taps:
            z, enc_taps = self.online(
                bands, geom, parcel_id, visible, tap_blocks=(6, 12)
            )  # (B, N, T, 256), {6,12: (B,N,T,256)}
        else:
            z = self.online(bands, geom, parcel_id, visible)  # (B, N, T, 256)

        # EMA teacher (own stem + encoder) over the full grid → targets
        with torch.no_grad():
            tgt = self.teacher(bands, geom, parcel_id, None)  # (B, N, T, 256)
        tgt = _ln_target(tgt) if self.target_ln else stop_grad(tgt)

        # predictor: mask-query at masked slots, full attention
        zp = self.enc_to_pred(z)  # (B, N, T, 128)
        m = mask[:, :, None, None]  # (B, N, 1, 1)
        pred_in = torch.where(m, self.mask_token, zp)  # masked → learnable query
        h = self.predictor(pred_in, geom, parcel_id, visible=None)  # (B, N, T, 128)
        pred = self.pred_to_target(h)  # (B, N, T, 256)

        # L1 over masked positions only (time-tubed)
        tube = mask[:, :, None].expand(pred.shape[:-1])  # (B, N, T) bool
        pred_m = pred[tube]  # (n_masked, 256)
        tgt_m = tgt[tube]
        loss = _l1_or_zero(pred_m, tgt_m, "l1")

        taps = None
        if collect_taps:
            taps = self._build_taps(enc_taps, pred, tgt, mask, visible, whole_contact)
        return JepaOutput(loss=loss, n_masked=int(tube.sum()), taps=taps)

    @staticmethod
    def _build_taps(
        enc_taps: dict[int, Tensor],
        pred: Tensor,
        tgt: Tensor,
        mask: Tensor,
        visible: Tensor,
        whole_contact: Tensor | None,
    ) -> dict[str, Tensor]:
        """Detached monitor taps. Encoder rows are VISIBLE tokens only (the learned
        representation); pred/target rows are the time-tubed masked positions split
        into whole-sensor and intra-sensor tiers. Target rows are already
        ``target_ln``'d (computed after the norm), matching the loss."""
        B, N, T = mask.shape[0], mask.shape[1], pred.shape[2]
        vis_t = visible[:, :, None].expand(B, N, T)  # (B, N, T)
        out: dict[str, Tensor] = {
            "enc6": enc_taps[6].detach()[vis_t],  # (n_vis·T, 256)
            "enc12": enc_taps[12].detach()[vis_t],
        }
        if whole_contact is None:
            return out
        whole_t = whole_contact[:, :, None].expand(B, N, T)
        intra_t = (mask & ~whole_contact)[:, :, None].expand(B, N, T)
        p, t = pred.detach(), tgt.detach()
        out.update(
            pred_whole=p[whole_t], tgt_whole=t[whole_t],
            pred_intra=p[intra_t], tgt_intra=t[intra_t],
        )
        return out

    @torch.no_grad()
    def update_teacher(self, step: int | None = None) -> float:
        return self.teacher.update_from(self.online, step=step)
