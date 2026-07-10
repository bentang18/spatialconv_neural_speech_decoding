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

The EMA copy wraps the ENCODER; the stem (a linear patch-embed applied upstream)
is shared, not EMA-blended — a KISS simplification of the V-JEPA target encoder,
negligible for a linear embed. Whole-sensor masks reappear in the predictor as
all-query shafts, reconstructed cross-shaft via L2 (the offloaded task).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from collections.abc import Sequence

from speech_decoding.models.v14_converged_v3.geometry import L1Geometry
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
    ) -> Tensor:
        tokens = self.stem(bands)  # (B, N, T, 256)
        return self.encoder(tokens, geom, parcel_id, visible=visible)


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
        # Learnable mask query, zero-init (V-JEPA-2.1 audit: mask_token zero-init).
        self.mask_token = nn.Parameter(torch.zeros(PRED_D_MODEL))
        self.target_ln = target_ln

    def forward(
        self,
        bands: Sequence[Tensor],
        geom: L1Geometry,
        parcel_id: Tensor,
        mask: Tensor,
    ) -> JepaOutput:
        """bands: 3-band |STFT| inputs; mask: (B, N) bool (True = target)."""
        visible = ~mask  # (B, N)

        # online tower (stem + encoder) over visible electrodes only
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
        return JepaOutput(loss=loss, n_masked=int(tube.sum()))

    @torch.no_grad()
    def update_teacher(self, step: int | None = None) -> float:
        return self.teacher.update_from(self.online, step=step)
