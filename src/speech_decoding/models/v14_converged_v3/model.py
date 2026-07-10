"""v14_converged_v3 — top-level assembly (Phase 7).

One forward turns the 3-band |STFT| frames into the plain-JEPA masked loss:

    bands ─▶ SpectralStem ─▶ tokens (B,N,T,256)
                               │  sample per-session electrode-tube mask (B,N)
                               ▼
                      V3JepaObjective(encoder / EMA teacher / predictor) ─▶ loss

The per-session ``L1Geometry`` (shaft gather + depths) and the DKT ``parcel_id``
come from the sidecar, built once per session upstream and passed in.

── v2 REUSE AUDIT (memo project-v14-converged-v3-sensor-architecture) ──────────
REUSED (well-tested v2 substrate, imported not copied):
  • ``ssl.ema.EmaTeacher`` / ``fixed_ema_schedule`` / ``stop_grad`` — EMA teacher.
  • ``ssl.masked_jepa._l1_or_zero`` — the masked L1 (empty-safe, graph-connected).
  • target_ln = affine-free ``F.layer_norm`` on targets — same recipe as
    ``v14_converged_v2._ln_target``.
  • STFT band extraction + ``_stft_band_k_range`` and the LOF / bad-window guards
    live in the extractor / cache path (Phase 0), unchanged.
  • Vectorized mask idioms (argsort-of-rand, cover-rank) generalized from
    ``v14_converged_v2._hga_fill_not_trim`` / ``sample_electrode_drop_v2``.
REWRITTEN FRESH (v3-specific, no legacy copy):
  • stem fold, ALL positional encoding (L1 index+time RoPE, L2 parcel identity),
    L1/L2 attention blocks, encoder/predictor tower layouts, electrode-tube
    masking, the JEPA assembly, and the shaft/index sidecar + geometry.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from speech_decoding.models.v14_converged_v3.geometry import L1Geometry
from speech_decoding.models.v14_converged_v3.masking import (
    V3MaskConfig,
    sample_contact_mask,
)
from speech_decoding.models.v14_converged_v3.objective import (
    JepaOutput,
    V3JepaObjective,
)


class V3ConvergedModel(nn.Module):
    def __init__(
        self,
        *,
        n_parcels: int,
        mask_cfg: V3MaskConfig = V3MaskConfig(),
        target_ln: bool = True,
    ) -> None:
        super().__init__()
        # The stem lives inside the objective's EMA-mirrored target tower (V-JEPA
        # EMAs the patch-embed too), so the model owns only the objective + mask cfg.
        self.objective = V3JepaObjective(n_parcels=n_parcels, target_ln=target_ln)
        self.mask_cfg = mask_cfg

    def forward(
        self,
        band_inputs: Sequence[Tensor],
        geom: L1Geometry,
        parcel_id: Tensor,
        *,
        generator: torch.Generator,
        collect_taps: bool = False,
        backend: str = "auto",
    ) -> JepaOutput:
        B, N = band_inputs[0].shape[0], band_inputs[0].shape[1]
        # masking is the sole augmentation; the whole-sensor tier tag is only needed
        # for the monitor tap split (collect_taps).
        if collect_taps:
            mask, whole_contact = sample_contact_mask(
                geom, N, n_rows=B, generator=generator, cfg=self.mask_cfg,
                return_tier=True,
            )
        else:
            mask = sample_contact_mask(
                geom, N, n_rows=B, generator=generator, cfg=self.mask_cfg
            )
            whole_contact = None
        # M = round(mask_frac·N) — the EXACT per-row held-out count the masking
        # guarantees (masking.py). Passing it (not mask.sum()) keeps the packed
        # objective host-sync-free and fixes M_vis for the online pack plan.
        m_masked = round(self.mask_cfg.mask_frac * N)
        return self.objective(
            band_inputs, geom, parcel_id, mask, m_masked=m_masked, backend=backend,
            collect_taps=collect_taps, whole_contact=whole_contact,
        )

    @torch.no_grad()
    def update_teacher(self, step: int | None = None) -> float:
        return self.objective.update_teacher(step=step)
