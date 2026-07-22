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
    sample_masks,
    sample_masks_r5,
)
from speech_decoding.models.v14_converged_v3.objective import (
    LAMBDA_CTX,
    LAMBDA_NLL,
    JepaOutput,
    V3JepaObjective,
)
from speech_decoding.models.v14_converged_v3.pack_r4 import (
    build_r4_grid,
    build_r5_grid,
    build_visible_pack,
    token_flags,
    token_flags_r5,
)
from speech_decoding.models.v14_converged_v3.stem import clock_length_32hz


class V3ConvergedModel(nn.Module):
    def __init__(
        self,
        *,
        n_parcels: int,
        mask_cfg: V3MaskConfig = V3MaskConfig(),
        target_ln: bool = True,
        deep_sup: bool = True,
        lambda_nll: float = LAMBDA_NLL,
        nll_floor: bool = True,
        secondary_loss: str = "nll",
        context_loss: bool = False,
        lambda_ctx: float = LAMBDA_CTX,
        mae: bool = False,
        native_fine_hga: bool = False,
        early_fusion: bool = False,
    ) -> None:
        super().__init__()
        # The stem lives inside the objective's EMA-mirrored target tower (V-JEPA
        # EMAs the patch-embed too), so the model owns only the objective + mask cfg.
        # deep_sup default ON (#61, Ben-greenlit copy-exactly); deep_sup=False = the
        # single-tap ablation arm. lambda_nll (§5 open knob) is the secondary
        # Gaussian-NLL weight in total = JEPA_L1 + λ·NLL; it only matters when the
        # per-session frozen state-stats are supplied (secondary opt-in).
        # nll_floor=False is r5 Arm 2 (floor-off): the head learns Sigma with no measured
        # noise floor. Like lambda_nll it only bites when the secondary is opted in.
        # secondary_loss="l1" is r5 Arm 3 (point loss): mu-only head, no covariance
        # parameters at all. L1 (not L2) is measured — see V3JepaObjective.__init__.
        # early_fusion (r5, Chang 2-stream): EarlyFusionStem + single-band grid + single
        # temporal mask + accept-the-bleed scoring. The mask sampler / grid / flags switch
        # to their r5 variants below; the objective owns the frontend + scoring swap.
        self.objective = V3JepaObjective(
            n_parcels=n_parcels, target_ln=target_ln, deep_sup=deep_sup,
            lambda_nll=lambda_nll, nll_floor=nll_floor,
            secondary_loss=secondary_loss,
            context_loss=context_loss, lambda_ctx=lambda_ctx,
            mae=mae, native_fine_hga=native_fine_hga, early_fusion=early_fusion,
        )
        self.native_fine_hga = bool(native_fine_hga)
        self.early_fusion = bool(early_fusion)
        self.mask_cfg = mask_cfg

    def forward(
        self,
        band_inputs: Sequence[Tensor],
        geom: L1Geometry,
        parcel_id: Tensor,
        *,
        generator: torch.Generator,
        collect_taps: bool = False,
        stat_mean: Tensor | None = None,
        stat_std: Tensor | None = None,
        grid_max_seqlen: int | None = None,
        m_vis: int | None = None,
        pack_max_seqlen: int | None = None,
    ) -> JepaOutput:
        B, N = band_inputs[0].shape[0], band_inputs[0].shape[1]
        T = clock_length_32hz(
            band_inputs,
            native_fine_hga=self.native_fine_hga,
            early_fusion=self.early_fusion,
        )
        # masking is the sole augmentation: per-shaft-balanced spatial contact drop + per-band
        # (SLOW/MID/HGA) independent temporal blocks (masking.V3Masks). The flat r4 objective
        # derives the visible/scored token sets from these directly (pack_r4.token_flags).
        # r5 (early_fusion): ONE temporal mask over the single 32 Hz band (sample_masks_r5).
        if self.early_fusion:
            masks = sample_masks_r5(
                geom, N, n_time=T, n_rows=B, generator=generator, cfg=self.mask_cfg
            )
        else:
            masks = sample_masks(
                geom, N, n_time=T, n_rows=B, generator=generator, cfg=self.mask_cfg
            )
        # grid_max_seqlen / m_vis / pack_max_seqlen are the per-session Python-int shape
        # constants ``session_plan`` precomputes (the module caches + passes them each step);
        # they let the objective skip the per-step ``.item()`` host syncs. None ⇒ eager path.
        # stat_mean/stat_std (per-session frozen state-norm stats) turn ON the secondary
        # Gaussian-NLL; absent ⇒ JEPA-only. They flow in per-session like geom/parcel_id.
        return self.objective(
            band_inputs,
            geom,
            parcel_id,
            masks,
            collect_taps=collect_taps,
            stat_mean=stat_mean,
            stat_std=stat_std,
            grid_max_seqlen=grid_max_seqlen,
            m_vis=m_vis,
            pack_max_seqlen=pack_max_seqlen,
        )

    @torch.no_grad()
    def session_plan(
        self, geom: L1Geometry, parcel_id: Tensor, n_time: int
    ) -> tuple[int, int, int]:
        """``(grid_max_seqlen, m_vis, pack_max_seqlen)`` — the per-session Python-int shape
        constants the compiled ``forward`` would otherwise recover with a per-step host sync.

        Computed ONCE per session (the module caches the result by ``session_key``): all the
        ``.item()`` syncs fire here, not every step. The counts are session-INVARIANT — exact
        per-shaft spatial masking (``d_s`` fixed) + GLOBAL per-band temporal masking ⇒ every
        clip masks ``d_s·k_full + (n_s−d_s)·T_masked`` tokens per shaft — so a single
        representative mask (fixed seed) yields the values every clip shares."""
        N = int(geom.valid.sum())
        gen = torch.Generator(device=geom.gather_idx.device).manual_seed(0)
        if self.early_fusion:  # r5 single-band grid + single-temporal-mask flags
            grid = build_r5_grid(geom, n_time=n_time)
            parcel_packed = parcel_id[grid.contact]
            masks = sample_masks_r5(
                geom, N, n_time=n_time, n_rows=1, generator=gen, cfg=self.mask_cfg
            )
            masked, _ = token_flags_r5(grid, masks)
        else:
            grid = build_r4_grid(geom, n_time=n_time)
            parcel_packed = parcel_id[grid.contact]
            masks = sample_masks(
                geom, N, n_time=n_time, n_rows=1, generator=gen, cfg=self.mask_cfg
            )
            masked, _ = token_flags(grid, masks)
        pack = build_visible_pack(grid, masked, parcel_packed)
        return grid.max_seqlen, pack.m_vis, pack.max_seqlen

    @torch.no_grad()
    def update_teacher(self, step: int | None = None) -> float:
        return self.objective.update_teacher(step=step)
