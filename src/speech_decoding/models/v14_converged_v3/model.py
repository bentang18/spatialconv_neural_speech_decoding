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
    sample_masks_r5nf,
    sample_masks_r6,
)
from speech_decoding.models.v14_converged_v3.objective import (
    D_MODEL,
    JepaOutput,
    V3JepaObjective,
)
from speech_decoding.models.v14_converged_v3.pack_r4 import (
    build_r4_grid,
    build_r5_grid,
    build_r5nf_grid,
    build_visible_pack,
    token_flags,
    token_flags_r5,
    token_flags_r5nf,
    token_flags_r6,
)
from speech_decoding.models.v14_converged_v3.stem import (
    NOFUSION_DECIMATE,
    clock_length_32hz,
    nf_token_geometry,
)


class V3ConvergedModel(nn.Module):
    def __init__(
        self,
        *,
        n_parcels: int,
        mask_cfg: V3MaskConfig = V3MaskConfig(),
        target_ln: bool = True,
        parcel_embed: bool = True,
        space_rope: bool = True,
        deep_sup: bool = True,
        mae: bool = False,
        native_fine_hga: bool = False,
        early_fusion: bool = False,
        no_fusion: bool = False,
        r6: bool = False,
        nf_decimate: int = NOFUSION_DECIMATE,
        mae_stream_weight: str = "equal",
        mae_force_norm_pix: bool = False,
        mae_hga_envelope: bool = False,
        d_model: int = D_MODEL,
    ) -> None:
        super().__init__()
        # The stem lives inside the objective's EMA-mirrored target tower (V-JEPA
        # EMAs the patch-embed too), so the model owns only the objective + mask cfg.
        # deep_sup default ON (#61, Ben-greenlit copy-exactly); deep_sup=False = the
        # single-tap ablation arm.
        # early_fusion (r5, Chang 2-stream): EarlyFusionStem + single-band grid + single
        # temporal mask + accept-the-bleed scoring. The mask sampler / grid / flags switch
        # to their r5 variants below; the objective owns the frontend + scoring swap.
        # no_fusion (v3r5nf): r5-fused's fully-separated sibling — NoFusionStem (2 stems) +
        # two-band grid + INDEPENDENT per-stream masks + accept-the-bleed scoring, MAE-only.
        # r6: arm0's frontend VERBATIM (same 32 Hz caches, same PerBandStem) with four deltas —
        # shaft-batched data regime, no norm_pix, no margin gate, PER-SENSOR band masks
        # (sample_masks_r6) + a predictor band embed. MAE-only. See V3JepaObjective.__init__.
        self.objective = V3JepaObjective(
            n_parcels=n_parcels, target_ln=target_ln, parcel_embed=parcel_embed,
            space_rope=space_rope, deep_sup=deep_sup,
            mae=mae, native_fine_hga=native_fine_hga, early_fusion=early_fusion,
            no_fusion=no_fusion, r6=r6, nf_decimate=nf_decimate,
            mae_stream_weight=mae_stream_weight,
            mae_force_norm_pix=mae_force_norm_pix,
            mae_hga_envelope=mae_hga_envelope,
            d_model=d_model,
        )
        self.native_fine_hga = bool(native_fine_hga)
        self.early_fusion = bool(early_fusion)
        self.no_fusion = bool(no_fusion)
        self.r6 = bool(r6)
        self.nf_decimate = int(nf_decimate)
        self.mask_cfg = mask_cfg

    def forward(
        self,
        band_inputs: Sequence[Tensor],
        geom: L1Geometry,
        parcel_id: Tensor,
        *,
        generator: torch.Generator,
        collect_taps: bool = False,
        grid_max_seqlen: int | None = None,
        m_vis: int | None = None,
        pack_max_seqlen: int | None = None,
    ) -> JepaOutput:
        B, N = band_inputs[0].shape[0], band_inputs[0].shape[1]
        T = clock_length_32hz(
            band_inputs,
            native_fine_hga=self.native_fine_hga,
            early_fusion=self.early_fusion,
            no_fusion=self.no_fusion,
        )
        # masking is the sole augmentation: per-shaft-balanced spatial contact drop + per-band
        # (SLOW/MID/HGA) independent temporal blocks (masking.V3Masks). The flat r4 objective
        # derives the visible/scored token sets from these directly (pack_r4.token_flags).
        # r5 (early_fusion): ONE temporal mask over the single 32 Hz band (sample_masks_r5).
        if self.early_fusion:
            masks = sample_masks_r5(
                geom, N, n_time=T, n_rows=B, generator=generator, cfg=self.mask_cfg
            )
        elif self.no_fusion:
            # mask on the TOKEN grid (T tokens @32 Hz for decimate 2; T/2 @16 Hz for fast). The
            # objective derives the SAME n_tok from T (nf_token_geometry), so mask + grid agree.
            n_tok, _ = nf_token_geometry(T, decimate=self.nf_decimate)
            masks = sample_masks_r5nf(
                geom, N, n_time=n_tok, n_rows=B, generator=generator, cfg=self.mask_cfg
            )
        elif self.r6:
            # r6: per-SENSOR-INDEPENDENT, per-band (SLOW/MID/HGA) width-4 blocks on the 32 Hz
            # clock T (the sampler derives each band's own length T/8, T/2, T).
            masks = sample_masks_r6(
                geom, N, n_time=T, n_rows=B, generator=generator, cfg=self.mask_cfg
            )
        else:
            masks = sample_masks(
                geom, N, n_time=T, n_rows=B, generator=generator, cfg=self.mask_cfg
            )
        # grid_max_seqlen / m_vis / pack_max_seqlen are the per-session Python-int shape
        # constants ``session_plan`` precomputes (the module caches + passes them each step);
        # they let the objective skip the per-step ``.item()`` host syncs. None ⇒ eager path.
        return self.objective(
            band_inputs,
            geom,
            parcel_id,
            masks,
            collect_taps=collect_taps,
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
        per-shaft spatial masking (``d_s`` fixed) + per-band temporal masking at an exact count
        ⇒ every clip masks ``d_s·k_full + (n_s−d_s)·T_masked`` tokens per shaft — so a single
        representative mask (fixed seed) yields the values every clip shares. That holds however
        finely the temporal mask is drawn: r4 masks the same band-b frames globally, r6 draws per
        SENSOR, but both hide exactly ``round(frac·T_b)`` band-b frames per contact, so the
        per-shaft total is the same constant either way."""
        N = int(geom.valid.sum())
        gen = torch.Generator(device=geom.gather_idx.device).manual_seed(0)
        if self.early_fusion:  # r5 single-band grid + single-temporal-mask flags
            grid = build_r5_grid(geom, n_time=n_time)
            parcel_packed = parcel_id[grid.contact]
            masks = sample_masks_r5(
                geom, N, n_time=n_time, n_rows=1, generator=gen, cfg=self.mask_cfg
            )
            masked, _ = token_flags_r5(grid, masks)
        elif self.no_fusion:  # v3r5nf two-band grid + independent per-stream mask flags
            # n_time arrives as the TOKEN count (the module converts via nf_token_geometry); the
            # grid still needs the RoPE stride (1 for decimate 2, 2 for fast) — but max_seqlen/
            # m_vis are stride-invariant (they count tokens), so this plan is decimate-correct.
            grid = build_r5nf_grid(geom, n_time=n_time, time_stride=self.nf_decimate // 2)
            parcel_packed = parcel_id[grid.contact]
            masks = sample_masks_r5nf(
                geom, N, n_time=n_time, n_rows=1, generator=gen, cfg=self.mask_cfg
            )
            masked, _ = token_flags_r5nf(grid, masks)
        elif self.r6:  # r6: r4 grid + per-sensor 3-band mask flags (no margin gate)
            grid = build_r4_grid(geom, n_time=n_time)
            parcel_packed = parcel_id[grid.contact]
            masks = sample_masks_r6(
                geom, N, n_time=n_time, n_rows=1, generator=gen, cfg=self.mask_cfg
            )
            masked, _ = token_flags_r6(grid, masks)
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
