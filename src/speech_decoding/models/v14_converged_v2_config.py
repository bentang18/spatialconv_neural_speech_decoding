"""NeuralTrain ``BaseModelConfig`` for the converged-v2 SSL model.

Registers :class:`speech_decoding.models.v14_converged_v2.V14ConvergedV2` under
the discriminator name ``"V14ConvergedV2Net"`` so the dispatch selects it with a
``{"name": "V14ConvergedV2Net", ...}`` dict on the Experiment's
``brain_model_config`` field — the same mechanism that selects the 3STFT
``"V14Converged"``.

Architecture-SHAPE fields carry NO defaults (``d_model``, ``n_heads``, the layer
counts, ``pred_dim``, ``n_parcels``) — per the discuss-before-code /
no-pre-committed-numeric-defaults rule, the dispatch (and Ben) name every shape
explicitly at launch. Only the LOCKED first-run science knobs (``k=2``,
``tube_ratio=0.25``, ``pool_op=None`` (auto), ``ema_tau=0.9992`` — impl-plan + memos)
carry defaults.

``build(n_in_channels, n_outputs)`` ignores both args: the band geometry is fixed
by ``BANDS_V2`` (not the electrode count) and SSL has no classification head.
"""

from __future__ import annotations

from torch import nn

from neuraltrain.models.base import BaseModelConfig

from speech_decoding.models.v14_converged_v2 import (
    V14ConvergedV2,
    V14ConvergedV2Config,
)


class V14ConvergedV2Net(BaseModelConfig):
    """Config for the self-contained converged-v2 SSL model.

    ``.build()`` returns a :class:`V14ConvergedV2` (2-band frontend + set-pool +
    latent + dual-depth M2/M4 predictors + frozen EMA teacher), which the
    ``V14ConvergedV2BrainModule`` wraps. The model owns the teacher, the
    shared-denominator loss, the static drop-not-pad forward, and mask sampling.
    """

    # --- architecture shape (REQUIRED — no silent run defaults) -------------
    d_model: int
    n_heads: int
    frontend_layers: int
    latent_layers: int
    m2_pred_layers: int
    m4_pred_layers: int
    pred_dim: int
    n_parcels: int

    # --- locked first-run science knobs (impl-plan + memos) -----------------
    k: int = 2
    tube_ratio: float = 0.25
    # Pool K/V read-operator typing; None ⇒ auto → "band" (n_op=2, LFS | HGA — the
    # physiology-backed default; see V14ConvergedV2Config.pool_op_resolved).
    # "shared"|"band"|"patch" (1/2/4) override — shared/patch are the tie/untie ablations.
    pool_op: str | None = None
    ema_tau: float = 0.99925   # V-JEPA 2 §2.4 lock; matches dispatch DEFAULT_EMA_TAU

    # --- Run-A bundle (default LEGACY for checkpoint resume) ----------------
    # ``m3_pred_layers`` is the master switch: set it (e.g. 6) ⇒ the Run-A
    # architecture (M3 pool-inpaint head + pool LayerNorm + tubed-only M4 +
    # per-head loss). None ⇒ legacy assembly resumes byte-identical.
    m3_pred_layers: int | None = None
    qk_norm: bool = False
    w_m2: float = 1.0
    w_m3: float = 1.0
    w_m4: float = 1.0
    support_weight: bool = False

    # --- Run-B bundle (masked-electrode-prediction pool head) ---------------
    # ``m3_drop_frac`` (ρ) is the master switch — set it (e.g. 0.3) to REPLACE the
    # M3 pool-inpaint head with the per-parcel masked-electrode-prediction head
    # (whole-electrode drop, survivor floor ``m3_min_keep``, native-RAS relational
    # reconstruction of the teacher frontend feature off the k pool seeds). Locked
    # later; exposed now so the drop rate + floor are named at launch, not hidden.
    m3_drop_frac: float | None = None
    m3_min_keep: int = 3
    w_melec: float = 1.0
    sigma_mm: float = 12.0
    geom_n_freqs: int = 4
    seed_offset_sigma_mm: float = 0.0  # isotropic k-seed offset init σ (mm); 0=zero-init
    m2_hetero: bool = True  # canonical Run-B: electrodes are the masking unit

    # --- Run-B variant: M4→teacher-M3 + V-JEPA 2.1 context loss -------------
    # ``m4_recon_m3`` retargets the M4 head to the teacher's M3 (pool + parcel-
    # embed; teacher latent NOT run) and reads the predictor's context-position
    # outputs as a second (context) loss, weighted by λ warmed 0→``context_lambda``
    # over ``context_warmup_steps``. M-elec kept. Off ⇒ byte-identical Run-B.
    m4_recon_m3: bool = False
    context_lambda: float = 0.2
    context_warmup_steps: int = 15000
    context_warmup_start_step: int = 0
    # Optional SEPARATE M4-context ramp (own delayed start + width); -1 ⇒ inherit
    # the shared context_warmup_* schedule (byte-identical single-ramp default).
    m4_context_warmup_start_step: int = -1
    m4_context_warmup_steps: int = -1
    # V-JEPA 2.1 context-loss master switch, DECOUPLED from ``m4_recon_m3`` so both
    # A/B arms carry identical context. ``context_taps`` ⊆ {M2, M4, MELEC}.
    context_loss: bool = False
    context_taps: tuple[str, ...] = ("M2", "M4")

    # --- V-JEPA parity knobs (default OFF = byte-identical resume) -----------
    target_ln: bool = False   # affine-free F.layer_norm on loss targets (forward_target)
    pred_ln: bool = False     # affine predictor_norm before head (models/predictor.py)

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:  # noqa: ARG002
        return V14ConvergedV2(
            V14ConvergedV2Config(
                d_model=self.d_model,
                n_heads=self.n_heads,
                frontend_layers=self.frontend_layers,
                latent_layers=self.latent_layers,
                m2_pred_layers=self.m2_pred_layers,
                m4_pred_layers=self.m4_pred_layers,
                pred_dim=self.pred_dim,
                n_parcels=self.n_parcels,
                k=self.k,
                tube_ratio=self.tube_ratio,
                pool_op=self.pool_op,
                ema_tau=self.ema_tau,
                m3_pred_layers=self.m3_pred_layers,
                qk_norm=self.qk_norm,
                w_m2=self.w_m2,
                w_m3=self.w_m3,
                w_m4=self.w_m4,
                support_weight=self.support_weight,
                m3_drop_frac=self.m3_drop_frac,
                m3_min_keep=self.m3_min_keep,
                w_melec=self.w_melec,
                sigma_mm=self.sigma_mm,
                geom_n_freqs=self.geom_n_freqs,
                seed_offset_sigma_mm=self.seed_offset_sigma_mm,
                m2_hetero=self.m2_hetero,
                m4_recon_m3=self.m4_recon_m3,
                context_lambda=self.context_lambda,
                context_warmup_steps=self.context_warmup_steps,
                context_warmup_start_step=self.context_warmup_start_step,
                m4_context_warmup_start_step=self.m4_context_warmup_start_step,
                m4_context_warmup_steps=self.m4_context_warmup_steps,
                context_loss=self.context_loss,
                context_taps=self.context_taps,
                target_ln=self.target_ln,
                pred_ln=self.pred_ln,
            )
        )


__all__ = ["V14ConvergedV2Net"]
