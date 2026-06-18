"""NeuralTrain ``BaseModelConfig`` for the converged v14 SSL model.

Registers ``V14ConvergedSSL`` (``models/v14_converged.py``) under the
discriminator name ``"V14Converged"`` so the dispatch can select it the same
way it selects the live ``V14ParcelPerceiver`` — a ``{"name": "V14Converged",
...}`` dict on the Experiment's ``brain_model_config`` field.

The architecture-SHAPE fields carry NO defaults (``d_model``, ``n_parcels``,
``n_heads``, the layer counts, the predictor dims/depths) — per the
discuss-before-code / no-pre-committed-numeric-defaults rule, the dispatch (and
Ben) must name every shape explicitly at launch; nothing here silently picks a
run width. Only the neutral identity ``lambda_m2 = lambda_m4 = 1.0`` and the
LOCKED-arch ``freq_pos = "learned"`` (the memo's "1d learnable freq tag, not
sinusoidal") carry defaults.

``build(n_in_channels, n_outputs)`` ignores both args: the band geometry is
fixed by ``BANDS`` (not the electrode count) and SSL has no classification head
(``n_outputs`` is informational, mirroring ``V14JointExperiment`` passing 1).
"""

from __future__ import annotations

import typing as tp

from torch import nn

from neuraltrain.models.base import BaseModelConfig

from speech_decoding.models.v14_converged import V14ConvergedSSL


class V14Converged(BaseModelConfig):
    """Config for the self-contained converged-arch SSL model.

    ``.build()`` returns a :class:`V14ConvergedSSL` (frontend + latent + the two
    paradigm-B predictors + the frontend-only EMA teacher), which the
    ``V14ConvergedExperiment`` wraps in a ``V14ConvergedBrainModule``.
    """

    # --- architecture shape (REQUIRED — no silent run defaults) -------------
    d_model: int
    n_parcels: int
    n_heads: int
    frontend_layers: int
    latent_layers: int
    m2_pred_dim: int
    m2_pred_layers: int
    m4_pred_dim: int
    m4_pred_layers: int

    # --- neutral identity / locked-arch defaults ----------------------------
    lambda_m2: float = 1.0
    lambda_m4: float = 1.0
    freq_pos: tp.Literal["learned", "sinusoidal"] = "learned"
    # M4 heteroscedastic down-weight (Ben 2026-06-18): electrode-MEAN parcel target
    # variance ∝ 1/n → down-weight low-n parcels. ON by default with the B37 lock
    # (α=1.0, n_ref=11); the `--converged-m4-precision-off` sister disables it.
    m4_precision_weight: bool = True
    m4_precision_alpha: float = 1.0
    m4_precision_n_ref: float = 11.0

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:  # noqa: ARG002
        return V14ConvergedSSL(
            self.d_model,
            self.n_parcels,
            n_heads=self.n_heads,
            frontend_layers=self.frontend_layers,
            latent_layers=self.latent_layers,
            m2_pred_dim=self.m2_pred_dim,
            m2_pred_layers=self.m2_pred_layers,
            m4_pred_dim=self.m4_pred_dim,
            m4_pred_layers=self.m4_pred_layers,
            lambda_m2=self.lambda_m2,
            lambda_m4=self.lambda_m4,
            freq_pos=self.freq_pos,
            m4_precision_weight=self.m4_precision_weight,
            m4_precision_alpha=self.m4_precision_alpha,
            m4_precision_n_ref=self.m4_precision_n_ref,
        )


__all__ = ["V14Converged"]
