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
``tube_ratio=0.25``, ``tie_lfs=True``, ``ema_tau=0.9992`` — impl-plan + memos)
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
    pred_layers: int
    pred_dim: int
    n_parcels: int

    # --- locked first-run science knobs (impl-plan + memos) -----------------
    k: int = 2
    tube_ratio: float = 0.25
    tie_lfs: bool = True
    ema_tau: float = 0.9992

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:  # noqa: ARG002
        return V14ConvergedV2(
            V14ConvergedV2Config(
                d_model=self.d_model,
                n_heads=self.n_heads,
                frontend_layers=self.frontend_layers,
                latent_layers=self.latent_layers,
                pred_layers=self.pred_layers,
                pred_dim=self.pred_dim,
                n_parcels=self.n_parcels,
                k=self.k,
                tube_ratio=self.tube_ratio,
                tie_lfs=self.tie_lfs,
                ema_tau=self.ema_tau,
            )
        )


__all__ = ["V14ConvergedV2Net"]
