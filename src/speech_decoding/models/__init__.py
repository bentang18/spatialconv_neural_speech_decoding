"""Speech-decoding model configs.

`TinyMLP` is the substrate-integration smoke-test model. `V14ParcelPerceiver`
is the first-pass v14 architecture (Perceiver-IO with parcel-id-tagged latents,
log(support+eps) cross-attn bias, DETR readout). See
``memory/project_v14_encoder_design_2026_05_13.md``.
"""

from speech_decoding.models.mlp import TinyMLP, TinyMLPModel
from speech_decoding.models.v14_encoder import (
    V14ClassifierHead,
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
    V14ParcelPerceiverWithHead,
)

__all__ = [
    "TinyMLP",
    "TinyMLPModel",
    "V14ClassifierHead",
    "V14ParcelPerceiver",
    "V14ParcelPerceiverModel",
    "V14ParcelPerceiverWithHead",
]
