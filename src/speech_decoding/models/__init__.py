"""Speech-decoding model configs.

`TinyMLP` is the substrate-integration smoke-test model. `V14ParcelPerceiver`
is the v14 architecture (Perceiver-IO with parcel-id-tagged latents,
log(support+eps) cross-attn bias, factorized t×parcel latent stack, frozen
PMA k=1 parcel collapse, Phase-4 flat head). See
``memory/project_v14_arch_revision_2026_05_19_v3.md`` (base) and
``memory/project_v14_arch_post_v3_amendment_2026_05_19.md`` (v4 amendment).
"""

from speech_decoding.models.mlp import TinyMLP, TinyMLPModel
from speech_decoding.models.v14_encoder import (
    V14ParcelCollapsePMA,
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
    V14ParcelPerceiverWithHead,
    V14Phase3DistillHead,
    V14Phase3TimePoolTriangular,
    V14Phase4FlatHead,
)

__all__ = [
    "TinyMLP",
    "TinyMLPModel",
    "V14ParcelCollapsePMA",
    "V14ParcelPerceiver",
    "V14ParcelPerceiverModel",
    "V14ParcelPerceiverWithHead",
    "V14Phase3DistillHead",
    "V14Phase3TimePoolTriangular",
    "V14Phase4FlatHead",
]
