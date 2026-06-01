"""Speech-decoding model configs.

`TinyMLP` is the substrate-integration smoke-test model. `V14ParcelPerceiver`
is the v14 architecture (Perceiver-IO with parcel-id-tagged latents,
log(support+eps) cross-attn bias, factorized t×parcel latent stack, and a
Phase-4 readout that collapses parcels with the frozen P3-PMA then
mean-over-time → Linear — B35). See
``memory/project_v14_arch_revision_2026_05_19_v3.md`` (base) and
``memory/project_v14_arch_post_v3_amendment_2026_05_19.md`` (v4 amendment).
"""

from speech_decoding.models.mlp import TinyMLP, TinyMLPModel
from speech_decoding.models.v14_encoder import (
    V14MeanPoolLinearHead,
    V14ParcelCollapsePMA,
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
    V14ParcelPerceiverWithHead,
    V14PerTaskAttentivePooler,
    V14Phase4FlatHead,
    V14PmaReadout,
)

__all__ = [
    "TinyMLP",
    "TinyMLPModel",
    "V14MeanPoolLinearHead",
    "V14ParcelCollapsePMA",
    "V14ParcelPerceiver",
    "V14ParcelPerceiverModel",
    "V14ParcelPerceiverWithHead",
    "V14PerTaskAttentivePooler",
    "V14Phase4FlatHead",
    "V14PmaReadout",
]
