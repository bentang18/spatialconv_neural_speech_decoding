"""Speech-decoding model configs.

`TinyMLP` is the substrate-integration smoke-test model. `V14ParcelPerceiver`
is the v14 architecture (parcel-id-tagged latents, a hard block-diagonal
per-parcel pool over the one-hot DK ``support`` — B36 2026-06-01 replaced the
soft log(support+eps) cross-attn bias — factorized t×parcel latent stack, and a
Phase-4 readout that collapses parcels with the frozen P3-PMA then
mean-over-time → Linear — B35). See
``memory/project_v14_b36_perparcel_pool_structured_jepa_2026_06_01.md`` (current
hard-pool lock) and ``memory/project_v14_arch_revision_2026_05_19_v3.md`` (base).
"""

from speech_decoding.models.mlp import TinyMLP, TinyMLPModel
from speech_decoding.models.v14_converged_config import V14Converged
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
    "V14Converged",
    "V14MeanPoolLinearHead",
    "V14ParcelCollapsePMA",
    "V14ParcelPerceiver",
    "V14ParcelPerceiverModel",
    "V14ParcelPerceiverWithHead",
    "V14PerTaskAttentivePooler",
    "V14Phase4FlatHead",
    "V14PmaReadout",
]
