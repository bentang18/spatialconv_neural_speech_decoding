"""Speech-decoding model configs.

`TinyMLP` is the substrate-integration smoke-test model. v14 (Perceiver IO +
parcel-id-tagged latents) lands here later.
"""

from speech_decoding.models.mlp import TinyMLP, TinyMLPModel

__all__ = ["TinyMLP", "TinyMLPModel"]
