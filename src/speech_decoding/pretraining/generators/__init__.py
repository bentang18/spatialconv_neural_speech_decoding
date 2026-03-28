"""Synthetic data generators for NCA-JEPA pretraining."""
from speech_decoding.pretraining.generators.base import Generator
from speech_decoding.pretraining.generators.smooth_ar import SmoothARGenerator
from speech_decoding.pretraining.generators.switching_lds import SwitchingLDSGenerator

GENERATORS = {
    "smooth_ar": SmoothARGenerator,
    "switching_lds": SwitchingLDSGenerator,
}
