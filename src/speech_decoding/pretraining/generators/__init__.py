"""Synthetic data generators for NCA-JEPA pretraining."""
from speech_decoding.pretraining.generators.base import Generator
from speech_decoding.pretraining.generators.smooth_ar import SmoothARGenerator

GENERATORS = {
    "smooth_ar": SmoothARGenerator,
}
