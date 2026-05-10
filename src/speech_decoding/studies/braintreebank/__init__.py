"""BrainTreebank cohort — local NeuroAI Study + raw h5 loader.

Used by the Neuroprobe hillclimb path. Continuous-trial sEEG events expose raw
2048 Hz voltage with no re-reference.
"""

from speech_decoding.studies.braintreebank.study import Wang2024Treebank

__all__ = ["Wang2024Treebank"]
