"""BrainTreebank cohort — NeuralSet Study + Ieeg + loader.

Used by the Neuroprobe hillclimb path. Continuous-trial sEEG events; raw 2048 Hz
voltage with no re-reference (Stage-1 ablation cells re-add CAR / Laplacian / HG
envelope on top — those are not loader behavior).
"""

from speech_decoding.studies.braintreebank.study import (
    BraintreebankIeeg,
    BraintreebankStudy,
)

__all__ = ["BraintreebankIeeg", "BraintreebankStudy"]
