"""NeuralTrain/Exca experiment scaffolding for v14 runs."""

from __future__ import annotations

import typing as tp

# ExperimentLogger & friends are stdlib + pydantic only — safe on any env,
# including the CPU-only DCC baseline venv.
from speech_decoding.experiments.experiment_logging import (
    ExperimentLogger,
    ExperimentRunRecord,
    collect_experiment_records,
)

# Data / Experiment / BrainModule pull in torch + lightning. `lightning` is a
# GPU-training-only dependency (absent from pyproject/uv.lock), so importing it
# at package-load time makes the whole `experiments` package unimportable on a
# CPU env — which breaks the Stage-0 baseline runner, which only needs
# ExperimentLogger. Defer them to first attribute access (PEP 562); fully
# backward compatible (`from speech_decoding.experiments import Experiment` still
# works, it just imports lightning on demand).
if tp.TYPE_CHECKING:
    from speech_decoding.experiments.data import Data
    from speech_decoding.experiments.experiment import Experiment
    from speech_decoding.experiments.module import BrainModule


def __getattr__(name: str):  # PEP 562 lazy loader for the lightning-backed symbols
    if name == "Data":
        from speech_decoding.experiments.data import Data

        return Data
    if name == "Experiment":
        from speech_decoding.experiments.experiment import Experiment

        return Experiment
    if name == "BrainModule":
        from speech_decoding.experiments.module import BrainModule

        return BrainModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "BrainModule",
    "Data",
    "Experiment",
    "ExperimentLogger",
    "ExperimentRunRecord",
    "collect_experiment_records",
]
