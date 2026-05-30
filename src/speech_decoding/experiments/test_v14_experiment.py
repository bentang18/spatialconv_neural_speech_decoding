"""SCAFFOLD-01 tests: V14Experiment phase-parametrized scaffold.

Per fix-list item SCAFFOLD-01 / NT01: ``V14Experiment`` exposes a
``phase: Literal[1, 2, "3a", "3b", 4]`` discriminator. One class branches
per phase. The runtime ``_train_and_test`` path is gated on the V14Data
contract landing, so these tests pin the *structural* contract only:

* The class accepts every legal phase value.
* Illegal phase values raise pydantic validation errors.

The per-phase loss-coefficient resolver was deleted (the loss SSOT is
:func:`speech_decoding.ssl.aggregator.compute_v14_ssl_losses`; B31 2-term
default selected via ``loss_variant``); its drift-prone guard-tests went
with it.
"""

from __future__ import annotations

import pytest

from speech_decoding.experiments.v14_experiment import V14Experiment


@pytest.mark.parametrize("phase", [1, 2, "3a", "3b", 4])
def test_v14_experiment_accepts_every_legal_phase(phase: int | str) -> None:
    """All five phase values land in the Literal."""
    cfg = V14Experiment.model_construct(phase=phase)
    assert cfg.phase == phase


@pytest.mark.parametrize("bad", [0, 3, 5, "3", "3c", "p1", None])
def test_v14_experiment_rejects_illegal_phase(bad: object) -> None:
    """Pydantic must guard against typos and stray phase IDs."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        V14Experiment(phase=bad)  # type: ignore[arg-type]
