"""SCAFFOLD-01 (NT01): ``V14Experiment`` — phase-parametrized NeuralTrain
experiment for the v14 cross-subject SSL stack.

Per ``docs/neuroprobe/v14_implementation_fix_list.md`` (SCAFFOLD-01) and
``docs/neuroprobe/v14_blockers.md`` NT01, the v14 training program runs
five distinct phases — Phase-1 (anatomy-OFF SSL), Phase-2 (anatomy-ON SSL
+ shaft-block electrode mask), Phase-3a (Whisper-L8 MLP-adapter warmup),
Phase-3b (slow-LR student unfreeze), and Phase-4 (frozen-encoder linear
probe). One class branches per phase via the ``phase`` discriminator.

This module is structural: the class declaration, the phase discriminator,
and the per-phase loss-coefficient resolver are stable. The full runtime
path (V14Data, EMA teacher, Predictor2Block warm-start, checkpoint reload)
is gated on SCAFFOLD-02..06 and the downstream blockers (B03/B05/B06).

Wiring the LOSS-01 composer (Wave 6) into the per-phase ``_train_step``
happens here once SCAFFOLD-02 (V14Data) lands; the resolver returns the
coefficients the trainer should apply, so the wiring can be tested at
unit granularity without instantiating the data pipeline.
"""

from __future__ import annotations

from typing import Literal

import pydantic

from speech_decoding.experiments.experiment import Experiment
from speech_decoding.ssl.total_loss import (
    W_DKOLEO_M4,
    W_MID_SLOT,
    W_POST_FRAME,
    W_POST_UTTERANCE,
    W_PRE_FRAME,
)


V14Phase = Literal[1, 2, "3a", "3b", 4]


# Phase-3 (Whisper distillation) coefficient. Single Smooth-L1 term,
# β=1.0 default (Wave 2 lock). Surfaced here so phase-3 wiring grep'able.
W_PHASE3_DISTILL: float = 1.0


def v14_phase_loss_coefficients(phase: V14Phase) -> tuple[float, ...]:
    """Per-phase loss-coefficient tuple.

    * P1 / P2: the LOSS-01 5-term lock
      ``(W_PRE_FRAME, W_MID_SLOT, W_POST_FRAME, W_POST_UTTERANCE,
      W_DKOLEO_M4)``.
    * P3a / P3b: ``(W_PHASE3_DISTILL,)`` — single Whisper Smooth-L1 term.
    * P4: ``()`` — frozen-encoder linear probe; no SSL loss.

    Raises :class:`ValueError` on unknown phases.
    """
    if phase in (1, 2):
        return (W_PRE_FRAME, W_MID_SLOT, W_POST_FRAME, W_POST_UTTERANCE, W_DKOLEO_M4)
    if phase in ("3a", "3b"):
        return (W_PHASE3_DISTILL,)
    if phase == 4:
        return ()
    raise ValueError(f"unknown phase: {phase!r}")


class V14Experiment(Experiment):
    """Phase-parametrized NeuralTrain experiment for v14 SSL.

    Inherits the full NeuralTrain/Exca scaffolding from
    :class:`speech_decoding.experiments.experiment.Experiment` (logger,
    callbacks, trainer, infra). Adds the ``phase`` discriminator.

    The runtime branching (EMA teacher in P1/P2; Predictor2Block
    warm-start at P1→P2; Whisper adapter+pool in P3; frozen-encoder
    probe in P4) lives in ``_train_and_test`` once SCAFFOLD-02 lands.

    Coefficients for the active phase are resolved via
    :func:`v14_phase_loss_coefficients` so the trainer never hard-codes
    them — a single source of truth keeps LOSS-01 and the phase
    branching coupled.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    phase: V14Phase

    def loss_coefficients(self) -> tuple[float, ...]:
        """Coefficients for the active phase. Convenience accessor."""
        return v14_phase_loss_coefficients(self.phase)
