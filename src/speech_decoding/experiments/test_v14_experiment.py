"""SCAFFOLD-01 tests: V14Experiment phase-parametrized scaffold.

Per fix-list item SCAFFOLD-01 / NT01: ``V14Experiment`` exposes a
``phase: Literal[1, 2, "3a", "3b", 4]`` discriminator. One class branches
per phase. The runtime ``_train_and_test`` path is gated on the V14Data
contract landing, so these tests pin the *structural* contract only:

* The class accepts every legal phase value.
* Illegal phase values raise pydantic validation errors.
* The per-phase loss-coefficient resolver returns the canonical
  ``(W_PRE_FRAME, W_MID_SLOT, W_POST_FRAME, W_POST_UTTERANCE, W_DKOLEO_M4)``
  tuple for P1+P2 (the unified-objective phases) and ``None`` / a single
  Whisper distillation weight for P3/P4 (so a probe/grep can detect
  drift without instantiating the data pipeline).
"""

from __future__ import annotations

import pytest

from speech_decoding.experiments.v14_experiment import (
    V14Experiment,
    v14_phase_loss_coefficients,
)
from speech_decoding.ssl.total_loss import (
    W_DKOLEO_M4,
    W_MID_SLOT,
    W_POST_FRAME,
    W_POST_UTTERANCE,
    W_PRE_FRAME,
)


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


def test_v14_phase_loss_coefficients_p1_p2_unified_5term() -> None:
    """P1 and P2 share the LOSS-01 5-term lock."""
    for phase in (1, 2):
        coeffs = v14_phase_loss_coefficients(phase)
        assert coeffs == (
            W_PRE_FRAME, W_MID_SLOT, W_POST_FRAME,
            W_POST_UTTERANCE, W_DKOLEO_M4,
        ), f"phase {phase} drifted from LOSS-01 lock"


def test_v14_phase_loss_coefficients_p3_returns_distill_weight() -> None:
    """P3a / P3b drop the 5-term objective; the resolver signals this by
    returning a single-element tuple carrying the Whisper distillation
    weight (locked at 1.0 — Smooth-L1 with β=1.0 default per Wave 2)."""
    for phase in ("3a", "3b"):
        coeffs = v14_phase_loss_coefficients(phase)
        assert coeffs == (1.0,), f"phase {phase} unexpected coeffs {coeffs}"


def test_v14_phase_loss_coefficients_p4_returns_empty() -> None:
    """Phase-4 = downstream linear probe; no SSL loss applies."""
    assert v14_phase_loss_coefficients(4) == ()


def test_v14_phase_loss_coefficients_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError, match="unknown phase"):
        v14_phase_loss_coefficients("p1")  # type: ignore[arg-type]
