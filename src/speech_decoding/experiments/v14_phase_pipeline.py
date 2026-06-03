"""B36 WS-E (E3): multi-phase driver — chain P1→P2→P3→P4 with ckpt handoff.

Each phase is a separate Exca-cached :meth:`Experiment.run`. The driver wires
one phase's ``snapshot_ckpt_to`` to the next phase's ``pretrained_ckpt`` so the
encoder (and, from P3, the PMA) warm-starts across the boundary while the
phase-local pieces (the predictor at P1/P2, the ``StudentWhisperProjector`` at
P3) are dropped — see :meth:`V14JointBrainModule.transferable_state`.

The driver does not assume what each phase is — it runs an ordered list of
already-configured :class:`Experiment` objects. The capstone (WS-I2) passes
``[p1, p2, p3, p4]``; a unit test passes ``[p1, p2]`` to exercise the
single-boundary handoff. Phase 0 keeps whatever ``pretrained_ckpt`` it was
configured with (normally ``None``).
"""

from __future__ import annotations

import typing as tp
from pathlib import Path

from speech_decoding.experiments.experiment import Experiment


def _phase_ckpt_path(work_dir: Path, index: int) -> Path:
    return work_dir / f"phase_{index}.ckpt"


def configure_phase_handoff(
    phases: tp.Sequence[Experiment], *, work_dir: str | Path,
) -> list[Experiment]:
    """Reconfigure ``phases`` so each snapshots to / loads from the next (E3).

    Pure: returns NEW experiment objects with ``snapshot_ckpt_to`` /
    ``pretrained_ckpt`` set; the inputs are untouched and nothing is written
    to disk. Phase ``i`` snapshots to ``work_dir/phase_{i}.ckpt``; phase
    ``i+1`` loads ``phase_{i}.ckpt``.

    The diff is applied via ``infra.clone_obj`` — NOT ``model_copy``. Exca
    binds each ``TaskInfra`` to the config instance at construction;
    ``model_copy`` sets the pydantic field on the copy but leaves the infra
    bound to the *original* config, so ``@infra.apply run()`` dispatches into
    a body that still reads the pre-copy ``pretrained_ckpt`` (``None``) — the
    handoff would be silently inert and same-config phases would collide on
    one UID. ``clone_obj`` rebuilds the object with the diff applied, rebinding
    the infra and giving each phase a distinct UID. Do not revert to
    ``model_copy``.
    """
    if not phases:
        raise ValueError("configure_phase_handoff needs at least one phase")
    work = Path(work_dir)
    configured: list[Experiment] = []
    for i, phase in enumerate(phases):
        update: dict[str, tp.Any] = {
            "snapshot_ckpt_to": str(_phase_ckpt_path(work, i)),
        }
        if i > 0:
            update["pretrained_ckpt"] = str(_phase_ckpt_path(work, i - 1))
        configured.append(phase.infra.clone_obj(update))
    return configured


def run_phase_pipeline(
    phases: tp.Sequence[Experiment], *, work_dir: str | Path,
) -> list[dict[str, tp.Any]]:
    """Run ``phases`` in order with the ckpt handoff wired (E3/E4).

    Returns the per-phase result dicts from :meth:`Experiment.run`. Each
    phase's snapshot lands at ``work_dir/phase_{i}.ckpt`` and is loaded by the
    next phase before its fit.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    configured = configure_phase_handoff(phases, work_dir=work)
    return [phase.run() for phase in configured]


__all__ = ["configure_phase_handoff", "run_phase_pipeline"]
