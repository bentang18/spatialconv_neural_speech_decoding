"""T3.1: --phase flag scaffold.

Phases 1/2/3 must raise NotImplementedError citing the gating blocker IDs;
Phase 4 (default) must fall through to the existing Phase-4 dispatch path
(here exercised with --dry-run so no Experiment is built)."""

from __future__ import annotations

import pytest

from speech_decoding.experiments.dispatch_v14 import main


@pytest.mark.parametrize(
    "phase, expected_substrings",
    [
        # B29 Item 1 (2026-05-27 PM-late) collapsed P1 + P2 into a single
        # joint SSL phase routed through V14JointExperiment. The legacy
        # B01/B02/IE09/M07/M15/M18/B06/M04/M05/M19 blocker IDs cited in
        # the prior gating message are all ✅ closed in docs/neuroprobe/
        # v14_blockers.md or were retired entirely. The current gating
        # surface is the in-flight wiring tasks (B2.x) and the B30
        # dispatch-sister-flags row.
        (1, ("B2.1", "B2.2", "B2.3", "B30-dispatch-sister-flags")),
        (2, ("B2.1", "B2.2", "B2.3", "B30-dispatch-sister-flags")),
        (3, ("Phase-3 distillation", "frozen SSL checkpoint", "B29 Item 1")),
    ],
)
def test_phase_1_2_3_raise_with_blocker_ids(
    phase: int, expected_substrings: tuple[str, ...]
) -> None:
    with pytest.raises(NotImplementedError) as exc_info:
        main(["--phase", str(phase), "--dry-run"])
    message = str(exc_info.value)
    for token in expected_substrings:
        assert token in message, f"phase {phase}: missing blocker id {token}"


def test_phase_4_is_default_and_falls_through_to_dispatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--phase 4 is the current Phase-4 downstream path; --dry-run exits
    cleanly (no Experiment built)."""
    rc = main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "V14 dispatch" in out


def test_phase_4_explicit_falls_through_to_dispatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["--phase", "4", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "V14 dispatch" in out


def test_invalid_phase_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit):
        main(["--phase", "0", "--dry-run"])
    with pytest.raises(SystemExit):
        main(["--phase", "5", "--dry-run"])
