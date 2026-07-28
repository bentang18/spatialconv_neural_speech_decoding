"""Pin our BrainTreebank pretraining contract to upstream Neuroprobe's ``SUBMIT.md``.

The leaderboard's pretraining rule is enforced by a SIGNED ATTESTATION, not by code:

    "I attest that the submitted model was not pretrained on any data that intersects
     with any data of Neuroprobe."

So the constants that define which sessions we may pretrain on are a legal contract, and a
drift between them and upstream is not a lint issue -- it is a false attestation. Upstream can
edit SUBMIT.md at any time (0.1.8 added MNI coordinates; the allowed list could move next), and
our copy of the list is a hand-transcription. This test re-derives the list FROM THE SOURCE TEXT
and fails if the transcription no longer matches.

Skipped when no upstream checkout is present, so it never blocks CI on a machine without it.
Point it at a clone with NEUROPROBE_REPO=/path/to/neuroprobe.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from speech_decoding.studies.braintreebank.manifest import (
    BT_LITE_SESSIONS,
    BT_PRETRAIN_ALLOWED_SESSIONS,
    BT_PRETRAIN_PARTIAL_SESSIONS,
)

_DEFAULT = Path(__file__).resolve().parents[4].parent / "neuroprobe"


def _repo() -> Path:
    env = os.environ.get("NEUROPROBE_REPO")
    return Path(env) if env else _DEFAULT


def _submit_md() -> str:
    p = _repo() / "SUBMIT.md"
    if not p.is_file():
        pytest.skip(f"no upstream neuroprobe checkout at {_repo()} (set NEUROPROBE_REPO)")
    return p.read_text()


def _btbank_sessions(line: str) -> set[tuple[int, int]]:
    return {(int(s), int(t)) for s, t in re.findall(r"btbank(\d+)_(\d+)", line)}


def _allowed_and_partial(text: str) -> tuple[set, set]:
    """Both fenced btbank lists in the 'Pretraining guidelines' section.

    The first fenced block is the standard allowed set; the second is the special "partial"
    sessions carved out of subjects 7 and 10. Parsed by content (which subjects appear), not by
    block order, so a reordering upstream cannot silently swap them.
    """
    blocks = [b for b in re.findall(r"```\s*\n(btbank[^`]*?)\n?```", text)]
    assert blocks, "no fenced btbank list found in SUBMIT.md"
    found = [_btbank_sessions(b) for b in blocks if _btbank_sessions(b)]
    partial = [s for s in found if all(t >= 100 for _, t in s)]
    standard = [s for s in found if any(t < 100 for _, t in s)]
    assert len(standard) == 1 and len(partial) == 1, (
        f"expected one standard and one partial btbank block, got "
        f"{[sorted(f) for f in found]}"
    )
    return standard[0], partial[0]


def test_allowed_pretrain_sessions_match_upstream():
    allowed, _ = _allowed_and_partial(_submit_md())
    assert set(BT_PRETRAIN_ALLOWED_SESSIONS) == allowed, (
        "BT_PRETRAIN_ALLOWED_SESSIONS has drifted from SUBMIT.md.\n"
        f"  ours only:     {sorted(set(BT_PRETRAIN_ALLOWED_SESSIONS) - allowed)}\n"
        f"  upstream only: {sorted(allowed - set(BT_PRETRAIN_ALLOWED_SESSIONS))}"
    )


def test_partial_pretrain_sessions_match_upstream():
    _, partial = _allowed_and_partial(_submit_md())
    assert set(BT_PRETRAIN_PARTIAL_SESSIONS) == partial


def test_off_limits_sessions_are_exactly_our_lite_set():
    """The prose bullet list of off-limits sessions must equal BT_LITE_SESSIONS.

    This is the other half of the contract: SUBMIT.md states the forbidden sessions as prose
    bullets ("Subject 1: Trials 1, 2"), independently of the allowed fenced block. If the two
    ever disagree, our single BT_LITE_SESSIONS constant cannot represent both.
    """
    text = _submit_md()
    off = set()
    for subj, trials in re.findall(r"^- Subject (\d+): Trials? ([\d, ]+)$", text, re.M):
        for t in re.findall(r"\d+", trials):
            off.add((int(subj), int(t)))
    assert off, "no 'Subject N: Trials ...' bullets parsed from SUBMIT.md"
    assert off == set(BT_LITE_SESSIONS), (
        f"off-limits bullets != BT_LITE_SESSIONS\n"
        f"  upstream only: {sorted(off - set(BT_LITE_SESSIONS))}\n"
        f"  ours only:     {sorted(set(BT_LITE_SESSIONS) - off)}"
    )


def test_allowed_and_off_limits_are_disjoint():
    allowed, partial = _allowed_and_partial(_submit_md())
    assert not (allowed | partial) & set(BT_LITE_SESSIONS)


def test_launched_pretrain_sessions_are_legal():
    """The 13 sessions every v3 pretraining launcher actually passes must be attestation-legal.

    Transcribed from the STORED batch scripts of the live ablation jobs (2756178/9/80) and the
    r6 launcher family on 2026-07-28 -- read back with `scontrol write batch_script`, not from a
    memo. All 79 launchers on the cluster used only subsets of this set. Pinned here so a future
    edit to the launcher template is caught against upstream rather than at submission time.
    """
    launched = {(1, 0), (2, 1), (2, 2), (2, 3), (2, 5), (2, 6),
                (3, 2), (4, 2), (6, 0), (6, 1), (6, 4), (8, 0), (9, 0)}
    allowed, _ = _allowed_and_partial(_submit_md())
    assert launched <= allowed, f"ILLEGAL pretraining sessions: {sorted(launched - allowed)}"
    assert not launched & set(BT_LITE_SESSIONS)
