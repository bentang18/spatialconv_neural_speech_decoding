"""Fail-closed Neuroprobe pretraining-leakage guard.

The Neuroprobe submission contract (``.cache/neuroprobe_upstream/SUBMIT.md``,
"Pretraining guidelines") requires that "the submitted model was not pretrained
on any data that intersects with any data of Neuroprobe". The off-limits eval
set is exactly :data:`BT_LITE_SESSIONS` (== ``NEUROPROBE_LITE_SUBJECT_TRIALS``).

This guard inspects the *realized* train/val DataLoaders of an SSL/distill phase
and aborts the run if any segment was drawn from an off-limits ``(subject_id,
trial_id)`` session. It is intentionally:

* **Fail-closed** — a BrainTreebank SSL phase whose loaders lack the
  ``subject_id``/``trial_id`` trigger columns cannot be verified, so it aborts
  rather than proceed unverified. No environment-variable bypass exists; a
  legitimate change to the off-limits set is a code change here.
* **Phase-scoped** — only the SSL/distill phases (P1/P2/P3) enforce it. The
  supervised P4 readout *is* the Neuroprobe probe and legitimately trains on the
  eval-split train portion, so it is exempt (its experiment class leaves
  ``enforces_pretrain_leakage_guard`` False).
* **Corpus-scoped** — the off-limits set is BrainTreebank-specific, so non-BT
  studies (SWEC / AJILE12 / D-cohort) are exempt.

Contract memo: ``memory/project_v14_leakage_free_pretraining_contract_2026_06_06.md``.
"""

from __future__ import annotations

import typing as tp

from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS

_WANG_STUDY_NAME = "Wang2024Treebank"


def _iter_steps(study: tp.Any) -> list[tp.Any]:
    """All leaf steps reachable from a study.

    Unwraps ``ns.Chain`` whose ``steps`` may be a list OR an
    ``OrderedDict[str, Step]`` (both are first-class NeuralSet forms — see
    ``neuralset.base.Chain``), and recurses into nested chains. A guard that
    only did ``list(steps)`` would, for the dict form, iterate the string KEYS
    and silently fail to detect the BrainTreebank study — a fail-open hole that
    would let an eval-leaking corpus pretrain unverified."""
    steps = getattr(study, "steps", None)
    if steps is None:
        return [study]
    if isinstance(steps, dict):
        steps = list(steps.values())
    flat: list[tp.Any] = []
    for step in steps:
        if getattr(step, "steps", None) is not None:
            flat.extend(_iter_steps(step))
        else:
            flat.append(step)
    return flat


def study_is_braintreebank(study: tp.Any) -> bool:
    """True iff the study (or any step of an ``ns.Chain``) is the BT study."""
    for step in _iter_steps(study):
        if type(step).__name__ == _WANG_STUDY_NAME:
            return True
        if getattr(step, "name", None) == _WANG_STUDY_NAME:
            return True
    return False


def assert_no_eval_leakage(
    loaders: tp.Mapping[str, tp.Any],
    *,
    study: tp.Any,
    phases: tp.Sequence[str] = ("train", "val", "test"),
    off_limits: tp.Sequence[tuple[int, int]] = BT_LITE_SESSIONS,
) -> None:
    """Abort if any ``phases`` loader trains on a Neuroprobe off-limits session.

    Parameters
    ----------
    loaders
        ``split -> DataLoader`` mapping as returned by :meth:`Data.build`. Each
        loader's ``.dataset.triggers`` is the realized per-segment trigger frame
        (``select`` returns only the chosen segments), so this checks exactly the
        data that will receive gradients, not the dispatch's *intended* split.
    study
        The ``ns.Step``/``ns.Chain`` feeding the loaders. Non-BrainTreebank
        studies are exempt (the off-limits set is BT-specific).
    phases
        Which loader splits to verify. Defaults to all three: SSL/distill take
        gradients on ``train`` and observe ``val``/``test`` for pretext
        monitoring — every split must be legal. The experiment passes the
        realized loader keys explicitly; the default is the safe superset.
    off_limits
        The off-limits ``(subject_id, trial_id)`` set. Defaults to the 12
        Neuroprobe eval sessions (:data:`BT_LITE_SESSIONS`).
    """
    if not study_is_braintreebank(study):
        return

    off = {(int(s), int(t)) for s, t in off_limits}
    seen: set[tuple[int, int]] = set()
    for phase in phases:
        loader = loaders.get(phase)
        if loader is None:
            continue
        dataset = getattr(loader, "dataset", None)
        triggers = getattr(dataset, "triggers", None)
        if triggers is None or not {"subject_id", "trial_id"} <= set(triggers.columns):
            raise RuntimeError(
                "leakage guard (fail-closed): BrainTreebank SSL/distill "
                f"{phase!r} loader exposes no subject_id/trial_id triggers, so "
                "its pretraining corpus cannot be verified against the "
                "Neuroprobe off-limits eval set. Refusing to train unverified."
            )
        for s, t in zip(triggers["subject_id"], triggers["trial_id"]):
            seen.add((int(s), int(t)))

    bad = sorted(seen & off)
    if bad:
        raise RuntimeError(
            "LEAKAGE GUARD TRIPPED: the SSL/distill pretraining corpus "
            f"intersects the Neuroprobe off-limits eval set at sessions {bad}. "
            "A leaderboard-legal model must be pretrained only on "
            "V14_PRETRAIN_SESSIONS (BT_FULL - BT_LITE, cohort-scoped). The SSL "
            "train/val splits are coupled to the eval split if you see this. "
            "Contract: memory/"
            "project_v14_leakage_free_pretraining_contract_2026_06_06.md"
        )


def assert_pretrain_corpus_spans_cohort(
    loaders: tp.Mapping[str, tp.Any],
    *,
    study: tp.Any,
    min_subjects: int = 7,
    min_sessions: int = 13,
    train_split: str = "train",
) -> None:
    """Abort if the realized SSL train corpus is starved of cross-subject signal.

    This is the *anti-starvation* half of the #82 contract — the complement of
    :func:`assert_no_eval_leakage`. The leakage guard catches only the LEAK axis
    (``seen & off``); a regression that narrowed the corpus to one *legal*
    subject — empty intersection with :data:`BT_LITE_SESSIONS` — would pass the
    leakage guard and every existing test while silently training a
    "cross-subject" foundation model on a single subject. That is exactly the
    venom-style failure (training collapsed onto one session) that survived every
    component-level audit because no audit checked subject CARDINALITY end-to-end.

    The whole cross-subject generalization claim is void if the realized train
    split is one subject, so this is a positive invariant on the realized
    ``train`` loader: it must span at least ``min_subjects`` distinct
    ``subject_id`` and ``min_sessions`` distinct ``(subject_id, trial_id)``.

    Like the leakage guard it is **fail-closed** (missing train loader / trigger
    columns abort), **phase-scoped** (only SSL/distill call it, via
    ``enforces_pretrain_leakage_guard``), and **corpus-scoped** (BrainTreebank
    only; the V14 pretrain cohort floor is BT-specific). Gradients flow on the
    ``train`` split, so cardinality is asserted there.

    Parameters
    ----------
    loaders
        ``split -> DataLoader`` mapping. ``loaders[train_split].dataset.triggers``
        is the realized per-segment trigger frame — the data that takes gradients.
    study
        The study feeding the loaders. Non-BrainTreebank studies are exempt.
    min_subjects, min_sessions
        The V14 pretrain cohort floor: 7 subjects {1,2,3,4,6,8,9} across the 13
        ``V14_PRETRAIN_SESSIONS``. Defaults match the BT-only joint corpus; a
        multi-corpus joint phase that mixes non-BT studies would refine these.
    train_split
        The gradient-bearing split key (default ``"train"``).
    """
    if not study_is_braintreebank(study):
        return

    loader = loaders.get(train_split)
    dataset = getattr(loader, "dataset", None) if loader is not None else None
    triggers = getattr(dataset, "triggers", None)
    if triggers is None or not {"subject_id", "trial_id"} <= set(triggers.columns):
        raise RuntimeError(
            "anti-starvation guard (fail-closed): BrainTreebank SSL/distill "
            f"{train_split!r} loader exposes no subject_id/trial_id triggers, so "
            "its cross-subject cardinality cannot be verified. Refusing to train "
            "unverified."
        )

    sessions = {
        (int(s), int(t))
        for s, t in zip(triggers["subject_id"], triggers["trial_id"])
    }
    subjects = {s for s, _ in sessions}
    if len(subjects) < min_subjects or len(sessions) < min_sessions:
        raise RuntimeError(
            "ANTI-STARVATION GUARD TRIPPED: the SSL/distill train corpus spans "
            f"only {len(subjects)} subject(s) {sorted(subjects)} across "
            f"{len(sessions)} session(s), below the V14 pretrain cohort floor of "
            f"{min_subjects} subjects / {min_sessions} sessions. A cross-subject "
            "foundation model trained on a starved corpus is scientifically void "
            "even with zero eval leakage. The realized train split was narrowed "
            "(corpus constant, splitter, segmenter, or --mode re-coupling). "
            "Contract: memory/"
            "project_v14_leakage_free_pretraining_contract_2026_06_06.md"
        )
