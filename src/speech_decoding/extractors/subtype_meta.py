"""B29 Item 9 / Item 11 per-clip subject-subtype + λ_anat metadata.

Lock memos:
  * Subtype: ``memory/project_v14_b29_joint_default_2026_05_27.md`` §Item 11
    (``subtype_embed: nn.Embedding(2, d=256)``; sister three-way ``{depth,
    grid, strip}``).
  * λ_anat per-clip gate: ``memory/project_v14_b29_joint_default_2026_05_27.md``
    §Item 12 (drop ``parcels_supervised`` gating, drive anatomy-bias from
    per-clip metadata) + ``docs/neuroprobe/v14_implementation_fix_list.md``
    §B28 Item 3 → SUPERSEDED by per-clip gate.

Two per-event extractors:

  * :class:`SubjectSubtypeExtractor` — emits ``subject_subtype`` per
    clip. Binary default = {0 = sEEG-depth (intracranial), 1 = ECoG
    (surface)}; three-way sister = {0 = depth, 1 = grid, 2 = strip}.
  * :class:`LambdaAnatExtractor` — emits per-clip ``λ_anat ∈ [0, 1]``
    scalar driving the anatomy-bias gate at the cross-attn layer.
    Default policy: anatomy-rich corpora (BT / D-cohort / AJILE12) →
    1.0; SWEC → 0.0 until ``R-swec-pseudo-parcel-per-shaft`` sister
    lands.

The per-subject subtype + per-corpus λ_anat tables live as module-level
dicts so dispatch can override them via kwargs. Real data wiring should
override these from the per-corpus manifests in
``studies/braintreebank/manifest.py`` and the equivalent SWEC/D-cohort
manifests once they land.
"""

from __future__ import annotations

import typing as tp

import numpy as np
from neuralset.base import TimedArray

from speech_decoding.extractors.reference import CARIeegExtractor


# ---------------------------------------------------------------------------
# Subject-subtype lookup
# ---------------------------------------------------------------------------

BINARY_SUBTYPE_VOCAB: tp.Final[tuple[str, ...]] = ("depth", "ecog")
THREE_WAY_SUBTYPE_VOCAB: tp.Final[tuple[str, ...]] = ("depth", "grid", "strip")

BINARY_SUBTYPE_TO_IDX: tp.Final[dict[str, int]] = {
    name: i for i, name in enumerate(BINARY_SUBTYPE_VOCAB)
}
THREE_WAY_SUBTYPE_TO_IDX: tp.Final[dict[str, int]] = {
    name: i for i, name in enumerate(THREE_WAY_SUBTYPE_VOCAB)
}

# Per-corpus default subtype (binary). Concrete per-subject overrides
# should be passed in via the ``per_subject`` dict kwarg at extractor
# build time, sourced from the per-corpus manifest at dispatch time.
DEFAULT_CORPUS_SUBTYPE: tp.Final[dict[str, str]] = {
    "braintreebank": "depth",      # sEEG depth
    "wang2024treebank": "depth",
    "d_cohort": "depth",           # sEEG depth (Duke cohort)
    "cogan_dcohort": "depth",
    "swec": "depth",               # sEEG depth (CH)
    "ajile12": "ecog",             # surface ECoG (Peterson 2022)
}

# Per-corpus λ_anat default per the B29 Item 12 spec. Anatomy-rich
# corpora gate ON; SWEC defaults to 0 until the per-shaft pseudo-parcel
# sister lands.
DEFAULT_CORPUS_LAMBDA_ANAT: tp.Final[dict[str, float]] = {
    "braintreebank": 1.0,
    "wang2024treebank": 1.0,
    "d_cohort": 1.0,
    "cogan_dcohort": 1.0,
    "ajile12": 1.0,
    "swec": 0.0,                  # turned ON only via R-swec-pseudo-parcel-per-shaft sister
}


def _resolve_corpus_name(event: tp.Any, corpus: str | None) -> str:
    """Best-effort lookup of the corpus/study name from an event.

    Falls back to the dispatch-provided ``corpus`` kwarg if the event
    has no ``study_name`` / ``corpus`` attribute.
    """
    for attr in ("corpus", "study_name", "study", "dataset"):
        value = getattr(event, attr, None)
        if isinstance(value, str) and value:
            return value
    if corpus is not None:
        return corpus
    raise ValueError(
        "could not resolve corpus name from event; pass ``corpus`` kwarg "
        "to the extractor (e.g. corpus='braintreebank' for BT pretrain)."
    )


def _resolve_subject_id(event: tp.Any) -> str | None:
    for attr in ("subject_id", "subject", "patient_id", "patient"):
        value = getattr(event, attr, None)
        if value is not None:
            return str(value)
    return None


class SubjectSubtypeExtractor(CARIeegExtractor):
    """Per-event ``subject_subtype`` scalar (0-d int64 TimedArray).

    Routes ``(corpus, subject_id)`` → subtype idx via the table kwargs.
    Binary default vocab (B29 Item 11): {0 = depth, 1 = ECoG}.
    Three-way sister vocab: {0 = depth, 1 = grid, 2 = strip}.
    """

    vocab: tp.Literal["binary", "three_way"] = "binary"
    # Per-corpus default subtype name; overrides ``DEFAULT_CORPUS_SUBTYPE``.
    corpus_subtype: dict[str, str] = {}
    # Per-subject overrides — keyed by ``str(subject_id)``.
    subject_subtype: dict[str, str] = {}
    # Used when the event has no corpus attr; dispatch must set this.
    corpus: str | None = None

    def model_post_init(self, log__):
        super().model_post_init(log__)
        if self.vocab not in ("binary", "three_way"):
            raise ValueError(
                f"vocab must be 'binary' or 'three_way'; got {self.vocab!r}"
            )
        vocab_table = (
            BINARY_SUBTYPE_TO_IDX if self.vocab == "binary"
            else THREE_WAY_SUBTYPE_TO_IDX
        )
        for source, table in (
            ("corpus_subtype", self.corpus_subtype),
            ("subject_subtype", self.subject_subtype),
        ):
            bad = [v for v in table.values() if v not in vocab_table]
            if bad:
                raise ValueError(
                    f"{source} contains values {bad!r} not in vocab "
                    f"{tuple(vocab_table.keys())}"
                )

    def _resolve_idx(self, event) -> int:
        vocab_table = (
            BINARY_SUBTYPE_TO_IDX if self.vocab == "binary"
            else THREE_WAY_SUBTYPE_TO_IDX
        )
        subject_id = _resolve_subject_id(event)
        if subject_id is not None and subject_id in self.subject_subtype:
            return vocab_table[self.subject_subtype[subject_id]]
        corpus = _resolve_corpus_name(event, self.corpus)
        if corpus in self.corpus_subtype:
            return vocab_table[self.corpus_subtype[corpus]]
        if corpus in DEFAULT_CORPUS_SUBTYPE:
            default = DEFAULT_CORPUS_SUBTYPE[corpus]
            # Demote three-way → binary cleanly when vocab is binary.
            if default == "grid" or default == "strip":
                if self.vocab == "binary":
                    default = "ecog"
            return vocab_table[default]
        raise KeyError(
            f"no subtype registered for corpus={corpus!r} subject={subject_id!r}; "
            f"set ``corpus_subtype`` or ``subject_subtype`` at build time."
        )

    def _get_timed_array(
        self, event, start: float, duration: float,
    ) -> TimedArray:
        idx = self._resolve_idx(event)
        # Shape (1, 1) = (1 channel, 1 sample) — see RefIdxExtractor for
        # the same NeuralSet BaseExtractor invariant. A 1-d data array
        # crashes ``torch.zeros(*tensor.shape[:-1])`` on the zero-duration
        # prepare call.
        return TimedArray(
            frequency=1.0 / float(duration),
            start=start,
            duration=duration,
            data=np.asarray([[idx]], dtype=np.int64),
        )


class LambdaAnatExtractor(CARIeegExtractor):
    """Per-event ``λ_anat ∈ [0, 1]`` scalar (0-d float32 TimedArray).

    Default policy (B29 Item 12): anatomy-rich corpora gate ON
    (``λ_anat = 1.0``); SWEC defaults to 0.0 until the per-shaft
    pseudo-parcel sister lands. Override either per-corpus
    (``corpus_lambda_anat``) or per-subject (``subject_lambda_anat``) at
    build time.

    The encoder forward already multiplies the anatomy bias by
    ``λ_anat`` (kwarg landed at B28); this extractor wires the per-clip
    metadata that drives it.
    """

    corpus_lambda_anat: dict[str, float] = {}
    subject_lambda_anat: dict[str, float] = {}
    corpus: str | None = None

    def model_post_init(self, log__):
        super().model_post_init(log__)
        for source, table in (
            ("corpus_lambda_anat", self.corpus_lambda_anat),
            ("subject_lambda_anat", self.subject_lambda_anat),
        ):
            for k, v in table.items():
                if not 0.0 <= float(v) <= 1.0:
                    raise ValueError(
                        f"{source}[{k!r}]={v!r} must lie in [0, 1]"
                    )

    def _resolve_value(self, event) -> float:
        subject_id = _resolve_subject_id(event)
        if subject_id is not None and subject_id in self.subject_lambda_anat:
            return float(self.subject_lambda_anat[subject_id])
        corpus = _resolve_corpus_name(event, self.corpus)
        if corpus in self.corpus_lambda_anat:
            return float(self.corpus_lambda_anat[corpus])
        if corpus in DEFAULT_CORPUS_LAMBDA_ANAT:
            return float(DEFAULT_CORPUS_LAMBDA_ANAT[corpus])
        raise KeyError(
            f"no λ_anat registered for corpus={corpus!r} subject={subject_id!r}; "
            f"set ``corpus_lambda_anat`` or ``subject_lambda_anat`` at build time."
        )

    def _get_timed_array(
        self, event, start: float, duration: float,
    ) -> TimedArray:
        value = self._resolve_value(event)
        return TimedArray(
            frequency=1.0 / float(duration),
            start=start,
            duration=duration,
            data=np.asarray([[value]], dtype=np.float32),
        )
