"""V14 Phase-3 Whisper teacher-target extractor (WS-H / T20).

For each trigger ``Word`` event, slice the cached whole-movie Whisper feature
stream at the word's MOVIE-CLOCK onset and return the ``(clip_frames, d)``
teacher target the P3 distillation loss consumes (``whisper_target`` batch key;
collated to ``(B, 250, 1280)``).

THE CLOCK TRAP (FLAG 9). The teacher cache is one dense stream PER MOVIE indexed
by movie-audio time (``t0_movie_s == 0``; frame ``f`` ↔ ``f / rate_hz`` seconds
into the movie). So the join MUST use the movie-clock onset — the transcript
``words_df['start']``, threaded onto the event as ``movie_onset_s`` — NOT the
neural-clock ``est_idx / 2048`` (the event's ``start``). The two diverge 235-904 s
within a single BT trial. Δlag slides only the NEURAL response window; the
audio-keyed teacher is unshifted, so this extractor never sees ``neural_lag_s``.

The dense stream is memory-mapped (``mmap=True``) and memoised per movie: a slice
reads only its 250×1280 window off disk, so workers share the OS page cache
instead of each holding a whole multi-GB fp16 movie in RAM.
"""

from __future__ import annotations

import logging
import typing as tp

import pydantic
import torch
from neuralset.events.etypes import Event
from neuralset.extractors.base import BaseStatic

from speech_decoding.bt_alignment.teacher_cache import (
    DEFAULT_TEACHER_HZ,
    movie_cache_path,
)

logger = logging.getLogger(__name__)


def _resolve_movie(subject_id: int, trial_id: int) -> str:
    """``(subject, trial)`` → BT movie slug.

    Lazy import: ``neuroprobe.config`` reads ``ROOT_DIR_BRAINTREEBANK`` at import
    and raises on laptops without BT mounted, so resolving is deferred to call
    time (the module stays importable for tests, which monkeypatch this)."""
    from neuroprobe.config import BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING

    return BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"btbank{subject_id}_{trial_id}"]


class WhisperTargetExtractor(BaseStatic):
    """Per-clip cached Whisper teacher target ``(n_frames, d_model)``.

    ``aggregation='trigger'`` so the slice is computed once per clip from the
    trigger ``Word`` event (a 5-s segment otherwise contains many words).
    """

    event_types: tp.Literal["Word"] = "Word"
    aggregation: tp.Literal["trigger"] = "trigger"

    cache_dir: str
    model: str = "openai/whisper-large-v3"
    layer_merge: str = "mean_all"
    clip_s: float = 5.0
    rate_hz: int = DEFAULT_TEACHER_HZ
    d_model: int = 1280

    # movie slug -> mmap'd (T, d) fp16 dense stream; built lazily inside each
    # dataloader worker (empty at pickle time, so the extractor pickles cleanly).
    _dense: dict[str, torch.Tensor] = pydantic.PrivateAttr(default_factory=dict)
    _n_clamped: int = pydantic.PrivateAttr(0)

    @property
    def n_frames(self) -> int:
        """Teacher window length in frames (pinned to 250 = 5 s × 50 Hz)."""
        return round(self.clip_s * self.rate_hz)

    def __getstate__(self) -> dict:
        # Drop the per-movie mmap memo before pickling. ``prepare()`` runs
        # get_static once in the MAIN process (to populate the output shape),
        # which loads one whole-movie dense stream into ``_dense``; without this,
        # that multi-GB tensor would be pickled into every spawned dataloader
        # worker, defeating the mmap design (each worker should rebuild it lazily
        # and share the OS page cache). Reset to the empty construction state.
        state = super().__getstate__()
        private = state.get("__pydantic_private__")
        if private and ("_dense" in private or "_n_clamped" in private):
            private = dict(private)
            private["_dense"] = {}
            private["_n_clamped"] = 0
            state["__pydantic_private__"] = private
        return state

    def _movie_dense(self, movie: str) -> torch.Tensor:
        dense = self._dense.get(movie)
        if dense is not None:
            return dense
        path = movie_cache_path(self.cache_dir, self.model, self.layer_merge, movie)
        if not path.is_file():
            raise FileNotFoundError(
                f"WhisperTargetExtractor: teacher cache missing for movie "
                f"{movie!r}: {path}. Build it with "
                "scripts/neuroprobe/build_bt_teacher_cache.py."
            )
        entry = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        dense = entry["features"]
        if int(entry["rate_hz"]) != self.rate_hz:
            raise ValueError(
                f"{movie}: cache rate {entry['rate_hz']} Hz != extractor "
                f"rate_hz {self.rate_hz}"
            )
        if float(entry.get("t0_movie_s", 0.0)) != 0.0:
            raise ValueError(
                f"{movie}: whole-movie cache must start at the movie origin "
                f"(t0_movie_s=0.0), got {entry.get('t0_movie_s')!r}; movie-clock "
                "frame indexing assumes a zero offset."
            )
        if dense.ndim != 2 or dense.shape[1] != self.d_model:
            raise ValueError(
                f"{movie}: cache features have shape {tuple(dense.shape)}, "
                f"expected (T, {self.d_model})."
            )
        self._dense[movie] = dense
        return dense

    def get_static(self, event: Event) -> torch.Tensor:
        subject_id = int(event._get_field_or_extra("subject_id"))
        trial_id = int(event._get_field_or_extra("trial_id"))
        movie_onset_s = float(event._get_field_or_extra("movie_onset_s"))

        movie = _resolve_movie(subject_id, trial_id)
        dense = self._movie_dense(movie)
        n = self.n_frames
        total = int(dense.shape[0])
        if total < n:
            raise ValueError(
                f"{movie}: dense cache has {total} frames < clip window {n}; "
                "movie shorter than the clip length?"
            )

        # t0_movie_s == 0 (asserted on load) ⇒ frame index = round(onset · rate).
        frame0 = round(movie_onset_s * self.rate_hz)
        clamped = min(max(frame0, 0), total - n)
        if clamped != frame0:
            # A word within the last clip_s of the cached stream: clamp back so
            # the window stays exactly n frames (the 50→8 Hz pool requires 250).
            # Negligibly rare (words end well before the movie does); log it so a
            # systematic mis-alignment can't hide.
            self._n_clamped += 1
            logger.warning(
                "WhisperTargetExtractor: clamped teacher window for %s "
                "(onset=%.2fs, frame0=%d, total=%d, clamped_to=%d); count=%d",
                movie, movie_onset_s, frame0, total, clamped, self._n_clamped,
            )

        # Exactly (n, d_model) by construction: clamped ∈ [0, total-n] (so the
        # first axis is n) and dense.shape[1] == d_model was asserted on load.
        return dense[clamped : clamped + n].to(torch.float32).contiguous()
