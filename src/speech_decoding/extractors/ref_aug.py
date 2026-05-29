"""REF-01 / REF-02 per-clip reference-operator augmentation.

Lock memo: ``memory/project_v14_ref_aug_input_distribution_lock_2026_05_27.md``
(REF-aug 3-cell per-clip uniform-random draw, 5/27 PM, Geeling-Chau-motivated).

This module wires the *dataloader-side* half (REF-01) of the REF-aug 3-cell
contract. The encoder-side half (REF-02 — ``ref_embed`` additive at A1 +
optional cross-attn K/V reuse) is already wired in
:mod:`speech_decoding.models.v14_encoder` via B29 Item 11.

The contract:

  * Each clip independently samples ``ref_idx ∈ {0, 1, 2}`` per the lock
    table ``REF_MODES = ("shaft_car", "bipolar", "laplacian")``. The draw
    is reproducible per ``(seed, event_key)`` so eval/sister rollouts can
    pin the sequence.
  * ``ref_idx`` propagates to the encoder forward as a per-clip scalar
    so the lookup ``ref_embed[ref_idx]`` matches the data-side operator.
  * Per the lock memo, SWEC degenerates to global-CAR-only (``ref_idx``
    forced to a single mode); the per-corpus restriction is enforced at
    the corpus-routing layer of the dispatch — not here.
  * Raw is intentionally NOT in ``REF_MODES`` per per-corpus
    definitional ambiguity (lock memo §B-3).

Scope of this file:

  * :func:`draw_ref_idx` — deterministic per-event uniform draw.
  * :class:`RefIdxExtractor` — emits ``ref_idx`` as a per-event 0-d
    :class:`~neuralset.base.TimedArray` so it can be plumbed through
    NeuralSet's segmenter alongside ``electrode_tokens``.
  * :class:`RefAugMultiStftView` — extends :class:`MultiStftView` with a
    per-call ref-operator switch. Default uses uniform draw over
    ``REF_MODES``; ``ref_modes=("shaft_car",)`` is the
    ``R-no-ref-aug`` sister cell that restores the pre-5/27 behavior.

The Laplacian branch is a known gap (see ``REF-01-LAP`` row in
``docs/neuroprobe/v14_implementation_fix_list.md``): the underlying
:class:`speech_decoding.extractors.reference.CARIeegExtractor` exposes
``car ∈ {"global", "shaft"}`` and ``reference="bipolar"`` but no native
Laplacian. Until that lands, drawing ``"laplacian"`` raises
:class:`NotImplementedError` with a memo pointer; sister and smoke runs
should pass ``ref_modes=("shaft_car", "bipolar")``.
"""

from __future__ import annotations

import hashlib
import typing as tp

import numpy as np
import torch
from neuralset.base import TimedArray

from speech_decoding.extractors.reference import CARIeegExtractor
from speech_decoding.extractors.view import MultiStftView


REF_MODES: tuple[str, ...] = ("shaft_car", "bipolar", "laplacian")
SUPPORTED_REF_MODES: tuple[str, ...] = ("shaft_car", "bipolar")
"""Subset of :data:`REF_MODES` the pipeline can actually apply today.

Lock memo defines the full 3-cell draw, but the laplacian branch raises
:class:`NotImplementedError` until ``REF-01-LAP`` lands an MNE-level
Laplacian operator. Drawing from the 3-cell enum would crash roughly a
third of clips at runtime; restrict the default to the supported subset
and document the gap so the 3-cell opt-in (`ref_modes=REF_MODES`) is an
explicit caller choice.
"""

REF_MODE_TO_IDX: dict[str, int] = {m: i for i, m in enumerate(REF_MODES)}


def _validate_ref_modes(ref_modes: tuple[str, ...]) -> None:
    bad = [m for m in ref_modes if m not in REF_MODE_TO_IDX]
    if bad:
        raise ValueError(
            f"ref_modes contains unknown modes {bad!r}; "
            f"valid modes are {REF_MODES}"
        )


def _event_key(event: tp.Any, start: float, duration: float) -> str:
    """Stable string key for an (event, start, duration) clip.

    The natural identity of a clip is the event's
    ``(timeline, start, duration)``. NeuralSet's
    :class:`neuralset.events.etypes.Event` (and subclasses ``Ieeg``,
    ``Word``, …) does NOT expose ``event_id`` / ``id`` — only
    ``start``, ``timeline``, ``duration``, ``extra``. An earlier draft
    of this helper looked up the wrong field and would have crashed
    every real dispatch; the correct path is to read ``timeline``
    directly and combine it with the slice offsets so the SHA digest
    is stable across runs and unique per clip.
    """
    timeline = getattr(event, "timeline", None)
    if timeline is None:
        raise ValueError(
            "event must expose ``timeline``; "
            f"got {type(event).__name__} without one. "
            "REF-aug needs a stable per-clip key — without ``timeline`` "
            "the draw collides across clips and ref_idx loses entropy."
        )
    return f"{timeline}:{float(start):.9f}:{float(duration):.9f}"


def draw_ref_idx(
    event: tp.Any,
    start: float,
    duration: float,
    *,
    seed: int = 0,
    ref_modes: tuple[str, ...] = SUPPORTED_REF_MODES,
) -> int:
    """Deterministic per-clip uniform draw over ``ref_modes``.

    Per the lock memo (5/27 PM): "per-clip uniform-random ref draw".
    Reproducibility key = ``(seed, event_id, start, duration)``.
    """
    if len(ref_modes) == 0:
        raise ValueError("ref_modes must be non-empty")
    if len(ref_modes) == 1:
        return REF_MODE_TO_IDX[ref_modes[0]]
    digest = hashlib.sha256(
        f"{int(seed)}:{_event_key(event, start, duration)}".encode("utf-8")
    ).digest()
    word = int.from_bytes(digest[:8], "little", signed=False)
    local_idx = word % len(ref_modes)
    return REF_MODE_TO_IDX[ref_modes[local_idx]]


class RefIdxExtractor(CARIeegExtractor):
    """Per-event ``ref_idx`` scalar (0-d TimedArray).

    Plumbs the REF-aug per-clip draw to the encoder via NeuralSet's
    segmenter ``extractors`` dict + ``x_name`` tuple. The encoder forward
    accepts ``ref_idx`` (B29 Item 11) and indexes ``self.ref_embed``.

    Inherits :class:`CARIeegExtractor` only so it shares the same pydantic
    base + Ieeg event-type fingerprint as the other v14 extractors; it
    does NOT load waveform data here (the parent's ``_get_timed_array``
    is bypassed in favor of producing a scalar).
    """

    seed: int = 0
    ref_modes: tuple[str, ...] = SUPPORTED_REF_MODES

    def model_post_init(self, log__):
        super().model_post_init(log__)
        _validate_ref_modes(self.ref_modes)

    def _get_timed_array(
        self, event, start: float, duration: float,
    ) -> TimedArray:
        ref_idx = draw_ref_idx(
            event, start, duration,
            seed=self.seed, ref_modes=tuple(self.ref_modes),
        )
        data = np.asarray([ref_idx], dtype=np.int64)
        return TimedArray(
            frequency=1.0 / float(duration),
            start=start,
            duration=duration,
            data=data,
        )


class RefAugMultiStftView(MultiStftView):
    """``MultiStftView`` with a per-clip ref-operator switch (REF-01).

    Default ``ref_modes = SUPPORTED_REF_MODES = ("shaft_car", "bipolar")``
    is the supported subset of the 5/27 PM lock's 3-cell enum (laplacian
    held back until ``REF-01-LAP`` lands). Pass ``ref_modes=("shaft_car",)``
    for the ``R-no-ref-aug`` sister cell (pre-5/27 default behavior), or
    ``ref_modes=REF_MODES`` once the laplacian path lights up.

    The actual reference-operator switch reuses the parent's
    :class:`CARIeegExtractor` config knobs: ``"shaft_car"`` =
    ``car="shaft"``, ``"bipolar"`` = ``reference="bipolar"``. The
    ``"laplacian"`` branch raises :class:`NotImplementedError` until the
    upstream extractor gains a native Laplacian path (see ``REF-01-LAP``
    in ``docs/neuroprobe/v14_implementation_fix_list.md``).
    """

    seed: int = 0
    ref_modes: tuple[str, ...] = SUPPORTED_REF_MODES

    def model_post_init(self, log__):
        super().model_post_init(log__)
        _validate_ref_modes(self.ref_modes)

    def _apply_ref_operator(
        self,
        waveform_t: torch.Tensor,
        sample_rate: int,
        ref_mode: str,
    ) -> torch.Tensor:
        """Re-reference an already-loaded (C, T) waveform tensor.

        ``shaft_car`` is the no-op for callers who already passed
        ``car="shaft"`` upstream — the waveform is already shaft-CAR'd by
        the parent's ``_preprocess_raw``. ``bipolar`` and ``laplacian``
        operate post-hoc on the waveform tensor and are scaffolded to
        match the lock-memo contract; the bipolar branch uses a
        first-neighbor-difference proxy (not the MNE bipolar reference
        path, which lives at the raw level — see TODO below).

        TODO(REF-01-LAP / REF-01-BIP-RAW): the canonical per-clip ref
        switch should re-run the underlying MNE preprocessing chain with
        the drawn ref operator (bipolar / Laplacian / shaft-CAR). This
        scaffolding lets the data path round-trip ``ref_idx`` and the
        embedding lookup; the surface-level (C, T) proxy is correct in
        shape but not yet correct in physics. Tracked under REF-01-LAP /
        REF-01-BIP-RAW gap rows in ``v14_implementation_fix_list.md``.
        """
        del sample_rate  # noqa
        if ref_mode == "shaft_car":
            return waveform_t
        if ref_mode == "bipolar":
            if waveform_t.shape[0] < 2:
                return waveform_t
            # Lock memo §Operators table: bipolar is ``x_e − x_{e+1}``;
            # ``|STFT|`` magnitude downstream masks any sign error, so a
            # unit test pins the convention. Last channel reused as a
            # shape-preservation placeholder until REF-01-BIP-RAW lands
            # the MNE-level rewire.
            diffs = waveform_t[:-1] - waveform_t[1:]
            return torch.cat([diffs, waveform_t[-1:]], dim=0)
        if ref_mode == "laplacian":
            raise NotImplementedError(
                "REF-01: 'laplacian' branch is scaffolded but not yet "
                "wired against CARIeegExtractor / MNE preprocessing. See "
                "REF-01-LAP in docs/neuroprobe/v14_implementation_fix_list.md "
                "and the lock memo project_v14_ref_aug_input_distribution_lock_2026_05_27.md. "
                "For smoke runs, pass ref_modes=('shaft_car', 'bipolar')."
            )
        raise AssertionError(  # unreachable: ref_mode = REF_MODES[ref_idx] by construction
            f"unhandled ref_mode {ref_mode!r}; REF_MODES = {REF_MODES!r}"
        )

    def _get_timed_array(
        self, event, start: float, duration: float,
    ) -> TimedArray:
        ref_idx = draw_ref_idx(
            event, start, duration,
            seed=self.seed, ref_modes=tuple(self.ref_modes),
        )
        ref_mode = REF_MODES[ref_idx]
        waveform_ta = super(MultiStftView, self)._get_timed_array(
            event, start, duration,
        )
        waveform_t = torch.from_numpy(np.asarray(waveform_ta.data)).float()
        waveform_t = self._apply_ref_operator(
            waveform_t, int(float(waveform_ta.frequency)), ref_mode,
        )
        from speech_decoding.extractors.view import _multi_stft_view
        spec = _multi_stft_view(
            waveform_t,
            sample_rate=int(float(waveform_ta.frequency)),
            hop_length=self.hop_length,
            nperseg_low=self.nperseg_low,
            nperseg_mid=self.nperseg_mid,
            nperseg_hi=self.nperseg_hi,
            n_bins=self.n_fbank_bins,
            f0_hz=self.fbank_f0_hz,
            octave_step=self.fbank_octave_step,
            half_bw_octaves=self.fbank_half_bw_octaves,
            routing=self.fbank_routing,
            log_eps=self.log_eps,
            apply_log=self.apply_log,
        )
        if self.c_max is not None:
            c_event = int(spec.shape[0])
            if c_event > self.c_max:
                raise ValueError(
                    f"event has {c_event} electrodes which exceeds c_max={self.c_max}"
                )
            padded = torch.zeros(
                self.c_max, int(spec.shape[1]), int(spec.shape[2]),
                dtype=spec.dtype,
            )
            padded[:c_event] = spec
            spec = padded
        n_time_bins = int(spec.shape[-1])
        new_frequency = float(n_time_bins) / float(duration)
        return TimedArray(
            frequency=new_frequency,
            start=start,
            duration=duration,
            data=spec.cpu().numpy(),
        )
