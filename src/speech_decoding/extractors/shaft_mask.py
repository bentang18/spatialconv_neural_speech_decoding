"""Phase-2 shaft-block electrode mask — 2026-05-27 PM final spec.

Supersedes the original EX03 ``K=3`` spec (2026-05-23) and the
same-day AM ``K = min(2, ceil(0.25 * N_shafts))`` fraction-based spec
(2026-05-27 AM, held ~5 hours). The cross-attn cost-savings target is
unchanged; the K cap is tightened to the minimum that still produces a
non-trivial shaft block.

The operator (one line, paradigm B drop)::

    K = k_override if k_override is not None else (1 if n_shafts >= 2 else 0)

* Default ``K = 1`` whenever ``N_shafts ≥ 2`` — single guard closes the
  ``N_shafts == 1`` catastrophic-drop hole the AM fraction spec had.
* Default ``extent_blocks = ("alpha",)`` — block α only (β and γ from
  prior K=3 / K≤2 specs are dropped from the default).
* Sisters reach K=2 / K=3 / stratified extents via the ``k_override`` +
  ``extent_blocks`` parameters — same module, no fork.

Block α specification (unchanged across the K-cap revisions):

* shaft-extent ``0.45–0.60`` of the per-shaft electrode count
* time-extent ``0.15–0.30`` (consumed by the patch-mask side; this
  module emits the electrode-space mask only)

Output of :meth:`ShaftMaskExtractor.__call__` is a ``(C,)`` bool tensor
— ``True`` at the shaft-blocked electrode positions. The caller (the
training loop in Phase-2) OR-s this into the student forward's
``shaft_mask`` argument so it lands in ``key_padding_mask`` at the
cross-attention layers (paradigm B drop — no ``[MASK]`` embedding).
The teacher forward gets ``shaft_mask=None`` (B03d asymmetry —
non-negotiable; symmetric teacher would destroy the V-JEPA supervision
signal). Phase-1 / Phase-3 / Phase-4 don't construct this mask at all.

Sister cells:

* ``R-shaft-K2`` — ``k_override=2, extent_blocks=("alpha", "beta")``.
* ``R-shaft-K3-mixed-3block`` — ``k_override=3,
  extent_blocks=("alpha", "beta", "gamma")``.
* ``R-shaft-K1-explicit`` — ``k_override=1`` (redundant with the
  default, but explicit; documents the K=1 lock in the dispatch log).
"""

from __future__ import annotations

import dataclasses
import hashlib
import typing as tp
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor


@dataclasses.dataclass(frozen=True, slots=True)
class BlockExtent:
    """One shaft-block extent specification.

    Fractions are over the *per-shaft* electrode count, not the global
    electrode count — a shaft of 10 contacts with extent ``0.45–0.60``
    means 4–6 contiguous contacts on that one shaft.

    The 4-tuple lives as ``(shaft_lo, shaft_hi, time_lo, time_hi)`` so
    the per-corpus ablation dispatch can read a single source.
    """

    name: str
    shaft_lo: float
    shaft_hi: float
    time_lo: float
    time_hi: float


BLOCK_ALPHA = BlockExtent("alpha", 0.45, 0.60, 0.15, 0.30)
"""B26-PM default: the only block in the active list when the default
``extent_blocks=("alpha",)`` is used. Shaft-extent 0.45–0.60 × time-extent
0.15–0.30."""

BLOCK_BETA = BlockExtent("beta", 0.20, 0.60, 0.00, 0.40)
"""Sister-only (``R-shaft-K2``): widens the shaft-extent floor + widens
the time-extent. Available via ``extent_blocks=("alpha", "beta")``."""

BLOCK_GAMMA = BlockExtent("gamma", 0.20, 0.60, 0.00, 0.40)
"""Sister-only (``R-shaft-K3-mixed-3block``): same extent ranges as β —
the *fact of having a third block* is the falsification, not the extent.
Available via ``extent_blocks=("alpha", "beta", "gamma")``."""


_BLOCK_BY_NAME: dict[str, BlockExtent] = {
    BLOCK_ALPHA.name: BLOCK_ALPHA,
    BLOCK_BETA.name: BLOCK_BETA,
    BLOCK_GAMMA.name: BLOCK_GAMMA,
}


def default_k(n_shafts: int) -> int:
    """5/27 PM default K formula::

        K = 1 if n_shafts >= 2 else 0

    No fraction, no ceiling. The single ``N_shafts ≥ 2`` guard closes
    the catastrophic-drop hole at ``N_shafts == 1`` (the AM fraction
    spec would have emitted K=1 for N=1, dropping the entire subject's
    electrode set into the shaft block).
    """
    return 1 if n_shafts >= 2 else 0


@dataclasses.dataclass(frozen=True, slots=True)
class SubjectAnatomy:
    """Minimal anatomy descriptor consumed by :class:`ShaftMaskExtractor`.

    ``shaft_ids`` (``(C,)`` long) labels each electrode's shaft index.
    ``contact_pos`` (``(C,)`` float, [0, 1]) is the normalized position
    along the shaft (0 = deep, 1 = surface). The Phase-2 dataloader is
    expected to assemble this from per-subject anatomy ahead of the
    forward.
    """

    shaft_ids: Tensor
    contact_pos: Tensor


def _resolve_extent_blocks(
    extent_blocks: Sequence[str | BlockExtent],
) -> list[BlockExtent]:
    """Resolve a mixed list of block names + :class:`BlockExtent` objects
    into a list of :class:`BlockExtent`. Unknown names raise ``ValueError``."""
    resolved: list[BlockExtent] = []
    for entry in extent_blocks:
        if isinstance(entry, BlockExtent):
            resolved.append(entry)
        else:
            if entry not in _BLOCK_BY_NAME:
                raise ValueError(
                    f"unknown block extent name {entry!r}; "
                    f"expected one of {sorted(_BLOCK_BY_NAME)}"
                )
            resolved.append(_BLOCK_BY_NAME[entry])
    return resolved


class ShaftMaskExtractor:
    """Per-clip shaft-block mask sampler (B26-PM final spec).

    Parameters
    ----------
    seed
        RNG seed for block placement. Pass a per-step seed when sampling
        a fresh mask each batch element.
    k_override
        Override the default ``K = 1 if N_shafts >= 2 else 0`` formula
        with a fixed integer ``K``. Used by the ``R-shaft-K2`` and
        ``R-shaft-K3-mixed-3block`` sister cells. When ``None`` (the
        default), the formula is applied.
    extent_blocks
        Ordered list of block extents to draw from. Each entry is either
        the string name of a registered block (``"alpha" | "beta" |
        "gamma"``) or a :class:`BlockExtent` instance directly. The
        sampler picks the first ``min(K, len(extent_blocks))`` entries
        from this list, one block per picked shaft. Defaults to
        ``("alpha",)`` — block α only.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        k_override: int | None = None,
        extent_blocks: Sequence[str | BlockExtent] = ("alpha",),
    ) -> None:
        if k_override is not None and k_override < 0:
            raise ValueError(f"k_override must be >= 0, got {k_override}")
        if len(extent_blocks) == 0:
            raise ValueError(
                "extent_blocks must be non-empty (default is (\"alpha\",))"
            )
        self.seed = seed
        self.k_override = k_override
        self.block_extents: list[BlockExtent] = _resolve_extent_blocks(extent_blocks)

    def _n_shafts(self, anatomy: SubjectAnatomy) -> int:
        if anatomy.shaft_ids.numel() == 0:
            return 0
        return int(anatomy.shaft_ids.max().item()) + 1

    def _k_for_anatomy(self, anatomy: SubjectAnatomy) -> int:
        n_shafts = self._n_shafts(anatomy)
        if self.k_override is not None:
            # Sister cells override the formula but still respect
            # n_shafts as an upper bound (K=3 on a 2-shaft subject
            # collapses to 2).
            return min(self.k_override, n_shafts)
        return default_k(n_shafts)

    def _pick_block_extents(self, k: int) -> list[BlockExtent]:
        """Return the first ``k`` block extents from the active list.

        With the 5/27 PM default ``extent_blocks=("alpha",)`` and
        ``k=1`` this is just ``[BLOCK_ALPHA]``. Sisters with longer
        lists pick the corresponding prefix.
        """
        return self.block_extents[: min(k, len(self.block_extents))]

    def __call__(self, anatomy: SubjectAnatomy) -> Tensor:
        """Return a ``(C,)`` bool mask. ``True`` = shaft-blocked."""
        if anatomy.shaft_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.bool)

        C = anatomy.shaft_ids.shape[0]
        mask = torch.zeros(C, dtype=torch.bool, device=anatomy.shaft_ids.device)
        k = self._k_for_anatomy(anatomy)
        if k <= 0:
            return mask

        gen = torch.Generator(device="cpu")
        if self.seed is not None:
            gen.manual_seed(self.seed)

        shaft_ids_unique = torch.unique(anatomy.shaft_ids).tolist()
        n_shafts = len(shaft_ids_unique)
        if n_shafts == 0:
            return mask

        blocks = self._pick_block_extents(k)
        if not blocks:
            return mask

        # Pick K distinct shafts to block. If K > N_shafts, fall back to
        # all shafts (no replacement beyond capacity).
        n_picks = min(k, n_shafts)
        shaft_perm = torch.randperm(n_shafts, generator=gen).tolist()
        picked_shafts = [shaft_ids_unique[i] for i in shaft_perm[:n_picks]]

        for block, shaft_id in zip(blocks, picked_shafts):
            shaft_mask = anatomy.shaft_ids == shaft_id
            shaft_indices = torch.nonzero(shaft_mask).flatten()
            n_on_shaft = shaft_indices.numel()
            if n_on_shaft == 0:
                continue
            # Sample a contiguous span: span_size = uniform from
            # [shaft_lo, shaft_hi] × n_on_shaft.
            lo_count = max(1, int(round(block.shaft_lo * n_on_shaft)))
            hi_count = max(lo_count, int(round(block.shaft_hi * n_on_shaft)))
            span_size = int(torch.randint(
                lo_count, hi_count + 1, (1,), generator=gen
            ).item())
            span_size = min(span_size, n_on_shaft)
            # Start position uniform from [0, n_on_shaft - span_size].
            max_start = n_on_shaft - span_size
            if max_start <= 0:
                start = 0
            else:
                start = int(torch.randint(0, max_start + 1, (1,), generator=gen).item())
            block_idx = shaft_indices[start : start + span_size]
            mask[block_idx] = True

        return mask


# ---------------------------------------------------------------------------
# NeuralFetch-style per-event wrapper for the BT cohort.
# ---------------------------------------------------------------------------


def _bt_shaft_ids_from_labels(electrode_labels: Sequence[str]) -> Tensor:
    """Map BT electrode labels (`"OFa3"`) → contiguous shaft ids (`(C,) long`).

    Uses :func:`speech_decoding.extractors.reference.parse_shaft` to peel the
    trailing contact number off each label; identical alphabetic prefixes share
    a shaft. Order-of-first-appearance defines the shaft index.
    """
    from speech_decoding.extractors.reference import parse_shaft

    seen: dict[str, int] = {}
    out = torch.zeros(len(electrode_labels), dtype=torch.long)
    for i, label in enumerate(electrode_labels):
        prefix, _ = parse_shaft(str(label))
        if prefix not in seen:
            seen[prefix] = len(seen)
        out[i] = seen[prefix]
    return out


def _bt_contact_pos_from_labels(electrode_labels: Sequence[str]) -> Tensor:
    """Normalise within-shaft contact index → [0, 1] float (`(C,)`)."""
    from speech_decoding.extractors.reference import parse_shaft

    shafts: list[tuple[str, tp.Optional[int]]] = [
        parse_shaft(str(label)) for label in electrode_labels
    ]
    counts: dict[str, int] = {}
    for prefix, _ in shafts:
        counts[prefix] = counts.get(prefix, 0) + 1
    cur: dict[str, int] = {}
    out = torch.zeros(len(electrode_labels), dtype=torch.float32)
    for i, (prefix, _) in enumerate(shafts):
        idx = cur.get(prefix, 0)
        n_on_shaft = max(counts[prefix], 1)
        out[i] = float(idx) / float(max(n_on_shaft - 1, 1))
        cur[prefix] = idx + 1
    return out


def _bt_event_clip_seed(
    base_seed: int, subject_id: int, event: tp.Any,
) -> int:
    """Deterministic per-event seed for the shaft-block sampler.

    Reads ``event.timeline`` and ``event.start`` so two events from the same
    subject get distinct shaft blocks; ``BaseStatic`` is per-event so the
    same mask broadcasts across all clips of one event (the per-clip B03c
    seed lives in REF-aug, not here — the shaft-block lives at event-level).
    """
    timeline = getattr(event, "timeline", "")
    start = float(getattr(event, "start", 0.0))
    payload = f"{int(base_seed)}:{int(subject_id)}:{timeline}:{start:.9f}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**31 - 1)


try:
    from neuralset.events.etypes import Event as _NeuralsetEvent
    from neuralset.extractors.base import BaseStatic as _BaseStatic
except ImportError:  # pragma: no cover — tests without neuralset import
    _NeuralsetEvent = tp.Any  # type: ignore[assignment,misc]
    _BaseStatic = object  # type: ignore[assignment,misc]


class BTShaftMaskExtractor(_BaseStatic):  # type: ignore[misc,valid-type]
    """Per-event ``(c_max,) bool`` shaft-mask aligned to the voltage electrode order.

    Composes BT voltage electrode labels (``voltage_electrode_order``) →
    :class:`SubjectAnatomy` → :class:`ShaftMaskExtractor.__call__` → pad to
    ``(c_max,)`` with ``False``. ``True`` at shaft-blocked electrodes.

    Drift-table row **B03 shaft-mask** (5/27 PM final spec): student-only mask
    (OR-ed into the student forward's ``shaft_mask`` kwarg, never passed to
    the EMA teacher per the B26 full-input contract). The
    :class:`V14JointBrainModule` already routes ``batch.data["shaft_mask"]``
    student-only.

    The electrode axis is the VOLTAGE order
    (``BrainTreebankSubject.electrode_labels``), so it lines up row-for-row with
    ``support`` / ``valid_mask`` / ``electrode_tokens`` (C1/C2 fix). Shaft
    membership is parsed from the physical electrode label and is independent of
    the DK vocabulary, so there is no unknown-label filter here — every voltage
    electrode sits on a physical shaft and participates.

    Sister cells:

    * ``R-shaft-K2`` — ``k_override=2, extent_blocks=("alpha", "beta")``.
    * ``R-shaft-K3-mixed-3block`` — ``k_override=3,
      extent_blocks=("alpha", "beta", "gamma")``.
    * ``R-shaft-K1-explicit`` — ``k_override=1``.
    """

    event_types: tp.Literal["Ieeg"] = "Ieeg"
    bt_root: str
    c_max: int = 384
    seed: int = 0
    k_override: int | None = None
    extent_blocks: tuple[str, ...] = ("alpha",)

    def get_static(self, event: tp.Any) -> Tensor:
        # Imports kept local so the module stays importable without
        # neuralset (tests import the inner ``ShaftMaskExtractor`` /
        # ``SubjectAnatomy`` directly).
        from speech_decoding.extractors.dk_support import (
            _coerce_subject_id,
            _coerce_trial_id,
        )
        from speech_decoding.studies.braintreebank.anatomy import (
            voltage_electrode_order,
        )

        subject_id = _coerce_subject_id(getattr(event, "subject"))
        trial_id = _coerce_trial_id(event)
        # Shaft membership is parsed from the physical label, so this aligns to
        # the VOLTAGE electrode order (same axis as support / valid_mask /
        # electrode_tokens) and needs no DK-vocab filter — an out-of-vocab
        # electrode still sits on a physical shaft (C1/C2 fix). ``trial_id`` threads
        # the per-session STATIC drop so the shaft mask aligns to the same
        # per-session montage as the front-end voltage (DP4).
        electrode_labels = voltage_electrode_order(self.bt_root, subject_id, trial_id)
        n_real = len(electrode_labels)
        if n_real > self.c_max:
            raise ValueError(
                f"subject {subject_id} has {n_real} electrodes which exceeds c_max={self.c_max}"
            )

        anatomy = SubjectAnatomy(
            shaft_ids=_bt_shaft_ids_from_labels(electrode_labels),
            contact_pos=_bt_contact_pos_from_labels(electrode_labels),
        )

        clip_seed = _bt_event_clip_seed(self.seed, subject_id, event)
        inner = ShaftMaskExtractor(
            seed=clip_seed,
            k_override=self.k_override,
            extent_blocks=tuple(self.extent_blocks),
        )
        real_mask = inner(anatomy)

        out = torch.zeros(self.c_max, dtype=torch.bool)
        out[:n_real] = real_mask
        return out
