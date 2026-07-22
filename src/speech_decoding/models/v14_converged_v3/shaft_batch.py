"""v14_converged_v3 — shaft-level (cross-patient) batching data layer (pretrain-only).

The L1-only towers carry NO session identity (a token is defined by its (depth, time_pos,
parcel_id) coords + its cu_seqlens block — proved correctness-neutral in
test_shaft_pack_isolation), so a batch can be a grab-bag of shafts from DIFFERENT patients.
This module is the data layer that assembles such a batch. The MODEL forward is untouched:
a cross-patient pack is represented as a **B=1 super-montage** — the K shafts unioned into one
``L1Geometry`` + one ``parcel_id`` + band data ``(1, ΣN, F, T)`` — which drops straight into
``V3ConvergedModel.forward(bands, geom, parcel_id, …)``.

Three rule-INDEPENDENT pieces live here (identical for bucketing or token-budget packing):
  1. ``build_shaft_pool`` — enumerate the global (session, shaft) pool + per-subject counts.
  2. ``TemperatureShaftSampler`` — draw a shaft via P(subject) ∝ n_shafts(subject)^α, then a
     shaft within the subject uniformly (α=0 ⇒ subject-uniform, α=1 ⇒ shaft-uniform).
  3. ``collate_shaft_pack`` — union K shaft-clips into a B=1 super-montage ``V3Batch``.

The grouping RULE (how many shafts per pack / how the shape is fixed) is deliberately NOT
here — that is the open bucket-vs-pack decision. ``collate_shaft_pack(pad_to_total=…)`` supports
the packing path's final-slack pad (append ONE fake always-visible shaft to hit an exact
``grid.total`` ⇒ a single compiled shape) — far less pad than bucketing's per-shaft fill,
because ``build_r4_grid`` counts VALID contacts only (pack_r4.py:115-118,140): the geom's
``valid=False`` pad does not enlarge the grid, so a fixed shape needs *valid* filler, and
packing needs it only for the buffer's tail slack.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from speech_decoding.models.v14_converged_v3.batch import V3Batch
from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

# Reserved parcel id for the packing tail-slack filler contacts (kept OUT of the DKT range
# so it never collides with a real parcel; the filler shaft is always-visible context, never
# a loss target — see collate_shaft_pack). Matches dispatch_v3's n_parcels=75 (0..74 real).
PAD_PARCEL_ID = 0  # any real id works — filler is masked out of loss by being always-visible.


@dataclass(frozen=True)
class ShaftRef:
    """One shaft in the global pool: where it lives + how big it is."""

    session_idx: int  # index into the sessions list
    shaft_id: int  # shaft id within that session's geometry
    n_contacts: int  # valid contacts on this shaft (drives the token budget)
    subject: object  # subject key for the temperature weight (sessions of one subject share it)


@dataclass(frozen=True)
class ShaftPool:
    """The global shaft pool + the per-subject grouping the temperature sampler needs."""

    refs: tuple[ShaftRef, ...]
    subjects: tuple[object, ...]  # distinct subjects, stable order
    subject_ref_idxs: tuple[tuple[int, ...], ...]  # subject i → indices into refs
    subject_n_shafts: Tensor  # (n_subjects,) long — shafts per subject (the temperature base)


def _default_subject_of(spec) -> object:
    """Group a subject's sessions (trials): the ``(subject, trial)`` session_key's first element."""
    key = getattr(spec, "session_key", None)
    return key[0] if isinstance(key, (tuple, list)) else key


def build_shaft_pool(
    sessions: Sequence, *, subject_of: Callable[[object], object] | None = None
) -> ShaftPool:
    """Enumerate every (session, shaft) in ``sessions`` into a global pool.

    ``sessions`` is the loaded ``list[V3SessionSpec]``; each spec exposes ``setup.geom.n_shafts``
    and per-shaft valid counts via ``setup.geom.valid.sum(1)``. ``subject_of(spec) → key`` groups
    a subject's sessions (trials) so the temperature weight is per SUBJECT, not per session; the
    default reads ``spec.session_key[0]`` (the (subject, trial) convention)."""
    subject_of = subject_of or _default_subject_of

    refs: list[ShaftRef] = []
    subj_order: list[object] = []
    subj_to_idxs: dict[object, list[int]] = {}
    for si, spec in enumerate(sessions):
        geom = spec.setup.geom
        per_shaft = geom.valid.sum(dim=1).tolist()  # valid contacts per shaft
        subj = subject_of(spec)
        for sh in range(geom.n_shafts):
            nc = int(per_shaft[sh])
            if nc == 0:
                continue  # a fully-dropped shaft carries no tokens — skip
            ref_i = len(refs)
            refs.append(ShaftRef(session_idx=si, shaft_id=sh, n_contacts=nc, subject=subj))
            if subj not in subj_to_idxs:
                subj_to_idxs[subj] = []
                subj_order.append(subj)
            subj_to_idxs[subj].append(ref_i)

    subject_n_shafts = torch.tensor(
        [len(subj_to_idxs[s]) for s in subj_order], dtype=torch.long
    )
    return ShaftPool(
        refs=tuple(refs),
        subjects=tuple(subj_order),
        subject_ref_idxs=tuple(tuple(subj_to_idxs[s]) for s in subj_order),
        subject_n_shafts=subject_n_shafts,
    )


class TemperatureShaftSampler:
    """Draw shafts via P(subject) ∝ n_shafts(subject)^α, then uniform within the subject.

    P(a specific shaft of subject u) = P(u)·(1/n_u) ∝ n_u^(α−1):
      * α = 1 ⇒ ∝ n_u^0 ⇒ every shaft equally likely (SHAFT-uniform).
      * α = 0 ⇒ P(subject) uniform, shaft uniform within (SUBJECT-uniform).
      * α = 0.5 ⇒ the √-tempered middle (the frozen default) — counters the electrode-count
        leak (r=−0.763) without starving shaft-rich subjects. The subject axis is the
        foundation-model axis, so balance is applied there, sessions/shafts nested.
    """

    def __init__(self, pool: ShaftPool, *, alpha: float = 0.5) -> None:
        if not pool.refs:
            raise ValueError("empty shaft pool")
        self.pool = pool
        self.alpha = float(alpha)
        # subject draw weights ∝ n_shafts^α (float64 for a clean multinomial).
        self._subj_w = pool.subject_n_shafts.to(torch.float64) ** self.alpha

    def draw(self, generator: torch.Generator) -> ShaftRef:
        u = int(torch.multinomial(self._subj_w, 1, generator=generator).item())
        idxs = self.pool.subject_ref_idxs[u]
        j = int(torch.randint(len(idxs), (1,), generator=generator).item())
        return self.pool.refs[idxs[j]]

    def draw_pack_to_budget(
        self, generator: torch.Generator, *, contact_budget: int
    ) -> list[tuple[ShaftRef, int]]:
        """Token-BUDGET packing draw, OVERFILL-AND-TRIM (drop, never pad).

        Draw shafts until the running total REACHES OR EXCEEDS ``contact_budget``, then
        TRIM the last shaft's kept-contact count so the pack sums to EXACTLY the budget.
        ``grid.total`` is thus pinned to one compiled shape with ZERO padding — the trimmed
        contacts are simply absent this step (they resample on a later step), never fake
        always-visible fill. Returns ``(ShaftRef, n_keep)`` pairs with Σ n_keep ==
        contact_budget; the caller loads the first ``n_keep`` contacts (depth order) of each
        shaft, trimming only the last. This is the packing RULE — swap it for a fixed-K
        bucketed draw to get the bucketing regime."""
        drawn: list[tuple[ShaftRef, int]] = []
        total = 0
        # each iter adds ≥1 contact ⇒ the budget closes within contact_budget draws (draw is
        # WITH replacement, so the pool never "runs dry" — the loop always reaches the budget).
        for _ in range(contact_budget):
            r = self.draw(generator)
            room = contact_budget - total
            if r.n_contacts >= room:  # this shaft closes (or overshoots) the budget → trim it
                drawn.append((r, room))
                return drawn
            drawn.append((r, r.n_contacts))
            total += r.n_contacts
        return drawn


@dataclass(frozen=True)
class ShaftClipSample:
    """One shaft's clip: its 3 |STFT| bands + the coords needed to place it in a super-montage.

    ``bands`` = 3 × (n_s, F_band, T). ``depth`` (n_s,) the ORIGINAL clinical contact numbers
    (gaps preserved — the index-RoPE coord, rebuilt into a per-pack label so this shaft stays a
    distinct block). ``parcel_id`` (n_s,) the DKT tags. ``session_key``/``shaft_id`` for logging.
    """

    bands: tuple[Tensor, ...]
    depth: Tensor  # (n_s,) long — original clinical depths
    parcel_id: Tensor  # (n_s,) long
    session_key: object
    shaft_id: int


def _pack_labels(depth: Tensor, slot: int) -> list[str]:
    """Rebuild labels for pack-slot ``slot`` that (a) keep each contact's clinical depth (so
    index-RoPE is unchanged) and (b) carry a UNIQUE prefix per slot so two patients' identically
    named shafts (both "LA…") do NOT merge into one block. ``build_sidecar`` parses prefix (shaft)
    + trailing number (depth); ``f"Z{slot}z{d}"`` ⇒ prefix ``Z{slot}z``, depth ``d``."""
    return [f"Z{slot}z{int(d)}" for d in depth.tolist()]


def collate_shaft_pack(
    samples: Sequence[ShaftClipSample], *, pad_to_total: int | None = None
) -> V3Batch:
    """Union K shaft-clips (from ARBITRARY sessions) into a B=1 super-montage ``V3Batch``.

    Each sample becomes its own shaft block (unique relabelled prefix); bands concat along the
    contact axis → ``(1, ΣN, F, T)``; parcel ids concat in the same order. The result feeds the
    UNCHANGED ``V3ConvergedModel.forward`` (B=1). ``session_key`` encodes the SHAPE (``ΣN`` after
    any pad) so the module's per-session plan-cache collapses to one entry per compiled shape —
    the plan-cache-by-shape the shaft path needs, for free.

    ``pad_to_total`` (packing path): append ONE filler shaft of ``pad_to_total − ΣN`` always-visible
    contacts so ``grid.total`` is EXACTLY fixed ⇒ a single compiled shape. Filler contacts are
    never loss targets: the masking makes them visible context only (they carry zero band data).
    ``None`` ⇒ no pad (variable-ΣN; the caller accepts a few shapes or is bucketed elsewhere)."""
    if not samples:
        raise ValueError("collate_shaft_pack got an empty pack")
    n_bands = len(samples[0].bands)

    labels: list[str] = []
    parcels: list[Tensor] = []
    band_chunks: list[list[Tensor]] = [[] for _ in range(n_bands)]
    total_n = 0
    for slot, s in enumerate(samples):
        n_s = s.bands[0].shape[0]
        labels.extend(_pack_labels(s.depth, slot))
        parcels.append(s.parcel_id.long())
        for b in range(n_bands):
            band_chunks[b].append(s.bands[b])
        total_n += n_s

    if pad_to_total is not None:
        pad_n = pad_to_total - total_n
        if pad_n < 0:
            raise ValueError(f"pack ΣN={total_n} exceeds pad_to_total={pad_to_total}")
        if pad_n > 0:
            slot = len(samples)
            labels.extend(_pack_labels(torch.arange(1, pad_n + 1), slot))
            parcels.append(torch.full((pad_n,), PAD_PARCEL_ID, dtype=torch.long))
            for b in range(n_bands):
                f, t = samples[0].bands[b].shape[1], samples[0].bands[b].shape[2]
                band_chunks[b].append(torch.zeros(pad_n, f, t))

    parcel_id = torch.cat(parcels)
    sidecar = build_sidecar(labels, parcel_id=parcel_id)
    geom = build_l1_geometry(sidecar)
    bands = [torch.cat(band_chunks[b], dim=0).unsqueeze(0) for b in range(n_bands)]  # (1,ΣN,F,T)
    return V3Batch(
        bands=bands,
        geom=geom,
        parcel_id=sidecar.parcel_id,
        session_key=("shaft_pack", int(parcel_id.shape[0])),  # shape key ⇒ plan-cache by shape
    )


__all__ = [
    "ShaftRef",
    "ShaftPool",
    "build_shaft_pool",
    "TemperatureShaftSampler",
    "ShaftClipSample",
    "collate_shaft_pack",
]
