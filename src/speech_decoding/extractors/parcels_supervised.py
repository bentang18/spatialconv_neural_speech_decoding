"""V14 ``parcels_supervised`` per-subject map (MASK-02).

B03 mask-discipline lock (2026-05-25 PM, ``project_v14_b03_mask_lock_2026_05_25``)
replaces the per-clip JEPA-inverted ``valid_parcels[clip]`` mask with a
**per-subject** ``parcels_supervised[subject]`` map: for each subject, the
set of DK parcels with ≥1 electrode. SSL loss-side gates (``L_mid_slot``,
``L_post_frame``, ``L_post_utterance``) scope only to slots whose parcel is
in this set. Slot-bank regularizers (``L_DKoleo``, monitor F1, reactive Gram)
revert to **all 320 slots** per B21.

This is a build-time computation: the set per subject does not change across
clips. Compute once when the study or extractor pipeline initializes; carry
it as a metadata field on the batch so the encoder forward + loss heads can
gate without re-reading anatomy at each step.

SWEC fallback (empty set): downstream loss sites are anatomy-blind for that
subject and supervise all 320 slots. The same convention applies to any
subject for which DK anatomy is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from speech_decoding.studies.braintreebank.anatomy import (
    V14_DK_PARCEL_LABELS,
    load_public_bt_anatomy,
)


def compute_parcels_supervised_from_support(
    support: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> set[int]:
    """Per-subject DK parcel set with ≥1 covered electrode.

    Pure tensor helper — input is an electrode×parcel support matrix of
    shape ``(C, K)`` (the same matrix the encoder consumes as the cross-attn
    bias source). A parcel is "supervised" if any electrode has nonzero
    support there.

    ``valid_mask`` (optional ``(C,)`` bool) masks out padded electrode rows
    before the reduction — keeps the count consistent with how the encoder
    applies ``valid_mask`` at cross-attn time.
    """
    if support.ndim != 2:
        raise ValueError(
            f"support must be 2-D (C, K); got shape {tuple(support.shape)}"
        )
    if valid_mask is not None:
        if valid_mask.shape[0] != support.shape[0]:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} does not match "
                f"support[0]={support.shape[0]}"
            )
        support = support * valid_mask.unsqueeze(-1).to(support.dtype)
    parcel_covered = (support > 0).any(dim=0)                   # (K,) bool
    return {int(p) for p in torch.nonzero(parcel_covered, as_tuple=False).flatten().tolist()}


def compute_parcels_supervised_bt(
    bt_root: str | Path,
    subject_id: int,
    *,
    parcel_labels: tuple[str, ...] = V14_DK_PARCEL_LABELS,
) -> set[int]:
    """Per-BrainTreebank-subject ``parcels_supervised`` set.

    Reads BT-shipped ``localization/sub_<id>/depth-wm.csv`` via
    ``load_public_bt_anatomy`` and returns ``set[parcel_id]`` for every DK
    label that maps into ``parcel_labels`` (the v14 K=80 vocabulary).

    Labels outside the vocabulary (e.g. ``Left-Inf-Lat-Vent``) are silently
    skipped here — at extractor time those electrodes are either dropped
    (``unknown_label_policy="skip"``) or raise (``"raise"``); from the
    parcel-set perspective they simply don't contribute coverage.
    """
    anatomy = load_public_bt_anatomy(str(bt_root), subject_id)
    label_to_idx = {label: i for i, label in enumerate(parcel_labels)}
    supervised: set[int] = set()
    for label in anatomy["DesikanKilliany"]:
        idx = label_to_idx.get(label)
        if idx is not None:
            supervised.add(idx)
    return supervised


def parcels_supervised_to_slot_mask(
    parcels_supervised: Iterable[int] | None,
    *,
    k_parcels: int,
    m_sub_slots: int,
) -> torch.Tensor:
    """Convert a per-subject parcel set to a flat ``(K*M,)`` bool slot mask.

    The encoder's latent stack and SSL loss heads operate over ``K*M`` flat
    slots; each parcel ``p`` owns ``M`` consecutive sub-slots
    ``[p*M, ..., p*M + M - 1]``. A parcel in ``parcels_supervised`` enables
    all ``M`` of its sub-slots.

    SWEC fallback (``parcels_supervised`` empty or None) returns an
    all-True mask: anatomy-blind subjects supervise every slot, so the loss
    head sees the full slot bank.
    """
    if parcels_supervised is None or len(set(parcels_supervised)) == 0:
        return torch.ones(k_parcels * m_sub_slots, dtype=torch.bool)
    parcel_mask = torch.zeros(k_parcels, dtype=torch.bool)
    for p in parcels_supervised:
        if not (0 <= int(p) < k_parcels):
            raise ValueError(
                f"parcels_supervised contains invalid id {p}; expected "
                f"0 ≤ p < {k_parcels}"
            )
        parcel_mask[int(p)] = True
    return (
        parcel_mask.unsqueeze(-1)
                   .expand(k_parcels, m_sub_slots)
                   .reshape(k_parcels * m_sub_slots)
                   .contiguous()
    )
