"""LOSS-05: slot-bank uniformity regularizer ``L_DKoleo @ M4``.

**Sister-only primitive** — B28 (2026-05-27) demoted DKoleo from the always-on
default to three opt-in sisters (``R-dkoleo-batch-cls-unit`` /
``R-dkoleo-intra-clip-slots`` / ``R-vicreg-slot-variance``). The B31 2-term
default (``L_pre_frame @ M2 + L_post_frame @ M4``) does NOT include it. The
loss composer (``ssl/total_loss.py``) wires it at weight 0.1 only when a
``l_dkoleo_m4`` tensor is supplied.

Operates over **all latent slots** of the M4 state (80 at the B29 M=1 default,
320 under the ``R-m4-slots`` M=4 sister) — slot-bank regularizers are
anatomy-blind, so they are NOT gated by ``latent_valid``; the full bank stays
well-spread even on subjects where most parcels have no coverage.

Metric (DINOv3 koleo, per slot::

    L_DKoleo = − (1 / L) · Σ_i log( min_{j ≠ i} ‖ z_i − z_j ‖₂ )

where ``z`` is the per-slot vector obtained by mean-pooling M4 over the
time-patch axis. Computed per batch element, then averaged over the batch.
"""

from __future__ import annotations

import torch
from torch import Tensor

from speech_decoding.ssl.koleo import koleo_loss


def dkoleo_m4_loss(m4: Tensor) -> Tensor:
    """Slot-bank uniformity loss over the M4 latent stream.

    ``m4`` shape ``(B, L, T_p, d)``. Reduces over the time-patch axis,
    then computes :func:`speech_decoding.ssl.koleo.koleo_loss` per batch
    element over the ``L`` slots and averages.

    Returns a scalar tensor.
    """
    if m4.ndim != 4:
        raise ValueError(
            f"expected (B, L, T_p, d); got {tuple(m4.shape)}"
        )
    B = m4.shape[0]
    # Mean-pool the time-patch axis to a single d-vector per slot.
    slot_vecs = m4.mean(dim=2)                              # (B, L, d)
    per_batch = torch.stack(
        [koleo_loss(slot_vecs[b]) for b in range(B)]
    )
    return per_batch.mean()
