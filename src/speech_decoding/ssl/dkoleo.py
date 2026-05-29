"""LOSS-05 (B21 collapse-prevention lock 2026-05-25): slot-bank uniformity
regularizer ``L_DKoleo @ M4``.

Operates over **all 320 slots** of the M4 latent state — NOT restricted to
``parcels_supervised[subject]`` (B21 default; slot-bank regularizers are
anatomy-blind so the full latent bank remains well-spread even on subjects
where most parcels are unsupervised).

Metric (DINOv3 koleo, per slot::

    L_DKoleo = − (1 / L) · Σ_i log( min_{j ≠ i} ‖ z_i − z_j ‖₂ )

where ``z`` is the per-slot vector obtained by mean-pooling M4 over the
time-patch axis. Computed per batch element, then averaged over the batch.

Weight 0.1 always-on in P1 + P2 (LOSS-01). Reactive cousin ``L_DKoleo@M3``
(weight 0.05) is off-default and lives in the loss composer, not here.
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
