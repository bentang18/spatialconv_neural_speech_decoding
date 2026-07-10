"""v14_converged_v3 — L1 shaft-gather geometry (Phase 4a).

The L1 block attends jointly over (contact, time) WITHIN a shaft, block-diagonal
by shaft. To run it as one vectorized batched attention (no python loop over
shafts in the hot path), we pre-gather the N contacts into a ragged
``(n_shafts, max_c)`` grid — each row is one shaft, short shafts zero-padded — and
carry a validity mask plus the per-slot depth (the index-RoPE coordinate). This is
the "per-shaft cu_seqlens, zero pad" of the memo, realised as a padded gather so
it runs on any backend (production may swap in FlashAttention varlen for the same
result). Built once per session at setup — not the training loop, so the ragged
construction here may use plain python.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from speech_decoding.models.v14_converged_v3.sidecar import SensorSidecar


@dataclass(frozen=True)
class L1Geometry:
    """Padded per-shaft gather plan for the L1 block-diagonal attention."""

    gather_idx: Tensor  # (n_shafts, max_c) long — contact index into N; pad → 0
    valid: Tensor  # (n_shafts, max_c) bool — False on pad slots
    depth: Tensor  # (n_shafts, max_c) long — clinical contact index per slot
    shaft_of_contact: Tensor  # (N,) long — shaft id per contact (tier-split monitor)
    n_shafts: int
    max_c: int


def build_l1_geometry(sidecar: SensorSidecar) -> L1Geometry:
    """Group the sidecar's contacts into a padded ``(n_shafts, max_c)`` gather plan.

    Row s holds the contact indices of shaft s (order of appearance), zero-padded
    to ``max_c`` = the largest shaft's contact count. ``valid`` masks pad slots;
    ``depth`` carries each slot's clinical contact number (gaps preserved) for the
    index-RoPE inside the L1 block.
    """
    shaft_id = sidecar.shaft_id
    depth = sidecar.depth
    n_shafts = sidecar.n_shafts
    n = shaft_id.shape[0]

    members: list[list[int]] = [[] for _ in range(n_shafts)]
    for i in range(n):
        members[int(shaft_id[i])].append(i)
    max_c = max(len(m) for m in members)

    gather_idx = torch.zeros((n_shafts, max_c), dtype=torch.long)
    valid = torch.zeros((n_shafts, max_c), dtype=torch.bool)
    depth_pad = torch.zeros((n_shafts, max_c), dtype=torch.long)
    for s, m in enumerate(members):
        idx = torch.tensor(m, dtype=torch.long)
        gather_idx[s, : len(m)] = idx
        valid[s, : len(m)] = True
        depth_pad[s, : len(m)] = depth[idx]

    return L1Geometry(
        gather_idx=gather_idx,
        valid=valid,
        depth=depth_pad,
        shaft_of_contact=shaft_id.clone(),
        n_shafts=n_shafts,
        max_c=max_c,
    )
