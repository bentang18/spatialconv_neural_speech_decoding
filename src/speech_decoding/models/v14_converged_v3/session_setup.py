"""v14_converged_v3 — per-session setup (Phase D1).

The pure core of the per-session build (memo project-v3-pipeline-build-contract-
2026-07-10). One call turns a session's FULL voltage electrode order + hard parcel
tags + the guard-1 bad-electrode set into everything the training loop needs that
is fixed for the session's lifetime:

    labels, parcel_id, drop_labels ─▶ guard-1 filter ─▶ SensorSidecar
                                                     ─▶ L1Geometry
                                                     ─▶ assert_mask_feasible (#36)
                                         keep_idx = the memmap channel-read plan

GUARD-1 (bad ELECTRODES) is voltage-domain and hop-independent, so it lands here at
sidecar/geometry build — NOT as a spec-cache recompute. The drop set is the union
of ``extra_bad`` (manual, from the spec .json ``key.timeline.extra_bad``) and the
LOF / ``drop_bads`` contacts; both name ELECTRODES, so the core drops by label.

This is the once-per-session cold path (montage is fixed for the session), so plain
python filtering is fine — the "always vectorize" rule governs the per-step hot
loop, not setup. The file-reading adapter that supplies ``labels`` / ``parcel_id`` /
``drop_labels`` from disk is a thin layer validated on real data at F2; the logic
here is exercised synthetically.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from speech_decoding.models.v14_converged_v3.geometry import (
    L1Geometry,
    build_l1_geometry,
)
from speech_decoding.models.v14_converged_v3.masking import (
    V3MaskConfig,
    assert_mask_feasible,
)
from speech_decoding.models.v14_converged_v3.sidecar import (
    SensorSidecar,
    build_sidecar,
)
from speech_decoding.models.v14_converged_v3.state_target import STATE_DIM


@dataclass(frozen=True)
class V3SessionSetup:
    """Everything fixed for one session's lifetime after guard-1."""

    keep_idx: Tensor  # (N,) long — survivor rows into the FULL voltage order = memmap read plan
    sidecar: SensorSidecar  # shaft_id / depth / parcel_id over survivors
    geom: L1Geometry  # padded per-shaft L1 gather plan
    parcel_id: Tensor  # (N,) long — survivor parcel tags (== sidecar.parcel_id)
    # frozen per-(parcel,dim) z-score stats for the secondary Gaussian-NLL, gathered to
    # (P, 6) aligned to the survivor unique(parcel_id) — None ⇒ secondary OFF (JEPA-only).
    stat_mean: Tensor | None = None
    stat_std: Tensor | None = None


def _align_state_stats(
    parcel_id: Tensor, stat_mean: Tensor | None, stat_std: Tensor | None
) -> tuple[Tensor | None, Tensor | None]:
    """Gather by-parcel-VALUE frozen state stats into (P, 6) aligned to unique(parcel_id).

    ``stat_mean``/``stat_std`` are indexed by parcel id VALUE (the same index space as the
    perceiver's parcel embedding and ``raw_state_vectors``' ``torch.unique(parcel_id)``), so
    the survivor set after guard-1 selects the rows. The secondary Gaussian-NLL is OPT-IN:
    both omitted ⇒ (None, None) ⇒ JEPA-only. Fails loud on shape / coverage (cold path)."""
    if stat_mean is None and stat_std is None:
        return None, None
    if stat_mean is None or stat_std is None:
        raise ValueError("stat_mean and stat_std must both be given, or both omitted")
    parcels = torch.unique(parcel_id)  # sorted survivor ids == raw_state_vectors order
    hi = int(parcels.max().item())
    for name, t in (("stat_mean", stat_mean), ("stat_std", stat_std)):
        if t.ndim != 2 or t.shape[1] != STATE_DIM:
            raise ValueError(
                f"{name} must be (n_parcels, {STATE_DIM}) indexed by parcel id; "
                f"got {tuple(t.shape)}"
            )
        if hi >= t.shape[0]:
            raise ValueError(
                f"{name} rows {t.shape[0]} do not cover parcel id {hi}"
            )
    return stat_mean[parcels], stat_std[parcels]


def build_session_setup(
    labels: Sequence[str],
    parcel_id: Tensor,
    *,
    drop_labels: Iterable[str],
    mask_cfg: V3MaskConfig = V3MaskConfig(),
    stat_mean: Tensor | None = None,
    stat_std: Tensor | None = None,
) -> V3SessionSetup:
    """Guard-1 filter → sidecar → geometry → feasibility assert.

    ``labels`` is the FULL session voltage electrode order (memmap row order);
    ``parcel_id`` (N_full,) is the hard DKT tag per full-order electrode.
    ``drop_labels`` = extra_bad ∪ LOF/drop_bads; labels not in the montage are a
    no-op (a source may name a channel already excluded upstream). Raises via
    ``assert_mask_feasible`` if the surviving montage + ``mask_cfg`` cannot hold a
    fixed-M mask (#36) — fail loud at setup, never inside the compiled forward.
    """
    labels = tuple(str(x) for x in labels)
    if parcel_id.shape != (len(labels),):
        raise ValueError(
            f"parcel_id shape {tuple(parcel_id.shape)} != ({len(labels)},) (full order)"
        )
    drop = set(drop_labels)
    keep_positions = [i for i, lab in enumerate(labels) if lab not in drop]
    if not keep_positions:
        raise ValueError("guard-1 dropped every electrode — no survivors")

    keep_idx = torch.tensor(keep_positions, dtype=torch.long)
    keep_labels = tuple(labels[i] for i in keep_positions)
    keep_parcel = parcel_id[keep_idx].long()

    sidecar = build_sidecar(keep_labels, parcel_id=keep_parcel)
    geom = build_l1_geometry(sidecar)
    assert_mask_feasible(geom, mask_cfg)
    sm, ss = _align_state_stats(sidecar.parcel_id, stat_mean, stat_std)

    return V3SessionSetup(
        keep_idx=keep_idx,
        sidecar=sidecar,
        geom=geom,
        parcel_id=sidecar.parcel_id,
        stat_mean=sm,
        stat_std=ss,
    )
