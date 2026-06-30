"""Stage-2 driver: dispatch a :class:`ProbeCell` to its cache grids + readout.

Stage 1 (GPU, DCC-gated) forwards the encoder over a cohort session's balanced anchors
and saves a :class:`SessionTapCache` — the per-token tap grids (M2 electrode-space,
M3/M4 parcel-space), the per-task labels, the per-mode split row indices, and the
electrode→parcel metadata. Stage 2 (CPU, re-runnable) is this module: given the cells
(:func:`enumerate_cells`) and the loaded caches, it runs the linear ridge and attentive
readouts per (cell, tap) and returns AUROCs. No GPU, no BT data — pure cache → readout.

The cache is PER SESSION, not per cell: one forward covers all 15 tasks (a clip's label
is NaN for tasks whose balanced set excludes it; the readouts drop NaN rows) and both
WS folds and the CS split. CrossSubject cells additionally consume the subject-2 anchor
cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from speech_decoding.experiments.pretrain_probe_attentive import (
    CellResult,
    attentive_cs_cell_result,
    attentive_ws_cell_result,
)
from speech_decoding.experiments.pretrain_probe_readout import (
    DEFAULT_LAM_GRID,
    linear_cs_cell_auroc,
    linear_ws_cell_auroc,
)
from speech_decoding.experiments.pretrain_probe_suite import ProbeCell
from speech_decoding.experiments.v2_attentive_train import HeadTrainConfig

__all__ = [
    "TAP_SPACE",
    "SessionTapCache",
    "save_cache",
    "load_cache",
    "run_linear_cell",
    "run_attentive_cell",
    "run_attentive_cell_result",
]

# Tap → feature space. M2 is electrode-space (pre-pool); M3/M4 are parcel-space (post-PMA).
TAP_SPACE: dict[str, str] = {"M2": "electrode", "M3": "parcel", "M4": "parcel"}


@dataclass
class SessionTapCache:
    """One cohort session's forwarded tap grids + labels + splits + electrode metadata.

    ``grids["M2"]`` is ``(N, C, S, d)`` (electrode-space); ``grids["M3"]``/``["M4"]`` are
    ``(N, P, k, S, d)`` (parcel-space). ``labels[task]`` is ``(N,)`` in {0,1,NaN}.
    ``ws_split[fold]`` is ``{"train","val","test"}`` row-index arrays (KFold2 held-out
    fold halved); ``cs_split`` is ``{"val","test"}`` for when this session is a CS test
    session (its 50/50 val/test halves)."""

    subject_id: int
    trial_id: int
    grids: dict[str, Tensor]
    labels: dict[str, object]                 # task -> np.ndarray (N,)
    parcel_per_electrode: Tensor              # (C,)
    electrode_mask: Tensor                    # (C,)
    n_parcels: int
    ws_split: dict[int, dict[str, object]]    # fold -> {train,val,test} -> np.ndarray
    cs_split: dict[str, object]               # {val,test} -> np.ndarray


def save_cache(cache: SessionTapCache, path: str) -> None:
    torch.save(cache, path)


def load_cache(path: str) -> SessionTapCache:
    return torch.load(path, weights_only=False)


def run_linear_cell(
    cell: ProbeCell,
    tap: str,
    *,
    test_cache: SessionTapCache,
    anchor_cache: SessionTapCache | None = None,
    lam_grid: tuple[float, ...] = DEFAULT_LAM_GRID,
) -> float:
    """Linear ridge AUROC for one (cell, tap). WS reads ``test_cache`` only; CS reads the
    anchor (train) cache + the test session cache (λ on the test val half)."""
    space = TAP_SPACE[tap]
    if cell.eval_mode == "WithinSession":
        sp = test_cache.ws_split[cell.fold_index]
        return linear_ws_cell_auroc(
            test_cache.grids[tap], test_cache.labels[cell.task],
            train_rows=sp["train"], val_rows=sp["val"], test_rows=sp["test"],
            tap_space=space,
            parcel_per_electrode=test_cache.parcel_per_electrode,
            electrode_mask=test_cache.electrode_mask,
            n_parcels=test_cache.n_parcels, lam_grid=lam_grid,
        )
    if cell.eval_mode == "CrossSubject":
        if anchor_cache is None:
            raise ValueError("CrossSubject linear cell needs an anchor_cache")
        sp = test_cache.cs_split
        return linear_cs_cell_auroc(
            anchor_cache.grids[tap], anchor_cache.labels[cell.task],
            test_cache.grids[tap], test_cache.labels[cell.task],
            val_rows=sp["val"], test_rows=sp["test"], tap_space=space,
            pe_anchor=anchor_cache.parcel_per_electrode, em_anchor=anchor_cache.electrode_mask,
            pe_test=test_cache.parcel_per_electrode, em_test=test_cache.electrode_mask,
            n_parcels=test_cache.n_parcels, lam_grid=lam_grid,
        )
    raise ValueError(f"unsupported eval mode {cell.eval_mode!r}")


def run_attentive_cell_result(
    cell: ProbeCell,
    tap: str,
    cfg: HeadTrainConfig,
    *,
    test_cache: SessionTapCache,
    anchor_cache: SessionTapCache | None = None,
    device: torch.device | None = None,
) -> CellResult:
    """Attentive (set-attention) val+test AUROC for one (cell, tap). Full grid, no
    intersect. ``val`` drives the sweep-once HP pick; ``test`` is the held-out report."""
    space = TAP_SPACE[tap]
    if cell.eval_mode == "WithinSession":
        sp = test_cache.ws_split[cell.fold_index]
        return attentive_ws_cell_result(
            test_cache.grids[tap], test_cache.labels[cell.task],
            train_rows=sp["train"], val_rows=sp["val"], test_rows=sp["test"],
            tap_space=space,
            parcel_per_electrode=test_cache.parcel_per_electrode,
            electrode_mask=test_cache.electrode_mask,
            n_parcels=test_cache.n_parcels, cfg=cfg, device=device,
        )
    if cell.eval_mode == "CrossSubject":
        if anchor_cache is None:
            raise ValueError("CrossSubject attentive cell needs an anchor_cache")
        sp = test_cache.cs_split
        return attentive_cs_cell_result(
            anchor_cache.grids[tap], anchor_cache.labels[cell.task],
            test_cache.grids[tap], test_cache.labels[cell.task],
            val_rows=sp["val"], test_rows=sp["test"], tap_space=space,
            pe_anchor=anchor_cache.parcel_per_electrode, em_anchor=anchor_cache.electrode_mask,
            pe_test=test_cache.parcel_per_electrode, em_test=test_cache.electrode_mask,
            n_parcels=test_cache.n_parcels, cfg=cfg, device=device,
        )
    raise ValueError(f"unsupported eval mode {cell.eval_mode!r}")


def run_attentive_cell(
    cell: ProbeCell,
    tap: str,
    cfg: HeadTrainConfig,
    *,
    test_cache: SessionTapCache,
    anchor_cache: SessionTapCache | None = None,
    device: torch.device | None = None,
) -> float:
    """Held-out test AUROC for one attentive (cell, tap)."""
    return run_attentive_cell_result(
        cell, tap, cfg, test_cache=test_cache, anchor_cache=anchor_cache, device=device
    ).test
