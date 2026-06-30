"""Per-cell attentive readouts for the pretrain probe suite — Stage 2b.

The set-attention (PMA) NONLINEAR analogue of the linear ridge in
:mod:`pretrain_probe_readout`, layered onto the SAME Stage-1 cache grids. Where the
linear ridge needs a fixed feature vector (so cross-subject must intersect to common
dims), the attentive head is permutation-invariant over a variable token set — so the
contract (project_pretrain_probe_suite_contract_2026_06_30) is:

  ATTENTIVE — full grid, NO pool, NO intersect:
    M2 (electrode-space): electrode tokens (no pool), keep-S, full set — WS and CS.
    M3/M4 (parcel-space): parcel tokens over the globally-fixed grid, keep-S — WS and CS.

Token layout makes the head's two positional mechanisms line up:
  - electrode tap grid ``(N, C, S, d)`` → ``(N, C·S, d)``; token = ``c·S + s`` so
    ``t % S = s`` (the learnable time tag, ``n_time_frames=S``) and the per-electrode
    anatomical parcel id rides in ``token_parcel_ids`` for irregular-group parcel dropout.
  - parcel tap grid ``(N, P, k, S, d)`` → ``(N, P·k·S, d)``; token = ``p·kS + ki·S + s`` so
    again ``t % S = s``, and the parcel id (``arange(P)`` repeated ``k·S``) drives dropout.

Unsupported parcels (no electrode coverage for this subject) ride as ``key_padding_mask``
False — honest per-subject coverage, NOT a cross-subject intersection. ``parcel_dropout``
then augments by dropping further parcels at train time (the cross-subject-shift mimic).

Per cell: fit on the train tokens, early-stop on the val half's AUROC (SWAD on), report
the held-out test half's AUROC — the better of the best-val / SWAD snapshot by val.
HP selection (weight_decay / parcel_dropout / dropout) is swept ONCE upstream, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

from speech_decoding.experiments.pretrain_probe_readout import parcel_support
from speech_decoding.experiments.v2_attentive_train import (
    HeadTrainConfig,
    _build_head,
    train_head,
)

__all__ = [
    "CellResult",
    "attentive_ws_cell_auroc",
    "attentive_cs_cell_auroc",
    "attentive_ws_cell_result",
    "attentive_cs_cell_result",
]


@dataclass(frozen=True)
class CellResult:
    """One attentive cell's held-out AUROCs. ``val`` selects HP (sweep-once); ``test``
    reports. ``val`` is the chosen snapshot's val AUROC (the same set used for early-stop
    + best-vs-SWAD); ``test`` is the never-touched test half."""

    val: float
    test: float


def _electrode_tokens(
    grid: Tensor, electrode_mask: Tensor, parcel_per_electrode: Tensor
) -> tuple[Tensor, None, Tensor, int]:
    """``(N,C,S,d)`` → ``(N, C_valid·S, d)`` tokens (valid electrodes only, no pool).

    Returns ``(tokens, mask=None, token_parcel_ids (C_valid·S,), S)``."""
    valid = electrode_mask.bool()
    g = grid[:, valid]                                    # (N, Cv, S, d)
    n, cv, s, d = g.shape
    tokens = g.reshape(n, cv * s, d)
    pids = parcel_per_electrode[valid].long().repeat_interleave(s)
    return tokens, None, pids, s


def _parcel_tokens(
    grid: Tensor, support: Tensor, parcel_labels: Tensor
) -> tuple[Tensor, Tensor, Tensor, int]:
    """``(N,P,k,S,d)`` → ``(N, P·k·S, d)`` tokens over a COMPACTED parcel grid.

    The grid's P axis is ``parcel_labels`` (the ~16 present parcels, NOT the full atlas).
    ``token_parcel_ids`` carry the real DKT atlas ids — so parcel-group dropout and the CS
    anchor/test alignment reason in anatomy, not compacted position. Unsupported parcels
    (in the compacted set but with no valid electrode this subject) are masked
    (``key_padding_mask`` False), not dropped. Returns ``(tokens, mask (N,P·k·S),
    token_parcel_ids, S)``."""
    n, p, k, s, d = grid.shape
    lab = parcel_labels.long()
    tokens = grid.reshape(n, p * k * s, d)
    pids = lab.repeat_interleave(k * s)
    tok_valid = support.bool()[lab].repeat_interleave(k * s)   # (P·k·S,)
    mask = tok_valid[None].expand(n, -1).contiguous()
    return tokens, mask, pids, s


def _build_tokens(
    grid: Tensor,
    tap_space: str,
    parcel_per_electrode: Tensor,
    electrode_mask: Tensor,
    n_parcels: int,
    parcel_labels: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, Tensor, int]:
    if tap_space == "electrode":
        return _electrode_tokens(grid, electrode_mask, parcel_per_electrode)
    if tap_space == "parcel":
        if parcel_labels is None:
            raise ValueError("parcel tap_space needs parcel_labels (compacted grid ids)")
        support = parcel_support(parcel_per_electrode, electrode_mask, n_parcels)
        return _parcel_tokens(grid, support, parcel_labels)
    raise ValueError(f"tap_space must be 'electrode'|'parcel'; got {tap_space!r}")


def _take(
    tokens: Tensor, mask: Tensor | None, y: np.ndarray, rows: np.ndarray
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Select ``rows`` with finite labels → ``(x, mask, y)``."""
    rows = np.asarray(rows)
    yr = np.asarray(y, dtype=float)[rows]
    fin = np.isfinite(yr)
    sel = torch.from_numpy(rows[fin]).long()
    x = tokens[sel]
    m = None if mask is None else mask[sel]
    yy = torch.from_numpy(yr[fin]).float()
    return x, m, yy


def _pick(res: dict) -> tuple[dict, float]:
    """The better of the best-val / SWAD snapshot, with its val AUROC (SWAD on)."""
    swad_val = res["swad_val"]
    if not np.isfinite(swad_val) or res["best_val"] >= swad_val:
        return res["best_state"], res["best_val"]
    return res["swad_state"], swad_val


@torch.no_grad()
def _score(
    cfg: HeadTrainConfig,
    state: dict,
    x: Tensor,
    y: Tensor,
    mask: Tensor | None,
    device: torch.device,
) -> float:
    yv = y.cpu().numpy()
    if len(np.unique(yv)) < 2 or x.shape[0] < 2:
        return float("nan")
    head = _build_head(cfg).to(device)
    head.load_state_dict(state)
    head.eval()
    s = head(x.to(device), None if mask is None else mask.to(device))
    return float(roc_auc_score(yv, s.squeeze(-1).cpu().numpy()))


def _cfg_for(cfg: HeadTrainConfig, s_frames: int) -> HeadTrainConfig:
    """Force the time tag on (``n_time_frames=S``) and the group-dropout path
    (``tokens_per_parcel=0`` — dropout is driven by ``token_parcel_ids``)."""
    return replace(cfg, n_time_frames=s_frames, tokens_per_parcel=0)


def _fit_and_score(
    x_tr: Tensor, m_tr: Tensor | None, y_tr: Tensor,
    x_val: Tensor, m_val: Tensor | None, y_val: Tensor,
    x_te: Tensor, m_te: Tensor | None, y_te: Tensor,
    pids_tr: Tensor, cfg2: HeadTrainConfig, device: torch.device,
) -> CellResult:
    if x_tr.shape[1] == 0 or x_tr.shape[0] < 2 or x_te.shape[0] < 2:
        return CellResult(float("nan"), float("nan"))
    res = train_head(
        x_tr, y_tr, x_val, y_val, cfg2,
        mask_tr=m_tr, mask_val=m_val, token_parcel_ids_tr=pids_tr, device=device,
    )
    state, val = _pick(res)
    return CellResult(val, _score(cfg2, state, x_te, y_te, m_te, device))


def attentive_ws_cell_result(
    grid: Tensor,
    y: np.ndarray,
    *,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
    tap_space: str,
    parcel_per_electrode: Tensor,
    electrode_mask: Tensor,
    n_parcels: int,
    parcel_labels: Tensor | None = None,
    cfg: HeadTrainConfig,
    device: torch.device | None = None,
) -> CellResult:
    """WithinSession attentive cell: fit on ``train_rows``, early-stop on ``val_rows``,
    report AUROC on ``test_rows`` (all one session's full-grid token set)."""
    device = device or torch.device("cpu")
    tokens, mask, pids, s = _build_tokens(
        grid, tap_space, parcel_per_electrode, electrode_mask, n_parcels, parcel_labels
    )
    x_tr, m_tr, y_tr = _take(tokens, mask, y, train_rows)
    x_val, m_val, y_val = _take(tokens, mask, y, val_rows)
    x_te, m_te, y_te = _take(tokens, mask, y, test_rows)
    return _fit_and_score(
        x_tr, m_tr, y_tr, x_val, m_val, y_val, x_te, m_te, y_te,
        pids, _cfg_for(cfg, s), device,
    )


def attentive_cs_cell_result(
    grid_anchor: Tensor,
    y_anchor: np.ndarray,
    grid_test: Tensor,
    y_test: np.ndarray,
    *,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
    tap_space: str,
    pe_anchor: Tensor,
    em_anchor: Tensor,
    pe_test: Tensor,
    em_test: Tensor,
    n_parcels: int,
    parcel_labels_anchor: Tensor | None = None,
    parcel_labels_test: Tensor | None = None,
    cfg: HeadTrainConfig,
    device: torch.device | None = None,
) -> CellResult:
    """CrossSubject attentive cell: fit on ALL the anchor tokens, early-stop on the test
    session's val half, report AUROC on its test half. Full grid, no pool, no intersect
    — the anchor and test token sets keep their own per-subject coverage (masked, not
    intersected); ``parcel_dropout`` augments the train side for the cross-subject shift."""
    device = device or torch.device("cpu")
    tok_a, mask_a, pids_a, s = _build_tokens(
        grid_anchor, tap_space, pe_anchor, em_anchor, n_parcels, parcel_labels_anchor
    )
    tok_t, mask_t, _, _ = _build_tokens(
        grid_test, tap_space, pe_test, em_test, n_parcels, parcel_labels_test
    )
    anchor_rows = np.arange(len(y_anchor))
    x_tr, m_tr, y_tr = _take(tok_a, mask_a, y_anchor, anchor_rows)
    x_val, m_val, y_val = _take(tok_t, mask_t, y_test, val_rows)
    x_te, m_te, y_te = _take(tok_t, mask_t, y_test, test_rows)
    return _fit_and_score(
        x_tr, m_tr, y_tr, x_val, m_val, y_val, x_te, m_te, y_te,
        pids_a, _cfg_for(cfg, s), device,
    )


def attentive_ws_cell_auroc(*args, **kwargs) -> float:
    """Test AUROC for a WithinSession attentive cell (see :func:`attentive_ws_cell_result`)."""
    return attentive_ws_cell_result(*args, **kwargs).test


def attentive_cs_cell_auroc(*args, **kwargs) -> float:
    """Test AUROC for a CrossSubject attentive cell (see :func:`attentive_cs_cell_result`)."""
    return attentive_cs_cell_result(*args, **kwargs).test
