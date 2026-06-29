"""Training + weight-averaging core for the attentive probe head.

Two pieces, both laptop-testable on synthetic token sets (the LOSO orchestration and
the WD/LS/dropout sweep are mechanical glue layered on top in the bench driver):

  - :func:`train_head` — train one :class:`AttentiveProbeHead` with BCE (+ optional
    binary label smoothing), nested-val EARLY STOP on val AUROC, and SWAD-style dense
    weight averaging (a running mean of post-warmup snapshots, the window bounded below
    by ``swad_warmup`` and above by early-stop — the dominant SWAD effect of averaging
    the dense tail BEFORE overfit, without the paper's finicky val-loss ts/te search).
    Returns both the best-val snapshot and the SWAD mean so the caller picks per val.
  - :func:`diwa_average` — DiWA: parameter-wise mean of several independently-trained
    head state-dicts (different seeds/HPs). Zero inference overhead, the top
    cross-subject extension (arXiv:2205.09739). Compose with per-run SWAD.

The val metric is AUROC (the eval target), NOT loss — early-stop on a held-out TRAIN
subject's AUROC is what selects for cross-subject transfer (the load-bearing reg)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

from speech_decoding.experiments.v2_attentive_probe import AttentiveProbeHead

__all__ = ["HeadTrainConfig", "train_head", "diwa_average"]


@dataclass
class HeadTrainConfig:
    d_model: int
    n_heads: int = 6
    n_queries: int = 1
    mlp_ratio: float = 2.0
    attn_dropout: float = 0.1
    mlp_dropout: float = 0.1
    residual_dropout: float = 0.1
    token_dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.1
    label_smoothing: float = 0.0
    batch_size: int = 256
    max_steps: int = 2000
    eval_every: int = 50
    patience: int = 8           # eval windows w/o val-AUROC improvement → stop
    swad_warmup: int = 200      # no SWAD snapshot before this step
    seed: int = 0


def _build_head(cfg: HeadTrainConfig) -> AttentiveProbeHead:
    return AttentiveProbeHead(
        cfg.d_model, n_heads=cfg.n_heads, n_queries=cfg.n_queries,
        mlp_ratio=cfg.mlp_ratio, attn_dropout=cfg.attn_dropout,
        mlp_dropout=cfg.mlp_dropout, residual_dropout=cfg.residual_dropout,
        token_dropout=cfg.token_dropout, n_out=1,
    )


@torch.no_grad()
def _auroc(head: AttentiveProbeHead, x: Tensor, y: np.ndarray,
           mask: Tensor | None, device: torch.device) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    head.eval()
    s = head(x.to(device), None if mask is None else mask.to(device))
    return float(roc_auc_score(y, s.squeeze(-1).cpu().numpy()))


def _clone_state(head: AttentiveProbeHead) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in head.state_dict().items()}


def train_head(
    x_tr: Tensor, y_tr: Tensor,
    x_val: Tensor, y_val: Tensor,
    cfg: HeadTrainConfig,
    *,
    mask_tr: Tensor | None = None,
    mask_val: Tensor | None = None,
    device: torch.device | None = None,
) -> dict:
    """Train one head; return ``{best_state, swad_state, best_val, swad_val, steps}``.

    ``x_*`` are ``(N, T, d)`` token sets, ``y_*`` ``(N,)`` in {0,1}, masks ``(N,T)``
    bool (True=valid) or None (constant T). Label smoothing is applied to the binary
    targets (``y(1-α)+α/2``). Early-stops on ``x_val`` AUROC; SWAD-averages post-warmup
    snapshots up to the stop."""
    device = device or torch.device("cpu")
    y_val_np = y_val.cpu().numpy().astype(float)
    head = _build_head(cfg).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lossf = torch.nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    n = x_tr.shape[0]
    yb_tr = y_tr.float().view(n, 1)
    swad_sum: dict[str, Tensor] | None = None
    swad_count = 0
    best_val = -float("inf")
    best_state = _clone_state(head)
    since_improve = 0
    step = 0
    while step < cfg.max_steps:
        head.train()
        idx = torch.randint(0, n, (min(cfg.batch_size, n),), generator=g)
        xb = x_tr[idx].to(device)
        ys = yb_tr[idx].to(device)
        ys = ys * (1.0 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
        mb = None if mask_tr is None else mask_tr[idx].to(device)
        loss = lossf(head(xb, mb), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1

        if step % cfg.eval_every == 0:
            va = _auroc(head, x_val, y_val_np, mask_val, device)
            if step >= cfg.swad_warmup:
                state = _clone_state(head)
                if swad_sum is None:
                    swad_sum = {k: v.clone() for k, v in state.items()}
                else:
                    for k, v in state.items():
                        swad_sum[k] += v
                swad_count += 1
            if np.isnan(va):
                continue
            if va > best_val + 1e-4:
                best_val = va
                best_state = _clone_state(head)
                since_improve = 0
            else:
                since_improve += 1
                if since_improve >= cfg.patience:
                    break

    if swad_sum is None:                       # never reached warmup → fall back to best
        swad_state = best_state
    else:
        swad_state = {k: v / swad_count for k, v in swad_sum.items()}
    swad_head = _build_head(cfg).to(device)
    swad_head.load_state_dict(swad_state)
    swad_val = _auroc(swad_head, x_val, y_val_np, mask_val, device)
    return {
        "best_state": best_state, "swad_state": swad_state,
        "best_val": float(best_val), "swad_val": swad_val, "steps": step,
    }


def diwa_average(states: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """DiWA: parameter-wise mean of several head state-dicts (different seeds/HPs)."""
    if not states:
        raise ValueError("diwa_average needs at least one state dict")
    keys = states[0].keys()
    out: dict[str, Tensor] = {}
    for k in keys:
        acc = states[0][k].clone().float()
        for s in states[1:]:
            acc += s[k].float()
        out[k] = (acc / len(states)).to(states[0][k].dtype)
    return out
