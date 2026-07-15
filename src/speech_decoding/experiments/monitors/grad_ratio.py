"""MON-LOSS-GRAD-RATIO: the ‖g_nll‖/‖g_jepa‖ balance between r4's two objectives.

The r4 total is ``jepa_L1 + λ·NLL``. Whether the secondary Gaussian-NLL is a healthy
minority pull on the shared encoder or is dominating/vanishing is NOT readable from the
loss VALUES — the JEPA term is an L1 (≥0) but the NLL is sign-indefinite (it goes negative
as the predicted variance shrinks), so a loss-value ratio is meaningless. What actually
competes for the shared online tower (stem + encoder) is the GRADIENT each objective puts
on it. This measures exactly that, live, on real data.

SINGLE-PROCESS ONLY. Computing ‖g_jepa‖ and ‖g_nll‖ needs a separate ``autograd.grad``
pass per loss (retain_graph so Lightning's own backward still runs). Under DDP those extra
backward traversals re-enter the reducer over the shared graph — the exact static_graph ×
grad-accum surface that crashed r3. The caller MUST gate this to ``devices == 1`` (no
distributed reducer). It is a 1-GPU diagnostic lever, never on the multi-GPU launch.

Cost: two extra backward passes through the online tower on the monitor cadence only
(every ``grad_ratio_every_n_steps``). Negligible on a short 1-GPU diagnostic; not free, so
opt-in and cadence-gated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def _grad_norm(loss: Tensor, params: Sequence[Tensor]) -> float:
    """‖∂loss/∂params‖₂ over ``params``. ``retain_graph=True`` keeps the graph alive for
    the sibling loss's pass AND Lightning's subsequent ``loss.backward()``. ``allow_unused``:
    a param the loss doesn't touch contributes 0 (the NLL only reaches the shared tower, not
    the JEPA-only predictor, and vice-versa)."""
    grads = torch.autograd.grad(
        loss, list(params), retain_graph=True, allow_unused=True, create_graph=False
    )
    sq = torch.zeros((), device=loss.device, dtype=torch.float32)
    for g in grads:
        if g is not None:
            sq = sq + g.detach().to(torch.float32).pow(2).sum()
    return float(sq.sqrt().item())


def loss_grad_ratio(
    jepa_loss: Tensor,
    nll_loss: Tensor,
    params: Sequence[Tensor],
    *,
    lambda_nll: float,
) -> dict[str, float]:
    """The live loss-balance readout on the shared tower ``params``.

    Returns ``g_jepa`` / ``g_nll`` (the raw grad-L2s), ``grad_ratio`` = ‖g_nll‖/‖g_jepa‖,
    and ``grad_ratio_weighted`` = λ·ratio (the effective pull the total actually applies).
    ``grad_ratio`` is ``inf`` if ‖g_jepa‖ is 0 (degenerate; flagged rather than hidden)."""
    params = [p for p in params if p.requires_grad]
    gj = _grad_norm(jepa_loss, params)
    gn = _grad_norm(nll_loss, params)
    ratio = gn / gj if gj > 0.0 else math.inf
    return {
        "loss_g_jepa": gj,
        "loss_g_nll": gn,
        "grad_ratio": ratio,
        "grad_ratio_weighted": float(lambda_nll) * ratio,
    }


__all__ = ["loss_grad_ratio"]
