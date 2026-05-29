"""MON-HEAD-BALANCE-005: per-head attention-usage canary (B29 Item 9 demoted).

Lock memo: ``memory/project_v14_b29_joint_default_2026_05_27.md`` §Item 9.

Pre-B29: ``MON-HEAD-BALANCE-005`` was a hard kill criterion: any
attention head that drifted outside ``[0.3, 3.0]`` relative usage vs
the per-block mean was treated as a Phase-1 failure. B29 demotes this
to a *health canary* — same bounds, but the verdict is "investigate,
don't gate." The training loop logs but does not interrupt the run.

The "relative usage" metric is per-head *attention concentration* —
the mean (across batch and queries) of the max attention probability the
head assigned to any key (``attention.max(dim=-1).values.mean(...)``),
normalized by the layer mean concentration (so the per-layer expected
value is 1.0). Raw row sums are degenerate (each softmax row sums to 1,
so all heads tie at ``N_q``); max-per-query is the standard concentration
proxy. Heads near 0.3 are diffuse / starving; heads near 3.0 are
dominating.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import torch
from torch import Tensor


# B29 Item 9 demotion: same bounds as pre-B29, but the verdict no
# longer gates dispatch.
HEAD_BALANCE_BOUNDS: tuple[float, float] = (0.3, 3.0)


@dataclass(frozen=True)
class HeadBalanceVerdict:
    """Per-head relative usage + which heads fall outside the bounds.

    Fields
    ------
    relative_usage: ``(H,)`` per-head usage normalized to layer mean.
    starving_heads: ids of heads with usage < ``bounds[0]``.
    dominating_heads: ids of heads with usage > ``bounds[1]``.
    bounds: locked ``(0.3, 3.0)`` bounds (sister can override).
    kill: always ``False`` post-B29 (canary not gate). Kept on the
        struct for symmetric introspection with the other monitor
        verdicts.
    investigate: whether any head fell outside the bounds — i.e. the
        "investigate, don't gate" signal.
    """

    relative_usage: Tensor
    starving_heads: tuple[int, ...]
    dominating_heads: tuple[int, ...]
    bounds: tuple[float, float]
    kill: bool
    investigate: bool


def head_balance_monitor(
    attention_weights: Tensor,
    *,
    bounds: tuple[float, float] = HEAD_BALANCE_BOUNDS,
) -> HeadBalanceVerdict:
    """Compute per-head relative-usage canary.

    Parameters
    ----------
    attention_weights: ``(B, H, N_q, N_k)`` softmax-normalized attention
        probabilities at a single layer. Multi-layer monitoring should
        average verdicts (or run this per-layer) — keep this function
        single-layer for testability.
    bounds: ``(low, high)`` relative-usage bounds. Default
        ``(0.3, 3.0)`` per the locked spec.

    Returns
    -------
    HeadBalanceVerdict
    """
    if attention_weights.dim() != 4:
        raise ValueError(
            "attention_weights must be (B, H, N_q, N_k); "
            f"got shape {tuple(attention_weights.shape)}"
        )
    # Per-head "concentration" metric: average of the max attention
    # weight that head assigned per query. Softmax-normalized attention
    # rows sum to 1, so the raw row-sum is degenerate (all heads tie).
    # Max-weight average is a standard concentration proxy: high =
    # head is confident / picks a single key (active); low = head is
    # diffuse (under-used). The relative-usage canary then compares
    # this concentration to the layer mean — heads with relative
    # concentration well above 1 are dominating; below 1 are starving.
    # Shape: (B, H, N_q) → (B, H) → (H,).
    per_head_concentration = attention_weights.max(dim=-1).values.mean(dim=-1).mean(dim=0)
    layer_mean = per_head_concentration.mean().clamp_min(
        torch.finfo(per_head_concentration.dtype).eps
    )
    relative = per_head_concentration / layer_mean

    low, high = bounds
    starving = tuple(int(i) for i in torch.where(relative < low)[0].tolist())
    dominating = tuple(int(i) for i in torch.where(relative > high)[0].tolist())
    investigate = bool(starving) or bool(dominating)
    return HeadBalanceVerdict(
        relative_usage=relative,
        starving_heads=starving,
        dominating_heads=dominating,
        bounds=bounds,
        kill=False,                      # B29 demotion: never kill
        investigate=investigate,
    )


__all__: tp.Final[tuple[str, ...]] = (
    "HEAD_BALANCE_BOUNDS",
    "HeadBalanceVerdict",
    "head_balance_monitor",
)
