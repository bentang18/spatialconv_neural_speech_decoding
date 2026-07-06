"""No-weight-decay optimizer param-group SPLIT (§7/B01 CONVENTION, task #40).

This is a convention for *which kinds of parameters* receive weight decay when it
is applied — NOT a lock on the weight_decay VALUE. The §7/B01 recipe UNLOCKED the
wd value (it is an M0 sweep center, not a fixed universal; the dispatch guard that
refused ``weight_decay > 0`` was removed in #40). What is locked is only the
standard exemption: biases + LayerNorm/RMSNorm gains (any param with ``ndim <= 1``)
ride in a ``weight_decay: 0.0`` group; every ≥2-D param — matmul/conv weights AND
learned embedding / identity / query tables — gets the swept wd.

This exactly matches V-JEPA 2/2.1 ``init_opt`` (``app/vjepa_2_1/utils.py``:305-331),
which exempts only ``("bias" in n) or (len(p.shape) == 1)`` and DECAYS all 2-D+
params including every embedding and (3-D) mask token — same rule nanoGPT uses.
(An earlier revision here additionally exempted 2-D embedding tables by name, citing
timm/ViT ``no_weight_decay``; the actual V-JEPA source decays them, so that
name-allowlist was removed to eliminate the silent divergence — 2026-07-06.)

The exemption itself is falsifiable: ``--no-wd-exclude-norms`` decays every param
uniformly (the uniform-decay falsifier — V-JEPA's ``zero_init_bias_wd=False``), so
the M0 sweep can measure whether the bias/norm exemption helps.

Mechanism — the top-level ``LightningOptimizer`` ``weight_decay`` is the DEFAULT
applied to every param group; per-group ``weight_decay: 0.0`` dicts override it
for the exempt subset. ``BaseTorchOptimizer.build(params)`` forwards param-group
dicts straight to ``torch_optimizer(params, lr=self.lr, **self.kwargs)``, so a
group dict carrying ``weight_decay: 0.0`` shadows ``kwargs["weight_decay"]`` for
that group while every other group inherits the swept value (verified against
``neuraltrain/optimizers/base.py``).

The split is gated (see :func:`maybe_split_no_decay`): it is a no-op when the
configured weight_decay is 0 (so the pre-#40 wd=0 path is bit-identical) and
when the caller passes ``exclude=False`` (the ``--no-wd-exclude-norms`` override
that decays every param uniformly, the falsifier for whether the exclusion
matters).
"""

from __future__ import annotations

import typing as tp

import torch
import torch.nn as nn

def is_no_decay(name: str, param: torch.Tensor) -> bool:  # noqa: ARG001
    """True if ``param`` is weight-decay-exempt: a bias / LayerNorm γβ / any other
    ``ndim <= 1`` param (1-D mask/identity tokens included). Every ≥2-D param —
    matmul/conv weights AND embedding / identity / query tables — is decayed, exactly
    as V-JEPA 2.1 ``init_opt`` (``utils.py``:305-331). ``name`` is unused (retained so
    :func:`no_decay_param_ids` can pass it straight from ``named_parameters``)."""
    return param.ndim <= 1


def no_decay_param_ids(*modules: nn.Module) -> set[int]:
    """``id()`` set of every weight-decay-exempt parameter reachable from
    ``modules`` (traversed via ``named_parameters`` so dotted names match the
    substrings).

    Pass the ``LightningModule``(s) whose params will be optimized; a superset
    is harmless because :func:`split_no_decay` only acts on params that actually
    appear in the optimizer groups (e.g. the EMA teacher's params are classified
    but never optimized, so their ids are simply never looked up).
    """
    ids: set[int] = set()
    for module in modules:
        for name, param in module.named_parameters():
            if is_no_decay(name, param):
                ids.add(id(param))
    return ids


def split_no_decay(
    groups: tp.Sequence[tp.Any], no_decay_ids: set[int],
) -> list[dict[str, tp.Any]]:
    """Split optimizer param groups into decay / no-decay sub-groups.

    ``groups`` is either a flat sequence of ``Parameter`` (one implicit group)
    or a sequence of param-group dicts (each a ``{"params": [...], "lr": ...}``).
    Each input group is partitioned by membership in ``no_decay_ids``; the
    no-decay sub-group carries ``weight_decay: 0.0`` and inherits every other
    key of its parent group (notably the discriminative ``lr``). Empty
    sub-groups are dropped, so a group whose params are all decay (or all
    no-decay) yields exactly one output group.
    """
    if groups and isinstance(groups[0], dict):
        group_dicts: list[dict[str, tp.Any]] = [dict(g) for g in groups]
    else:
        group_dicts = [{"params": list(groups)}]
    out: list[dict[str, tp.Any]] = []
    for g in group_dicts:
        params = list(g["params"])
        extra = {k: v for k, v in g.items() if k != "params"}
        decay = [p for p in params if id(p) not in no_decay_ids]
        no_decay = [p for p in params if id(p) in no_decay_ids]
        if decay:
            out.append({"params": decay, **extra})
        if no_decay:
            # ``**extra`` first so an explicit weight_decay 0.0 always wins.
            out.append({**extra, "params": no_decay, "weight_decay": 0.0})
    return out


def optimizer_weight_decay(optim_config: tp.Any) -> float:
    """Top-level weight_decay configured on a ``LightningOptimizer`` (0.0 if
    unset). The split is a no-op when this is 0, so callers gate on ``> 0``."""
    inner = getattr(optim_config, "optimizer", None)
    kwargs = getattr(inner, "kwargs", None) or {}
    return float(kwargs.get("weight_decay", 0.0) or 0.0)


def maybe_split_no_decay(
    groups: tp.Sequence[tp.Any],
    *,
    modules: tp.Sequence[nn.Module],
    optim_config: tp.Any,
    exclude: bool,
) -> tp.Any:
    """Apply :func:`split_no_decay` iff a non-zero weight_decay is configured
    AND ``exclude`` is set (the default ``--wd-exclude-norms`` path).

    Returns ``groups`` unchanged otherwise: the wd=0 path stays bit-identical to
    pre-#40, and ``--no-wd-exclude-norms`` (``exclude=False``) decays every param
    uniformly so the sweep can measure whether the exclusion helps.
    """
    if not exclude:
        return groups
    if optimizer_weight_decay(optim_config) <= 0.0:
        return groups
    return split_no_decay(groups, no_decay_param_ids(*modules))


__all__ = [
    "is_no_decay",
    "no_decay_param_ids",
    "split_no_decay",
    "optimizer_weight_decay",
    "maybe_split_no_decay",
]
