"""No-weight-decay optimizer param-group SPLIT (§7/B01 locked CONVENTION, task #40).

This is a convention for *which kinds of parameters* receive weight decay when it
is applied — NOT a lock on the weight_decay VALUE. The §7/B01 recipe UNLOCKED the
wd value (it is an M0 sweep center, not a fixed universal; the dispatch guard that
refused ``weight_decay > 0`` was removed in #40). What is locked is only the
standard exemption: the following ride in a ``weight_decay: 0.0`` group while only
matmul/conv weights get the swept wd —

  * biases + LayerNorm/RMSNorm gains (any param with ``ndim <= 1``) — the nanoGPT
    convention; and
  * learned embedding / identity / query tokens (≥2-D tables that are not matmul
    weights) — the timm / ViT / V-JEPA-2 ``no_weight_decay`` convention. (nanoGPT
    itself DECAYS 2-D embeddings — it exempts only ``ndim <= 1``; we follow timm
    for the table/token part, not nanoGPT.)

The exemption itself is falsifiable: ``--no-wd-exclude-norms`` decays every param
uniformly (the uniform-decay falsifier), so the M0 sweep can measure whether the
exemption helps.

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

# Embedding / identity / query tables the §7 convention exempts from weight decay
# even though they are >1-D (the ``ndim <= 1`` rule below misses them). Matched as
# whole dot-separated COMPONENTS of the parameter name (not raw substrings), so
# ``student.encoder.freq_embed`` and ``predictor.id_embed.weight`` qualify while a
# hypothetical ``grid_embedder.weight`` does NOT falsely match ``id_embed`` — the
# component split keeps the exemption from creeping onto matmul weights by name
# accident.
_NO_DECAY_NAME_COMPONENTS: frozenset[str] = frozenset({
    "freq_embed",               # encoder per-freq-patch additive table
    "learnable_parcel_embed",   # encoder parcel-id latent table
    "learnable_subslot_embed",  # encoder sub-slot table (M>1)
    "subtype_embed",            # encoder sensor-subtype nn.Embedding (.weight)
    "ref_embed",                # encoder reference-operator nn.Embedding (.weight)
    "id_embed",                 # JepaPredictor learned identity/tag table (6/04)
    "query",                    # PMA / attentive-pooler learned query token (3-D)
})


def is_no_decay(name: str, param: torch.Tensor) -> bool:
    """True if ``param`` is weight-decay-exempt: a bias / LayerNorm γβ / any
    other ``ndim <= 1`` param (mask tokens included), or a named embedding /
    identity / query table whose dotted name contains a component in
    :data:`_NO_DECAY_NAME_COMPONENTS`."""
    if param.ndim <= 1:
        return True
    return any(c in _NO_DECAY_NAME_COMPONENTS for c in name.split("."))


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
