"""Tests for the §7/B01 no-weight-decay param-group split (task #40).

Contract under test (optim_param_groups):
 - which params are exempt (biases / LayerNorm γβ / ndim<=1 / named embeds);
 - the split preserves per-group ``lr`` and forces ``weight_decay: 0.0`` on the
   exempt sub-group only;
 - ``maybe_split_no_decay`` is a no-op at wd=0 (bit-identical to pre-#40) and
   under ``exclude=False`` (the ``--no-wd-exclude-norms`` falsifier);
 - it splits when a non-zero weight_decay is configured AND exclude is set.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neuraltrain.optimizers import LightningOptimizer

from speech_decoding.experiments.optim_param_groups import (
    is_no_decay,
    maybe_split_no_decay,
    no_decay_param_ids,
    optimizer_weight_decay,
    split_no_decay,
)


class _TinyNet(nn.Module):
    """A module spanning every classification branch: a matmul weight (decay) +
    its bias (no-decay), a LayerNorm (γβ, both no-decay), a >1-D table caught by
    name only (``freq_embed``), an ``nn.Embedding`` (``id_embed.weight``), a
    learned ``query`` token (3-D, name-only — the PMA case), and a ``grid_embedder``
    matmul decoy whose name *substring*-contains ``id_embed`` (gr**id_embed**der)
    but must still DECAY under component-exact matching."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)               # lin.weight (2-D), lin.bias (1-D)
        self.ln = nn.LayerNorm(8)                # ln.weight, ln.bias (both 1-D)
        self.freq_embed = nn.Parameter(torch.randn(4, 8))   # 2-D, name-only
        self.id_embed = nn.Embedding(4, 8)       # id_embed.weight (2-D, name-only)
        self.query = nn.Parameter(torch.randn(1, 1, 8))     # 3-D PMA query token
        self.grid_embedder = nn.Linear(8, 8)     # decoy: name superstring of id_embed


def _named(net: nn.Module) -> dict[str, torch.Tensor]:
    return dict(net.named_parameters())


# ---------------------------------------------------------------------------
# is_no_decay
# ---------------------------------------------------------------------------
def test_is_no_decay_ndim_rule() -> None:
    assert is_no_decay("lin.bias", torch.zeros(8)) is True          # 1-D
    assert is_no_decay("ln.weight", torch.zeros(8)) is True         # 1-D γ
    assert is_no_decay("mask_token", torch.zeros(16)) is True       # 1-D misc
    assert is_no_decay("lin.weight", torch.zeros(8, 8)) is False    # 2-D matmul


def test_is_no_decay_name_rule_for_2d_tables() -> None:
    w2d = torch.zeros(4, 8)
    # >1-D, so only the name rule can exempt these.
    assert is_no_decay("student.encoder.freq_embed", w2d) is True
    assert is_no_decay("predictor.id_embed.weight", w2d) is True
    assert is_no_decay("encoder.learnable_parcel_embed", w2d) is True
    assert is_no_decay("encoder.subtype_embed.weight", w2d) is True
    assert is_no_decay("encoder.ref_embed.weight", w2d) is True
    # The PMA / attentive-pooler learned query token (3-D, name-only).
    assert is_no_decay("encoder.pma.query", torch.zeros(1, 1, 8)) is True
    # A plain 2-D weight whose name matches nothing stays a decay param.
    assert is_no_decay("encoder.proj.weight", w2d) is False


def test_is_no_decay_matches_whole_components_not_substrings() -> None:
    """Component-exact matching: a matmul weight whose name *substring*-contains an
    exempt token must still DECAY. ``grid_embedder.weight`` contains ``id_embed``
    (gr**id_embed**der) and ``q_proj.weight`` contains ``q`` but neither is a
    whole dotted component, so both decay."""
    w2d = torch.zeros(8, 8)
    assert is_no_decay("encoder.grid_embedder.weight", w2d) is False
    assert is_no_decay("pma.q_proj.weight", w2d) is False
    # Sanity: the exact component still exempts.
    assert is_no_decay("encoder.grid_embedder.bias", torch.zeros(8)) is True  # ndim<=1


# ---------------------------------------------------------------------------
# no_decay_param_ids
# ---------------------------------------------------------------------------
def test_no_decay_param_ids_membership() -> None:
    net = _TinyNet()
    p = _named(net)
    ids = no_decay_param_ids(net)
    # Exempt: bias, LN γβ, freq_embed (2-D name), id_embed.weight (2-D name),
    # the 3-D query token, and grid_embedder.bias (1-D).
    for name in ("lin.bias", "ln.weight", "ln.bias", "freq_embed",
                 "id_embed.weight", "query", "grid_embedder.bias"):
        assert id(p[name]) in ids, name
    # Not exempt: the matmul weights (the decoy grid_embedder.weight included).
    assert id(p["lin.weight"]) not in ids
    assert id(p["grid_embedder.weight"]) not in ids


# ---------------------------------------------------------------------------
# split_no_decay — flat param list
# ---------------------------------------------------------------------------
def test_split_flat_param_list() -> None:
    net = _TinyNet()
    ids = no_decay_param_ids(net)
    groups = split_no_decay(list(net.parameters()), ids)
    assert len(groups) == 2
    decay = next(g for g in groups if "weight_decay" not in g)
    no_decay = next(g for g in groups if g.get("weight_decay") == 0.0)
    # Only the two matmul weights decay (lin.weight + the grid_embedder decoy);
    # everything else is exempt (lin.bias, ln.weight, ln.bias, freq_embed,
    # id_embed.weight, query, grid_embedder.bias = 7).
    assert {id(x) for x in decay["params"]} == {
        id(net.lin.weight), id(net.grid_embedder.weight)}
    assert len(no_decay["params"]) == 7
    # Every param appears exactly once across the two groups (no drop / dupe).
    all_ids = [id(x) for g in groups for x in g["params"]]
    assert sorted(all_ids) == sorted(id(x) for x in net.parameters())


# ---------------------------------------------------------------------------
# split_no_decay — dict groups (discriminative LR preserved)
# ---------------------------------------------------------------------------
def test_split_dict_groups_preserves_lr() -> None:
    net = _TinyNet()
    ids = no_decay_param_ids(net)
    # Two discriminative groups, mirroring P2/P3b: one at base/10, one at base.
    groups = [
        {"params": [net.freq_embed, net.lin.weight], "lr": 1e-4},
        {"params": [net.ln.weight, net.ln.bias, net.lin.bias], "lr": 1e-3},
    ]
    out = split_no_decay(groups, ids)
    # Group 1 splits (freq_embed exempt, lin.weight decays) → both keep lr 1e-4.
    g1 = [g for g in out if g["params"] and id(g["params"][0]) in
          {id(net.freq_embed), id(net.lin.weight)}]
    assert all(g["lr"] == 1e-4 for g in g1)
    g1_decay = next(g for g in g1 if "weight_decay" not in g)
    g1_nodecay = next(g for g in g1 if g.get("weight_decay") == 0.0)
    assert [id(x) for x in g1_decay["params"]] == [id(net.lin.weight)]
    assert [id(x) for x in g1_nodecay["params"]] == [id(net.freq_embed)]
    assert g1_nodecay["lr"] == 1e-4   # discriminative LR survives the split
    # Group 2 is entirely no-decay → exactly one output group at lr 1e-3, wd 0.0.
    g2 = [g for g in out if g.get("lr") == 1e-3]
    assert len(g2) == 1
    assert g2[0]["weight_decay"] == 0.0
    assert len(g2[0]["params"]) == 3


def test_split_all_decay_group_yields_single_group() -> None:
    net = _TinyNet()
    ids = no_decay_param_ids(net)
    groups = [{"params": [net.lin.weight], "lr": 5e-4}]  # all decay
    out = split_no_decay(groups, ids)
    assert len(out) == 1
    assert "weight_decay" not in out[0]
    assert out[0]["lr"] == 5e-4


def test_split_explicit_parent_wd_overridden_to_zero_on_exempt() -> None:
    net = _TinyNet()
    ids = no_decay_param_ids(net)
    # Parent group already carries a non-zero weight_decay; the exempt sub-group
    # must still come out at 0.0 (``**extra`` first, explicit wd wins).
    groups = [{"params": [net.lin.weight, net.lin.bias], "weight_decay": 0.5}]
    out = split_no_decay(groups, ids)
    decay = next(g for g in out if id(g["params"][0]) == id(net.lin.weight))
    no_decay = next(g for g in out if id(g["params"][0]) == id(net.lin.bias))
    assert decay["weight_decay"] == 0.5     # matmul keeps the swept wd
    assert no_decay["weight_decay"] == 0.0  # bias forced to 0.0


# ---------------------------------------------------------------------------
# optimizer_weight_decay
# ---------------------------------------------------------------------------
def test_optimizer_weight_decay_reads_kwargs() -> None:
    cfg = LightningOptimizer(
        optimizer={"name": "AdamW", "lr": 1e-3, "kwargs": {"weight_decay": 0.05}}
    )
    assert optimizer_weight_decay(cfg) == 0.05


def test_optimizer_weight_decay_zero_when_unset() -> None:
    cfg = LightningOptimizer(optimizer={"name": "Adam", "lr": 1e-3})
    assert optimizer_weight_decay(cfg) == 0.0


# ---------------------------------------------------------------------------
# maybe_split_no_decay — the gated entry point
# ---------------------------------------------------------------------------
def test_maybe_split_noop_when_wd_zero() -> None:
    net = _TinyNet()
    cfg = LightningOptimizer(optimizer={"name": "Adam", "lr": 1e-3})  # wd=0
    groups = list(net.parameters())
    out = maybe_split_no_decay(groups, modules=(net,), optim_config=cfg, exclude=True)
    assert out is groups  # identity → bit-identical to the pre-#40 path


def test_maybe_split_noop_when_exclude_false() -> None:
    net = _TinyNet()
    cfg = LightningOptimizer(
        optimizer={"name": "AdamW", "lr": 1e-3, "kwargs": {"weight_decay": 0.1}}
    )
    groups = list(net.parameters())
    out = maybe_split_no_decay(groups, modules=(net,), optim_config=cfg, exclude=False)
    assert out is groups  # --no-wd-exclude-norms → uniform decay, no split


def test_maybe_split_applies_when_wd_positive_and_exclude() -> None:
    net = _TinyNet()
    cfg = LightningOptimizer(
        optimizer={"name": "AdamW", "lr": 1e-3, "kwargs": {"weight_decay": 0.1}}
    )
    out = maybe_split_no_decay(
        list(net.parameters()), modules=(net,), optim_config=cfg, exclude=True
    )
    assert isinstance(out, list) and all(isinstance(g, dict) for g in out)
    no_decay = next(g for g in out if g.get("weight_decay") == 0.0)
    assert id(net.lin.bias) in {id(x) for x in no_decay["params"]}
    decay = next(g for g in out if "weight_decay" not in g)
    assert {id(x) for x in decay["params"]} == {
        id(net.lin.weight), id(net.grid_embedder.weight)}


# ---------------------------------------------------------------------------
# Integration: the real PMA query token must be exempt (the M0 P3 wd-sweep case)
# ---------------------------------------------------------------------------
def test_real_pma_query_token_is_exempt_at_positive_wd() -> None:
    """The learned PMA query (``V14ParcelCollapsePMA.query``, 3-D) must ride in the
    wd=0 group while its q/kv/out matmul projections decay — the exact case the M0
    P3 wd sweep exercises. Also guards that ``q_proj`` is NOT falsely matched by
    the new ``query`` component."""
    from speech_decoding.models.v14_encoder import V14ParcelCollapsePMA

    pma = V14ParcelCollapsePMA(d_model=16, n_heads=4)
    p = dict(pma.named_parameters())
    ids = no_decay_param_ids(pma)
    # The query token + both LayerNorm γβ are exempt.
    assert id(p["query"]) in ids
    assert id(p["ln_q.weight"]) in ids and id(p["ln_kv.bias"]) in ids
    # The three matmul projections decay (q_proj component != query).
    for name in ("q_proj.weight", "kv_proj.weight", "out_proj.weight"):
        assert id(p[name]) not in ids, name

    cfg = LightningOptimizer(
        optimizer={"name": "AdamW", "lr": 1e-3, "kwargs": {"weight_decay": 0.05}}
    )
    out = maybe_split_no_decay(
        list(pma.parameters()), modules=(pma,), optim_config=cfg, exclude=True
    )
    no_decay = next(g for g in out if g.get("weight_decay") == 0.0)
    decay = next(g for g in out if "weight_decay" not in g)
    assert id(p["query"]) in {id(x) for x in no_decay["params"]}
    assert {id(x) for x in decay["params"]} == {
        id(p["q_proj.weight"]), id(p["kv_proj.weight"]), id(p["out_proj.weight"])}
