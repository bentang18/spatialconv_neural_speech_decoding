"""Tests for the §7/B01 no-weight-decay param-group split (task #40).

Contract under test (optim_param_groups):
 - which params are exempt: biases / LayerNorm γβ / any ``ndim <= 1`` param ONLY —
   every ≥2-D param (matmul weights AND embedding / query tables) is decayed, matching
   V-JEPA 2.1 ``init_opt``;
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
    """A module spanning every classification branch under the V-JEPA rule
    (``ndim <= 1`` exempt, everything ≥2-D decayed): a matmul weight (decay) + its
    bias (exempt), a LayerNorm (γβ, both exempt), a 2-D additive table
    (``freq_embed`` — DECAYED), an ``nn.Embedding`` (``id_embed.weight`` — DECAYED),
    a learned 3-D ``query`` token (DECAYED), and a second Linear whose bias is exempt
    while its weight decays."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)               # lin.weight (2-D decay), lin.bias (1-D exempt)
        self.ln = nn.LayerNorm(8)                # ln.weight, ln.bias (both 1-D exempt)
        self.freq_embed = nn.Parameter(torch.randn(4, 8))   # 2-D table → DECAY
        self.id_embed = nn.Embedding(4, 8)       # id_embed.weight (2-D) → DECAY
        self.query = nn.Parameter(torch.randn(1, 1, 8))     # 3-D query token → DECAY
        self.grid_embedder = nn.Linear(8, 8)     # 2-D weight → DECAY, bias → exempt


def _named(net: nn.Module) -> dict[str, torch.Tensor]:
    return dict(net.named_parameters())


# ---------------------------------------------------------------------------
# is_no_decay
# ---------------------------------------------------------------------------
def test_is_no_decay_ndim_rule() -> None:
    assert is_no_decay("lin.bias", torch.zeros(8)) is True          # 1-D
    assert is_no_decay("ln.weight", torch.zeros(8)) is True         # 1-D γ
    assert is_no_decay("mask_token", torch.zeros(16)) is True       # 1-D misc token
    assert is_no_decay("lin.weight", torch.zeros(8, 8)) is False    # 2-D matmul


def test_is_no_decay_decays_all_2d_tables_matching_vjepa() -> None:
    """V-JEPA rule: every ≥2-D param is DECAYED — embeddings, identity tables, and
    query tokens included. Only ``ndim <= 1`` is exempt. This is the divergence-fix:
    the old name-allowlist that exempted these is gone."""
    w2d = torch.zeros(4, 8)
    assert is_no_decay("student.encoder.freq_embed", w2d) is False
    assert is_no_decay("predictor.id_embed.weight", w2d) is False
    assert is_no_decay("encoder.learnable_parcel_embed", w2d) is False
    assert is_no_decay("encoder.subtype_embed.weight", w2d) is False
    assert is_no_decay("encoder.ref_embed.weight", w2d) is False
    # A 3-D learned query token (PMA / attentive pooler) is likewise decayed.
    assert is_no_decay("encoder.pma.query", torch.zeros(1, 1, 8)) is False
    assert is_no_decay("encoder.proj.weight", w2d) is False
    # …while any ndim<=1 param stays exempt regardless of name.
    assert is_no_decay("encoder.grid_embedder.bias", torch.zeros(8)) is True


# ---------------------------------------------------------------------------
# no_decay_param_ids
# ---------------------------------------------------------------------------
def test_no_decay_param_ids_membership() -> None:
    net = _TinyNet()
    p = _named(net)
    ids = no_decay_param_ids(net)
    # Exempt = the ndim<=1 params ONLY: biases + LN γβ.
    for name in ("lin.bias", "ln.weight", "ln.bias", "grid_embedder.bias"):
        assert id(p[name]) in ids, name
    # Decayed = every ≥2-D param: matmul weights AND the embedding / query tables.
    for name in ("lin.weight", "grid_embedder.weight", "freq_embed",
                 "id_embed.weight", "query"):
        assert id(p[name]) not in ids, name


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
    # Decay = every ≥2-D param: the two matmul weights + freq_embed + id_embed.weight
    # + the 3-D query token = 5. Exempt = the four ndim<=1 params (two biases,
    # LN γβ) = 4.
    assert {id(x) for x in decay["params"]} == {
        id(net.lin.weight), id(net.grid_embedder.weight),
        id(net.freq_embed), id(net.id_embed.weight), id(net.query)}
    assert len(no_decay["params"]) == 4
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
    # grid_embedder.bias (1-D) is the exempt member of group 1; lin.weight decays.
    groups = [
        {"params": [net.grid_embedder.bias, net.lin.weight], "lr": 1e-4},
        {"params": [net.ln.weight, net.ln.bias, net.lin.bias], "lr": 1e-3},
    ]
    out = split_no_decay(groups, ids)
    # Group 1 splits (bias exempt, lin.weight decays) → both keep lr 1e-4.
    g1 = [g for g in out if g["params"] and id(g["params"][0]) in
          {id(net.grid_embedder.bias), id(net.lin.weight)}]
    assert all(g["lr"] == 1e-4 for g in g1)
    g1_decay = next(g for g in g1 if "weight_decay" not in g)
    g1_nodecay = next(g for g in g1 if g.get("weight_decay") == 0.0)
    assert [id(x) for x in g1_decay["params"]] == [id(net.lin.weight)]
    assert [id(x) for x in g1_nodecay["params"]] == [id(net.grid_embedder.bias)]
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
    # Every ≥2-D param decays: both matmul weights + the 2-D/3-D tables.
    assert {id(x) for x in decay["params"]} == {
        id(net.lin.weight), id(net.grid_embedder.weight),
        id(net.freq_embed), id(net.id_embed.weight), id(net.query)}


# ---------------------------------------------------------------------------
# Integration: under the V-JEPA rule the real PMA query token DECAYS (only the
# LayerNorm γβ are exempt) — the M0 P3 wd-sweep case, re-baselined 2026-07-06.
# ---------------------------------------------------------------------------
def test_real_pma_query_token_is_decayed_at_positive_wd() -> None:
    """The learned PMA query (``V14ParcelCollapsePMA.query``, 3-D) is DECAYED like any
    ≥2-D param, exactly as V-JEPA decays its (3-D) mask tokens. Only the two LayerNorm
    γβ pairs ride the wd=0 group; the query + all matmul projections decay."""
    from speech_decoding.models.v14_encoder import V14ParcelCollapsePMA

    pma = V14ParcelCollapsePMA(d_model=16, n_heads=4)
    p = dict(pma.named_parameters())
    ids = no_decay_param_ids(pma)
    # The 3-D query token now DECAYS (ndim>1); both LayerNorm γβ stay exempt.
    assert id(p["query"]) not in ids
    assert id(p["ln_q.weight"]) in ids and id(p["ln_kv.bias"]) in ids
    # The three matmul projections decay.
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
    # query rides the DECAY group now, alongside the three projections.
    assert id(p["query"]) in {id(x) for x in decay["params"]}
    assert {id(p["q_proj.weight"]), id(p["kv_proj.weight"]),
            id(p["out_proj.weight"]), id(p["query"])} <= {id(x) for x in decay["params"]}
    # Only ndim<=1 params are exempt.
    assert all(x.ndim <= 1 for x in no_decay["params"])
