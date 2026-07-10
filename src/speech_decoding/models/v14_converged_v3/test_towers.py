"""v14_converged_v3 Phase 4c — encoder / predictor tower assembly (TDD).

Memo project-v14-converged-v3-sensor-architecture (ENCODER/PREDICTOR OFFLOAD):

  Encoder  12 WIDE blocks, d_model 256, 4 heads (head_dim 64). Layout
           ``L1×6 · (L2 L1)×3`` = 9 L1 : 3 L2 (6-block local front-load, then a
           1:1 local/global interleave — Ben 2026-07-10, KISS; the local,
           overfit-safe capacity the encoder RETAINS). SINGLE tap = the final
           block output.
  Predictor 12 NARROW blocks, d_model 128 (0.5×), 4 heads (head_dim 32). Layout
           ``[L2L1L1]×4`` = 8 L1 : 4 L2, L2-FIRST (the cross-sensor capacity the
           predictor carries, discarded at inference — MAE/V-JEPA asymmetry).

The tower is a plain pre-norm block stack that dispatches each block to its
geometry (L1 ← shaft gather; L2 ← parcel identity). These tests pin the exact
layout (count + order + width) and that a forward preserves (B, N, T, d).
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.attention import L1Block, L2Block
from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar
from speech_decoding.models.v14_converged_v3.towers import (
    ENC_LAYOUT,
    PRED_LAYOUT,
    build_encoder,
    build_predictor,
)

T = 6


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        parcel_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    return sc, build_l1_geometry(sc)


def _kinds(tower):
    return [
        "L1" if isinstance(b, L1Block) else "L2" if isinstance(b, L2Block) else "?"
        for b in tower.blocks
    ]


def test_encoder_layout_is_l1x6_then_1to1_interleave() -> None:
    # 6-block local front-load, then a 1:1 local/global interleave (Ben 2026-07-10,
    # KISS): L2-first in the mixed region, replacing the old 2:1 L2L1L1 tail.
    assert ENC_LAYOUT == (
        "L1", "L1", "L1", "L1", "L1", "L1",
        "L2", "L1", "L2", "L1", "L2", "L1",
    )
    enc = build_encoder(n_parcels=8)
    kinds = _kinds(enc)
    assert len(kinds) == 12
    assert kinds.count("L1") == 9 and kinds.count("L2") == 3
    assert kinds[:6] == ["L1"] * 6  # front-load intact
    assert kinds == list(ENC_LAYOUT)


def test_predictor_layout_is_four_l2l1l1_l2_first() -> None:
    assert PRED_LAYOUT == ("L2", "L1", "L1") * 4
    pred = build_predictor(n_parcels=8)
    kinds = _kinds(pred)
    assert len(kinds) == 12
    assert kinds.count("L1") == 8 and kinds.count("L2") == 4
    assert kinds[0] == "L2"  # L2-first


def test_encoder_is_wide_256_four_heads() -> None:
    enc = build_encoder(n_parcels=8)
    l1 = next(b for b in enc.blocks if isinstance(b, L1Block))
    assert l1.n_heads == 4 and l1.head_dim == 64  # d_model 256


def test_predictor_is_narrow_128_four_heads() -> None:
    pred = build_predictor(n_parcels=8)
    l1 = next(b for b in pred.blocks if isinstance(b, L1Block))
    assert l1.n_heads == 4 and l1.head_dim == 32  # d_model 128


def test_encoder_forward_preserves_shape() -> None:
    sc, geom = _session()
    enc = build_encoder(n_parcels=8).eval()
    x = torch.randn(1, 5, T, 256)
    out = enc(x, geom, sc.parcel_id)
    assert out.shape == (1, 5, T, 256)


def test_predictor_forward_preserves_shape() -> None:
    sc, geom = _session()
    pred = build_predictor(n_parcels=8).eval()
    x = torch.randn(1, 5, T, 128)
    out = pred(x, geom, sc.parcel_id)
    assert out.shape == (1, 5, T, 128)


def test_parcel_embed_added_once_at_tower_input() -> None:
    # V-JEPA 2.1 modality-embed style: one shared parcel table per tower, added to
    # the token stream at the tower input (not per-L2-block). Relabelling a contact's
    # parcel changes the tower output; and the tower owns exactly one parcel table.
    sc, geom = _session()
    enc = build_encoder(n_parcels=8).eval()
    # exactly one parcel embedding table on the tower (not one per L2 block)
    n_embeds = sum(1 for _ in enc.parcel_embed.parameters())
    assert n_embeds == 1
    x = torch.randn(1, 5, T, 256)
    pid_b = sc.parcel_id.clone()
    pid_b[0] = 5  # contact 0 → different parcel
    out_a = enc(x, geom, sc.parcel_id)
    out_b = enc(x, geom, pid_b)
    # Wiring check (init-magnitude-agnostic): relabelling a parcel must change the
    # output at all. atol=0 catches the near-zero-init (1e-6) parcel signal, which a
    # fixed 1e-4 tolerance would mask.
    assert not torch.allclose(out_a, out_b, atol=0.0, rtol=0.0)


def test_tower_has_terminal_layernorm() -> None:
    # V-JEPA 2 applies an affine terminal LayerNorm to BOTH towers before the
    # downstream projection: encoder `self.norm` (vision_transformer.py:210) and
    # predictor `predictor_norm` (predictor.py:241). v2 carried the same
    # (pred_norm / ln_out); the v3 rewrite dropped it — restore it here so the
    # predictor output is unit-scale against the LN'd target (no scale runaway),
    # and the encoder tap / teacher target match upstream `affinefree(affine(h))`.
    import torch.nn as nn

    from speech_decoding.models.v14_converged_v3.towers import (
        build_encoder,
        build_predictor,
    )

    for build in (build_encoder, build_predictor):
        tower = build(n_parcels=8).eval()
        assert isinstance(tower.norm_out, nn.LayerNorm)
    sc, geom = _session()
    enc = build_encoder(n_parcels=8).eval()
    x = torch.randn(1, 5, T, 256) * 7.0  # deliberately off-scale input
    out = enc(x, geom, sc.parcel_id)
    # fresh affine LN (gamma=1, beta=0) ⇒ each token unit-scaled over the feature dim
    assert out.mean(dim=-1).abs().max() < 1e-4
    assert (out.var(dim=-1, unbiased=False) - 1.0).abs().max() < 1e-2


def test_tower_init_matches_vjepa2() -> None:
    # V-JEPA 2 init: Linear weights trunc_normal(0.02) + zero bias (qkv_bias ON),
    # then depth-scaled residual rescale of attn out-proj + mlp.fc2 by sqrt(2·id).
    import math

    enc = build_encoder(n_parcels=8)
    # qkv/out biases restored (upstream qkv_bias=True, proj bias)
    b0 = enc.blocks[0]
    assert b0.qkv.bias is not None and b0.out.bias is not None
    assert b0.qkv.bias.abs().max().item() == 0.0  # zero-init
    # trunc_normal(0.02) on an un-rescaled Linear (mlp.fc1)
    assert abs(b0.mlp.fc1.weight.std().item() - 0.02) < 0.004
    assert b0.mlp.fc1.bias.abs().max().item() == 0.0
    # depth rescale: out-proj std shrinks as 1/sqrt(2·(id+1)) ⇒ block0/block5 ~ sqrt(6)
    s0 = enc.blocks[0].out.weight.std().item()
    s5 = enc.blocks[5].out.weight.std().item()
    assert s0 > s5
    assert abs(s0 / s5 - math.sqrt(6.0)) < 0.4


def test_tower_dispatches_l1_to_geom_and_l2_to_parcel() -> None:
    # A whole-session forward must be block-diagonal-then-mixed: after the first
    # 6 L1 blocks (encoder), shaft A still cannot have seen shaft B; only once an
    # L2 block runs does cross-shaft information flow. Verify the encoder as a
    # whole DOES mix across shafts (an L2 fired) but a 6-L1 prefix does NOT.
    sc, geom = _session()
    enc = build_encoder(n_parcels=8).eval()
    x = torch.randn(1, 5, T, 256)

    # full encoder mixes across shafts
    out = enc(x, geom, sc.parcel_id)
    x2 = x.clone()
    x2[0, 3] += torch.randn(T, 256) * 3.0  # perturb shaft B
    out2 = enc(x2, geom, sc.parcel_id)
    assert not torch.allclose(out[0, 0], out2[0, 0], atol=1e-4)  # shaft A moved

    # the L1-only prefix does not
    h = x
    for b in enc.blocks[:6]:
        h = b(h, geom)
    h2 = x2
    for b in enc.blocks[:6]:
        h2 = b(h2, geom)
    assert torch.allclose(h[0, :3], h2[0, :3], atol=1e-5)  # shaft A untouched
