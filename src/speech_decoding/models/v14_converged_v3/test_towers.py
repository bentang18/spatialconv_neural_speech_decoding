"""v14_converged_v3 Phase 4c — encoder / predictor tower assembly (TDD).

Memo project-v14-converged-v3-sensor-architecture (ENCODER/PREDICTOR OFFLOAD):

  Encoder  12 WIDE blocks, d_model 256, 4 heads (head_dim 64). Layout
           ``L1×6 · L2L1L1 · L2L1L1`` = 10 L1 : 2 L2 (5:1, L1-heavy — the local,
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


def test_encoder_layout_is_l1x6_then_two_l2l1l1() -> None:
    assert ENC_LAYOUT == (
        "L1", "L1", "L1", "L1", "L1", "L1",
        "L2", "L1", "L1", "L2", "L1", "L1",
    )
    enc = build_encoder(n_parcels=8)
    kinds = _kinds(enc)
    assert len(kinds) == 12
    assert kinds.count("L1") == 10 and kinds.count("L2") == 2
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
