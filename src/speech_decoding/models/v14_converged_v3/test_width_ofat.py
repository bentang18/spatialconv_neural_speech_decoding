"""Width OFAT — one integer moves the model along the ViT ladder (TDD).

The arm (Ben 2026-08-04): "Try ViT small OFAT - change d=384, heads = 6 - and keep
predictor width at 1/2 encoder width yeah? heads the same yeah?"

The locked config is enc 256/4 (head_dim 64) and pred 128/4 (head_dim 32), which ALREADY
satisfies three relations: heads = d/64, pred_d = enc_d/2, pred_heads = enc_heads. So the
arm is not a new sizing policy, it is the existing one evaluated at a second width. These
tests pin that reading two ways:

  1. ``width_spec(256)`` must reproduce the four locked constants EXACTLY. If it does not,
     the "default is byte-identical" claim is false and every prior run becomes
     incomparable to this arm.
  2. ``width_spec(384)`` must give enc 384/6 and pred 192/6, i.e. head_dim 64 and 32 held.

Depth is NOT a knob here — the layout stays 12 L1 encoder blocks and 6 L1 predictor blocks
(Design B), so the contrast is width alone.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.attention import L1Block
from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective
from speech_decoding.models.v14_converged_v3.stem import PER_BAND_SPECS
from speech_decoding.models.v14_converged_v3.towers import (
    ENC_D_MODEL,
    ENC_N_HEADS,
    N_LEVELS,
    PRED_D_MODEL,
    PRED_N_HEADS,
    build_encoder,
    build_predictor,
    width_spec,
)

VIT_SMALL = 384


def test_width_spec_at_256_reproduces_the_locked_constants() -> None:
    # THE INVARIANCE CLAIM. The four module constants are the shipped config; the derivation
    # must land on them, not merely near them, or "default ⇒ unchanged" is not true.
    assert width_spec(ENC_D_MODEL) == (ENC_D_MODEL, ENC_N_HEADS, PRED_D_MODEL, PRED_N_HEADS)
    assert width_spec(256) == (256, 4, 128, 4)


def test_width_spec_at_384_is_vit_small() -> None:
    enc_d, enc_h, pred_d, pred_h = width_spec(VIT_SMALL)
    assert (enc_d, enc_h, pred_d, pred_h) == (384, 6, 192, 6)
    assert enc_d // enc_h == 64  # ViT convention: head_dim fixed, only d moves
    assert pred_d // pred_h == 32  # the predictor's head_dim, held at the locked value


@pytest.mark.parametrize("d", [128, 256, 384, 512, 768])
def test_head_dims_are_invariant_across_the_whole_ladder(d: int) -> None:
    enc_d, enc_h, pred_d, pred_h = width_spec(d)
    assert enc_d // enc_h == 64 and pred_d // pred_h == 32
    assert pred_d * 2 == enc_d and pred_h == enc_h


@pytest.mark.parametrize("bad", [192, 320, 300, 64, 0])
def test_off_ladder_width_fails_loud_at_construction(bad: int) -> None:
    # 192 and 320 divide by 64 but NOT by 128, so their halved predictor would not divide by
    # the shared head count. Catching it here beats catching it at the first matmul.
    with pytest.raises(ValueError, match="must be a multiple of 128"):
        width_spec(bad)


def test_builders_default_to_the_locked_widths() -> None:
    enc = build_encoder(n_parcels=8)
    pred = build_predictor(n_parcels=8)
    e = next(b for b in enc.blocks if isinstance(b, L1Block))
    p = next(b for b in pred.blocks if isinstance(b, L1Block))
    assert (e.n_heads, e.head_dim) == (4, 64)
    assert (p.n_heads, p.head_dim) == (4, 32)


def test_builders_at_vit_small_widen_without_deepening() -> None:
    enc = build_encoder(n_parcels=8, d_model=VIT_SMALL)
    pred = build_predictor(n_parcels=8, enc_d_model=VIT_SMALL)
    assert len(enc.blocks) == 12 and len(pred.blocks) == 6  # DEPTH UNCHANGED
    for b in enc.blocks:
        assert isinstance(b, L1Block) and (b.n_heads, b.head_dim) == (6, 64)
    for b in pred.blocks:
        assert isinstance(b, L1Block) and (b.n_heads, b.head_dim) == (6, 32)
    assert enc.blocks[0].qkv.in_features == 384
    assert pred.blocks[0].qkv.in_features == 192


def test_objective_shapes_follow_the_encoder_width() -> None:
    # Every derived shape in the objective has to move together: a stale one loads fine at
    # 256 and silently mismatches at 384, which is exactly the failure this pins.
    obj = V3JepaObjective(n_parcels=8, mae=True, r6=True, d_model=VIT_SMALL)
    assert obj.d_model == 384 and obj.d_pred == 192
    assert obj.online.stem.d_model == 384 if hasattr(obj.online.stem, "d_model") else True
    # deep-sup fusion MLP: (n_levels·d → d → d_pred)
    assert obj.enc_to_pred[0].in_features == N_LEVELS * 384  # 1536, was 1024 at d=256
    assert obj.enc_to_pred[0].out_features == 384
    assert obj.enc_to_pred[2].out_features == 192
    # MAE reconstruction heads read the PREDICTOR width and emit each band's own bin count.
    assert [h.in_features for h in obj.mae_heads] == [192] * len(PER_BAND_SPECS)
    assert [h.out_features for h in obj.mae_heads] == [nb for nb, _ in PER_BAND_SPECS]
    # predictor-space embeddings + the mask query all live at d_pred
    assert obj.mask_token.shape == (1, 1, 192)
    assert obj.pred_band_emb.shape == (len(PER_BAND_SPECS), 192)


def test_default_objective_is_unchanged_at_256() -> None:
    obj = V3JepaObjective(n_parcels=8, mae=True, r6=True)
    assert obj.d_model == 256 and obj.d_pred == 128
    assert obj.enc_to_pred[0].in_features == N_LEVELS * 256  # the shipped 1024
    assert obj.mask_token.shape == (1, 1, 128)


def test_vit_small_is_a_real_capacity_increase() -> None:
    # Guards against a "wider" arm that silently builds the same model. 1.5x width on a
    # transformer is ~2.25x the block parameters, so the total must rise substantially.
    small = sum(p.numel() for p in V3JepaObjective(n_parcels=8, mae=True, r6=True).parameters())
    big = sum(
        p.numel()
        for p in V3JepaObjective(n_parcels=8, mae=True, r6=True, d_model=VIT_SMALL).parameters()
    )
    assert big > 2.0 * small
    print(f"[check] width OFAT params: d256 {small/1e6:.2f}M -> d384 {big/1e6:.2f}M "
          f"({big/small:.2f}x)")


def test_encoder_width_is_readable_off_a_state_dict() -> None:
    # The encode path INFERS the width from the ckpt rather than taking a flag (unlike
    # --no-space-rope, which leaves no trace). This pins the key it reads.
    enc = build_encoder(n_parcels=8, d_model=VIT_SMALL)
    sd = enc.state_dict()
    assert sd["blocks.0.norm1.weight"].shape[0] == 384
    assert build_encoder(n_parcels=8).state_dict()["blocks.0.norm1.weight"].shape[0] == 256


def test_wrong_width_shell_cannot_silently_load_a_ckpt() -> None:
    # The self-verifying property that makes a CLI flag unnecessary at encode time.
    wide = build_encoder(n_parcels=8, d_model=VIT_SMALL).state_dict()
    narrow = build_encoder(n_parcels=8)
    with pytest.raises(RuntimeError, match="size mismatch"):
        narrow.load_state_dict(wide, strict=False)


def test_vit_small_encoder_runs_and_preserves_the_token_axis() -> None:
    from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
    from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

    sc = build_sidecar(["LA1", "LA2", "LA3", "LB1", "LB2"], parcel_id=torch.tensor([0, 0, 0, 1, 1]))
    geom = build_l1_geometry(sc)
    enc = build_encoder(n_parcels=8, d_model=VIT_SMALL).eval()
    x = torch.randn(2, 5, 6, 384)
    with torch.no_grad():
        out = enc(x, geom, sc.parcel_id)
    assert out.shape == (2, 5, 6, N_LEVELS * 384)  # deep-sup concat at the new width
