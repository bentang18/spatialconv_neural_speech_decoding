"""v14_converged_v3 — packed (varlen) tower == padded tower (#24) TDD.

The whole tower runs packed in production. Because every block preserves the
per-contact equivalence (L1 same key-set + RoPE; L2 permutation-invariant over
keys; MLP/LN/residual per-token), the packed tower reproduces the padded tower at
every contact present in both — this is the end-to-end proof the objective relies
on when it swaps the online encoder to packed over M_vis and the teacher/predictor
to packed over N.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.packing import build_pack_plan
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar
from speech_decoding.models.v14_converged_v3.towers import build_encoder, build_predictor

T = 6


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        parcel_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    return sc, build_l1_geometry(sc)


def _run_packed(tower, x_full, plan, parcel_id, N):
    idx = plan.order[:, :, None, None].expand(-1, -1, T, x_full.shape[-1])
    x_packed = x_full.gather(1, idx)
    parcel_packed = parcel_id[plan.order]  # (B, P)
    out = tower.forward_packed(x_packed, plan, parcel_packed, backend="reference")
    full = out.new_zeros(x_full.shape[0], N, T, out.shape[-1])
    full.scatter_(1, plan.order[:, :, None, None].expand(-1, -1, T, out.shape[-1]), out)
    return full


def test_deep_sup_encoder_returns_four_concatenated_levels() -> None:
    # #61: deep-sup encoder taps {3,6,9,12}, each through its own affine norms_block,
    # concatenated → last dim = 4·256. No terminal norm_out (norms_block[-1] doubles).
    from speech_decoding.models.v14_converged_v3.towers import ENC_SUP_TAPS, N_LEVELS

    sc, geom = _session()
    enc = build_encoder(n_parcels=8, deep_sup=True).eval()
    assert enc.deep_sup and enc.sup_taps == ENC_SUP_TAPS
    assert enc.norm_out is None
    assert len(enc.norms_block) == N_LEVELS == 4
    x = torch.randn(2, 5, T, 256)
    out = enc(x, geom, sc.parcel_id)
    assert out.shape == (2, 5, T, 4 * 256)


def test_deep_sup_encoder_levels_are_each_layernormed() -> None:
    # Each 256-chunk is an affine-LN output; at init (weight 1 / bias 0) every level
    # is ~unit-variance and ~zero-mean along the feature dim.
    sc, geom = _session()
    enc = build_encoder(n_parcels=8, deep_sup=True).eval()
    x = torch.randn(2, 5, T, 256)
    out = enc(x, geom, sc.parcel_id)
    for lvl in range(4):
        chunk = out[..., lvl * 256 : (lvl + 1) * 256]
        assert chunk.mean(-1).abs().max() < 1e-4  # zero-mean per token
        assert (chunk.std(-1, unbiased=False) - 1.0).abs().max() < 5e-2  # unit-var


def test_deep_sup_encoder_packed_matches_padded() -> None:
    # The #24 packed==padded equivalence must survive deep-sup (the concatenated,
    # per-level-normed output reproduces the padded tower at every present contact).
    sc, geom = _session()
    N = 5
    enc = build_encoder(n_parcels=8, deep_sup=True).eval()
    x = torch.randn(2, N, T, 256)
    padded = enc(x, geom, sc.parcel_id)  # (B, N, T, 1024)
    plan = build_pack_plan(geom, n_time=T, batch=2, n_selected=N, visible=None)
    full = _run_packed(enc, x, plan, sc.parcel_id, N)
    assert full.shape[-1] == 4 * 256
    assert torch.allclose(full, padded, atol=1e-4)


def test_single_tap_encoder_still_returns_d_model() -> None:
    # deep_sup=False = the single-tap ablation arm: terminal norm_out, [.,256].
    sc, geom = _session()
    enc = build_encoder(n_parcels=8, deep_sup=False).eval()
    assert not enc.deep_sup and enc.norm_out is not None and enc.norms_block is None
    x = torch.randn(2, 5, T, 256)
    assert enc(x, geom, sc.parcel_id).shape == (2, 5, T, 256)


def test_encoder_packed_unmasked_matches_padded() -> None:
    sc, geom = _session()
    N = 5
    enc = build_encoder(n_parcels=8).eval()
    x = torch.randn(2, N, T, 256)
    padded = enc(x, geom, sc.parcel_id)  # (B, N, T, 256)
    plan = build_pack_plan(geom, n_time=T, batch=2, n_selected=N, visible=None)
    full = _run_packed(enc, x, plan, sc.parcel_id, N)
    assert torch.allclose(full, padded, atol=1e-4)


def test_encoder_packed_masked_matches_padded_on_visible() -> None:
    sc, geom = _session()
    N = 5
    enc = build_encoder(n_parcels=8).eval()
    x = torch.randn(2, N, T, 256)
    visible = torch.tensor([
        [True, False, True, False, True],
        [False, True, True, True, False],
    ])
    padded = enc(x, geom, sc.parcel_id, visible=visible)
    plan = build_pack_plan(geom, n_time=T, batch=2, n_selected=3, visible=visible)
    full = _run_packed(enc, x, plan, sc.parcel_id, N)
    vis = visible[:, :, None, None].expand_as(full)
    assert torch.allclose(full[vis], padded[vis], atol=1e-4)


def test_predictor_packed_unmasked_matches_padded() -> None:
    sc, geom = _session()
    N = 5
    pred = build_predictor(n_parcels=8).eval()
    x = torch.randn(2, N, T, 128)
    padded = pred(x, geom, sc.parcel_id)
    plan = build_pack_plan(geom, n_time=T, batch=2, n_selected=N, visible=None)
    full = _run_packed(pred, x, plan, sc.parcel_id, N)
    assert torch.allclose(full, padded, atol=1e-4)


def test_encoder_packed_taps_are_visible_rows() -> None:
    # tap_blocks capture packed (B, M_vis, T, d) block outputs — already only the
    # visible contacts; scatter them and confirm they equal the padded taps there.
    sc, geom = _session()
    N = 5
    enc = build_encoder(n_parcels=8).eval()
    x = torch.randn(1, N, T, 256)
    visible = torch.tensor([[True, False, True, True, False]])
    _, padded_taps = enc(x, geom, sc.parcel_id, visible=visible, tap_blocks=(3, 12))
    plan = build_pack_plan(geom, n_time=T, batch=1, n_selected=3, visible=visible)
    idx = plan.order[:, :, None, None].expand(-1, -1, T, 256)
    x_packed = x.gather(1, idx)
    parcel_packed = sc.parcel_id[plan.order]
    _, packed_taps = enc.forward_packed(
        x_packed, plan, parcel_packed, backend="reference", tap_blocks=(3, 12)
    )
    for blk in (3, 12):
        full = packed_taps[blk].new_zeros(1, N, T, 256)
        full.scatter_(1, idx, packed_taps[blk])
        vis = visible[:, :, None, None].expand(1, N, T, 256)
        assert torch.allclose(full[vis], padded_taps[blk][vis], atol=1e-4)
