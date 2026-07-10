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
