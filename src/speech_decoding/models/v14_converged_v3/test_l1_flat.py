"""v14_converged_v3 r4 — pin the flat per-band L1 path against the padded oracle.

``L1Block.forward_flat`` (r4, ragged per-band tokens) must be the SAME attention as the
trusted padded ``forward`` (test_attention's CPU reference) whenever the flat token set
happens to be a full rectangle: same tokens, same RoPE coords (depth, arange-time), same
per-shaft key set ⇒ per-contact-identical output. If they diverge, the flat path is wrong.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.attention import L1Block
from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

T = 6


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        parcel_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    return sc, build_l1_geometry(sc)


def test_forward_flat_matches_padded_oracle_on_full_rectangle() -> None:
    torch.manual_seed(0)
    sc, geom = _session()
    blk = L1Block(256, 4).eval()
    x = torch.randn(1, 5, T, 256)

    # padded oracle (the trusted CPU reference).
    oracle = blk.forward(x, geom, None)  # (1, 5, T, 256)

    # SAME tokens laid out flat, shaft-contiguous, contact-major (c*T+t within shaft),
    # depth = clinical index, time_pos = arange(T) — exactly the rectangle, unrolled.
    canon = geom.gather_idx[geom.valid]  # (N,) contact idx, canonical order
    depth_canon = geom.depth[geom.valid]  # (N,)
    n = int(canon.shape[0])
    x_flat = x[0, canon].reshape(n * T, 256)
    depth_flat = depth_canon.repeat_interleave(T)
    time_flat = torch.arange(T).repeat(n)
    n_per = geom.valid.sum(1)  # (S,) contacts per shaft
    cu = torch.zeros(geom.n_shafts + 1, dtype=torch.int32)
    cu[1:] = (n_per * T).cumsum(0).to(torch.int32)
    max_seqlen = int((n_per * T).max())

    out_flat = blk.forward_flat(
        x_flat, depth_flat, time_flat, cu, cu.to(torch.int64), max_seqlen
    ).reshape(n, T, 256)

    ref = oracle[0, canon]  # (N, T, 256)
    max_err = (out_flat - ref).abs().max().item()
    ok = torch.allclose(out_flat, ref, atol=1e-4)
    print(f"[check] flat L1 == padded oracle at all {n} contacts (max err {max_err:.2e}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_flat_is_block_diagonal_across_shafts() -> None:
    # Perturbing shaft LB's tokens must not change shaft LA's flat output (block-diagonal).
    torch.manual_seed(1)
    sc, geom = _session()
    blk = L1Block(256, 4).eval()
    canon = geom.gather_idx[geom.valid]
    depth_canon = geom.depth[geom.valid]
    rows = torch.arange(geom.n_shafts)[:, None].expand(geom.n_shafts, geom.max_c)[geom.valid]
    n = int(canon.shape[0])
    depth_flat = depth_canon.repeat_interleave(T)
    time_flat = torch.arange(T).repeat(n)
    n_per = geom.valid.sum(1)
    cu = torch.zeros(geom.n_shafts + 1, dtype=torch.int32)
    cu[1:] = (n_per * T).cumsum(0).to(torch.int32)
    max_seqlen = int((n_per * T).max())

    x = torch.randn(n * T, 256)
    out_a = blk.forward_flat(x, depth_flat, time_flat, cu, cu.to(torch.int64), max_seqlen)
    x2 = x.clone().reshape(n, T, 256)
    shaft_of_canon = rows.repeat_interleave(T).reshape(n, T)[:, 0]  # (N,)
    x2[shaft_of_canon == 1] += torch.randn_like(x2[shaft_of_canon == 1]) * 3.0  # perturb LB
    out_b = blk.forward_flat(
        x2.reshape(n * T, 256), depth_flat, time_flat, cu, cu.to(torch.int64), max_seqlen
    )
    la = (shaft_of_canon == 0).repeat_interleave(T)
    assert torch.allclose(out_a[la], out_b[la], atol=1e-5)  # LA untouched
    assert not torch.allclose(out_a[~la], out_b[~la], atol=1e-4)  # LB moved
    print("[check] flat L1 is block-diagonal: perturbing LB leaves LA bit-stable OK")
