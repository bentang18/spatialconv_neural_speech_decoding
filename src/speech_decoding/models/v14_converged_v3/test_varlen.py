"""v14_converged_v3 — varlen block-diagonal attention primitive (#24) TDD.

The reference backend is the CPU/testing path; it must reproduce EXACTLY the
block-diagonal attention flash runs on GPU. We pin it against an independent,
hand-rolled per-segment SDPA (the ground truth), and check the two structural
guarantees flash relies on: (1) no cross-block leakage, (2) zero-length segments
(whole-masked shafts) are inert.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from speech_decoding.models.v14_converged_v3.varlen import (
    _segment_ids,
    varlen_block_diag_attention,
)


def _per_segment_truth(q, k, v, cu):
    """Independent ground truth: dense softmax attention WITHIN each segment."""
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(q.shape[-1])
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
        if b == a:
            continue
        qs = q[a:b].transpose(0, 1)  # (H, L, hd)
        ks = k[a:b].transpose(0, 1)
        vs = v[a:b].transpose(0, 1)
        att = torch.softmax((qs @ ks.transpose(-1, -2)) * scale, dim=-1)
        out[a:b] = (att @ vs).transpose(0, 1)
    return out


def test_segment_ids_from_cu_seqlens() -> None:
    cu = torch.tensor([0, 3, 3, 5], dtype=torch.int32)  # seg lens 3, 0, 2
    seg = _segment_ids(cu, total=5)
    assert seg.tolist() == [0, 0, 0, 2, 2]  # empty seg 1 contributes nothing


def test_reference_matches_per_segment_softmax() -> None:
    torch.manual_seed(0)
    total, h, hd = 7, 2, 8
    q, k, v = (torch.randn(total, h, hd, dtype=torch.float64) for _ in range(3))
    cu = torch.tensor([0, 4, 7], dtype=torch.int32)  # two shafts: 4 and 3 tokens
    got = varlen_block_diag_attention(q, k, v, cu, max_seqlen=4, backend="reference")
    want = _per_segment_truth(q, k, v, cu)
    assert torch.allclose(got, want, atol=1e-10)


def test_no_cross_block_leakage() -> None:
    # Token in block 0 must be unaffected by wildly different values in block 1.
    torch.manual_seed(1)
    total, h, hd = 6, 1, 4
    q = torch.randn(total, h, hd, dtype=torch.float64)
    k = torch.randn(total, h, hd, dtype=torch.float64)
    v = torch.randn(total, h, hd, dtype=torch.float64)
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    out1 = varlen_block_diag_attention(q, k, v, cu, max_seqlen=3, backend="reference")
    v2 = v.clone()
    v2[3:] += 100.0  # perturb block 1 values only
    out2 = varlen_block_diag_attention(q, k, v2, cu, max_seqlen=3, backend="reference")
    assert torch.allclose(out1[:3], out2[:3], atol=1e-10)  # block 0 untouched
    assert not torch.allclose(out1[3:], out2[3:])  # block 1 changed


def test_zero_length_segments_are_inert() -> None:
    # Inserting empty segments (whole-masked shafts) between real blocks must not
    # change the result — cu_seqlens length grows, tokens/output identical.
    torch.manual_seed(2)
    total, h, hd = 5, 2, 4
    q, k, v = (torch.randn(total, h, hd, dtype=torch.float64) for _ in range(3))
    cu_dense = torch.tensor([0, 3, 5], dtype=torch.int32)
    cu_gappy = torch.tensor([0, 3, 3, 3, 5], dtype=torch.int32)  # two 0-len shafts
    a = varlen_block_diag_attention(q, k, v, cu_dense, max_seqlen=3, backend="reference")
    b = varlen_block_diag_attention(q, k, v, cu_gappy, max_seqlen=3, backend="reference")
    assert torch.allclose(a, b, atol=1e-12)


def test_single_block_equals_full_attention() -> None:
    # One segment spanning everything = plain dense attention (no masking).
    torch.manual_seed(3)
    total, h, hd = 5, 2, 8
    q, k, v = (torch.randn(total, h, hd, dtype=torch.float64) for _ in range(3))
    cu = torch.tensor([0, total], dtype=torch.int32)
    got = varlen_block_diag_attention(q, k, v, cu, max_seqlen=total, backend="reference")
    scale = 1.0 / math.sqrt(hd)
    qh, kh, vh = (t.transpose(0, 1)[None] for t in (q, k, v))
    want = F.scaled_dot_product_attention(qh, kh, vh, scale=scale)[0].transpose(0, 1)
    assert torch.allclose(got, want, atol=1e-10)


def test_auto_backend_is_reference_on_cpu() -> None:
    q = torch.randn(4, 1, 4)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    # backend="auto" on CPU must not attempt to import/use flash.
    out = varlen_block_diag_attention(q, q, q, cu, max_seqlen=4, backend="auto")
    assert out.shape == (4, 1, 4)


def test_flex_matches_reference() -> None:
    # FlexAttention (eager, no compile) realises the SAME block-diagonal softmax as
    # the reference — the numerical pin for the GPU-without-flash backend (F2b). Run
    # eager (compiled=False) so no torch.compile in CI; identical numerics either way.
    import pytest

    pytest.importorskip("torch.nn.attention.flex_attention")
    from speech_decoding.models.v14_converged_v3.varlen import _flex_block_diag

    torch.manual_seed(4)
    total, h, hd = 9, 2, 16
    q, k, v = (torch.randn(total, h, hd, dtype=torch.float32) for _ in range(3))
    cu = torch.tensor([0, 4, 4, 9], dtype=torch.int32)  # incl a 0-length shaft
    got = _flex_block_diag(q, k, v, cu, compiled=False)
    want = _per_segment_truth(q, k, v, cu)
    assert torch.allclose(got, want, atol=1e-5)


def _grid_pos_cu(contacts, max_c, T, vis):
    """(grid_pos (1,P), cu (S+1,)) for one clip, given per-shaft contact counts and
    the visible-shaft set. Masked shafts contribute a zero-length cu segment."""
    S = len(contacts)
    slots = [s * max_c + c for s in vis for c in range(contacts[s])]
    grid_pos = torch.tensor(slots, dtype=torch.int64)[None]
    blens = [contacts[s] * T if s in vis else 0 for s in range(S)]
    cu = torch.tensor([0] + list(_accum(blens)), dtype=torch.int32)
    return grid_pos, cu


def _accum(xs):
    t = 0
    for x in xs:
        t += x
        yield t


def test_njt_matches_padded_oracle_gpu() -> None:
    # njt_block_diag (the GPU production path) is numerically identical to the padded
    # sdpa_block_diag_packed oracle (the CPU/test path), all-visible AND with dropped
    # (whole-masked) shafts, and its backward is finite. GPU-gated: the ragged
    # flash-varlen kernel is CUDA-only. This is the #24-style equivalence re-proof.
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("njt varlen path is GPU-only")
    from speech_decoding.models.v14_converged_v3.varlen import (
        njt_block_diag,
        sdpa_block_diag_packed,
    )

    dev = torch.device("cuda")
    contacts = [4, 3, 5, 2]  # 4 shafts, ragged
    S, max_c, T = len(contacts), max(contacts), 6
    for vis in ([0, 1, 2, 3], [0, 2, 3]):  # all-visible, then drop shaft 1
        grid_pos = _grid_pos_cu(contacts, max_c, T, vis)[0].to(dev)
        cu = _grid_pos_cu(contacts, max_c, T, vis)[1].to(dev)
        P = grid_pos.shape[1]
        torch.manual_seed(7)
        q, k, v = (
            torch.randn(P * T, 4, 32, device=dev, dtype=torch.float32, requires_grad=True)
            for _ in range(3)
        )
        o_pad = sdpa_block_diag_packed(q, k, v, grid_pos, S, max_c, T)
        # njt now consumes the PRECOMPUTED compacted offsets + explicit max_seqlen
        # (the build_pack_plan contract) — drop the zero-length shafts here.
        off = cu.to(torch.int64)
        cu_drop = torch.cat([off[:1], off[1:][off[1:] != off[:-1]]])
        o_njt = njt_block_diag(q, k, v, cu_drop, max_c * T)
        assert torch.allclose(o_pad, o_njt, atol=1e-4), (vis, (o_pad - o_njt).abs().max())
        o_njt.sum().backward()
        assert all(torch.isfinite(t.grad).all() for t in (q, k, v)), vis
