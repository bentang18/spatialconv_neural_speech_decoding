"""v3r5nf (no-fusion) two-band grid + per-stream accept-the-bleed flags.

``build_r5nf_grid`` = two stride-1 bands (0=HGA, 1=LFS), ``k_full == 2T``, ``bandpos ==
time_pos == token index within band``; ``token_flags_r5nf`` scores EVERY masked token
(``in_loss == masked``) and routes each token to its OWN stream's masks.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import sample_masks_r5nf
from speech_decoding.models.v14_converged_v3.pack_r4 import (
    build_r5nf_grid,
    build_visible_pack,
    pack_band_tokens,
    token_flags_r5nf,
)
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

T = 16  # 32 Hz tokens per contact.
D = 8


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        parcel_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    return sc, build_l1_geometry(sc)


def test_two_bands_stride1_lattice() -> None:
    sc, geom = _session()
    grid = build_r5nf_grid(geom, n_time=T)
    n = int(geom.valid.sum())  # 5
    assert grid.k_full == 2 * T and grid.band_lengths == (T, T)
    assert grid.total == n * 2 * T == 160
    assert int((grid.band == 0).sum()) == n * T  # HGA tokens
    assert int((grid.band == 1).sum()) == n * T  # LFS tokens
    assert set(grid.band.unique().tolist()) == {0, 1}
    # stride-1: bandpos == time_pos == token index within band, spanning [0, T).
    assert torch.equal(grid.time_pos, grid.bandpos)
    for b in (0, 1):
        sel = grid.band == b
        assert int(grid.bandpos[sel].max()) == T - 1
        assert int(grid.bandpos[sel].min()) == 0
    print(f"[check] OK two bands, k_full={grid.k_full}, time_pos==bandpos, stride 1")


def test_band_major_order_hga_then_lfs_per_contact() -> None:
    sc, geom = _session()
    grid = build_r5nf_grid(geom, n_time=T)
    n = int(geom.valid.sum())
    band_blk = grid.band.reshape(n, grid.k_full)[0]  # per-contact band pattern
    assert torch.equal(band_blk[:T], torch.zeros(T, dtype=torch.long))  # HGA first
    assert torch.equal(band_blk[T:], torch.ones(T, dtype=torch.long))   # LFS second
    print("[check] OK per-contact order is [HGA 0..T-1, LFS 0..T-1]")


def test_shaft_contiguous_and_cu_static() -> None:
    sc, geom = _session()
    grid = build_r5nf_grid(geom, n_time=T)
    # LA has 3 contacts (3·2T=96 tokens), LB has 2 (64) ⇒ [0, 96, 160].
    assert grid.cu_seqlens.tolist() == [0, 96, 160]
    assert int(grid.cu_seqlens[-1]) == grid.total
    assert torch.all(grid.shaft[1:] >= grid.shaft[:-1])  # varlen requirement
    print("[check] OK shaft-contiguous, cu [0,96,160]")


def test_accept_the_bleed_in_loss_equals_masked() -> None:
    sc, geom = _session()
    grid = build_r5nf_grid(geom, n_time=T)
    g = torch.Generator().manual_seed(0)
    masks = sample_masks_r5nf(geom, int(geom.valid.sum()), n_time=T, n_rows=8, generator=g)
    masked, in_loss = token_flags_r5nf(grid, masks)
    assert torch.equal(masked, in_loss)  # no margin gate
    assert masked.any() and not masked.all()  # non-degenerate
    print(f"[check] OK in_loss == masked (no gate); masked frac {masked.float().mean():.0%}")


def test_per_stream_masking_routes_correctly() -> None:
    # construct a mask where HGA(contact e, frame t) is masked but LFS(e, t) is NOT: only the
    # HGA token at (e, t) must flip, its LFS twin must stay visible (the independence proof).
    sc, geom = _session()
    grid = build_r5nf_grid(geom, n_time=T)
    n = int(geom.valid.sum())
    S = geom.n_shafts
    # empty masks except one HGA temporal cell (row 0, shaft 0, frame 3).
    zeros_c = torch.zeros(1, n, dtype=torch.bool)
    zeros_t = torch.zeros(1, S, T, dtype=torch.bool)
    hga_t = zeros_t.clone()
    hga_t[0, 0, 3] = True  # HGA shaft-0 frame-3 masked; LFS untouched

    class _M:
        hga_contact_mask = zeros_c
        lfs_contact_mask = zeros_c
        hga_temporal_mask = hga_t
        lfs_temporal_mask = zeros_t

    masked, in_loss = token_flags_r5nf(grid, _M())
    # tokens that should be masked: HGA (band 0) tokens on shaft 0 at bandpos 3.
    want = (grid.band == 0) & (grid.shaft == 0) & (grid.bandpos == 3)
    assert torch.equal(masked[0], want)
    # the LFS twins at shaft 0 bandpos 3 must be VISIBLE (independence).
    lfs_twin = (grid.band == 1) & (grid.shaft == 0) & (grid.bandpos == 3)
    assert not masked[0][lfs_twin].any()
    assert torch.equal(masked, in_loss)
    print(f"[check] OK HGA-only mask flips {int(want.sum())} HGA token(s), LFS twin visible")


def test_pack_band_tokens_accepts_two_bands() -> None:
    sc, geom = _session()
    grid = build_r5nf_grid(geom, n_time=T)
    n = int(geom.valid.sum())
    B = 2
    hga_tok = torch.arange(B * n * T * D).float().reshape(B, n, T, D)
    lfs_tok = (torch.arange(B * n * T * D).float() + 1000.0).reshape(B, n, T, D)
    x = pack_band_tokens((hga_tok, lfs_tok), grid)  # 2-tuple
    assert x.shape == (B, grid.total, D)
    # per-contact block must be [HGA 0..T-1, LFS 0..T-1].
    canon = grid.contact.reshape(n, grid.k_full)[:, 0]
    expect = torch.cat([hga_tok[:, canon], lfs_tok[:, canon]], dim=2).reshape(B, grid.total, D)
    assert torch.equal(x, expect)
    print("[check] OK pack_band_tokens two-band round-trips to [HGA;LFS] grid order")


def test_visible_pack_rides_r5nf_grid() -> None:
    sc, geom = _session()
    grid = build_r5nf_grid(geom, n_time=T)
    g = torch.Generator().manual_seed(1)
    masks = sample_masks_r5nf(geom, int(geom.valid.sum()), n_time=T, n_rows=4, generator=g)
    masked, _ = token_flags_r5nf(grid, masks)
    parcel_packed = torch.zeros(grid.total, dtype=torch.long)
    pack = build_visible_pack(grid, masked, parcel_packed)
    for r in range(masked.shape[0]):
        assert torch.all(~masked[r][pack.idx[r]])
    assert pack.m_vis == int((~masked).sum(1).max())
    print(f"[check] OK visible pack on r5nf grid, m_vis={pack.m_vis}")
