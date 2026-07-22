"""r5 (Chang 2-stream) single-band grid + accept-the-bleed flags.

TDD for the additive r5 path: ``build_r5_grid`` is the degenerate one-band case of
``build_r4_grid`` (single band, stride-1 lattice, ``bandpos == time_pos``), and
``token_flags_r5`` scores EVERY masked token (``in_loss == masked``, no margin gate).
``pack_band_tokens`` must accept the 1-band grid (the guard is band-count agnostic).
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import sample_masks_r5
from speech_decoding.models.v14_converged_v3.pack_r4 import (
    build_r5_grid,
    build_visible_pack,
    pack_band_tokens,
    token_flags_r5,
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


def test_single_band_stride1_lattice() -> None:
    sc, geom = _session()
    grid = build_r5_grid(geom, n_time=T)
    n = int(geom.valid.sum())  # 5
    # ONE band, T tokens per contact.
    assert grid.k_full == T and grid.band_lengths == (T,)
    assert grid.total == n * T == 80
    assert int((grid.band == 0).sum()) == grid.total  # every token band 0
    # stride-1: lattice == token index == bandpos, spanning [0, T).
    assert torch.equal(grid.time_pos, grid.bandpos)
    sel = grid.band == 0
    assert int(grid.bandpos[sel].max()) == T - 1
    print(f"[check] OK single band, k_full={grid.k_full}, time_pos==bandpos, stride 1")


def test_shaft_contiguous_and_cu_static() -> None:
    sc, geom = _session()
    grid = build_r5_grid(geom, n_time=T)
    # LA has 3 contacts (48 tokens), LB has 2 (32) ⇒ [0, 48, 80].
    assert grid.cu_seqlens.tolist() == [0, 48, 80]
    assert int(grid.cu_seqlens[-1]) == grid.total
    assert torch.all(grid.shaft[1:] >= grid.shaft[:-1])  # varlen requirement
    # depth constant within a contact's k_full block, set == clinical depths.
    n = int(geom.valid.sum())
    per_contact_depth = grid.depth.reshape(n, grid.k_full)
    assert torch.all(per_contact_depth == per_contact_depth[:, :1])
    assert torch.equal(
        per_contact_depth[:, 0].sort().values, geom.depth[geom.valid].sort().values
    )
    print("[check] OK shaft-contiguous, cu [0,48,80], depth constant-per-contact")


def test_accept_the_bleed_in_loss_equals_masked() -> None:
    sc, geom = _session()
    grid = build_r5_grid(geom, n_time=T)
    g = torch.Generator().manual_seed(0)
    masks = sample_masks_r5(geom, int(geom.valid.sum()), n_time=T, n_rows=8, generator=g)
    masked, in_loss = token_flags_r5(grid, masks)
    # ACCEPT THE BLEED: no margin gate ⇒ every masked token scored.
    assert torch.equal(masked, in_loss)
    # masked == contact OR per-shaft temporal, reconstructed independently.
    contact_masked = masks.contact_mask[:, grid.contact]
    temporal_masked = masks.temporal_mask[:, grid.shaft, grid.bandpos]  # per-shaft time axis
    assert torch.equal(masked, contact_masked | temporal_masked)
    assert masked.any() and not masked.all()  # non-degenerate
    print(f"[check] OK in_loss == masked (no gate); masked frac {masked.float().mean():.0%}")


def test_pack_band_tokens_accepts_single_band() -> None:
    sc, geom = _session()
    grid = build_r5_grid(geom, n_time=T)
    n = int(geom.valid.sum())
    B = 2
    tok = torch.arange(B * n * T * D).float().reshape(B, n, T, D)
    x = pack_band_tokens((tok,), grid)  # 1-tuple accepted
    assert x.shape == (B, grid.total, D)
    # token (contact c, bandpos p) must land where grid says.
    canon = grid.contact.reshape(n, grid.k_full)[:, 0]  # canonical contact order
    expect = tok[:, canon].reshape(B, grid.total, D)
    assert torch.equal(x, expect)
    print("[check] OK pack_band_tokens single-band round-trips to grid order")


def test_wrong_stream_count_raises() -> None:
    sc, geom = _session()
    grid = build_r5_grid(geom, n_time=T)
    n = int(geom.valid.sum())
    tok = torch.zeros(1, n, T, D)
    try:
        pack_band_tokens((tok, tok), grid)  # 2 streams, grid expects 1
    except ValueError as e:
        assert "expected 1 band" in str(e)
        print("[check] OK 2 streams into 1-band grid raises")
        return
    raise AssertionError("expected ValueError on stream-count mismatch")


def test_visible_pack_rides_r5_grid() -> None:
    """build_visible_pack is band-agnostic ⇒ works unchanged on the r5 grid."""
    sc, geom = _session()
    grid = build_r5_grid(geom, n_time=T)
    g = torch.Generator().manual_seed(1)
    masks = sample_masks_r5(geom, int(geom.valid.sum()), n_time=T, n_rows=4, generator=g)
    masked, _ = token_flags_r5(grid, masks)
    parcel_packed = torch.zeros(grid.total, dtype=torch.long)
    pack = build_visible_pack(grid, masked, parcel_packed)
    # every packed idx is a visible (~masked) column, per row.
    for r in range(masked.shape[0]):
        assert torch.all(~masked[r][pack.idx[r]])
    assert pack.m_vis == int((~masked).sum(1).max())
    print(f"[check] OK visible pack on r5 grid, m_vis={pack.m_vis}")
