"""v14_converged_v3 r4 — flat per-band pack-plan invariants (TDD, #36/#26 foundation).

The flat plan is the contract the varlen kernels consume; a wrong layout is a silent
miscompute, so every structural invariant is named, asserted, and printed
(feedback-build-the-invariant-into-the-probe):

  1. Token universe = N·k_full, shaft-contiguous, cu_seqlens sums to it (static VALUES).
  2. Per-token coords are exactly the SLOW/MID/HGA lattice (band counts, time_pos = j·stride,
     depth = the contact's clinical index).
  3. masked ⊇ in_loss (loss only on queries); spatially-masked contacts are UNCONDITIONALLY
     in-loss (leak-proof by construction); temporally-masked-only tokens are in-loss IFF
     their own-band margin ≥ 2 (M14 overlap-factor-2).
  4. masked/visible partition the grid; masked COUNTS are per-session constants (static shapes).
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import sample_masks
from speech_decoding.models.v14_converged_v3.pack_r4 import (
    MARGIN,
    _dist_to_visible,
    band_token_counts,
    build_r4_grid,
    build_visible_pack,
    pack_band_tokens,
    scatter_visible,
    token_flags,
)
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

T = 16  # multiple of SLOW_STRIDE=8 ⇒ SLOW 2, MID 8, HGA 16 tokens per contact.


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        parcel_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    return sc, build_l1_geometry(sc)


def test_token_universe_is_shaft_contiguous_and_cu_static() -> None:
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    slow, mid, hga = band_token_counts(T)
    assert (slow, mid, hga) == (2, 8, 16)
    k_full = slow + mid + hga  # 26
    n = int(geom.valid.sum())  # 5
    assert grid.k_full == k_full and grid.total == n * k_full == 130
    # cu_seqlens: LA has 3 contacts (78 tokens), LB has 2 (52) ⇒ [0, 78, 130].
    assert grid.cu_seqlens.tolist() == [0, 78, 130]
    assert int(grid.cu_seqlens[-1]) == grid.total
    # shaft id is non-decreasing ⇒ each shaft block is contiguous (varlen requirement).
    assert torch.all(grid.shaft[1:] >= grid.shaft[:-1])
    print(f"[check] universe {grid.total} tok, shaft-contiguous, cu {grid.cu_seqlens.tolist()} OK")


def test_per_token_coords_are_the_band_lattice() -> None:
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    slow, mid, hga = grid.band_lengths
    n = int(geom.valid.sum())
    # band counts: exactly N per-band tokens.
    assert int((grid.band == 0).sum()) == n * slow
    assert int((grid.band == 1).sum()) == n * mid
    assert int((grid.band == 2).sum()) == n * hga
    # time_pos = bandpos * stride on the shared 32 Hz lattice (SLOW 8, MID 2, HGA 1).
    for b, stride in ((0, 8), (1, 2), (2, 1)):
        sel = grid.band == b
        assert torch.equal(grid.time_pos[sel], grid.bandpos[sel] * stride)
        assert int(grid.bandpos[sel].max()) == grid.band_lengths[b] - 1
    # depth (L1 index-RoPE coord): constant across a contact's k_full-token block, and the
    # set of per-contact depths == the montage's clinical depths geom.depth[valid].
    per_contact_depth = grid.depth.reshape(n, grid.k_full)
    assert torch.all(per_contact_depth == per_contact_depth[:, :1])  # constant within contact
    assert torch.equal(per_contact_depth[:, 0].sort().values, geom.depth[geom.valid].sort().values)
    print("[check] band counts + time_pos=j·stride + depth constant-per-contact OK")


def test_masked_superset_of_in_loss_and_margin_gate() -> None:
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    g = torch.Generator().manual_seed(0)
    masks = sample_masks(geom, int(geom.valid.sum()), n_time=T, n_rows=8, generator=g)
    masked, in_loss = token_flags(grid, masks)

    # loss only on queries.
    assert torch.all(in_loss <= masked)

    contact_masked = masks.contact_mask[:, grid.contact]  # (R, total)
    # spatially-masked contacts: EVERY token in-loss (no own visible same-band frame).
    assert torch.all(in_loss[contact_masked])

    # temporally-masked-only tokens (visible contact): in-loss IFF own-band margin ≥ 2.
    band_masks = (masks.slow_mask, masks.mid_mask, masks.hga_mask)
    ok_margin = True
    for b, bm in enumerate(band_masks):
        sel = grid.band == b
        pos = grid.bandpos[sel]
        dist = _dist_to_visible(bm)  # (R, T_b)
        want = bm[:, pos] & (dist[:, pos] >= MARGIN)  # temporal in-loss on the own grid
        vis_contact = ~contact_masked[:, sel]
        got = in_loss[:, sel] & vis_contact
        ok_margin &= torch.equal(got, want & vis_contact)
    assert ok_margin
    frac_leak_proof = float(contact_masked.float().mean())
    print(f"[check] in_loss ⊆ masked; spatial→all-in-loss; temporal margin≥{MARGIN} exact; "
          f"{frac_leak_proof:.0%} tokens leak-proof-by-spatial OK")


def test_masked_visible_partition_and_static_counts() -> None:
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    g = torch.Generator().manual_seed(1)
    masks = sample_masks(geom, int(geom.valid.sum()), n_time=T, n_rows=16, generator=g)
    masked, _ = token_flags(grid, masks)
    visible = ~masked
    # partition: every token is exactly one of masked / visible.
    assert torch.all(masked ^ visible)
    # per-session STATIC counts: masked total identical across all clips (compile-safe shapes).
    per_row = masked.sum(1)
    assert int(per_row.min()) == int(per_row.max())
    vis_per_row = visible.sum(1)
    assert int(vis_per_row.min()) == int(vis_per_row.max())
    print(f"[check] masked/visible partition; static counts masked={int(per_row[0])} "
          f"visible={int(vis_per_row[0])} (all {masked.shape[0]} rows equal) OK")


def test_pack_band_tokens_places_every_token_at_its_grid_slot() -> None:
    # pack_band_tokens must land band_tokens[b][:, c, j] at the grid slot whose
    # (contact, band, bandpos) == (c, b, j). Tag each source token with a unique scalar
    # f(c,b,j) and assert the flat output reproduces f over the WHOLE grid (order-exact).
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    B, d, N = 2, 3, int(geom.valid.sum())  # N=5 full contacts
    band_tokens = []
    for b, length in enumerate(grid.band_lengths):
        c = torch.arange(N)[None, :, None, None]          # contact
        j = torch.arange(length)[None, None, :, None]     # bandpos
        tag = (c * 100 + b * 10 + j).float()              # unique per (c,b,j)
        band_tokens.append(tag.expand(B, N, length, d).clone())
    packed = pack_band_tokens(band_tokens, grid)  # (B, total, d)
    expected = (grid.contact * 100 + grid.band * 10 + grid.bandpos).float()  # (total,)
    ok = (
        packed.shape == (B, grid.total, d)
        and torch.equal(packed[0, :, 0], expected)
        and torch.equal(packed[1, :, 0], expected)  # batch-consistent
        and torch.equal(packed[0, :, 0], packed[0, :, d - 1])  # d broadcast intact
    )
    print(f"[check] pack_band_tokens order-exact over all {grid.total} tokens, "
          f"batch+width consistent {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_visible_pack_is_leak_safe_and_static_shape() -> None:
    # THE leak-safety invariant: the online encoder must see NO masked token, so every
    # column in the pack must be visible. Plus static M_vis (compile-safe shapes) and
    # shaft-contiguity (the varlen kernel's block requirement).
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    pid = sc.parcel_id[grid.contact]  # (total,)
    g = torch.Generator().manual_seed(3)
    masks = sample_masks(geom, int(geom.valid.sum()), n_time=T, n_rows=8, generator=g)
    masked, _ = token_flags(grid, masks)
    pack = build_visible_pack(grid, masked, pid)

    # (1) leak-safe: no packed column is a masked token.
    leaked = int(masked.gather(1, pack.idx).sum())
    # (2) static shape: M_vis == visible count, identical across clips.
    vis_per = (~masked).sum(1)
    static = pack.m_vis == int(vis_per[0]) and int(vis_per.min()) == int(vis_per.max())
    # (3) coords consistent with the grid at the packed columns.
    coords_ok = (
        torch.equal(pack.depth, grid.depth[pack.idx])
        and torch.equal(pack.time_pos, grid.time_pos[pack.idx])
        and torch.equal(pack.parcel, pid[pack.idx])
    )
    # (4) shaft-contiguous per clip (ascending grid order ⇒ non-decreasing shaft id) and
    #     cu_seqlens segment lengths == actual visible tokens per shaft.
    shaft_pk = grid.shaft[pack.idx]  # (B, M_vis)
    contiguous = bool(torch.all(shaft_pk[:, 1:] >= shaft_pk[:, :-1]))
    seg = (pack.cu_seqlens[1:] - pack.cu_seqlens[:-1]).tolist()
    per_shaft_true = [int((shaft_pk[0] == s).sum()) for s in range(len(seg))]
    cu_ok = seg == per_shaft_true and int(pack.cu_seqlens[-1]) == pack.m_vis
    ok = leaked == 0 and static and coords_ok and contiguous and cu_ok
    print(f"[check] visible-pack leaked={leaked} (must be 0); M_vis={pack.m_vis} static={static}; "
          f"coords={coords_ok}; shaft-contiguous={contiguous}; cu={seg} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_scatter_visible_round_trips_and_fills_masked() -> None:
    # scatter_visible must (a) put each visible latent back at its own grid column and
    # (b) leave EVERY masked slot exactly equal to the mask-query fill — this is what
    # feeds the predictor (visible latents + mask queries at targets).
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    pid = sc.parcel_id[grid.contact]
    g = torch.Generator().manual_seed(4)
    masks = sample_masks(geom, int(geom.valid.sum()), n_time=T, n_rows=6, generator=g)
    masked, _ = token_flags(grid, masks)
    B, d = masked.shape[0], 5
    pack = build_visible_pack(grid, masked, pid)

    # tag each visible latent by its full-grid column so we can verify placement.
    h_vis = pack.idx[:, :, None].expand(-1, -1, d).to(torch.float32).clone()
    fill = torch.full((1, 1, d), -1.0)
    full = scatter_visible(h_vis, pack.idx, grid.total, fill)  # (B, total, d)

    col = torch.arange(grid.total).to(torch.float32)[None, :, None].expand(B, grid.total, d)
    visible = ~masked
    placed_ok = torch.equal(full[visible.unsqueeze(-1).expand_as(full)], col[visible.unsqueeze(-1).expand_as(col)])
    filled_ok = bool(torch.all(full[masked] == -1.0))
    ok = placed_ok and filled_ok
    print(f"[check] scatter_visible: visible latents at own column={placed_ok}; "
          f"masked slots == mask-query fill={filled_ok} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_dist_to_visible_is_correct_on_a_known_mask() -> None:
    # [v m m m m v] → distances to nearest visible: 0,1,2,2,1,0. margin≥2 hits the 2 interior.
    mask = torch.tensor([[False, True, True, True, True, False]])
    dist = _dist_to_visible(mask)
    assert dist.tolist() == [[0, 1, 2, 2, 1, 0]]
    in_loss = mask & (dist >= MARGIN)
    assert in_loss.tolist() == [[False, False, True, True, False, False]]
    print(f"[check] dist_to_visible {dist.tolist()[0]}; width-4 block ⇒ 2 interior in-loss OK")
