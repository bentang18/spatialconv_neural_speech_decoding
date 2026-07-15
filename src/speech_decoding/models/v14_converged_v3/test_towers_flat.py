"""v14_converged_v3 r4 — tower-level flat forward (Design B, #36 assembly keystone).

``V3Tower.forward_flat`` composes the tested leaf primitives (``L1Block.forward_flat``
pinned to the padded oracle in test_l1_flat; ``pack_r4.build_r4_grid`` structure in
test_pack_r4) into the production forward the JEPA teacher / encoder / predictor all run.
The composition adds three things the leaves don't: parcel identity once, a shared rope
table, and the SINGLE-CLIP grid lifted to a batch. Each is a place a silent miscompute can
hide, so every invariant is named + asserted + printed (feedback-build-the-invariant):

  1. Output width: deep-sup encoder → 4·d concat; single-norm predictor → d.
  2. Block-diagonal by (clip, shaft): perturbing one clip/shaft leaves the others bit-stable
     — the batched cu_seqlens + rope tiling must not cross clips or shafts.
  3. Batch consistency: identical clips → identical outputs, and a clip's output is the SAME
     whether run alone or inside a batch (the lift adds no coupling).
  4. Fed from the stem: stem → pack_band_tokens → forward_flat runs finite and grads flow
     back to the stem projections (the real training wire).
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import sample_masks
from speech_decoding.models.v14_converged_v3.pack_r4 import (
    build_r4_grid,
    build_visible_pack,
    pack_band_tokens,
    token_flags,
)
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar
from speech_decoding.models.v14_converged_v3.stem import PerBandStem
from speech_decoding.models.v14_converged_v3.towers import build_encoder, build_predictor

T = 16  # SLOW 2, MID 8, HGA 16 tokens per contact ⇒ k_full 26, total 130 (5 contacts).
D_ENC = 256


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        parcel_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    geom = build_l1_geometry(sc)
    return sc, geom, build_r4_grid(geom, n_time=T)


def test_output_width_deep_sup_encoder_and_single_norm_predictor() -> None:
    sc, geom, grid = _session()
    pid = sc.parcel_id[grid.contact]  # (total,)
    enc = build_encoder(n_parcels=8).eval()  # deep_sup default ON
    pred = build_predictor(n_parcels=8).eval()

    out_e = enc.forward_flat(torch.randn(2, grid.total, D_ENC), grid, pid)
    out_p = pred.forward_flat(torch.randn(2, grid.total, 128), grid, pid)
    ok = out_e.shape == (2, grid.total, 4 * D_ENC) and out_p.shape == (2, grid.total, 128)
    print(f"[check] enc deep-sup {tuple(out_e.shape)} == (2,{grid.total},1024); "
          f"pred {tuple(out_p.shape)} == (2,{grid.total},128) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_block_diagonal_by_clip_and_shaft() -> None:
    torch.manual_seed(0)
    sc, geom, grid = _session()
    pid = sc.parcel_id[grid.contact]
    enc = build_encoder(n_parcels=8).eval()
    x = torch.randn(2, grid.total, D_ENC)
    out = enc.forward_flat(x, grid, pid)

    # (a) perturb clip 1 only ⇒ clip 0 bit-stable, clip 1 moves.
    x1 = x.clone()
    x1[1] += torch.randn_like(x1[1]) * 3.0
    o1 = enc.forward_flat(x1, grid, pid)
    clip_iso = torch.allclose(out[0], o1[0], atol=1e-5) and not torch.allclose(out[1], o1[1], atol=1e-4)

    # (b) perturb shaft LB (grid.shaft==1) in clip 0 ⇒ shaft LA clip 0 stable, clip 1 fully stable.
    lb = grid.shaft == 1
    la = grid.shaft == 0
    x2 = x.clone()
    x2[0, lb] += torch.randn_like(x2[0, lb]) * 3.0
    o2 = enc.forward_flat(x2, grid, pid)
    shaft_iso = (
        torch.allclose(out[0, la], o2[0, la], atol=1e-5)
        and not torch.allclose(out[0, lb], o2[0, lb], atol=1e-4)
        and torch.allclose(out[1], o2[1], atol=1e-5)
    )
    ok = clip_iso and shaft_iso
    print(f"[check] block-diagonal by (clip, shaft): clip-iso={clip_iso}, shaft-iso={shaft_iso} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_batch_lift_adds_no_coupling() -> None:
    torch.manual_seed(1)
    sc, geom, grid = _session()
    pid = sc.parcel_id[grid.contact]
    enc = build_encoder(n_parcels=8).eval()

    one = torch.randn(1, grid.total, D_ENC)
    both = one.repeat(2, 1, 1)  # two identical clips
    out_both = enc.forward_flat(both, grid, pid)
    out_one = enc.forward_flat(one, grid, pid)  # single-clip run
    identical_clips = torch.allclose(out_both[0], out_both[1], atol=1e-5)
    alone_eq_batched = torch.allclose(out_one[0], out_both[0], atol=1e-5)
    ok = identical_clips and alone_eq_batched
    print(f"[check] identical clips → identical out ({identical_clips}); "
          f"alone == in-batch ({alone_eq_batched}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_tap_blocks_return_raw_depth_features() -> None:
    sc, geom, grid = _session()
    pid = sc.parcel_id[grid.contact]
    enc = build_encoder(n_parcels=8).eval()
    out, taps = enc.forward_flat(torch.randn(2, grid.total, D_ENC), grid, pid, tap_blocks=(3, 12))
    ok = set(taps) == {3, 12} and all(t.shape == (2, grid.total, D_ENC) for t in taps.values())
    print(f"[check] tap_blocks (3,12) → raw {D_ENC}-d feats at each, out still 1024-d "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_encoder_over_visible_pack_reduces_to_full_grid_when_nothing_masked() -> None:
    # forward_flat_pack is the online-encoder path over the per-clip VISIBLE subset. Its
    # correctness oracle: with an EMPTY mask every token is visible, so the visible pack IS
    # the full grid and forward_flat_pack must equal forward_flat token-for-token (same keys,
    # same order, same rope). This pins the per-clip-coord path against the tested full path.
    torch.manual_seed(3)
    sc, geom, grid = _session()
    pid = sc.parcel_id[grid.contact]
    enc = build_encoder(n_parcels=8).eval()
    x = torch.randn(2, grid.total, D_ENC)

    masked = torch.zeros(2, grid.total, dtype=torch.bool)  # nothing masked
    pack = build_visible_pack(grid, masked, pid)
    assert pack.m_vis == grid.total
    out_pack = enc.forward_flat_pack(x, pack)
    out_full = enc.forward_flat(x, grid, pid)
    ok = torch.allclose(out_pack, out_full, atol=1e-5)
    print(f"[check] empty-mask visible-pack ≡ full-grid forward: max|Δ|="
          f"{(out_pack - out_full).abs().max().item():.2e} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_encoder_over_visible_pack_is_block_diagonal_and_static_shape() -> None:
    # With a REAL mask the pack drops tokens per clip; the encoder must still be
    # block-diagonal by (clip, shaft) over the SURVIVING tokens and emit a static M_vis.
    torch.manual_seed(4)
    sc, geom, grid = _session()
    pid = sc.parcel_id[grid.contact]
    enc = build_encoder(n_parcels=8).eval()
    g = torch.Generator().manual_seed(5)
    masks = sample_masks(geom, int(geom.valid.sum()), n_time=T, n_rows=2, generator=g)
    masked, _ = token_flags(grid, masks)
    pack = build_visible_pack(grid, masked, pid)
    x = torch.randn(2, pack.m_vis, D_ENC)

    out = enc.forward_flat_pack(x, pack)
    shape_ok = out.shape == (2, pack.m_vis, 4 * D_ENC)
    # perturb shaft LB's visible tokens in clip 0 ⇒ shaft LA clip 0 stable, clip 1 stable.
    shaft_pk = grid.shaft[pack.idx]  # (2, M_vis)
    lb0 = shaft_pk[0] == 1
    la0 = shaft_pk[0] == 0
    x2 = x.clone()
    x2[0, lb0] += torch.randn_like(x2[0, lb0]) * 3.0
    o2 = enc.forward_flat_pack(x2, pack)
    iso = (
        torch.allclose(out[0, la0], o2[0, la0], atol=1e-5)
        and not torch.allclose(out[0, lb0], o2[0, lb0], atol=1e-4)
        and torch.allclose(out[1], o2[1], atol=1e-5)
    )
    ok = shape_ok and iso
    print(f"[check] visible-pack encoder: shape={tuple(out.shape)} static; block-diagonal "
          f"(clip+shaft) iso={iso} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_fed_from_stem_runs_finite_and_grads_reach_stem() -> None:
    torch.manual_seed(2)
    sc, geom, grid = _session()
    pid = sc.parcel_id[grid.contact]
    stem = PerBandStem(d_model=D_ENC)
    enc = build_encoder(n_parcels=8)  # train mode ⇒ grads

    # band inputs on the shared 32 Hz clock at T32 = n_time = 16, bins 7/6/7, 5 contacts.
    bands = (torch.randn(2, 5, 7, T), torch.randn(2, 5, 6, T), torch.randn(2, 5, 7, T))
    tokens, _ = stem(bands)  # SLOW (2,5,2,d), MID (2,5,8,d), HGA (2,5,16,d)
    x_flat = pack_band_tokens(tokens, grid)  # (2, total, d)
    assert x_flat.shape == (2, grid.total, D_ENC)

    out = enc.forward_flat(x_flat, grid, pid)
    out.sum().backward()
    finite = bool(torch.isfinite(out).all())
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in stem.projs.parameters())
    ok = finite and grad_ok
    print(f"[check] stem→pack→forward_flat finite={finite}, grads reach stem projs={grad_ok} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok
