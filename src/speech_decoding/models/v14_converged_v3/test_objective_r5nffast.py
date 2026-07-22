"""v3r5nffast (NoFusionStem decimate=4) — the 16 Hz-token OFAT of v3r5nf.

The ONLY change vs v3r5nf is the stem's first conv stride (1→2), so the net decimation is 4
(64→16 Hz) instead of 2 (64→32 Hz): half the time-tokens per contact. Every downstream shape
that keys off ``decimate`` — the token count, the RoPE lattice stride, the per-stream MAE head
widths (DEC·bins) and the raw-frame target gather — must track it, while the 32 Hz path
(decimate=2) stays byte-identical (guarded by test_objective_r5nf.py, unchanged).
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import sample_masks_r5nf
from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective
from speech_decoding.models.v14_converged_v3.pack_r4 import build_r5nf_grid
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar
from speech_decoding.models.v14_converged_v3.stem import (
    NOFUSION_BINS,
    NoFusionStem,
    nf_token_geometry,
)
from speech_decoding.models.v14_converged_v3.towers import PRED_D_MODEL

T32 = 16  # the 32 Hz clip clock (clock_length_32hz).
L = 2 * T32  # 64 Hz frames on a clip = 2·T32 = 32.
DEC = 4  # fast net decimation.
N_TOK = L // DEC  # 16 Hz tokens per contact per stream = 8.
STRIDE = DEC // 2  # RoPE lattice stride = 2.
B = 2
N = 5
F_HGA = DEC * NOFUSION_BINS[0]  # 16
F_LFS = DEC * NOFUSION_BINS[1]  # 4


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        parcel_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    return sc, build_l1_geometry(sc)


def _streams(seed: int = 0):
    """(HGA (B,N,4,L), LFS (B,N,1,L)) at 64 Hz frames (L = 2·T32)."""
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(B, N, NOFUSION_BINS[0], L, generator=g),
        torch.randn(B, N, NOFUSION_BINS[1], L, generator=g),
    ]


def _fast(**kw) -> V3JepaObjective:
    return V3JepaObjective(n_parcels=8, no_fusion=True, mae=True, nf_decimate=DEC, **kw)


# --------------------------------------------------------------------------- #
def test_nf_token_geometry_maps_clock_to_tokens_and_stride() -> None:
    # decimate 2 (v3r5nf): 32 Hz tokens, stride 1 (byte-identical). 4 (fast): 16 Hz, stride 2.
    assert nf_token_geometry(T32, decimate=2) == (T32, 1)
    assert nf_token_geometry(T32, decimate=4) == (T32 // 2, 2)
    with pytest.raises(ValueError, match="decimate must be 2 or 4"):
        nf_token_geometry(T32, decimate=3)
    with pytest.raises(ValueError, match="not divisible"):
        nf_token_geometry(3, decimate=4)  # L=6, 6%4≠0
    print(f"[check] OK nf_token_geometry: dec2=({T32},1) dec4=({T32 // 2},2), guards fire")


def test_fast_stem_emits_half_the_tokens() -> None:
    # net stride = (decimate//2)·2 = decimate ⇒ token count = L//decimate. Fast halves v3r5nf.
    hga, lfs = _streams(seed=1)
    fast = NoFusionStem(256, decimate=DEC)
    (hga_tok, lfs_tok), (pos_h, pos_l) = fast(fast_bands := [hga, lfs])
    slow = NoFusionStem(256, decimate=2)
    (hga_tok2, _), _ = slow(fast_bands)
    ok = (
        hga_tok.shape == (B, N, N_TOK, 256)
        and lfs_tok.shape == (B, N, N_TOK, 256)
        and pos_h.shape == (N_TOK,)
        and pos_l.shape == (N_TOK,)
        and hga_tok2.shape[2] == L // 2  # decimate 2 ⇒ 32 Hz = 2× the fast token count
        and fast.decimate == DEC
    )
    print(f"[check] fast stem tokens={tuple(hga_tok.shape)} (16 Hz, {N_TOK}/contact) vs "
          f"dec2 {hga_tok2.shape[2]} (32 Hz) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_fast_stem_asserts_frame_count_divisible_by_decimate() -> None:
    bad = [torch.randn(B, N, NOFUSION_BINS[0], L + 2), torch.randn(B, N, NOFUSION_BINS[1], L + 2)]
    # L+2 = 34, 34 % 4 ≠ 0 ⇒ the stem must reject it loud.
    with pytest.raises(ValueError, match="divisible by decimate"):
        NoFusionStem(256, decimate=DEC)(bad)
    print("[check] OK fast stem rejects 64 Hz frame count not divisible by decimate")


def test_fast_grid_places_tokens_on_the_32hz_lattice() -> None:
    sc, geom = _session()
    grid = build_r5nf_grid(geom, n_time=N_TOK, time_stride=STRIDE)
    # two bands, each N_TOK tokens; bandpos 0..N_TOK-1 (mask index); time_pos = bandpos·2 (RoPE).
    n_valid = int(geom.valid.sum())
    ok = (
        grid.band_lengths == (N_TOK, N_TOK)
        and grid.k_full == 2 * N_TOK
        and grid.total == n_valid * 2 * N_TOK
        and int(grid.bandpos.max()) == N_TOK - 1
        and int(grid.time_pos.max()) == (N_TOK - 1) * STRIDE  # 14, on the 32 Hz lattice
        and torch.equal(grid.time_pos, grid.bandpos * STRIDE)
    )
    print(f"[check] fast grid: bandpos.max={int(grid.bandpos.max())} (mask idx) "
          f"time_pos.max={int(grid.time_pos.max())} (RoPE lattice, stride {STRIDE}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_fast_heads_scale_with_decimate() -> None:
    obj = _fast()
    ok = (
        obj.nf_decimate == DEC
        and obj.mae_head_hga.out_features == F_HGA  # DEC·4 = 16
        and obj.mae_head_lfs.out_features == F_LFS  # DEC·1 = 4
        and isinstance(obj.online.stem, NoFusionStem)
        and obj.online.stem.decimate == DEC
    )
    print(f"[check] fast heads hga={obj.mae_head_hga.out_features} lfs={obj.mae_head_lfs.out_features} "
          f"(=DEC·bins {F_HGA}/{F_LFS}) stem.decimate={obj.online.stem.decimate} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_fast_target_gathers_all_four_frames_per_token() -> None:
    # token (contact c, bandpos p) → its OWN stream's DEC=4 raw frames {4p..4p+3}; over p=0..7
    # this tiles all L=32 frames exactly. HGA → 16 real feats, LFS → 4 real + 12 zero-pad.
    sc, geom = _session()
    obj = _fast()
    grid = build_r5nf_grid(geom, n_time=N_TOK, time_stride=STRIDE)
    hga, lfs = _streams(seed=3)
    target, feat_valid, feat_count = obj._mae_gather_target_r5nf([hga, lfs], grid)
    assert target.shape == (B, grid.total, F_HGA)
    max_dev = 0.0
    for t in range(grid.total):
        c = int(grid.contact[t]); p = int(grid.bandpos[t]); band = int(grid.band[t])
        src = hga if band == 0 else lfs
        nreal = F_HGA if band == 0 else F_LFS
        ref = torch.stack([src[:, c, :, DEC * p + f] for f in range(DEC)], dim=1).reshape(B, nreal)
        if nreal < F_HGA:
            ref = torch.nn.functional.pad(ref, (0, F_HGA - nreal))
        assert int(feat_count[t]) == nreal
        assert int(feat_valid[t].sum()) == nreal
        max_dev = max(max_dev, (target[:, t, :] - ref).abs().max().item())
    exact = max_dev < 1e-6
    print(f"[check] fast target == own 4 frames of own stream, all {grid.total} tokens: "
          f"max|Δ|={max_dev:.2e} feat_count 16/4 {'OK' if exact else 'VIOLATED'}")
    assert exact


def test_fast_forward_finite_and_grads_reach_stems_and_heads() -> None:
    sc, geom = _session()
    obj = _fast()
    obj.train()
    g = torch.Generator().manual_seed(7)
    masks = sample_masks_r5nf(geom, N, n_time=N_TOK, n_rows=B, generator=g)  # masks at 16 Hz grid
    out = obj(_streams(), geom, sc.parcel_id, masks)
    loss_val = out.loss.detach()
    out.loss.backward()
    finite = bool(torch.isfinite(loss_val)) and loss_val.ndim == 0 and float(loss_val) >= 0.0
    checks = {
        "stem.hga_stem": obj.online.stem.hga_stem,
        "stem.lfs_stem": obj.online.stem.lfs_stem,
        "encoder": obj.online.encoder,
        "predictor": obj.predictor,
        "mae_head_hga": obj.mae_head_hga,
        "mae_head_lfs": obj.mae_head_lfs,
    }
    grad_ok = {
        name: all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        for name, m in checks.items()
    }
    ok = finite and all(grad_ok.values())
    print(f"[check] fast loss={float(loss_val):.4f} finite={finite}; grads {grad_ok} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok
