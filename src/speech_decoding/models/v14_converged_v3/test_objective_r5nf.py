"""v3r5nf (no-fusion) objective arm — MAE-only, per-stream raw-frame reconstruction.

v3r5nf splits the r5 early-fused token into TWO token streams (NoFusionStem) masked
INDEPENDENTLY, and reconstructs each masked token's OWN raw 64 Hz frames with its OWN head
(HGA 8 feats, LFS 2). Loss POOLS per-channel MSE over all masked tokens both streams ⇒
HGA:LFS = 4:1 by construction. Invariants named + asserted + printed.
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
    NOFUSION_DECIMATE,
    NoFusionStem,
)
from speech_decoding.models.v14_converged_v3.towers import PRED_D_MODEL

T = 16  # 32 Hz tokens per contact.
L = NOFUSION_DECIMATE * T  # 64 Hz frames = 2T = 32.
B = 2
N = 5
DEC = NOFUSION_DECIMATE
F_HGA = DEC * NOFUSION_BINS[0]  # 8
F_LFS = DEC * NOFUSION_BINS[1]  # 2


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        parcel_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    return sc, build_l1_geometry(sc)


def _streams(seed: int = 0):
    """(HGA (B,N,4,L), LFS (B,N,1,L)) at 64 Hz frames (L = 2T)."""
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(B, N, NOFUSION_BINS[0], L, generator=g),
        torch.randn(B, N, NOFUSION_BINS[1], L, generator=g),
    ]


def _masks(geom, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return sample_masks_r5nf(geom, N, n_time=T, n_rows=B, generator=g)


def _nf(**kw) -> V3JepaObjective:
    return V3JepaObjective(n_parcels=8, no_fusion=True, mae=True, **kw)


# --------------------------------------------------------------------------- #
def test_nf_is_mae_only() -> None:
    with pytest.raises(ValueError, match="MAE-only"):
        V3JepaObjective(n_parcels=8, no_fusion=True, mae=False)
    print("[check] OK no_fusion + not mae raises 'v3r5nf is MAE-only'")


def test_nf_mutually_exclusive_with_other_frontends() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        V3JepaObjective(n_parcels=8, no_fusion=True, early_fusion=True, mae=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        V3JepaObjective(n_parcels=8, no_fusion=True, native_fine_hga=True, mae=True)
    print("[check] OK no_fusion ⊥ early_fusion / native_fine_hga")


def test_nf_stem_two_heads_no_teacher() -> None:
    obj = _nf()
    is_nf = isinstance(obj.online.stem, NoFusionStem)
    no_teacher = obj.teacher is None
    two_heads = (
        obj.mae_head_hga.out_features == F_HGA
        and obj.mae_head_lfs.out_features == F_LFS
        and obj.mae_head_r5 is None
        and obj.mae_heads is None
    )
    ok = is_nf and no_teacher and two_heads
    print(f"[check] NoFusionStem ({is_nf}), teacher=None ({no_teacher}), "
          f"hga_head={obj.mae_head_hga.out_features} lfs_head={obj.mae_head_lfs.out_features} "
          f"({two_heads}) {'OK' if ok else 'VIOLATED'}")
    assert ok
    assert obj.update_teacher() == 0.0  # no-op every opt-step


def test_final_heads_are_independent_weights_per_stream() -> None:
    # (d) each HGA token → mae_head_hga (→8), each LFS token → mae_head_lfs (→2), selected by
    # grid.band; the two heads are DISTINCT nn.Linear with NO weight sharing.
    obj = _nf()
    assert isinstance(obj.mae_head_hga, torch.nn.Linear)
    assert isinstance(obj.mae_head_lfs, torch.nn.Linear)
    assert obj.mae_head_hga.weight.data_ptr() != obj.mae_head_lfs.weight.data_ptr()
    assert obj.mae_head_hga.bias.data_ptr() != obj.mae_head_lfs.bias.data_ptr()
    print("[check] OK mae_head_hga ⊥ mae_head_lfs (distinct Linear, no shared storage)")


def test_gather_target_pulls_own_frames_and_pads_lfs() -> None:
    # THE alignment invariant: token t (contact c, bandpos p) → its OWN stream's 2 raw frames
    # {DEC·p, DEC·p+1}. HGA token → 8 real values; LFS token → 2 real + 6 zero-pad.
    sc, geom = _session()
    obj = _nf()
    grid = build_r5nf_grid(geom, n_time=T)
    hga, lfs = _streams(seed=3)
    target, feat_valid, feat_count = obj._mae_gather_target_r5nf([hga, lfs], grid)
    assert target.shape == (B, grid.total, F_HGA)

    max_dev = 0.0
    for t in range(grid.total):
        c = int(grid.contact[t]); p = int(grid.bandpos[t]); band = int(grid.band[t])
        if band == 0:  # HGA
            ref = torch.stack(
                [hga[:, c, :, DEC * p + f] for f in range(DEC)], dim=1
            ).reshape(B, F_HGA)  # [f0 ch0..3, f1 ch0..3]
            assert int(feat_count[t]) == F_HGA
            assert bool(feat_valid[t].all())
        else:  # LFS
            two = torch.stack(
                [lfs[:, c, :, DEC * p + f] for f in range(DEC)], dim=1
            ).reshape(B, F_LFS)  # [f0 ch0, f1 ch0]
            ref = torch.nn.functional.pad(two, (0, F_HGA - F_LFS))  # 6 zero-pad
            assert int(feat_count[t]) == F_LFS
            assert feat_valid[t].tolist() == [True, True, False, False, False, False, False, False]
        max_dev = max(max_dev, (target[:, t, :] - ref).abs().max().item())
    exact = max_dev < 1e-6
    print(f"[check] r5nf target == own 2 frames of own stream for all {grid.total} tokens: "
          f"max|Δ|={max_dev:.2e}, feat_count 8/2, LFS pad zeroed {'OK' if exact else 'VIOLATED'}")
    assert exact


def _all_masked_loss(obj, hga_bias: float, lfs_bias: float):
    """Force HGA pred-target == hga_bias on each of its 8 feats and LFS == lfs_bias on each of
    its 2 (zero head weights + bias, zero bands ⇒ target 0), mask ALL tokens (equal per-stream
    masked counts). Returns (loss, mae_hga_tap, mae_lfs_tap)."""
    sc, geom = _session()
    with torch.no_grad():
        obj.mae_head_hga.weight.zero_(); obj.mae_head_hga.bias.fill_(hga_bias)
        obj.mae_head_lfs.weight.zero_(); obj.mae_head_lfs.bias.fill_(lfs_bias)
    grid = build_r5nf_grid(geom, n_time=T)
    bands = [torch.zeros(B, N, NOFUSION_BINS[0], L), torch.zeros(B, N, NOFUSION_BINS[1], L)]
    h = torch.zeros(B, grid.total, PRED_D_MODEL)  # irrelevant (weights zeroed)
    in_loss = torch.ones(B, grid.total, dtype=torch.bool)  # mask ALL ⇒ equal HGA/LFS counts
    enc_taps = {12: torch.zeros(1, 256)}
    out = obj._mae_output_r5nf(bands, grid, h, in_loss, enc_taps=enc_taps, collect_taps=True)
    return float(out.loss.detach()), float(out.taps["mae_hga"]), float(out.taps["mae_lfs"])


def test_taps_emit_per_stream_recon_stats() -> None:
    # the nf tap block must emit per-stream explained-var + var-ratio (collapse detector), not
    # just the mean-MSE split — the health signal that was missing on the nf arm.
    import math

    sc, geom = _session()
    obj = _nf()
    grid = build_r5nf_grid(geom, n_time=T)
    bands = [torch.randn(B, N, NOFUSION_BINS[0], L), torch.randn(B, N, NOFUSION_BINS[1], L)]
    h = torch.randn(B, grid.total, PRED_D_MODEL)
    in_loss = torch.ones(B, grid.total, dtype=torch.bool)
    out = obj._mae_output_r5nf(
        bands, grid, h, in_loss, enc_taps={12: torch.zeros(1, 256)}, collect_taps=True
    )
    assert out.taps is not None
    keys = ("jepa_hga_explained_var", "jepa_hga_pred_target_var_ratio",
            "jepa_lfs_explained_var", "jepa_lfs_pred_target_var_ratio",
            "mae_hga", "mae_lfs")
    ok = all(k in out.taps and math.isfinite(float(out.taps[k])) for k in keys)
    print(f"[check] nf taps carry per-stream EV+var-ratio+MSE, all finite → {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_default_stream_weight_is_equal() -> None:
    assert _nf().mae_stream_weight == "equal"
    with pytest.raises(ValueError, match="equal|pooled"):
        _nf(mae_stream_weight="bogus")
    print("[check] OK default mae_stream_weight='equal'; bad value raises")


def test_pooled_loss_is_four_to_one_per_channel() -> None:
    # POOLED sums per-channel SE over both streams / total valid channels ⇒ HGA weighted 4× per
    # token (8 feats vs 2). HGA err 1 (mean-MSE 1), LFS err 2 (mean-MSE 4), equal masked counts:
    # pooled = (8·1 + 2·4)/(8 + 2) = 1.6. Per-stream mean-MSE taps are mode-independent (1 / 4).
    loss, hga, lfs = _all_masked_loss(_nf(mae_stream_weight="pooled"), 1.0, 2.0)
    ok = abs(loss - 1.6) < 1e-6 and abs(hga - 1.0) < 1e-6 and abs(lfs - 4.0) < 1e-6
    print(f"[check] pooled 4:1 — loss={loss:.6f} (==1.6=0.8·1+0.2·4), mae_hga={hga:.4f} (1.0) "
          f"mae_lfs={lfs:.4f} (4.0) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_equal_loss_is_fifty_fifty() -> None:
    # EQUAL (the default) averages the two per-stream mean-MSEs ⇒ 0.5·(1 + 4) = 2.5, INVARIANT
    # to the 8-vs-2 feat-count asymmetry that drives pooled's 4:1. Same per-stream errors.
    loss, hga, lfs = _all_masked_loss(_nf(mae_stream_weight="equal"), 1.0, 2.0)
    ok = abs(loss - 2.5) < 1e-6 and abs(hga - 1.0) < 1e-6 and abs(lfs - 4.0) < 1e-6
    print(f"[check] equal 1:1 — loss={loss:.6f} (==2.5=0.5·1+0.5·4), mae_hga={hga:.4f} (1.0) "
          f"mae_lfs={lfs:.4f} (4.0) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_forward_finite_and_grads_reach_both_stems_and_both_heads() -> None:
    sc, geom = _session()
    obj = _nf()
    obj.train()
    out = obj(_streams(), geom, sc.parcel_id, _masks(geom))
    loss_val = out.loss.detach()
    out.loss.backward()
    finite = bool(torch.isfinite(loss_val)) and loss_val.ndim == 0 and float(loss_val) >= 0.0
    checks = {
        "stem.hga_stem": obj.online.stem.hga_stem,
        "stem.lfs_stem": obj.online.stem.lfs_stem,
        "encoder": obj.online.encoder,
        "predictor": obj.predictor,
        "enc_to_pred": obj.enc_to_pred,
        "mae_head_hga": obj.mae_head_hga,
        "mae_head_lfs": obj.mae_head_lfs,
    }
    grad_ok = {
        name: all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        for name, m in checks.items()
    }
    # band_type_emb (a Parameter, inside stem.parameters()) also gets a grad.
    bt_grad = obj.online.stem.band_type_emb.grad is not None
    mask_grad = obj.mask_token.grad is not None and torch.isfinite(obj.mask_token.grad).all()
    ok = finite and all(grad_ok.values()) and bt_grad and mask_grad
    print(f"[check] nf loss={float(loss_val):.4f} finite={finite}; grads {grad_ok}; "
          f"band_type_emb grad={bt_grad}; mask_token grad={mask_grad} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_accept_the_bleed_scores_every_masked_token() -> None:
    sc, geom = _session()
    grid = build_r5nf_grid(geom, n_time=T)
    masks = _masks(geom)
    from speech_decoding.models.v14_converged_v3.pack_r4 import token_flags_r5nf

    masked, in_loss = token_flags_r5nf(grid, masks)
    obj = _nf()
    out = obj(_streams(), geom, sc.parcel_id, masks)
    scored = float(out.n_masked)
    full = float(masked.sum())
    ok = torch.equal(masked, in_loss) and scored == full and full > 0
    print(f"[check] accept-the-bleed: scored={scored} == masked={full} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


# ── mask-query band embedding (predictor space) ──────────────────────────────────
def test_mask_band_emb_shape_and_predictor_space() -> None:
    obj = _nf()
    assert obj.mask_band_emb.shape == (2, PRED_D_MODEL)  # 2 streams, d_pred (NOT d_model=256)
    # SEPARATE param from the stem's 256-dim band_type_emb.
    assert obj.mask_band_emb.data_ptr() != obj.online.stem.band_type_emb.data_ptr()
    assert obj.mask_band_emb.shape[-1] != obj.online.stem.band_type_emb.shape[-1]
    print(f"[check] OK mask_band_emb {tuple(obj.mask_band_emb.shape)} in predictor space, "
          f"separate from stem band_type_emb {tuple(obj.online.stem.band_type_emb.shape)}")


def _capture_pred_in(obj, bands, geom, parcel_id, masks):
    """Spy on predictor.forward_flat to grab the pred_in it receives (post band-emb inject)."""
    captured = {}
    orig = obj.predictor.forward_flat

    def spy(pred_in, grid, parcel_packed):
        captured["pred_in"] = pred_in.detach().clone()
        return orig(pred_in, grid, parcel_packed)

    obj.predictor.forward_flat = spy
    obj(bands, geom, parcel_id, masks)
    obj.predictor.forward_flat = orig
    return captured["pred_in"]


def test_mask_band_emb_differentiates_colocated_masked_queries() -> None:
    # The band identity rides ALL predictor tokens (symmetric with parcel, MAE/I-JEPA convention).
    # Two co-located tokens (same contact + pos, band 0 vs 1) must get DIFFERENT pred_in rows,
    # AND toggling mask_band_emb must move BOTH visible and masked slots (it's on every token).
    from speech_decoding.models.v14_converged_v3.masking import V3MasksR5NF

    sc, geom = _session()
    obj = _nf().eval()
    grid = build_r5nf_grid(geom, n_time=T)
    n = int(geom.valid.sum())
    S = geom.n_shafts
    # mask contact 0 in BOTH streams ⇒ all its HGA+LFS tokens masked, co-located pairs exist.
    zeros_c = torch.zeros(B, n, dtype=torch.bool)
    zeros_t = torch.zeros(B, S, T, dtype=torch.bool)
    hga_c = zeros_c.clone(); hga_c[:, 0] = True
    lfs_c = zeros_c.clone(); lfs_c[:, 0] = True
    masks = V3MasksR5NF(
        hga_contact_mask=hga_c, hga_temporal_mask=zeros_t.clone(),
        lfs_contact_mask=lfs_c, lfs_temporal_mask=zeros_t.clone(),
    )
    from speech_decoding.models.v14_converged_v3.pack_r4 import token_flags_r5nf

    masked, _ = token_flags_r5nf(grid, masks)
    bands = _streams(seed=4)

    pin1 = _capture_pred_in(obj, bands, geom, sc.parcel_id, masks)  # with band emb
    with torch.no_grad():
        obj.mask_band_emb.zero_()
    pin2 = _capture_pred_in(obj, bands, geom, sc.parcel_id, masks)  # band emb zeroed

    # co-located masked pair at contact 0, bandpos 0: HGA token vs LFS token.
    hga_tok = ((grid.contact == 0) & (grid.band == 0) & (grid.bandpos == 0)).nonzero().item()
    lfs_tok = ((grid.contact == 0) & (grid.band == 1) & (grid.bandpos == 0)).nonzero().item()
    assert bool(masked[0, hga_tok]) and bool(masked[0, lfs_tok])
    colocated_differ = not torch.allclose(pin1[:, hga_tok], pin1[:, lfs_tok])
    # band emb is on ALL tokens ⇒ toggling it moves BOTH visible and masked slots.
    vis = ~masked
    visible_changed = not torch.allclose(pin1[vis], pin2[vis])
    masked_changed = not torch.allclose(pin1[masked], pin2[masked])
    ok = colocated_differ and visible_changed and masked_changed
    print(f"[check] band emb rides all tokens: co-located HGA/LFS differ ({colocated_differ}); "
          f"visible moved ({visible_changed}); masked moved ({masked_changed}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_mask_band_emb_receives_gradient() -> None:
    sc, geom = _session()
    obj = _nf()
    obj.train()
    out = obj(_streams(), geom, sc.parcel_id, _masks(geom))
    out.loss.backward()
    grad_ok = obj.mask_band_emb.grad is not None and torch.isfinite(obj.mask_band_emb.grad).all()
    print(f"[check] mask_band_emb grad flows ({grad_ok}) {'OK' if grad_ok else 'VIOLATED'}")
    assert grad_ok
