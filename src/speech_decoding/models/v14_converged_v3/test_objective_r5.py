"""r5 (Chang 2-stream early-fusion) objective arm — JEPA + MAE on the single-band grid.

r5 swaps the FRONTEND (EarlyFusionStem: HGA⊕LFS → one 32 Hz token stream) and the
SCORING (accept-the-bleed: ``in_loss == masked``, no margin gate); the SSL objective stays
swappable (JEPA-latent or MAE-recon). Invariants a silent miscompute would violate, named +
asserted + printed (feedback-build-the-invariant-into-the-probe):

  1. Stem = EarlyFusionStem; grid single-band stride-1; secondary CUT (perceiver None).
  2. JEPA arm: EMA teacher present, forward finite, grads reach stem.fuse / encoder /
     predictor / enc_to_pred / pred_to_target / mask_token.
  3. MAE arm: NO teacher; head = ONE Linear(d_pred → DEC·C=10); mae_heads None; single loss;
     grads reach mae_head_r5.
  4. r5 MAE TARGET ALIGNMENT: the static gather maps flat token t to its OWN contact's 2 raw
     64 Hz frames {DEC·p, DEC·p+1} × 5 channels, ordered [f0 ch0..4, f1 ch0..4].
  5. ACCEPT THE BLEED: n_masked == the full masked-token count (no gate drops any).
  6. LEAK-SAFETY: masked tokens dropped from the encoder ⇒ online latent invariant to the
     input frames those masked tokens pool from.
  7. early_fusion ⊥ native_fine_hga (mutually exclusive).
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import sample_masks_r5
from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective
from speech_decoding.models.v14_converged_v3.pack_r4 import (
    build_r5_grid,
    build_visible_pack,
    token_flags_r5,
)
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar
from speech_decoding.models.v14_converged_v3.stem import (
    EARLY_FUSION_CHANNELS,
    EARLY_FUSION_DECIMATE,
    EarlyFusionStem,
)

T = 16  # 32 Hz tokens per contact.
L = EARLY_FUSION_DECIMATE * T  # 64 Hz frames = 2T = 32.
B = 2
N = 5
DEC_C = EARLY_FUSION_DECIMATE * EARLY_FUSION_CHANNELS  # 10


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
        torch.randn(B, N, 4, L, generator=g),
        torch.randn(B, N, 1, L, generator=g),
    ]


def _masks(geom, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return sample_masks_r5(geom, N, n_time=T, n_rows=B, generator=g)


def _jepa(**kw) -> V3JepaObjective:
    return V3JepaObjective(n_parcels=8, early_fusion=True, **kw)


def _mae(**kw) -> V3JepaObjective:
    return V3JepaObjective(n_parcels=8, early_fusion=True, mae=True, **kw)


# --------------------------------------------------------------------------- #
def test_stem_is_early_fusion_grid_single_band_secondary_cut() -> None:
    obj = _jepa()
    is_ef = isinstance(obj.online.stem, EarlyFusionStem)
    no_perc = obj.perceiver is None  # r5 CUTS the secondary
    has_teacher = obj.teacher is not None  # JEPA arm keeps the EMA teacher
    ok = is_ef and no_perc and has_teacher
    print(f"[check] EarlyFusionStem ({is_ef}), perceiver=None ({no_perc}), "
          f"teacher present ({has_teacher}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_jepa_forward_runs_and_grads_reach_every_trainable_module() -> None:
    sc, geom = _session()
    obj = _jepa()
    obj.train()
    out = obj(_streams(), geom, sc.parcel_id, _masks(geom))
    loss_val = out.loss.detach()
    out.loss.backward()

    finite = bool(torch.isfinite(loss_val)) and loss_val.ndim == 0 and float(loss_val) >= 0.0
    checks = {
        "stem.fuse": obj.online.stem.fuse,
        "encoder": obj.online.encoder,
        "predictor": obj.predictor,
        "enc_to_pred": obj.enc_to_pred,
        "pred_to_target": obj.pred_to_target,
    }
    grad_ok = {
        name: all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        for name, m in checks.items()
    }
    mask_grad = obj.mask_token.grad is not None and torch.isfinite(obj.mask_token.grad).all()
    ok = finite and all(grad_ok.values()) and mask_grad
    print(f"[check] jepa loss={float(loss_val):.4f} finite={finite}; grads {grad_ok}; "
          f"mask_token grad={mask_grad} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_mae_arm_head_shape_no_teacher_single_loss() -> None:
    sc, geom = _session()
    obj = _mae()
    obj.train()
    no_teacher = obj.teacher is None
    no_heads = obj.mae_heads is None
    head_shape = obj.mae_head_r5.out_features == DEC_C
    out = obj(_streams(), geom, sc.parcel_id, _masks(geom))
    loss_val = out.loss.detach()
    out.loss.backward()
    single = out.jepa_loss is None and out.nll_loss is None and out.ctx_loss is None
    finite = bool(torch.isfinite(loss_val)) and float(loss_val) >= 0.0
    grad = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in obj.mae_head_r5.parameters()
    )
    noop = obj.update_teacher() == 0.0
    ok = no_teacher and no_heads and head_shape and single and finite and grad and noop
    print(f"[check] MAE: teacher=None ({no_teacher}), mae_heads=None ({no_heads}), "
          f"head out={obj.mae_head_r5.out_features} ({head_shape}), single-loss ({single}), "
          f"loss={float(loss_val):.4f} finite ({finite}), head grad ({grad}), "
          f"update no-op ({noop}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_mae_target_is_own_contacts_two_frames_five_channels() -> None:
    # THE r5 alignment invariant: token t (contact c, time_pos p) → the DEC·C fused input
    # values from frames {DEC·p, DEC·p+1}, channel order [f0 ch0..4, f1 ch0..4].
    sc, geom = _session()
    obj = _mae()
    grid = build_r5_grid(geom, n_time=T)
    hga, lfs = _streams(seed=3)
    fused = torch.cat([hga, lfs], dim=2)  # (B, N, 5, L)
    tgt = obj._mae_gather_target_r5([hga, lfs], grid)  # (B, total, DEC·C)
    assert tgt.shape == (B, grid.total, DEC_C)

    max_dev = 0.0
    for t in range(grid.total):
        c = int(grid.contact[t]); p = int(grid.time_pos[t])
        assert p == int(grid.bandpos[t])  # single band, stride 1
        ref = torch.stack(
            [fused[:, c, :, EARLY_FUSION_DECIMATE * p + f] for f in range(EARLY_FUSION_DECIMATE)],
            dim=1,
        ).reshape(B, DEC_C)  # (B, DEC·C) [f0 ch0..4, f1 ch0..4]
        max_dev = max(max_dev, (tgt[:, t, :] - ref).abs().max().item())
    exact = max_dev < 1e-6
    print(f"[check] r5 target == own 2 frames × 5 ch for all {grid.total} tokens: "
          f"max|Δ|={max_dev:.2e} {'OK' if exact else 'VIOLATED'}")
    assert exact


def test_accept_the_bleed_scores_every_masked_token() -> None:
    sc, geom = _session()
    grid = build_r5_grid(geom, n_time=T)
    masks = _masks(geom)
    masked, in_loss = token_flags_r5(grid, masks)
    obj = _jepa()
    out = obj(_streams(), geom, sc.parcel_id, masks)
    # n_masked (the scored count) == the full masked count — no margin gate drops any.
    scored = float(out.n_masked)
    full = float(masked.sum())
    ok = torch.equal(masked, in_loss) and scored == full and full > 0
    print(f"[check] accept-the-bleed: scored={scored} == masked={full} "
          f"(in_loss==masked) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_leak_safe_spatially_masked_contact_is_fully_dropped() -> None:
    # r5's SPATIAL leak-safety (the complete drop): a spatially-masked contact has ALL its
    # tokens masked ⇒ dropped from the visible pack ⇒ the online latent is bit-identical
    # however violently that contact's input frames are corrupted. (Temporal masking does NOT
    # give this — the EarlyFusionStem convolves, so a masked token's frames legitimately bleed
    # into adjacent VISIBLE tokens; that accepted frontend bleed is the r5 design, NOT a leak.)
    sc, geom = _session()
    obj = _mae().eval()
    grid = build_r5_grid(geom, n_time=T)
    parcel_packed = sc.parcel_id[grid.contact]
    masks = _masks(geom)
    masked, _ = token_flags_r5(grid, masks)
    pack = build_visible_pack(grid, masked, parcel_packed)
    assert masks.contact_mask.any()  # need at least one spatially-masked contact to test

    hga, lfs = _streams()
    z1 = obj.online([hga, lfs], grid, parcel_packed, pack=pack)
    hga2, lfs2 = hga.clone(), lfs.clone()
    for clip in range(B):
        for c in torch.nonzero(masks.contact_mask[clip]).flatten().tolist():
            hga2[clip, c] += 5.0  # corrupt EVERY frame of the dropped contact
            lfs2[clip, c] += 5.0
    z2 = obj.online([hga2, lfs2], grid, parcel_packed, pack=pack)
    leak_safe = torch.allclose(z1, z2, atol=1e-6)
    print(f"[check] online latent invariant to spatially-dropped contacts={leak_safe} "
          f"(max|Δ|={(z1 - z2).abs().max().item():.2e}) {'OK' if leak_safe else 'VIOLATED'}")
    assert leak_safe


def test_early_fusion_and_native_fine_hga_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        V3JepaObjective(n_parcels=8, early_fusion=True, native_fine_hga=True)
    print("[check] OK early_fusion ⊥ native_fine_hga raises")
