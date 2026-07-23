"""r6 objective arm — MAE-only, 3-band native-rate |STFT|-bin reconstruction, RAW target.

r6 = the winning r4-MAE frontend (3 native-rate |STFT| bands, NativePerBandStem, per-band recon
heads) on r5's shaft-batched regime, with the r5 "no norm_pix" (raw target) decision applied to
r4's spectral bins. The gather is r4's SPECTRAL family re-indexed for ragged native T (native
frame == r4's decimated frame). Invariants named + asserted + printed.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import sample_masks_r6
from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective
from speech_decoding.models.v14_converged_v3.pack_r4 import build_r4_grid
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar
from speech_decoding.models.v14_converged_v3.stem import (
    NativePerBandStem,
    PER_BAND_SPECS,
    PerBandStem,
)
from speech_decoding.models.v14_converged_v3.towers import PRED_D_MODEL

T = 32  # 32 Hz clock; native bands SLOW T/8=4, MID T/2=16, HGA T=32.
B = 2
N = 5
F_MAX = max(nb for nb, _ in PER_BAND_SPECS)  # 7
BINS = tuple(nb for nb, _ in PER_BAND_SPECS)  # (7, 6, 7)
STRIDES = tuple(st for _, st in PER_BAND_SPECS)  # (8, 2, 1)
NATIVE_LEN = tuple(T // st for st in STRIDES)  # (4, 16, 32)


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        parcel_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    return sc, build_l1_geometry(sc)


def _native_bands(seed: int = 0):
    """(SLOW (B,N,7,4), MID (B,N,6,16), HGA (B,N,7,32)) at native rates."""
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(B, N, BINS[b], NATIVE_LEN[b], generator=g) for b in range(3)
    ]


def _masks(geom, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return sample_masks_r6(geom, N, n_time=T, n_rows=B, generator=g)


def _r6(**kw) -> V3JepaObjective:
    return V3JepaObjective(n_parcels=8, native_perband=True, mae=True, **kw)


# --------------------------------------------------------------------------- #
def test_r6_is_mae_only() -> None:
    with pytest.raises(ValueError, match="MAE-only"):
        V3JepaObjective(n_parcels=8, native_perband=True, mae=False)
    print("[check] OK native_perband + not mae raises 'r6 (native_perband) is MAE-only'")


def test_r6_mutually_exclusive_with_other_frontends() -> None:
    for other in ("early_fusion", "no_fusion", "native_fine_hga"):
        with pytest.raises(ValueError, match="mutually exclusive"):
            V3JepaObjective(n_parcels=8, native_perband=True, mae=True, **{other: True})
    print("[check] OK native_perband ⊥ early_fusion / no_fusion / native_fine_hga")


def test_r6_stem_three_heads_no_teacher() -> None:
    obj = _r6()
    is_native = isinstance(obj.online.stem, NativePerBandStem)
    no_teacher = obj.teacher is None
    heads_ok = (
        obj.mae_heads is not None
        and [h.out_features for h in obj.mae_heads] == list(BINS)  # 7/6/7 per-band
        and obj.mae_head_r5 is None
    )
    ok = is_native and no_teacher and heads_ok
    print(f"[check] NativePerBandStem ({is_native}), teacher=None ({no_teacher}), "
          f"heads={[h.out_features for h in obj.mae_heads]} ({heads_ok}) {'OK' if ok else 'VIOLATED'}")
    assert ok
    assert obj.update_teacher() == 0.0  # no-op every opt-step (no EMA teacher on MAE)


def test_gather_pulls_own_native_frame_per_token() -> None:
    # THE alignment invariant: token (band b, contact c, bandpos p) → its band's OWN native frame p
    # (NO ·stride — native), padded to F_MAX. feat_count = 7/6/7 per band.
    sc, geom = _session()
    obj = _r6()
    grid = build_r4_grid(geom, n_time=T)
    bands = _native_bands(seed=3)
    target, feat_valid, feat_count = obj._mae_gather_target_r6(bands, grid)
    assert target.shape == (B, grid.total, F_MAX)
    max_dev = 0.0
    for t in range(grid.total):
        b = int(grid.band[t]); c = int(grid.contact[t]); p = int(grid.bandpos[t])
        ref = bands[b][:, c, :, p]  # (B, F_b) raw native frame
        nb = BINS[b]
        assert int(feat_count[t]) == nb
        assert feat_valid[t][:nb].all() and not feat_valid[t][nb:].any()
        max_dev = max(max_dev, (target[:, t, :nb] - ref).abs().max().item())
        if nb < F_MAX:  # pad slots zeroed
            max_dev = max(max_dev, target[:, t, nb:].abs().max().item())
    exact = max_dev < 1e-6
    print(f"[check] r6 target == own native frame p for all {grid.total} tokens: "
          f"max|Δ|={max_dev:.2e}, feat_count 7/6/7, pad zeroed {'OK' if exact else 'VIOLATED'}")
    assert exact


def test_native_gather_equals_r4_decimated_gather() -> None:
    # THE piece-1 equivalence at the objective: r6's native gather == r4's spectral gather (BEFORE
    # norm_pix) of the equivalent 32 Hz bands decimated ::8/::2/::1. The native bake IS the decimate.
    sc, geom = _session()
    r4 = V3JepaObjective(n_parcels=8, mae=True)  # uniform PerBandStem, r4 spectral gather
    r6 = _r6()
    grid = build_r4_grid(geom, n_time=T)
    g = torch.Generator().manual_seed(9)
    bands32 = [torch.randn(B, N, BINS[b], T, generator=g) for b in range(3)]  # all at 32 Hz
    native = [bands32[b][..., ::STRIDES[b]] for b in range(3)]  # decimate → native (4/16/32)
    t4, v4, c4 = r4._mae_gather_target(bands32, grid)   # r4 raw bins (pre-norm_pix)
    t6, v6, c6 = r6._mae_gather_target_r6(native, grid)  # r6 native bins
    ok = torch.allclose(t4, t6, atol=1e-6) and torch.equal(v4, v6) and torch.equal(c4, c6)
    print(f"[check] r6 native gather == r4 decimated gather (pre-norm_pix): "
          f"max|Δ|={(t4 - t6).abs().max().item():.2e} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_r6_skips_norm_pix_raw_target() -> None:
    # NO norm_pix: fill every bin with a CONSTANT c ⇒ per-token var=0 ⇒ norm_pix would map the
    # target to 0 (loss 0); the RAW target keeps c (loss c² when pred=0). Distinguishes the two.
    sc, geom = _session()
    obj = _r6()
    c = 2.0
    with torch.no_grad():
        for head in obj.mae_heads:  # zero heads ⇒ pred == 0 everywhere
            head.weight.zero_(); head.bias.zero_()
    grid = build_r4_grid(geom, n_time=T)
    bands = [torch.full((B, N, BINS[b], NATIVE_LEN[b]), c) for b in range(3)]
    h = torch.zeros(B, grid.total, PRED_D_MODEL)  # irrelevant (heads zeroed)
    in_loss = torch.ones(B, grid.total, dtype=torch.bool)  # score all tokens
    out = obj._mae_output_r6(bands, grid, h, in_loss, enc_taps=None, collect_taps=False)
    loss = float(out.loss.detach())
    ok = abs(loss - c * c) < 1e-6  # raw ⇒ c²=4 ; norm_pix would give 0
    print(f"[check] no norm_pix: constant bins c={c} ⇒ loss={loss:.6f} (raw c²={c*c}, "
          f"norm_pix would be 0) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_r6_keeps_margin_gate_in_loss_subset_of_masked() -> None:
    # r6 KEEPS the M14 margin gate (unlike r5's accept-the-bleed): in_loss ⊊ masked in general.
    from speech_decoding.models.v14_converged_v3.pack_r4 import token_flags_r6

    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    masks = _masks(geom, seed=5)
    masked, in_loss = token_flags_r6(grid, masks)
    ok = torch.all(in_loss <= masked) and int(in_loss.sum()) < int(masked.sum())
    print(f"[check] margin gate kept: in_loss {int(in_loss.sum())} ⊊ masked {int(masked.sum())} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_count_proportional_band_weighting_hga_dominates() -> None:
    # count-proportional (verbatim r4): the flat masked-mean gives HGA (T tokens) ~8:1 dominance
    # over SLOW (T/8) by SCORED-TOKEN COUNT — no per-band reweight. Verify HGA scored ≫ SLOW scored.
    from speech_decoding.models.v14_converged_v3.pack_r4 import token_flags_r6

    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    _, in_loss = token_flags_r6(grid, _masks(geom, seed=2))
    per_band = {b: int((in_loss[0] & (grid.band == b)).sum()) for b in range(3)}
    ratio = per_band[2] / max(per_band[0], 1)  # HGA / SLOW scored count
    ok = per_band[2] > per_band[1] > per_band[0] and ratio > 3.0
    print(f"[check] count-proportional: scored SLOW={per_band[0]} MID={per_band[1]} "
          f"HGA={per_band[2]} (HGA/SLOW={ratio:.1f}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_forward_finite_and_grads_reach_stem_heads_and_predictor() -> None:
    sc, geom = _session()
    obj = _r6()
    obj.train()
    out = obj(_native_bands(), geom, sc.parcel_id, _masks(geom))
    loss_val = out.loss.detach()
    out.loss.backward()
    finite = bool(torch.isfinite(loss_val)) and loss_val.ndim == 0 and float(loss_val) >= 0.0
    checks = {
        "stem": obj.online.stem,
        "encoder": obj.online.encoder,
        "predictor": obj.predictor,
        "enc_to_pred": obj.enc_to_pred,
        "mae_heads": obj.mae_heads,
    }
    grad_ok = {
        name: all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        for name, m in checks.items()
    }
    bt_grad = obj.online.stem.band_type_emb.grad is not None
    mask_grad = obj.mask_token.grad is not None and torch.isfinite(obj.mask_token.grad).all()
    ok = finite and all(grad_ok.values()) and bt_grad and mask_grad
    print(f"[check] r6 loss={float(loss_val):.4f} finite={finite}; grads {grad_ok}; "
          f"band_type_emb grad={bt_grad}; mask_token grad={mask_grad} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_stem_is_perband_subclass_no_conv() -> None:
    # r6 stem = NativePerBandStem (PerBandStem subclass, linear projections, NO conv modules).
    obj = _r6()
    stem = obj.online.stem
    assert isinstance(stem, PerBandStem)  # subclass ⇒ same projs / band_type_emb structure
    has_conv = any(isinstance(m, torch.nn.Conv1d) for m in stem.modules())
    assert not has_conv  # conv stem OUT (OFAT #1)
    print(f"[check] OK NativePerBandStem is PerBandStem subclass, no Conv1d (conv OUT)")
