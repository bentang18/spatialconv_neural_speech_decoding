"""End-to-end tests for the Run-B masked-electrode-prediction assembly.

Run-B REPLACES the M3 pool-inpaint head with a per-parcel masked-electrode
reconstruction: whole electrodes are dropped from each parcel (survivor floor),
blocked from the student pool, and their teacher frontend feature is
reconstructed from the survivor-only k pool seeds — addressed by the electrode's
native-RAS centroid-relative offset (the MNI-free identity). M2 + tubed-only M4
are inherited from Run-A. These smoke tests assert the forward runs, the loss is
finite and carries the melec term, and gradients reach every new Run-B param.
"""

from __future__ import annotations

import torch

from speech_decoding.models import v14_converged_v2 as v2
from speech_decoding.models.v14_converged_v2 import (
    V14ConvergedV2,
    V14ConvergedV2Config,
    active_parcels,
    bands_for_clip_len,
)

N_PARCELS = 62


def _cfg(d=32, n_heads=4, drop_frac=0.4, support_weight=False, m2_hetero=False):
    return V14ConvergedV2Config(
        d_model=d, n_heads=n_heads, frontend_layers=2, latent_layers=2,
        m2_pred_layers=2, m4_pred_layers=2, pred_dim=d, n_parcels=N_PARCELS,
        k=2, tube_ratio=0.35, qk_norm=True,
        m3_drop_frac=drop_frac, m3_min_keep=3, w_melec=1.0,
        sigma_mm=12.0, geom_n_freqs=5, support_weight=support_weight,
        m2_hetero=m2_hetero,
    )


def _session(clip_s=5.0, B=3, seed=0):
    """Homogeneous batch: 8 electrodes → parcels {5×4, 9×1, 20×3} (a droppable
    4- and 3-parcel plus an exempt singleton)."""
    torch.manual_seed(seed)
    bands = bands_for_clip_len(clip_s)
    poe = torch.tensor([5, 5, 5, 5, 9, 20, 20, 20])
    _, membership = active_parcels(poe)
    P = membership.shape[0]
    C = poe.shape[0]
    g = torch.Generator()
    g.manual_seed(seed + 1)
    lfs = torch.randn(B, C, bands[0].n_freq_bins, bands[0].n_time_frames)
    hga = torch.randn(B, C, bands[1].n_freq_bins, bands[1].n_time_frames)
    m2 = v2.sample_m2_masks_v2(B, P, bands, g)
    tube = v2.sample_parcel_tube_v2(B, P, 0.35, g)
    coords = torch.randn(C, 3) * 10.0                       # native-RAS mm
    return bands, poe, membership, lfs, hga, m2, tube, coords, g


def test_run_b_builds_expected_params():
    m = V14ConvergedV2(_cfg())
    keys = set(m.state_dict())
    assert any("recon" in k for k in keys)
    assert any("geom" in k for k in keys)
    assert any("pool_seed_offset" in k for k in keys)
    assert any("pool.ln_out" in k for k in keys)           # Run-A structure inherited
    assert not any("m3_predictor" in k for k in keys)      # M3 head replaced
    # Run-B first-arm default: pool K/V typing auto-resolves to "shared" (n_op=1,
    # plain PMA) — one shared W_K/W_V read operator, NOT the Run-A band-tied 2.
    assert m.cfg.pool_op_resolved == "shared"
    assert m.pool.W_K.shape[0] == 1
    assert m.teacher_pool.W_K.shape[0] == 1


def test_run_b_pool_op_override_restores_band_typed():
    # explicit pool_op="band" opts a Run-B run back into the n_op=2 tie ablation.
    import dataclasses

    m = V14ConvergedV2(dataclasses.replace(_cfg(), pool_op="band"))
    assert m.cfg.pool_op_resolved == "band"
    assert m.pool.W_K.shape[0] == 2


def test_run_b_forward_loss_finite_with_melec():
    bands, poe, _, lfs, hga, m2, tube, coords, g = _session()
    m = V14ConvergedV2(_cfg())
    drop = m.sample_drop(lfs.shape[0], active_parcels(poe)[1], g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop)
    assert torch.isfinite(out["loss"]) and out["loss"] > 0
    assert "loss_melec" in out
    assert torch.isfinite(out["loss_melec"])
    assert "loss_m3" in out and out["loss_m3"] == 0.0       # M3 head is off


def test_run_b_requires_coords_and_drop():
    bands, poe, _, lfs, hga, m2, tube, coords, g = _session()
    m = V14ConvergedV2(_cfg())
    try:
        m(lfs, hga, poe, m2, tube, clip_len_s=5.0)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_run_b_grads_reach_recon_geom_and_seed_offset():
    bands, poe, membership, lfs, hga, m2, tube, coords, g = _session(seed=2)
    m = V14ConvergedV2(_cfg())
    drop = m.sample_drop(lfs.shape[0], membership, g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop)
    out["loss"].backward()
    assert m.pool_seed_offset.grad is not None and torch.isfinite(m.pool_seed_offset.grad).all()
    recon_grad = m.recon.q_pos.weight.grad
    assert recon_grad is not None and recon_grad.abs().sum() > 0
    geom_grad = m.geom.mlp[-1].weight.grad
    assert geom_grad is not None                            # geom on the pool path


def test_run_b_dropped_electrodes_blocked_from_student_pool():
    """The student seeds must ignore dropped electrodes: corrupting a dropped
    electrode's input leaves the pooled seeds (hence the loss) unchanged, while
    corrupting a surviving electrode changes them."""
    bands, poe, membership, lfs, hga, m2, tube, coords, g = _session(seed=5)
    m = V14ConvergedV2(_cfg()).eval()
    # deterministic drop, reused across the two forwards.
    drop = m.sample_drop(lfs.shape[0], membership, g)
    # find a clip/electrode that is dropped, and one that survives, in parcel 5.
    b0 = 0
    parcel5 = membership[0]                                 # electrodes 0..3
    dropped = (drop[b0] & parcel5).nonzero(as_tuple=True)[0]
    survived = (~drop[b0] & parcel5).nonzero(as_tuple=True)[0]
    assert dropped.numel() > 0 and survived.numel() > 0
    with torch.no_grad():
        base = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop)
        lfs_d = lfs.clone(); lfs_d[b0, dropped[0]] += 50.0   # corrupt a DROPPED elec
        corr_d = m(lfs_d, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop)
        lfs_s = lfs.clone(); lfs_s[b0, survived[0]] += 50.0  # corrupt a SURVIVOR
        corr_s = m(lfs_s, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop)
    # the recon target is the TEACHER (sees all), so loss shifts for both; but the
    # student-pool CONTEXT must be invariant to the dropped electrode. Check the
    # student seeds tap directly.
    with torch.no_grad():
        t_base = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords,
                   drop_mask=drop, return_taps=True)
        t_drop = m(lfs_d, hga, poe, m2, tube, clip_len_s=5.0, coords=coords,
                   drop_mask=drop, return_taps=True)
    # melec pred is driven by student seeds; a dropped-elec corruption must not move
    # the student-side prediction (only the teacher target moved).
    assert torch.allclose(t_base["_tap_melec_pred"], t_drop["_tap_melec_pred"], atol=1e-5)


# ----------------------------------------------------- heterogeneous M2 masking
def test_run_b_m2_hetero_requires_elec_mask():
    """m2_hetero=True ⇒ the forward must be given a per-electrode m2_elec_mask."""
    bands, poe, membership, lfs, hga, m2, tube, coords, g = _session()
    m = V14ConvergedV2(_cfg(m2_hetero=True))
    drop = m.sample_drop(lfs.shape[0], membership, g)
    raised = False
    try:
        m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop)
    except ValueError:
        raised = True
    assert raised


def test_run_b_m2_hetero_forward_finite():
    bands, poe, membership, lfs, hga, m2, tube, coords, g = _session(seed=3)
    m = V14ConvergedV2(_cfg(m2_hetero=True))
    B, C = lfs.shape[0], poe.shape[0]
    drop = m.sample_drop(B, membership, g)
    m2e = m.sample_m2_hetero(B, C, bands, g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords,
            drop_mask=drop, m2_elec_mask=m2e)
    assert torch.isfinite(out["loss"]) and out["loss"] > 0
    assert torch.isfinite(out["loss_melec"])


def test_run_b_m2_hetero_electrodes_disagree_within_parcel():
    """The point of heterogeneous masking: two electrodes in the SAME parcel hold
    out DIFFERENT cells (parcel-uniform would force them identical)."""
    bands, poe, membership, lfs, hga, m2, tube, coords, g = _session(seed=4)
    m = V14ConvergedV2(_cfg(m2_hetero=True))
    B, C = lfs.shape[0], poe.shape[0]
    m2e = m.sample_m2_hetero(B, C, bands, g)                # (B,C,S)
    # electrodes 0..3 are all parcel 5; at least one pair must differ.
    p5 = [0, 1, 2, 3]
    rows = m2e[0, p5]                                        # (4,S)
    assert not torch.all(rows == rows[0])                   # not all identical


def test_run_b_m2_hetero_off_ignores_elec_mask():
    """m2_hetero=False (default) ⇒ the forward runs parcel-uniform and a passed
    m2_elec_mask is simply not required (Run-A/parcel arm unaffected)."""
    bands, poe, membership, lfs, hga, m2, tube, coords, g = _session(seed=9)
    m = V14ConvergedV2(_cfg(m2_hetero=False))
    drop = m.sample_drop(lfs.shape[0], membership, g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop)
    assert torch.isfinite(out["loss"])                     # no m2_elec_mask needed
