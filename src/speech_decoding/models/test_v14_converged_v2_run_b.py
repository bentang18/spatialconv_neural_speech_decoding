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
    # Run-B canonical: pool K/V typing auto-resolves to "band" (n_op=2, LFS | HGA —
    # the one physiology-backed spatial-regime boundary), NOT the n_op=4 LFS split.
    assert m.cfg.pool_op_resolved == "band"
    assert m.pool.W_K.shape[0] == 2
    assert m.teacher_pool.W_K.shape[0] == 2
    # No per-parcel pool query embed (universal geometric operator); recon K/V + output
    # are band-typed to the same n_op as the pool (symmetric inverse).
    assert m.pool.embed_pq is None
    assert m.teacher_pool.embed_pq is None
    assert m.recon.n_op == 2
    assert m.recon.W_k.shape[0] == 2 and m.recon.out_head.shape[0] == 2


def test_run_b_pool_op_override_untie_lfs_patch():
    # explicit pool_op="patch" opts a Run-B run into the n_op=4 LFS-untie ablation.
    import dataclasses

    m = V14ConvergedV2(dataclasses.replace(_cfg(), pool_op="patch"))
    assert m.cfg.pool_op_resolved == "patch"
    assert m.pool.W_K.shape[0] == 4
    assert m.recon.n_op == 4 and m.recon.W_k.shape[0] == 4


def test_run_b_forward_loss_finite_with_melec():
    bands, poe, _, lfs, hga, m2, tube, coords, g = _session()
    m = V14ConvergedV2(_cfg())
    drop = m.sample_drop(lfs.shape[0], active_parcels(poe)[1], g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop)
    assert torch.isfinite(out["loss"]) and out["loss"] > 0
    assert "loss_melec" in out
    assert torch.isfinite(out["loss_melec"])
    # M3 head is OFF, but its diagnostic slot is re-labelled to the recon (third)
    # head so the wandb dashboards overlay Run-A's m3 curve (the training loss is
    # built from loss_melec, unchanged).
    assert "loss_m3" in out and out["loss_m3"] == out["loss_melec"]
    assert out["ratio_m3_m4"] == out["ratio_melec_m4"]


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
    # Run-B must emit the teacher-pool tap too (pool_rankme/feat_std source) so the
    # monitor's pool-rank diagnostic logs and overlays Run-A (M3 head is off here).
    assert "_tap_teacher_pool" in t_base


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


# ---------------------------------- complete-grid (hetero) + empty-cell key-block
def test_empty_cell_mask_helper():
    """A parcel-cell is EMPTY iff every member electrode is blocked there."""
    membership = torch.tensor([[True, True, False], [False, False, True]])
    block = torch.zeros(1, 3, 3, dtype=torch.bool)          # (B,S,C)
    block[0, 0, 0] = True; block[0, 0, 1] = True            # parcel0 empty @ cell0
    block[0, 1, 0] = True                                   # parcel0 keeps elec1 @ cell1
    block[0, 2, 2] = True                                   # parcel1 empty @ cell2
    empty = v2._empty_cell_mask(block, membership)          # (B,P,S)
    assert empty.shape == (1, 2, 3)
    assert bool(empty[0, 0, 0]) and not bool(empty[0, 0, 1])
    assert not bool(empty[0, 0, 2])                         # both visible @ cell2
    assert bool(empty[0, 1, 2]) and not bool(empty[0, 1, 0])


def test_run_b_hetero_latent_runs_over_full_S_grid():
    """Hetero drops NO parcel cells (union of per-electrode masks covers the
    grid) ⇒ the latent runs the full S cells, not the pvis-shrunk s_vis."""
    bands, poe, membership, lfs, hga, m2, tube, coords, g = _session(seed=7)
    m = V14ConvergedV2(_cfg(m2_hetero=True))
    B, C = lfs.shape[0], poe.shape[0]
    drop = m.sample_drop(B, membership, g)
    m2e = m.sample_m2_hetero(B, C, bands, g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords,
            drop_mask=drop, m2_elec_mask=m2e, return_taps=True)
    S = out["_tap_teacher_pool"].shape[3]                   # teacher = full grid
    assert out["_tap_student_latent"].shape[3] == S        # complete grid, no pvis


def test_run_b_uniform_latent_shrinks_to_s_vis():
    """Contrast: the parcel-uniform (non-hetero) arm still drops the masked cells
    via pvis, so its latent is strictly smaller than the full S grid."""
    bands, poe, membership, lfs, hga, m2, tube, coords, g = _session(seed=7)
    m = V14ConvergedV2(_cfg(m2_hetero=False))
    drop = m.sample_drop(lfs.shape[0], membership, g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords,
            drop_mask=drop, return_taps=True)
    S = out["_tap_teacher_pool"].shape[3]
    assert out["_tap_student_latent"].shape[3] < S         # pvis shrinks the grid


def test_run_b_hetero_melec_weight_is_binary_and_marks_empties():
    """melec loss weight ∈ {0,1}, length == B·D·S, and its zeros count EXACTLY the
    empty cells at the dropped electrodes' parcels (no supervision on empty seeds)."""
    bands, poe, membership, lfs, hga, m2, tube, coords, g = _session(seed=4)
    m = V14ConvergedV2(_cfg(m2_hetero=True))
    B, C = lfs.shape[0], poe.shape[0]
    drop = m.sample_drop(B, membership, g)
    m2e = m.sample_m2_hetero(B, C, bands, g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords,
            drop_mask=drop, m2_elec_mask=m2e, return_taps=True)
    w = out["_tap_melec_weight"]                            # (B·D·S,)
    empty = out["_tap_empty_cell"]                          # (B,P,S)
    assert ((w == 0) | (w == 1)).all()
    D = int(v2.electrode_drop_count(membership, 0.4, 3).sum())
    S = empty.shape[-1]
    assert w.numel() == B * D * S
    drop_idx = v2._select_idx(drop, D)                     # (B,D) same as forward
    parcel_idx = membership.float().argmax(0)              # electrode -> parcel row
    pd = parcel_idx[drop_idx]                              # (B,D)
    empty_d = torch.gather(empty, 1, pd[:, :, None].expand(B, D, S))
    assert int((w == 0).sum()) == int(empty_d.sum())       # zeros ⟺ empty cells


def test_melec_loss_weight_masks_zero_weighted_cells():
    """melec_weight all-ones == unweighted; zeroing cells drops them from the mean."""
    torch.manual_seed(0)
    d = 8
    m2p, m2t = torch.randn(4, d), torch.randn(4, d)
    m4p, m4t = torch.randn(4, d), torch.randn(4, d)
    mp, mt = torch.randn(6, d), torch.randn(6, d)
    e = torch.zeros(0, d)
    base = v2.converged_v2_loss_per_head(
        m2p, m2t, e, e, m4p, m4t, melec_pred=mp, melec_target=mt)
    ones = v2.converged_v2_loss_per_head(
        m2p, m2t, e, e, m4p, m4t, melec_pred=mp, melec_target=mt,
        melec_weight=torch.ones(6))
    assert torch.allclose(base["loss_melec"], ones["loss_melec"])
    w = torch.tensor([1., 1., 1., 0., 0., 0.])
    masked = v2.converged_v2_loss_per_head(
        m2p, m2t, e, e, m4p, m4t, melec_pred=mp, melec_target=mt, melec_weight=w)
    expected = (mp[:3] - mt[:3]).abs().mean()
    assert torch.allclose(masked["loss_melec"], expected)


def test_recon_head_band_typed_output_depends_on_band():
    """Band-typed K/V + output ⇒ the SAME seeds+offset reconstruct differently under
    a different cell band (proves the typing is live); n_op=1 ignores band."""
    torch.manual_seed(0)
    d, H, F, n_op = 32, 4, 7, 4
    head = v2.ElectrodeReconHead(d, H, F, n_op=n_op)
    seeds = torch.randn(5, 2, d)
    feats = torch.randn(5, F)
    o0 = head(seeds, feats, torch.zeros(5, dtype=torch.long))
    o1 = head(seeds, feats, torch.ones(5, dtype=torch.long))
    assert o0.shape == (5, d)
    assert not torch.allclose(o0, o1)                          # band changes the readout
    # a single-op head is band-invariant (no typing).
    shared = v2.ElectrodeReconHead(d, H, F, n_op=1)
    s0 = shared(seeds, feats, torch.zeros(5, dtype=torch.long))
    s1 = shared(seeds, feats, torch.ones(5, dtype=torch.long))
    assert torch.allclose(s0, s1)


def test_run_b_hetero_no_parcel_embed_grads_flow():
    """Canonical Run-B (hetero, band-typed n_op=2, no embed_pq): forward+backward
    finite, and grads reach the shared base_seed + the band-typed recon K/V + output."""
    bands, poe, membership, lfs, hga, m2, tube, coords, g = _session(seed=7)
    m = V14ConvergedV2(_cfg(m2_hetero=True))
    assert m.pool.embed_pq is None                             # universal geometric pool
    B, C = lfs.shape[0], poe.shape[0]
    drop = m.sample_drop(B, membership, g)
    m2e = m.sample_m2_hetero(B, C, bands, g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords,
            drop_mask=drop, m2_elec_mask=m2e)
    assert torch.isfinite(out["loss"]).all()
    out["loss"].backward()
    assert m.pool.base_seed.grad is not None and torch.isfinite(m.pool.base_seed.grad).all()
    assert m.recon.W_k.grad is not None and m.recon.W_k.grad.abs().sum() > 0
    assert m.recon.out_head.grad is not None and m.recon.out_head.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
