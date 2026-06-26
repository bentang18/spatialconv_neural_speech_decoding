"""P3.1 gates for the full :class:`V14ConvergedV2` assembly forward.

Four guards (per the assembly memo):
  * **M4 index alignment** — the new bookkeeping risk: each M4 query's gathered
    teacher target must be the latent of the SAME (parcel, seed, cell) the query's
    metadata names. Checked against a synthetic teacher-latent oracle.
  * **full-model dense==static loss oracle** — the static hot path (pack visible /
    gather untubed + per-row RoPE) must yield the EXACT loss of an independent dense
    formulation that keeps full shapes and uses key-masks. Composes every stage +
    the per-row latent-slot branch + the M4 query/target wiring.
  * **loss-scale at init** — Refinement-5: LN targets put loss_m2 ≈ loss_m4 and
    tubed ≈ untubed within 10× (no depth λ needed).
  * **e2e + EMA** — forward/backward finite, grads to student-not-teacher, EMA
    moves the teacher toward the student.
"""

from __future__ import annotations

import torch

from speech_decoding.models import v14_converged_v2 as v2
from speech_decoding.models.v14_converged_v2 import (
    V14ConvergedV2,
    V14ConvergedV2Config,
    _select_idx,
    active_parcels,
    bands_for_clip_len,
    compute_static_shapes_v2,
)

N_PARCELS = 62


def _cfg(d=32, n_heads=4):
    return V14ConvergedV2Config(
        d_model=d,
        n_heads=n_heads,
        frontend_layers=2,
        latent_layers=2,
        pred_layers=2,
        pred_dim=d,
        n_parcels=N_PARCELS,
        k=2,
        tube_ratio=0.35,
    )


def _session(clip_s=5.0, B=3, seed=0):
    """A realistic homogeneous batch: 6 electrodes → parcels {5×3, 9×1, 20×2}."""
    torch.manual_seed(seed)
    bands = bands_for_clip_len(clip_s)
    poe = torch.tensor([5, 5, 5, 9, 20, 20])
    _, membership = active_parcels(poe)
    P = membership.shape[0]
    g = torch.Generator()
    g.manual_seed(seed + 1)
    lfs = torch.randn(B, 6, bands[0].n_freq_bins, bands[0].n_time_frames)
    hga = torch.randn(B, 6, bands[1].n_freq_bins, bands[1].n_time_frames)
    m2 = v2.sample_m2_masks_v2(B, P, bands, g)
    tube = v2.sample_parcel_tube_v2(B, P, 0.35, g)
    return bands, poe, membership, lfs, hga, m2, tube


def test_m4_index_alignment_target_matches_query_metadata():
    """Build a synthetic teacher latent ``t[b,p,j,s] = encode(p,j,s)`` and verify
    the M4 target gathered at flat ``(pos·k+seed)·S+cell`` decodes back to the SAME
    (pos, seed, cell) the query carries — the core query↔target bookkeeping."""
    bands, _, membership, _, _, m2, tube = _session(B=4, seed=3)
    model = V14ConvergedV2(_cfg())
    sh = compute_static_shapes_v2(m2, tube, membership, bands, k=2)
    B, P, S, k = sh.b, sh.P, sh.S, sh.k
    pos, cell, seed, tubed = model._m4_indices(tube, m2, sh)

    # synthetic teacher latent encoding (p, j, s) into distinct integers.
    p_ax = torch.arange(P)[None, :, None, None]
    j_ax = torch.arange(k)[None, None, :, None]
    s_ax = torch.arange(S)[None, None, None, :]
    code = (p_ax * k + j_ax) * S + s_ax                       # (1,P,k,S)
    t_latent = code.expand(B, P, k, S).float()[..., None]     # (B,P,k,S,1)
    flat = (pos * k + seed) * S + cell                        # (B,Lq)
    gathered = torch.gather(
        t_latent.reshape(B, P * k * S, 1), 1, flat[..., None]
    ).squeeze(-1)                                             # (B,Lq)
    expected = (pos * k + seed) * S + cell
    assert torch.equal(gathered.long(), expected)

    # tubed block = all S cells of n_tube parcels; untubed = n_mask masked cells.
    Lt = sh.n_tube * k * S
    assert tubed[:Lt].all() and not tubed[Lt:].any()
    # untubed queries land ONLY on that parcel's M2-masked cells.
    pmask = _select_idx(m2, sh.n_mask)                        # (B,P,n_mask)
    for b in range(B):
        ut = ~tubed
        for i in ut.nonzero().squeeze(-1).tolist():
            p_pos, c = int(pos[b, i]), int(cell[b, i])
            assert c in pmask[b, p_pos].tolist()


def _dense_loss(model, lfs, hga, poe, m2, tube, clip_len_s):
    """Independent DENSE reference: full shapes + key-masks, no packing/gather
    (except the unavoidable query positions). Must match the static forward's loss."""
    bands = bands_for_clip_len(clip_len_s, base=model.base_bands)
    lay = model.session_layout(poe, bands)
    labels, membership, parcel_idx = lay.labels, lay.membership, lay.parcel_idx
    freq_id, slot, cell_patch = lay.freq_id, lay.slot, lay.cell_patch
    sh = compute_static_shapes_v2(m2, tube, membership, bands, k=model.cfg.k)
    B, C, P, S, d = sh.b, sh.c, sh.P, sh.S, model.cfg.d_model
    k, n_mask = sh.k, sh.n_mask

    with torch.no_grad():
        t_tok = model.teacher_frontend.tokenizer(lfs, hga)
        t_front = model.teacher_frontend.encode_tokens(t_tok, freq_id, slot)
        t_seeds = model.teacher_pool(t_front, membership, labels, cell_patch)
        t_latent = model.teacher_latent(t_seeds, labels, slot)

    elec_mask = m2[:, parcel_idx, :]                          # (B,C,S)
    visible = ~elec_mask
    tok = model.frontend.tokenizer(lfs, hga)
    s_front = model.frontend.encode_tokens(
        tok, freq_id, slot, key_mask=visible
    )                                                        # (B,C,S,d) DENSE
    mask_idx = _select_idx(elec_mask, n_mask)                 # (B,C,n_mask)

    # M2 dense: full-S context key-masked to visible; queries = masked cells.
    ctx = s_front.reshape(B * C, S, d)
    ctx_slot = slot[None].expand(B * C, S)
    ckm = visible.reshape(B * C, S)
    q2_slot = slot[mask_idx].reshape(B * C, n_mask)
    q2_freq = freq_id[mask_idx].reshape(B * C, n_mask)
    m2_pred = model.m2_predictor(
        ctx, ctx_slot, q2_slot, q2_freq, ctx_key_mask=ckm
    ).reshape(B, C, n_mask, d)
    m2_target = torch.gather(t_front, 2, mask_idx[..., None].expand(B, C, n_mask, d))

    # student latent DENSE: pool all S, key-mask hides tubed parcels AND masked cells.
    s_seeds = model.pool(s_front, membership, labels, cell_patch)   # (B,P,k,S,d)
    km = ((~tube)[:, :, None] & (~m2))[:, :, None, :].expand(B, P, k, S)  # (B,P,k,S)
    s_latent = model.latent(s_seeds, labels, slot, key_mask=km)     # (B,P,k,S,d)

    ctx4 = s_latent.reshape(B, P * k * S, d)
    ctx4_slot = slot.repeat(P * k)[None].expand(B, P * k * S)  # P-major,k-mid,S-minor
    ctx4_km = km.reshape(B, P * k * S)
    pos, cell, seed, tubed = model._m4_indices(tube, m2, sh)
    q_parcel, q_freq, q_slot = labels[pos], freq_id[cell], slot[cell]
    m4_pred = model.m4_predictor(
        ctx4, ctx4_slot, q_slot, q_freq,
        q_parcel=q_parcel, q_seed=seed, ctx_key_mask=ctx4_km,
    )
    Lq = pos.shape[1]
    flat = (pos * k + seed) * S + cell
    m4_target = torch.gather(
        t_latent.reshape(B, P * k * S, d), 1, flat[..., None].expand(B, Lq, d)
    )
    m4_weight = membership.sum(1)[pos].float()
    return v2.converged_v2_loss(
        m2_pred.reshape(-1, d), m2_target.reshape(-1, d),
        m4_pred.reshape(-1, d), m4_target.reshape(-1, d),
        m4_weight.reshape(-1), tubed[None, :].expand(B, Lq).reshape(-1),
    )


def test_full_model_dense_equals_static_loss():
    """The static hot path (forward) == the dense key-mask reference, on the loss
    AND every per-term diagnostic. Composes every stage + the per-row latent slot."""
    bands, poe, _membership, lfs, hga, m2, tube = _session(B=3, seed=5)
    model = V14ConvergedV2(_cfg())
    model.eval()
    with torch.no_grad():
        static = model(lfs, hga, poe, m2, tube, clip_len_s=5.0)
        dense = _dense_loss(model, lfs, hga, poe, m2, tube, 5.0)
    for key in ("loss", "loss_m2", "loss_m4", "loss_m4_tubed", "loss_m4_untubed"):
        assert torch.allclose(static[key], dense[key], atol=1e-5), (
            f"{key}: static {static[key].item()} != dense {dense[key].item()}"
        )


def test_per_row_latent_slot_matches_shared_when_uniform():
    """Guard the NEW per-row latent-slot branch: if every parcel masks the SAME
    cells (so per-row slots == the shared (S,) tiled), the per-row path reproduces
    the shared-slot path exactly."""
    bands = bands_for_clip_len(5.0)
    _, _, slot = v2.token_metadata(bands)
    B, P, k, d = 2, 4, 2, 32
    enc = v2.LatentEncoderV2(d, 4, n_layers=2, n_parcels=N_PARCELS)
    enc.eval()
    # take a visible subset of cells, SAME for every parcel.
    sub = torch.tensor([0, 2, 5, 9, 14, 20, 33, 60, 88, 109])
    seeds = torch.randn(B, P, k, sub.numel(), d)
    shared_slot = slot[sub]                                   # (s_vis,)
    per_row = shared_slot[None, None, None, :].expand(B, P, k, sub.numel()).reshape(
        B, P * k * sub.numel()
    )
    with torch.no_grad():
        out_shared = enc(seeds, None, shared_slot)
        out_per_row = enc(seeds, None, per_row)
    assert torch.allclose(out_shared, out_per_row, atol=1e-6)


def test_loss_scale_balanced_at_init():
    """Refinement-5: at init the LN'd targets keep loss_m2 ≈ loss_m4 and tubed ≈
    untubed within 10× — the construction check that no depth λ is needed."""
    _, poe, _, lfs, hga, m2, tube = _session(B=4, seed=7)
    model = V14ConvergedV2(_cfg())
    model.eval()
    with torch.no_grad():
        out = model(lfs, hga, poe, m2, tube, clip_len_s=5.0)
    assert 0.1 < out["ratio_m2_m4"].item() < 10.0
    assert 0.1 < out["ratio_tubed_untubed"].item() < 10.0


def test_e2e_forward_backward_finite_and_grad_targets():
    """Forward/backward runs, loss finite, grads reach the STUDENT towers but NOT
    the frozen teacher."""
    bands, poe, _membership, lfs, hga, m2, tube = _session(B=2, seed=9)
    model = V14ConvergedV2(_cfg())
    out = model(lfs, hga, poe, m2, tube, clip_len_s=5.0)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    g = [p.grad for p in model.frontend.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g)
    assert any(p.grad is not None for p in model.m4_predictor.parameters())
    assert all(p.grad is None for p in model.teacher_frontend.parameters())
    assert all(p.grad is None for p in model.teacher_latent.parameters())


def test_clip_len_1s_runs():
    """1 s eval clip (S=22) runs through the same assembly (clip-len parameterized)."""
    bands, poe, _membership, lfs, hga, m2, tube = _session(clip_s=1.0, B=2, seed=11)
    model = V14ConvergedV2(_cfg())
    model.eval()
    with torch.no_grad():
        out = model(lfs, hga, poe, m2, tube, clip_len_s=1.0)
    assert torch.isfinite(out["loss"])


def test_ema_step_moves_teacher_toward_student():
    """EMA advances the teacher a (1−τ) fraction toward the (perturbed) student."""
    model = V14ConvergedV2(_cfg())
    with torch.no_grad():
        for p in model.frontend.parameters():
            p.add_(1.0)                                       # diverge student
    tp = next(model.teacher_frontend.parameters())
    sp = next(model.frontend.parameters())
    before = (tp - sp).abs().mean().item()
    model.ema_step()
    after = (tp - sp).abs().mean().item()
    assert after < before
