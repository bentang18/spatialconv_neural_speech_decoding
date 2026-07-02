"""Run-A bundle gates for :class:`V14ConvergedV2` (Ben 2026-06-27).

The Run-A architecture (master switch ``m3_pred_layers``) adds: an M3 pool-inpaint
head (untubed parcels, M2-masked cells, teacher-POOL target), a terminal LayerNorm
on the pool, M4 restricted to TUBED parcels, a per-head self-normalized loss on RAW
EMA targets, and QK-norm on every tower. Legacy mode (``m3_pred_layers=None``) must
resume byte-identical — covered by :mod:`test_v14_converged_v2_assembly`.

Guards here:
  * **legacy resume safety** — a legacy state_dict round-trips strict into a fresh
    legacy model; the new Run-A params (M3 / pool-LN / qk-gains) exist ONLY in
    Run-A mode (no leaked keys → the live checkpoint still loads).
  * **M3 index/target alignment** — each M3 query's gathered teacher-POOL target
    is the SAME (parcel, seed, cell) the query's metadata names, and every M3
    query lands on an UNTUBED parcel's M2-masked cell.
  * **M4 tubed-only** — Run-A M4 queries cover exactly the tubed parcels, all S.
  * **per-head loss** — three independent self-normalized L1 means + explicit
    weights; empty head → 0.
  * **pool LayerNorm** — the Run-A pool output is unit-scale (the M3 target);
    legacy pool has none.
  * **dense == static** — the static hot path's Run-A loss equals an independent
    dense key-mask reference (the in-situ M3/M4 gather-alignment proof).
  * **e2e** — forward/backward finite, grads reach M3 + pool-LN + qk-gains.
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
    converged_v2_loss_per_head,
)

N_PARCELS = 62


def _cfg(run_a: bool, d=32, n_heads=4, qk_norm=None, support_weight=False):
    kw = dict(
        d_model=d, n_heads=n_heads, frontend_layers=2, latent_layers=2,
        m2_pred_layers=2, m4_pred_layers=2, pred_dim=d, n_parcels=N_PARCELS,
        k=2, tube_ratio=0.35,
        qk_norm=run_a if qk_norm is None else qk_norm,
    )
    if run_a:
        kw.update(m3_pred_layers=2, w_m2=1.0, w_m3=1.0, w_m4=1.0,
                  support_weight=support_weight)
    return V14ConvergedV2Config(**kw)


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


# --------------------------------------------------------------- resume safety
def test_legacy_state_dict_round_trips_strict():
    """A legacy model's state_dict loads strict into a fresh legacy model — the
    resume path for the live run. Run-A params must NOT appear in legacy mode."""
    a = V14ConvergedV2(_cfg(run_a=False))
    b = V14ConvergedV2(_cfg(run_a=False))
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=True)
    assert not missing and not unexpected
    keys = set(a.state_dict())
    assert not any("m3_predictor" in k for k in keys)
    assert not any("norm_gain" in k for k in keys)        # no qk-norm
    assert not any("pool.ln_out" in k for k in keys)      # no pool LN


def test_run_a_adds_exactly_the_bundle_params():
    """Run-A introduces the M3 head, pool LayerNorm, and qk-norm gains — and a
    legacy ckpt must NOT load strict into it (different arch, as expected)."""
    m = V14ConvergedV2(_cfg(run_a=True))
    keys = set(m.state_dict())
    assert any("m3_predictor" in k for k in keys)
    assert any("pool.ln_out" in k for k in keys)
    assert any("norm_gain" in k for k in keys)
    legacy = V14ConvergedV2(_cfg(run_a=False)).state_dict()
    try:
        m.load_state_dict(legacy, strict=True)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


# ------------------------------------------------------------- index alignment
def test_m3_index_alignment_and_untubed_masked():
    """M3 target gathered at flat ``(pos·k+seed)·S+cell`` decodes back to the
    query's (pos, seed, cell), and every M3 query lands on an UNTUBED parcel's
    M2-masked cell."""
    bands, _, membership, _, _, m2, tube = _session(B=4, seed=3)
    model = V14ConvergedV2(_cfg(run_a=True))
    sh = compute_static_shapes_v2(m2, tube, membership, bands, k=2)
    B, P, S, k = sh.b, sh.P, sh.S, sh.k
    pos, cell, seed = model._m3_indices(tube, m2, sh)
    assert pos.shape[1] == sh.m3_q // B

    p_ax = torch.arange(P)[None, :, None, None]
    j_ax = torch.arange(k)[None, None, :, None]
    s_ax = torch.arange(S)[None, None, None, :]
    code = (p_ax * k + j_ax) * S + s_ax                       # (1,P,k,S)
    t_seeds = code.expand(B, P, k, S).float()[..., None]      # synthetic POOL
    flat = (pos * k + seed) * S + cell
    gathered = torch.gather(
        t_seeds.reshape(B, P * k * S, 1), 1, flat[..., None]
    ).squeeze(-1)
    assert torch.equal(gathered.long(), flat)

    pmask = _select_idx(m2, sh.n_mask)                        # (B,P,n_mask)
    for b in range(B):
        untubed_p = (~tube[b]).nonzero().squeeze(-1).tolist()
        for i in range(pos.shape[1]):
            p_pos, c = int(pos[b, i]), int(cell[b, i])
            assert p_pos in untubed_p                         # untubed only
            assert c in pmask[b, p_pos].tolist()              # masked cell only


def test_m4_tubed_only_covers_tubed_parcels_all_cells():
    """Run-A M4 (tubed_only=True) = exactly the tubed parcels, all S cells, ×k."""
    bands, _, membership, _, _, m2, tube = _session(B=4, seed=4)
    model = V14ConvergedV2(_cfg(run_a=True))
    sh = compute_static_shapes_v2(m2, tube, membership, bands, k=2)
    B = sh.b
    pos, cell, seed, tubed = model._m4_indices(tube, m2, sh, tubed_only=True)
    assert tubed.all()
    assert pos.shape[1] == sh.m4_q_tubed // B
    for b in range(B):
        tubed_p = tube[b].nonzero().squeeze(-1).tolist()
        assert set(pos[b].tolist()) == set(tubed_p)
        # each tubed parcel covers every cell 0..S-1 for each seed.
        for p in tubed_p:
            cells = cell[b][pos[b] == p]
            assert sorted(set(cells.tolist())) == list(range(sh.S))


# --------------------------------------------------------------- per-head loss
def test_per_head_loss_self_normalized_and_weighted():
    """Each head = mean |pred − sg(target)| over its OWN (Q×d); loss = Σ w·L."""
    torch.manual_seed(0)
    d = 8
    m2p, m2t = torch.zeros(10, d), torch.full((10, d), 0.5)   # err 0.5
    m3p, m3t = torch.zeros(7, d), torch.full((7, d), 2.0)     # err 2.0
    m4p, m4t = torch.zeros(5, d), torch.full((5, d), 1.0)     # err 1.0
    out = converged_v2_loss_per_head(
        m2p, m2t, m3p, m3t, m4p, m4t, w_m2=1.0, w_m3=2.0, w_m4=3.0
    )
    assert torch.allclose(out["loss_m2"], torch.tensor(0.5))
    assert torch.allclose(out["loss_m3"], torch.tensor(2.0))
    assert torch.allclose(out["loss_m4"], torch.tensor(1.0))
    assert torch.allclose(out["loss"], torch.tensor(0.5 + 2.0 * 2.0 + 3.0 * 1.0))


def test_per_head_loss_melec_term_added():
    """Run-B: melec head adds w_melec·L_melec (self-normalized L1); omitting it
    leaves the loss byte-identical (Run-A back-compat)."""
    d = 8
    m2p, m2t = torch.zeros(10, d), torch.full((10, d), 0.5)
    m3p, m3t = torch.zeros(7, d), torch.full((7, d), 2.0)
    m4p, m4t = torch.zeros(5, d), torch.full((5, d), 1.0)
    mep, met = torch.zeros(6, d), torch.full((6, d), 3.0)     # err 3.0
    base = converged_v2_loss_per_head(m2p, m2t, m3p, m3t, m4p, m4t)
    out = converged_v2_loss_per_head(
        m2p, m2t, m3p, m3t, m4p, m4t, melec_pred=mep, melec_target=met, w_melec=0.5
    )
    assert "loss_melec" not in base                          # off by default
    assert torch.allclose(out["loss_melec"], torch.tensor(3.0))
    assert torch.allclose(out["loss"], base["loss"] + 0.5 * 3.0)
    assert torch.allclose(out["ratio_melec_m4"], torch.tensor(3.0))  # l_melec/l_m4


def test_per_head_loss_melec_detaches_target():
    p = torch.zeros(4, 6, requires_grad=True)
    t = torch.ones(4, 6, requires_grad=True)
    mp = torch.zeros(4, 6, requires_grad=True)
    mt = torch.ones(4, 6, requires_grad=True)
    out = converged_v2_loss_per_head(p, t, p, t, p, t, melec_pred=mp, melec_target=mt)
    out["loss"].backward()
    assert mp.grad is not None and mt.grad is None


def test_per_head_loss_detaches_target():
    """The target is stop-grad'd: grad flows to preds only."""
    p = torch.zeros(4, 6, requires_grad=True)
    t = torch.ones(4, 6, requires_grad=True)
    out = converged_v2_loss_per_head(p, t, p, t, p, t)
    out["loss"].backward()
    assert p.grad is not None and t.grad is None


def test_per_head_empty_head_is_zero():
    d = 4
    empty = torch.zeros(0, d)
    nonempty_p, nonempty_t = torch.zeros(3, d), torch.ones(3, d)
    out = converged_v2_loss_per_head(
        nonempty_p, nonempty_t, empty, empty, nonempty_p, nonempty_t
    )
    assert torch.allclose(out["loss_m3"], torch.tensor(0.0))


# --------------------------------------------------------------- pool LayerNorm
def test_pool_ln_out_unit_scale_run_a_only():
    """The Run-A pool output (M3 target) is per-token unit-scale; legacy pool has
    no terminal LN (output not normalized)."""
    _, poe, _, lfs, hga, _, _ = _session(B=2, seed=6)
    bands = bands_for_clip_len(5.0)
    m_a = V14ConvergedV2(_cfg(run_a=True)).eval()
    lay = m_a.session_layout(poe, bands)
    with torch.no_grad():
        tok = m_a.teacher_frontend.tokenizer(lfs, hga)
        front = m_a.teacher_frontend.encode_tokens(tok, lay.freq_id, lay.slot)
        seeds = m_a.teacher_pool(front, lay.membership, lay.labels, lay.cell_patch)
    flat = seeds.reshape(-1, seeds.shape[-1]).to(torch.float32)
    assert flat.mean(-1).abs().max() < 1e-4                   # LN centers
    assert torch.allclose(
        flat.std(-1, unbiased=False), torch.ones(flat.shape[0]), atol=1e-2
    )
    assert V14ConvergedV2(_cfg(run_a=False)).pool.ln_out is None


# --------------------------------------------------------------- dense == static
def _dense_loss_run_a(model, lfs, hga, poe, m2, tube, clip_len_s):
    """Independent DENSE Run-A reference: full shapes + key-masks (no packing/
    gather except the unavoidable query positions). Must match the static loss."""
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

    elec_mask = m2[:, parcel_idx, :]
    visible = ~elec_mask
    tok = model.frontend.tokenizer(lfs, hga)
    s_front = model.frontend.encode_tokens(tok, freq_id, slot, key_mask=visible)
    mask_idx = _select_idx(elec_mask, n_mask)

    # M2: full-S context key-masked to visible; queries = masked cells.
    ctx = s_front.reshape(B * C, S, d)
    ctx_slot = slot[None].expand(B * C, S)
    ckm = visible.reshape(B * C, S)
    q2_slot = slot[mask_idx].reshape(B * C, n_mask)
    q2_freq = freq_id[mask_idx].reshape(B * C, n_mask)
    m2_pred = model.m2_predictor(
        ctx, ctx_slot, q2_slot, q2_freq, ctx_key_mask=ckm
    ).reshape(B, C, n_mask, d)
    m2_target = torch.gather(t_front, 2, mask_idx[..., None].expand(B, C, n_mask, d))

    # student pool (dense) + latent; key-mask hides tubed parcels AND masked cells.
    s_seeds = model.pool(s_front, membership, labels, cell_patch)
    km = ((~tube)[:, :, None] & (~m2))[:, :, None, :].expand(B, P, k, S)
    s_latent = model.latent(s_seeds, labels, slot, key_mask=km)

    # M4 tubed-only.
    ctx4 = s_latent.reshape(B, P * k * S, d)
    ctx4_slot = slot.repeat(P * k)[None].expand(B, P * k * S)
    ctx4_km = km.reshape(B, P * k * S)
    pos, cell, seed, _ = model._m4_indices(tube, m2, sh, tubed_only=True)
    m4_pred = model.m4_predictor(
        ctx4, ctx4_slot, slot[cell], freq_id[cell],
        q_parcel=labels[pos], q_seed=seed, ctx_key_mask=ctx4_km,
    )
    Lq4 = pos.shape[1]
    flat4 = (pos * k + seed) * S + cell
    m4_target = torch.gather(
        t_latent.reshape(B, P * k * S, d), 1, flat4[..., None].expand(B, Lq4, d)
    )

    # M3: context = parcel-tagged pool seeds, key-masked to untubed×visible (==km);
    # queries = untubed parcels' masked cells; target = teacher POOL.
    s_tag = s_seeds + model.latent.parcel_embed(labels)[None, :, None, None, :]
    ctx3 = s_tag.reshape(B, P * k * S, d)
    ctx3_slot = slot.repeat(P * k)[None].expand(B, P * k * S)
    ctx3_km = km.reshape(B, P * k * S)
    m3_pos, m3_cell, m3_seed = model._m3_indices(tube, m2, sh)
    m3_pred = model.m3_predictor(
        ctx3, ctx3_slot, slot[m3_cell], freq_id[m3_cell],
        q_parcel=labels[m3_pos], q_seed=m3_seed, ctx_key_mask=ctx3_km,
    )
    Lq3 = m3_pos.shape[1]
    flat3 = (m3_pos * k + m3_seed) * S + m3_cell
    m3_target = torch.gather(
        t_seeds.reshape(B, P * k * S, d), 1, flat3[..., None].expand(B, Lq3, d)
    )

    m3_w = m4_w = None
    if model.cfg.support_weight:
        n_elec = membership.sum(-1).to(m3_pred.dtype)
        m3_w = n_elec[m3_pos].reshape(-1)
        m4_w = n_elec[pos].reshape(-1)
    return converged_v2_loss_per_head(
        m2_pred.reshape(-1, d), m2_target.reshape(-1, d),
        m3_pred.reshape(-1, d), m3_target.reshape(-1, d),
        m4_pred.reshape(-1, d), m4_target.reshape(-1, d),
        w_m2=model.cfg.w_m2, w_m3=model.cfg.w_m3, w_m4=model.cfg.w_m4,
        m3_weight=m3_w, m4_weight=m4_w,
    )


def test_run_a_dense_equals_static_loss():
    """The static hot path (forward) == the dense key-mask reference, on the loss
    AND every per-head diagnostic — the in-situ M3/M4 gather-alignment proof."""
    _, poe, _, lfs, hga, m2, tube = _session(B=3, seed=5)
    model = V14ConvergedV2(_cfg(run_a=True)).eval()
    with torch.no_grad():
        static = model(lfs, hga, poe, m2, tube, clip_len_s=5.0)
        dense = _dense_loss_run_a(model, lfs, hga, poe, m2, tube, 5.0)
    for key in ("loss", "loss_m2", "loss_m3", "loss_m4"):
        assert torch.allclose(static[key], dense[key], atol=1e-5), (
            f"{key}: static {static[key].item()} != dense {dense[key].item()}"
        )


# -------------------------------------------------------------------------- e2e
def test_run_a_e2e_grads_reach_m3_pool_ln_and_qk_gains():
    _, poe, _, lfs, hga, m2, tube = _session(B=2, seed=9)
    model = V14ConvergedV2(_cfg(run_a=True))
    out = model(lfs, hga, poe, m2, tube, clip_len_s=5.0)
    assert torch.isfinite(out["loss"])
    assert "loss_m3" in out
    out["loss"].backward()
    assert any(p.grad is not None for p in model.m3_predictor.parameters())
    assert model.pool.ln_out.weight.grad is not None
    qk = [p for n, p in model.named_parameters()
          if "norm_gain" in n and p.requires_grad]
    assert qk and any(p.grad is not None for p in qk)
    assert all(p.grad is None for p in model.teacher_frontend.parameters())


# ----------------------------------------------------- electrode-support weight
def test_support_weight_is_convex_weighted_mean():
    """A weighted head = Σ w·rowloss / Σ w (rowloss = mean over d). Convex, so
    it stays within the per-row min/max and reduces to the flat mean when the
    weights are uniform."""
    torch.manual_seed(0)
    d = 6
    pred = torch.zeros(4, d)
    target = torch.tensor([0.0, 1.0, 2.0, 3.0])[:, None].expand(4, d).contiguous()
    w = torch.tensor([1.0, 3.0, 0.0, 4.0])              # row 2 ignored
    out = converged_v2_loss_per_head(
        pred, target, pred, target, pred, target, m3_weight=w, m4_weight=w
    )
    rowloss = target.abs().mean(-1)                     # [0,1,2,3]
    expect = (rowloss * w).sum() / w.sum()              # (0+3+0+12)/8 = 1.875
    assert torch.allclose(out["loss_m3"], expect)
    assert torch.allclose(out["loss_m4"], expect)
    assert torch.allclose(out["loss_m2"], rowloss.mean())   # M2 stays flat
    # uniform weights == flat mean
    flat = converged_v2_loss_per_head(pred, target, pred, target, pred, target)
    uni = converged_v2_loss_per_head(
        pred, target, pred, target, pred, target,
        m3_weight=torch.ones(4), m4_weight=torch.ones(4),
    )
    assert torch.allclose(flat["loss_m3"], uni["loss_m3"])


def test_support_weight_dense_equals_static():
    """The electrode-weighted static path == the dense reference (same weights),
    on the loss and every per-head diagnostic."""
    _, poe, _, lfs, hga, m2, tube = _session(B=3, seed=7)
    model = V14ConvergedV2(_cfg(run_a=True, support_weight=True)).eval()
    assert model.cfg.support_weight
    with torch.no_grad():
        static = model(lfs, hga, poe, m2, tube, clip_len_s=5.0)
        dense = _dense_loss_run_a(model, lfs, hga, poe, m2, tube, 5.0)
    for key in ("loss", "loss_m2", "loss_m3", "loss_m4"):
        assert torch.allclose(static[key], dense[key], atol=1e-5), key


def test_support_weight_preserves_scale_and_changes_value():
    """All three heads stay ~same scale at init (every target is LayerNorm-
    terminated ⇒ |err|~0.8), and weighting actually moves loss_m3/loss_m4 vs the
    flat mean (parcels here have unequal electrode counts: 3,1,2)."""
    _, poe, _, lfs, hga, m2, tube = _session(B=3, seed=8)
    flat = V14ConvergedV2(_cfg(run_a=True, support_weight=False)).eval()
    wtd = V14ConvergedV2(_cfg(run_a=True, support_weight=True)).eval()
    wtd.load_state_dict(flat.state_dict())              # identical weights
    with torch.no_grad():
        of = flat(lfs, hga, poe, m2, tube, clip_len_s=5.0)
        ow = wtd(lfs, hga, poe, m2, tube, clip_len_s=5.0)
    for o in (of, ow):
        for kk in ("loss_m2", "loss_m3", "loss_m4"):
            assert 0.3 < o[kk].item() < 1.6, (kk, o[kk].item())
        assert 0.5 < (o["loss_m3"] / o["loss_m2"]).item() < 2.0
        assert 0.5 < (o["loss_m4"] / o["loss_m2"]).item() < 2.0
    # weighting redistributes ⇒ m3/m4 differ from flat; m2 (unweighted) identical.
    assert torch.allclose(of["loss_m2"], ow["loss_m2"])
    assert not torch.allclose(of["loss_m3"], ow["loss_m3"])
    assert not torch.allclose(of["loss_m4"], ow["loss_m4"])
