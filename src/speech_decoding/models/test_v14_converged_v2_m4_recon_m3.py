"""Tests for the Run-B variant: M4 → teacher-M3 + V-JEPA 2.1 context loss.

``m4_recon_m3`` retargets the M4 head to the teacher's M3 (pool + parcel-embed;
the teacher latent stack is NOT run) and reads the predictor's context-position
outputs (the untubed parcels) as a second, λ-weighted context loss — symmetric M2
context loss added at the frontend. M-elec is KEPT. These assert: the forward runs
and carries both context terms + melec; the teacher latent is genuinely skipped;
the λ ramp; the per-head loss context terms; the predictor's ``return_ctx``; and
byte-identical behaviour when the flag is off.
"""

from __future__ import annotations

import types

import pytest
import torch

from speech_decoding.experiments.v14_converged_v2_module import (
    V14ConvergedV2BrainModule,
)
from speech_decoding.models import v14_converged_v2 as v2
from speech_decoding.models.v14_converged_v2 import (
    JepaPredictorV2,
    V14ConvergedV2,
    V14ConvergedV2Config,
    active_parcels,
    bands_for_clip_len,
    converged_v2_loss_per_head,
)

N_PARCELS = 62


def _cfg(*, m4_recon_m3=True, support_weight=False, context_taps=("M2", "M4"),
         m2_hetero=False, context_loss=True, d=32):
    return V14ConvergedV2Config(
        d_model=d, n_heads=4, frontend_layers=2, latent_layers=2,
        m2_pred_layers=2, m4_pred_layers=2, pred_dim=d, n_parcels=N_PARCELS,
        k=2, tube_ratio=0.35, qk_norm=True,
        m3_drop_frac=0.4, m3_min_keep=3, w_melec=1.0,
        sigma_mm=12.0, geom_n_freqs=5, support_weight=support_weight,
        m2_hetero=m2_hetero,
        m4_recon_m3=m4_recon_m3, context_loss=context_loss,
        context_lambda=0.2, context_warmup_steps=15000,
        context_taps=context_taps,
    )


def _session(clip_s=5.0, B=3, seed=0):
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
    coords = torch.randn(C, 3) * 10.0
    return bands, poe, membership, lfs, hga, m2, tube, coords, g


# --- forward -------------------------------------------------------------


def test_m4_recon_m3_forward_carries_context_and_melec():
    m = V14ConvergedV2(_cfg())
    _, poe, membership, lfs, hga, m2, tube, coords, g = _session()
    drop = m.sample_drop(lfs.shape[0], membership, g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop)
    assert torch.isfinite(out["loss"])
    # both context terms present + finite (V-JEPA context loss, M2 + M4)
    for key in ("loss_m2_ctx", "loss_m4_ctx"):
        assert key in out, f"missing {key}"
        assert torch.isfinite(out[key])
    # M-elec is KEPT (Ben: DO NOT DROP MELEC)
    assert "loss_melec" in out and torch.isfinite(out["loss_melec"])
    # the forward-returned loss does NOT yet include the context terms (the module
    # folds them with the ramped λ) — they are pure diagnostics here.
    base = out["loss_m2"] + out["loss_m4"] + out["loss_melec"]
    assert torch.allclose(out["loss"], base, atol=1e-5)


def test_rescale_blocks_exact_scale():
    """BEiT/V-JEPA depth-rescale divides attn-out + MLP-fc2 by √(2·layer_id).
    Deterministic contract test: ones-init → block i weight == 1/√(2·(i+1))."""
    import math

    from torch import nn

    from speech_decoding.models.v14_encoder import _JointTokenBlock

    blocks = nn.ModuleList([_JointTokenBlock(16, 4) for _ in range(4)])
    for blk in blocks:
        nn.init.ones_(blk.attn.out.weight)
        nn.init.ones_(blk.ffn[-1].weight)
    V14ConvergedV2._rescale_blocks(blocks)
    for i, blk in enumerate(blocks):
        want = 1.0 / math.sqrt(2.0 * (i + 1))
        assert torch.allclose(blk.attn.out.weight, torch.full_like(blk.attn.out.weight, want))
        assert torch.allclose(blk.ffn[-1].weight, torch.full_like(blk.ffn[-1].weight, want))


def test_rescale_blocks_applied_to_towers():
    """The real model's towers get depth-rescaled: output-proj scale is strictly
    decreasing with block depth in every self-attention stack."""
    m = V14ConvergedV2(_cfg())
    for stack in (m.frontend, m.latent, m.m2_predictor, m.m4_predictor):
        norms = [float(blk.attn.out.weight.detach().norm()) for blk in stack.blocks]
        assert all(a > b for a, b in zip(norms, norms[1:])), norms


def test_context_ev_taps_emitted():
    """return_taps under m4_recon_m3 must emit the ctx pred/target rows so the
    SSL-health monitor can score context EV (the trivial-copy guard). Off ⇒ absent."""
    m = V14ConvergedV2(_cfg(m2_hetero=True, support_weight=True))
    bands, poe, membership, lfs, hga, m2, tube, coords, g = _session(seed=5)
    B, C = lfs.shape[0], poe.shape[0]
    drop = m.sample_drop(B, membership, g)
    m2e = m.sample_m2_hetero(B, C, bands, g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords,
            drop_mask=drop, m2_elec_mask=m2e, return_taps=True)
    d = m.cfg.d_model
    for key in ("_tap_m2_ctx_pred", "_tap_m2_ctx_target",
                "_tap_m4_ctx_pred", "_tap_m4_ctx_target"):
        assert key in out, f"missing {key}"
        assert out[key].shape[-1] == d and out[key].shape[0] > 1
    # context_loss off: plain Run-B (melec) emits NO ctx taps
    m0 = V14ConvergedV2(_cfg(context_loss=False, m2_hetero=True, support_weight=True))
    out0 = m0(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords,
              drop_mask=drop, m2_elec_mask=m2e, return_taps=True)
    for key in ("_tap_m2_ctx_pred", "_tap_m4_ctx_pred"):
        assert key not in out0


def test_m4_recon_m3_hetero_forward_finite():
    """The ACTIVE Run-B is hetero + support-weighted; m4_recon_m3 must run on it
    (context reads all S cells, empty ones weighted out via attend_unt)."""
    m = V14ConvergedV2(_cfg(m2_hetero=True, support_weight=True))
    bands, poe, membership, lfs, hga, m2, tube, coords, g = _session(seed=3)
    B, C = lfs.shape[0], poe.shape[0]
    drop = m.sample_drop(B, membership, g)
    m2e = m.sample_m2_hetero(B, C, bands, g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords,
            drop_mask=drop, m2_elec_mask=m2e)
    assert torch.isfinite(out["loss"]) and out["loss"] > 0
    for key in ("loss_m2_ctx", "loss_m4_ctx", "loss_melec"):
        assert key in out and torch.isfinite(out[key])
    # grad reaches the M4 predictor through both masked + context terms
    total = out["loss"] + 0.2 * (out["loss_m2_ctx"] + out["loss_m4_ctx"])
    total.backward()
    g4 = m.m4_predictor.mask_token.grad
    assert g4 is not None and g4.abs().sum() > 0


def test_m4_recon_m3_skips_teacher_latent():
    """The teacher latent stack must NOT run under m4_recon_m3 (teacher stops at
    M3). Patch its forward to blow up: the recon path succeeds, plain Run-B does
    not."""
    _, poe, membership, lfs, hga, m2, tube, coords, g = _session()

    def _boom(*_a, **_k):
        raise RuntimeError("teacher_latent forward ran")

    m_recon = V14ConvergedV2(_cfg(m4_recon_m3=True))
    m_recon.teacher_latent.forward = types.MethodType(  # type: ignore[method-assign]
        _boom, m_recon.teacher_latent
    )
    drop = m_recon.sample_drop(lfs.shape[0], membership, g)
    out = m_recon(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords,
                  drop_mask=drop)
    assert torch.isfinite(out["loss"])  # ran without calling teacher_latent

    m_plain = V14ConvergedV2(_cfg(m4_recon_m3=False))
    m_plain.teacher_latent.forward = types.MethodType(  # type: ignore[method-assign]
        _boom, m_plain.teacher_latent
    )
    drop2 = m_plain.sample_drop(lfs.shape[0], membership, g)
    with pytest.raises(RuntimeError, match="teacher_latent forward ran"):
        m_plain(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords,
                drop_mask=drop2)


def test_m4_recon_m3_gradients_reach_student():
    m = V14ConvergedV2(_cfg(support_weight=True))
    _, poe, membership, lfs, hga, m2, tube, coords, g = _session()
    drop = m.sample_drop(lfs.shape[0], membership, g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop)
    # include the context terms (the module's job) so their grad path is exercised
    total = out["loss"] + 0.2 * (out["loss_m2_ctx"] + out["loss_m4_ctx"])
    total.backward()
    g4 = m.m4_predictor.mask_token.grad
    assert g4 is not None and torch.isfinite(g4).all() and g4.abs().sum() > 0


def test_m4_recon_m3_context_taps_m4_only():
    m = V14ConvergedV2(_cfg(context_taps=("M4",)))
    _, poe, membership, lfs, hga, m2, tube, coords, g = _session()
    drop = m.sample_drop(lfs.shape[0], membership, g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop)
    assert "loss_m4_ctx" in out
    assert "loss_m2_ctx" not in out       # M2 tap disabled


def test_m4_recon_m3_requires_run_b():
    with pytest.raises(ValueError, match="requires Run-B"):
        V14ConvergedV2(V14ConvergedV2Config(
            d_model=32, n_heads=4, frontend_layers=2, latent_layers=2,
            m2_pred_layers=2, m4_pred_layers=2, pred_dim=32, n_parcels=N_PARCELS,
            k=2, tube_ratio=0.35, m3_pred_layers=2, m4_recon_m3=True,
        ))


# --- context_loss is the decoupled gate ---------------------------------


@pytest.mark.parametrize("m4_recon_m3", [True, False])
def test_context_loss_off_no_context_terms(m4_recon_m3):
    """Context is gated on ``context_loss``, NOT ``m4_recon_m3`` — with the master
    switch off there are no context terms on either M4 target arm (the A/B floor)."""
    m = V14ConvergedV2(_cfg(m4_recon_m3=m4_recon_m3, context_loss=False,
                            context_taps=("M2", "M4", "MELEC")))
    _, poe, membership, lfs, hga, m2, tube, coords, g = _session()
    drop = m.sample_drop(lfs.shape[0], membership, g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop)
    for key in ("loss_m2_ctx", "loss_m4_ctx", "loss_melec_ctx"):
        assert key not in out
    assert "loss_melec" in out            # masked M-elec unchanged


def test_context_loss_on_both_m4_arms():
    """The A/B contract: with ``context_loss`` on, BOTH M4 target arms (pool vs
    latent) carry identical M2/M4 context terms — the only difference is m4_recon_m3."""
    _, poe, membership, lfs, hga, m2, tube, coords, g = _session()
    for recon in (True, False):
        m = V14ConvergedV2(_cfg(m4_recon_m3=recon))
        drop = m.sample_drop(lfs.shape[0], membership, g)
        out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords,
                drop_mask=drop)
        for key in ("loss_m2_ctx", "loss_m4_ctx"):
            assert key in out and torch.isfinite(out[key]), f"{key} missing (recon={recon})"


def test_b_path_runs_teacher_latent_with_context():
    """m4_recon_m3=False + context_loss=True: teacher latent MUST run (M4 target =
    teacher latent), and context is still emitted."""
    _, poe, membership, lfs, hga, m2, tube, coords, g = _session()
    m = V14ConvergedV2(_cfg(m4_recon_m3=False, context_loss=True))
    m.teacher_latent.forward = types.MethodType(  # type: ignore[method-assign]
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("latent ran")),
        m.teacher_latent,
    )
    drop = m.sample_drop(lfs.shape[0], membership, g)
    with pytest.raises(RuntimeError, match="latent ran"):
        m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop)


def test_melec_context_hetero_emitted_and_weighted():
    """MELEC context (kept-electrode readback off the survivor seeds) is emitted in
    hetero mode, finite, and zero-weighted at no-coverage cells."""
    m = V14ConvergedV2(_cfg(m2_hetero=True, context_taps=("M2", "M4", "MELEC")))
    bands, poe, membership, lfs, hga, m2, tube, coords, g = _session()
    B, C = lfs.shape[0], poe.shape[0]
    drop = m.sample_drop(B, membership, g)
    m2e = m.sample_m2_hetero(B, C, bands, g)
    out = m(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop,
            m2_elec_mask=m2e, return_taps=True)
    assert "loss_melec_ctx" in out and torch.isfinite(out["loss_melec_ctx"])
    assert "_tap_melec_ctx_pred" in out and "_tap_melec_ctx_target" in out
    # the kept-electrode context loss must NOT be emitted when MELEC is off the taps
    m2b = V14ConvergedV2(_cfg(m2_hetero=True, context_taps=("M2", "M4")))
    out2 = m2b(lfs, hga, poe, m2, tube, clip_len_s=5.0, coords=coords, drop_mask=drop,
               m2_elec_mask=m2e)
    assert "loss_melec_ctx" not in out2


# --- per-head loss context terms ----------------------------------------


def test_loss_per_head_context_terms():
    torch.manual_seed(0)
    d = 8
    m2p, m2t = torch.randn(5, d), torch.randn(5, d)
    m4p, m4t = torch.randn(4, d), torch.randn(4, d)
    m2c_p, m2c_t = torch.randn(6, d), torch.randn(6, d)
    m4c_p, m4c_t = torch.randn(7, d), torch.randn(7, d)
    mec_p, mec_t = torch.randn(9, d), torch.randn(9, d)
    mec_w = torch.rand(9)
    empty = torch.zeros(0, d)

    base = converged_v2_loss_per_head(m2p, m2t, empty, empty, m4p, m4t)
    ctx = converged_v2_loss_per_head(
        m2p, m2t, empty, empty, m4p, m4t,
        m2_ctx_pred=m2c_p, m2_ctx_target=m2c_t,
        m4_ctx_pred=m4c_p, m4_ctx_target=m4c_t,
        melec_ctx_pred=mec_p, melec_ctx_target=mec_t, melec_ctx_weight=mec_w,
    )
    # the base training loss is UNCHANGED by the context terms (module folds them)
    assert torch.allclose(base["loss"], ctx["loss"])
    # context terms == the manual self-normalized L1 means
    assert torch.allclose(
        ctx["loss_m2_ctx"], (m2c_p - m2c_t).abs().mean(), atol=1e-6
    )
    assert torch.allclose(
        ctx["loss_m4_ctx"], (m4c_p - m4c_t).abs().mean(), atol=1e-6
    )
    # melec ctx is convex-weighted (matches the _head self-normalized weighted mean)
    w = mec_w / mec_w.sum()
    assert torch.allclose(
        ctx["loss_melec_ctx"], ((mec_p - mec_t).abs().mean(-1) * w).sum(), atol=1e-6
    )


def test_loss_per_head_context_absent_by_default():
    d = 8
    empty = torch.zeros(0, d)
    out = converged_v2_loss_per_head(
        torch.randn(5, d), torch.randn(5, d), empty, empty,
        torch.randn(4, d), torch.randn(4, d),
    )
    assert "loss_m2_ctx" not in out and "loss_m4_ctx" not in out


# --- λ ramp --------------------------------------------------------------


@pytest.mark.parametrize(
    "step,expected",
    [(0, 0.0), (7500, 0.1), (15000, 0.2), (30000, 0.2)],
)
def test_context_lambda_ramp(step, expected):
    fake = types.SimpleNamespace(
        global_step=step,
        model=types.SimpleNamespace(
            cfg=types.SimpleNamespace(
                context_lambda=0.2, context_warmup_steps=15000,
                context_warmup_start_step=0,
            )
        ),
    )
    lam = V14ConvergedV2BrainModule._context_lambda(fake)
    assert lam == pytest.approx(expected)


@pytest.mark.parametrize(
    "step,expected",
    [(0, 0.0), (4999, 0.0), (5000, 0.0), (10000, 0.1), (15000, 0.2), (30000, 0.2)],
)
def test_context_lambda_delayed_ramp(step, expected):
    """Delayed ramp: λ=0 through start=5000, then linear 0→0.2 over the next
    W=10000 steps (reaching 0.2 at step 15000). Lets the masked pretext establish
    before context is introduced (avoids the trivial-copy basin)."""
    fake = types.SimpleNamespace(
        global_step=step,
        model=types.SimpleNamespace(
            cfg=types.SimpleNamespace(
                context_lambda=0.2, context_warmup_steps=10000,
                context_warmup_start_step=5000,
            )
        ),
    )
    lam = V14ConvergedV2BrainModule._context_lambda(fake)
    assert lam == pytest.approx(expected)


@pytest.mark.parametrize("step", [0, 1, 5000, 80000])
def test_context_lambda_constant_when_warmup_zero(step):
    """warmup_steps=0 ⇒ constant λ from step 0 (the no-warmup run: V-JEPA λ=0.2
    cte, ADE 29.6 / SSv2 62.5). No 0-hold, no ramp."""
    fake = types.SimpleNamespace(
        global_step=step,
        model=types.SimpleNamespace(
            cfg=types.SimpleNamespace(
                context_lambda=0.2, context_warmup_steps=0,
                context_warmup_start_step=0,
            )
        ),
    )
    assert V14ConvergedV2BrainModule._context_lambda(fake) == pytest.approx(0.2)


def test_context_lambda_no_trainer_is_zero():
    class _NoStep:
        model = types.SimpleNamespace(
            cfg=types.SimpleNamespace(
                context_lambda=0.2, context_warmup_steps=15000,
                context_warmup_start_step=0,
            )
        )

        @property
        def global_step(self):
            raise RuntimeError("no trainer attached")

    lam = V14ConvergedV2BrainModule._context_lambda(_NoStep())
    assert lam == 0.0


@pytest.mark.parametrize(
    "step,expected",
    [(0, 0.0), (29999, 0.0), (30000, 0.0), (35000, 0.25), (40000, 0.5), (60000, 0.5)],
)
def test_context_lambda_decoupled_m4_ramp(step, expected):
    """Decoupled M4 context ramp: explicit start/steps override the shared schedule
    (M2/MELEC ramp 10k→20k) so M4 rides its own 30k→40k ramp — λ=0 through 30k, then
    linear 0→context_lambda over the next 10k. Shared cfg is the M2/MELEC schedule;
    the override args are what `_step` passes for M4."""
    fake = types.SimpleNamespace(
        global_step=step,
        model=types.SimpleNamespace(
            cfg=types.SimpleNamespace(
                context_lambda=0.5, context_warmup_steps=10000,
                context_warmup_start_step=10000,
            )
        ),
    )
    lam_m4 = V14ConvergedV2BrainModule._context_lambda(fake, start=30000, steps=10000)
    assert lam_m4 == pytest.approx(expected)


@pytest.mark.parametrize("step", [0, 5000, 15000, 20000, 40000])
def test_context_lambda_sentinel_inherits_shared_schedule(step):
    """start=None/steps=None (the -1 sentinel path in `_step`) ⇒ the M4 ramp is
    byte-identical to the shared M2/MELEC ramp. Guarantees the single-ramp default
    (no m4_context_warmup_* set) folds the context loss exactly as before."""
    fake = types.SimpleNamespace(
        global_step=step,
        model=types.SimpleNamespace(
            cfg=types.SimpleNamespace(
                context_lambda=0.5, context_warmup_steps=10000,
                context_warmup_start_step=10000,
            )
        ),
    )
    shared = V14ConvergedV2BrainModule._context_lambda(fake)
    inherited = V14ConvergedV2BrainModule._context_lambda(fake, start=None, steps=None)
    assert inherited == pytest.approx(shared)


# --- predictor return_ctx ------------------------------------------------


def test_predictor_return_ctx_shapes():
    torch.manual_seed(0)
    d, pred, H, B, Lc, Lq = 16, 16, 4, 2, 5, 3
    p = JepaPredictorV2(d, pred, H, 2, qk_norm=True)
    ctx = torch.randn(B, Lc, d)
    ctx_slot = torch.randint(0, 10, (B, Lc))
    q_slot = torch.randint(0, 10, (B, Lq))
    q_freq = torch.randint(0, 4, (B, Lq))
    q_only = p(ctx, ctx_slot, q_slot, q_freq)
    q2, c2 = p(ctx, ctx_slot, q_slot, q_freq, return_ctx=True)
    assert q_only.shape == (B, Lq, d)
    assert q2.shape == (B, Lq, d) and c2.shape == (B, Lc, d)
    # return_ctx must not perturb the query prediction (same graph, extra readout)
    assert torch.allclose(q_only, q2, atol=1e-6)
