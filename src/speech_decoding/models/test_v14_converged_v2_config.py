"""Guards for the pydantic ``V14ConvergedV2Net`` → ``V14ConvergedV2Config`` seam.

The dispatch builds a ``brain_model_config`` dict and hands it to the pydantic
``V14ConvergedV2Net`` (``extra_forbidden``), whose ``build()`` threads each field
into the dataclass ``V14ConvergedV2Config``. A field added to the dataclass + the
dispatch dict but NOT mirrored here fails at validation time (extra key) or silently
drops (missing passthrough) — neither is covered by the model unit tests, which
construct the dataclass directly. These tests close that gap.
"""

from __future__ import annotations

import dataclasses

import torch

from speech_decoding.models.v14_converged_v2 import V14ConvergedV2Config
from speech_decoding.models.v14_converged_v2_config import V14ConvergedV2Net

_NET_KW = dict(
    d_model=32, n_heads=4, frontend_layers=2, latent_layers=2,
    m2_pred_layers=2, m4_pred_layers=2, pred_dim=32, n_parcels=62, k=2,
    tube_ratio=0.35, qk_norm=True, m3_drop_frac=0.5, m3_min_keep=3, w_melec=1.0,
    sigma_mm=12.0, geom_n_freqs=5, support_weight=True, m2_hetero=True,
)


def test_net_accepts_and_threads_context_fields():
    net = V14ConvergedV2Net(
        **_NET_KW, m4_recon_m3=True, context_lambda=0.2, context_warmup_steps=0,
        context_loss=True, context_taps=("M2", "M4", "MELEC"),
    )
    cfg = net.build(1, 1).cfg
    assert cfg.context_loss is True
    assert cfg.context_taps == ("M2", "M4", "MELEC")
    assert cfg.m4_recon_m3 is True
    assert cfg.context_lambda == 0.2 and cfg.context_warmup_steps == 0


def test_net_threads_vjepa_parity_flags():
    """target_ln + pred_ln must reach the built model AND the predictor modules —
    the dispatch hands these keys to the extra_forbidden Net, so a missing field or
    passthrough would raise at launch (or drop the fix). Guards that end-to-end seam."""
    net = V14ConvergedV2Net(**_NET_KW, target_ln=True, pred_ln=True)
    model = net.build(1, 1)
    assert model.cfg.target_ln is True
    assert model.cfg.pred_ln is True
    assert model.m2_predictor.pred_norm is not None
    assert model.m4_predictor.pred_norm is not None
    # pred_ln also bundles the V-JEPA predictor-tail projection biases
    assert model.m2_predictor.ctx_proj.bias is not None
    assert model.m2_predictor.head.bias is not None


def test_net_parity_flags_default_off():
    net = V14ConvergedV2Net(**_NET_KW)
    model = net.build(1, 1)
    assert model.cfg.target_ln is False and model.cfg.pred_ln is False
    assert model.m2_predictor.pred_norm is None


def test_seed_offset_sigma_default_is_zero_init():
    """σ=0 (default) ⇒ pool_seed_offset all zeros (coincident at centroid), learnable."""
    model = V14ConvergedV2Net(**_NET_KW).build(1, 1)
    off = model.pool_seed_offset                              # model-level param
    assert off.shape == (_NET_KW["k"], 3)
    assert off.requires_grad
    assert torch.count_nonzero(off) == 0


def test_seed_offset_sigma_isotropic_spread():
    """σ>0 ⇒ isotropic-Gaussian spread offsets: non-zero, k distinct rows (symmetry
    broken), learnable, and spread scale tracks σ (a single shared model-level param
    drives both student + teacher pooling)."""
    torch.manual_seed(33)
    model = V14ConvergedV2Net(**_NET_KW, seed_offset_sigma_mm=5.0).build(1, 1)
    off = model.pool_seed_offset
    assert off.shape == (_NET_KW["k"], 3) and off.requires_grad
    assert torch.count_nonzero(off) > 0                       # not zero-init
    assert not torch.allclose(off[0], off[1])                 # k seeds distinct
    # per-axis std is O(σ) — a spread, not a degenerate near-zero draw
    assert 1.0 < off.std().item() < 15.0


def test_net_build_mirrors_every_shared_field():
    """Every field the Net names that also exists on the dataclass must reach the
    built cfg unchanged — catches a future field added to the Net but forgotten in
    ``build()`` (or vice-versa)."""
    net = V14ConvergedV2Net(**_NET_KW, m4_recon_m3=True, context_loss=True,
                            context_taps=("MELEC",))
    cfg = net.build(1, 1).cfg
    cfg_fields = {f.name for f in dataclasses.fields(V14ConvergedV2Config)}
    for name in type(net).model_fields:
        if name in cfg_fields:
            assert getattr(cfg, name) == getattr(net, name), f"{name} not threaded"
